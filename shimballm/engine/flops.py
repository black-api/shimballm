"""FLOPs accounting.

Training compute follows the paper's Equations (1)-(2) (Kaplan et al. 2020):

    per-token forward at context l:  2 * (N + L_layers * l * d)
    training cost:                   6 * D * (N + L_layers * l * d)

The ``project_crop`` function reproduces the paper's Table 1 projection
(1.5B student / 100B-token budget -> 88.0% conservative, 72.1% under H2);
``tests/test_flops.py`` asserts those exact numbers so the accounting stays
falsifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


def per_token_flops(n_params: float, n_layers: int, d_model: int, ctx: int) -> float:
    """Forward FLOPs per token at context ``ctx`` (Eq. 1 + attention term)."""
    return 2.0 * (n_params + n_layers * ctx * d_model)


def train_flops(n_tokens: float, n_params: float, n_layers: int, d_model: int,
                ctx: int) -> float:
    """Forward+backward training FLOPs for ``n_tokens`` at context ``ctx``."""
    return 3.0 * n_tokens * per_token_flops(n_params, n_layers, d_model, ctx)


def format_flops(x: float) -> str:
    for exp, suffix in ((21, "e21"), (18, "e18"), (15, "e15"), (12, "e12"), (9, "e9")):
        if abs(x) >= 10.0 ** exp:
            return f"{x / 10.0 ** exp:.2f}{suffix}"
    return f"{x:.3e}"


@dataclass
class FLOPsMeter:
    """Live FLOPs accounting during a run.

    Counts the student's forward+backward cost at the *actual* context length
    of each batch, the reference model's forward-only scoring cost, and the
    token count split into trained vs masked (masked tokens cost compute but
    contribute no gradient -- the distinction between Stage A and Stage B).
    """

    n_params: float
    n_layers: int
    d_model: int
    ref_n_params: float = 0.0
    budget_tokens: int = 0
    tokens_trained: int = 0       # optimizer-processed positions (grad flows)
    tokens_masked: int = 0        # positions whose loss weight was 0
    ref_tokens_scored: int = 0
    _flops_student: float = 0.0
    _flops_ref: float = 0.0

    def add_student_step(self, batch_tokens: int, ctx: int, n_masked: int = 0) -> None:
        self._flops_student += train_flops(batch_tokens, self.n_params, self.n_layers, self.d_model, ctx)
        self.tokens_trained += batch_tokens
        self.tokens_masked += n_masked

    def add_ref_forward(self, batch_tokens: int, ctx: int, n_params: Optional[float] = None) -> None:
        n = self.ref_n_params if n_params is None else n_params
        self._flops_ref += 2.0 * n * batch_tokens  # forward-only per token
        self.ref_tokens_scored += batch_tokens

    @property
    def flops_student(self) -> float:
        return self._flops_student

    @property
    def flops_ref(self) -> float:
        return self._flops_ref

    @property
    def flops_total(self) -> float:
        return self._flops_student + self._flops_ref

    def summary(self) -> Dict[str, float]:
        eff = self.tokens_trained - self.tokens_masked
        return {
            "tokens_trained": float(self.tokens_trained),
            "tokens_masked": float(self.tokens_masked),
            "tokens_effective": float(eff),
            "masked_fraction": (self.tokens_masked / self.tokens_trained) if self.tokens_trained else 0.0,
            "ref_tokens_scored": float(self.ref_tokens_scored),
            "flops_student": self._flops_student,
            "flops_ref": self._flops_ref,
            "flops_total": self.flops_total,
        }

    def state_dict(self) -> dict:
        return {
            "tokens_trained": self.tokens_trained,
            "tokens_masked": self.tokens_masked,
            "ref_tokens_scored": self.ref_tokens_scored,
            "flops_student": self._flops_student,
            "flops_ref": self._flops_ref,
        }

    def load_state_dict(self, st: dict) -> None:
        self.tokens_trained = int(st["tokens_trained"])
        self.tokens_masked = int(st["tokens_masked"])
        self.ref_tokens_scored = int(st["ref_tokens_scored"])
        self._flops_student = float(st["flops_student"])
        self._flops_ref = float(st["flops_ref"])


# --------------------------------------------------------------------------- #
# Paper Table 1 projection
# --------------------------------------------------------------------------- #
def project_crop(
    n_params: float = 1.5e9,
    d_model: int = 2048,
    n_layers: int = 28,
    budget_tokens: float = 100e9,
    stage_shares: Tuple[float, float, float] = (0.55, 0.30, 0.15),
    ctx_start: int = 2000,
    ctx_end: int = 4000,
    doc_drop: float = 0.30,
    n_ref: float = 0.3e9,
    corpus_inflation: float = 1.4,
    h2_token_shift: float = 0.20,
) -> Dict[str, float]:
    """Reproduce the paper's compute analysis (Table 1) from Eq. (1)-(2).

    Stage contexts: A at ``ctx_start``; B at the ramp midpoint
    ``(ctx_start + ctx_end) / 2``; C at ``ctx_end`` (matches the paper's
    "Stage-B context ramp whose midpoint is 3K").

    Attribution convention (same as the paper's narrative):
      * doc-gating saving = dropped share x Stage-A budget x 6(N + L*l_A*d)
      * context-ramp saving = uniform-4K cost - the same budget at 2K
    """
    sa, sb, sc = stage_shares
    ctx_b = (ctx_start + ctx_end) // 2

    def cost(tokens: float, ctx: int) -> float:
        return train_flops(tokens, n_params, n_layers, d_model, ctx)

    c_base = cost(budget_tokens, ctx_end)
    c_a = cost(sa * budget_tokens * (1 - doc_drop), ctx_start)
    c_b = cost(sb * budget_tokens, ctx_b)
    c_c = cost(sc * budget_tokens, ctx_end)
    c_train = c_a + c_b + c_c
    c_ref = 2.0 * n_ref * corpus_inflation * budget_tokens

    gating_saving = doc_drop * sa * cost(budget_tokens, ctx_start)
    ctx_saving = c_base - cost(budget_tokens, ctx_start)
    masking_saving = 0.0

    c_cons = c_train + c_ref
    c_h2 = (1.0 - h2_token_shift) * c_train + c_ref

    return {
        "baseline": c_base,
        "stage_A": c_a,
        "stage_B": c_b,
        "stage_C": c_c,
        "training_total": c_train,
        "reference_scoring": c_ref,
        "conservative_total": c_cons,
        "conservative_vs_baseline": c_cons / c_base,
        "h2_total": c_h2,
        "h2_vs_baseline": c_h2 / c_base,
        "doc_gating_saving": gating_saving,
        "doc_gating_saving_frac": gating_saving / c_base,
        "context_ramp_saving": ctx_saving,
        "context_ramp_saving_frac": ctx_saving / c_base,
        "token_masking_saving": masking_saving,
        "reference_overhead_frac": c_ref / c_base,
    }


def render_projection(p: Dict[str, float]) -> str:
    """Render a Table-1-style report."""
    lines = [
        "CROP compute projection (paper Table 1 reproduction)",
        "-" * 64,
        f"uniform baseline            {p['baseline']:.3e}   100.0%",
        f"document gating (30% drop)  -{p['doc_gating_saving']:.3e}   -{100 * p['doc_gating_saving_frac']:.1f}%",
        f"context ramp (2K->4K)       -{p['context_ramp_saving']:.3e}   -{100 * p['context_ramp_saving_frac']:.1f}%",
        f"token masking               {p['token_masking_saving']:.1e}    0.0%  (learning-efficiency lever)",
        f"reference scoring           +{p['reference_scoring']:.3e}   +{100 * p['reference_overhead_frac']:.1f}%",
        "-" * 64,
        f"CROP conservative           {p['conservative_total']:.3e}   {100 * p['conservative_vs_baseline']:.1f}%",
        f"CROP H2 scenario            {p['h2_total']:.3e}   {100 * p['h2_vs_baseline']:.1f}%",
    ]
    return "\n".join(lines)
