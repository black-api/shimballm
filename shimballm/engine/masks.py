"""Stage B: selective token replay (Rho-1 style masked loss).

Implements Eq. (5) of the paper:

    L_rho(theta) = sum_i w_i * l_i / sum_i w_i,
    w_i = 1[ s(x_i) > q_rho(t) ]

where s(x_i) is the frozen reference model's per-token loss and q_rho(t) is
the rho-quantile of the batch's scores. The easiest-token share rho(t) decays
linearly from rho_0 to rho_1 across Stage B: the mask tightens as the student
improves, then relaxes toward the end to re-anchor on the full distribution.

Masked tokens still flow through the network (so raw FLOPs are unchanged) but
contribute no gradient -- the learning-efficiency lever that shifts the
loss-vs-FLOPs curve downward.
"""

from __future__ import annotations

from typing import Tuple
import torch


class ReplayMaskScheduler:
    """Linear decay of the masked-easiest-token share rho(t) across Stage B."""

    def __init__(self, rho_start: float, rho_end: float, stage_budget_tokens: int):
        assert 0.0 <= rho_end <= rho_start <= 1.0
        self.rho_start = rho_start
        self.rho_end = rho_end
        self.stage_budget_tokens = max(int(stage_budget_tokens), 1)

    def rho_at(self, tokens_into_stage: int) -> float:
        frac = min(max(tokens_into_stage / self.stage_budget_tokens, 0.0), 1.0)
        return self.rho_start + (self.rho_end - self.rho_start) * frac

    def mask_from_scores(self, token_scores: torch.Tensor, rho: float,
                         min_keep_fraction: float = 0.05) -> torch.Tensor:
        """Binary weights w_i for one batch of token scores [B, T].

        Falls back to keeping all tokens if the quantile cut would keep fewer
        than ``min_keep_fraction`` of positions (degenerate batches, e.g. all
        padding or all-equal scores)."""
        if rho <= 0.0 or token_scores.numel() == 0:
            return torch.ones_like(token_scores)
        flat = token_scores.reshape(-1).float()
        q = torch.quantile(flat, rho)
        w = (token_scores > q)
        if w.float().mean() < min_keep_fraction:
            return torch.ones_like(token_scores)
        return w


def masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Cross-entropy averaged over kept tokens only.

    Args:
        logits: [B, T, V]
        targets: [B, T]
        weights: [B, T] in {0, 1} (float or bool)

    Returns:
        (loss, per_token_ce, n_kept)
    """
    import torch.nn.functional as F

    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    w = weights.to(ce.dtype)
    denom = w.sum()
    if denom.item() < 1:  # degenerate: keep everything (logged by caller)
        w = torch.ones_like(w)
        denom = w.sum()
    loss = (ce * w).sum() / (denom + eps)
    return loss, ce, int(denom.item())
