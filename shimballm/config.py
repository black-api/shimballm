"""Configuration for ShimbaLLM / CROP training runs.

Every knob maps to a piece of the CROP recipe:

  * ``CROPSettings``   -- the two new hyper-parameters CROP introduces
                          (doc gate quantile theta_doc, plateau slope gamma)
                          plus mask decay rho_0 -> rho_1 and tier cuts.
  * ``ScheduleConfig`` -- WSD schedule, stage shares, context ramp, budget D*.
  * ``RefConfig``      -- the small frozen reference model used for scoring.

Configs are plain YAML files; see ``examples/`` for complete templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List
import json
import os
import yaml


@dataclass
class ModelConfig:
    """Student model. Use ``arch="tiny"`` for the built-in TinyGPT, or set
    ``arch="hf"`` with ``hf_name`` to load any Hugging Face causal LM."""

    arch: str = "tiny"                # "tiny" | "hf"
    hf_name: Optional[str] = None     # e.g. "HuggingFaceTB/SmolLM2-135M"
    vocab_size: int = 96
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: Optional[int] = None  # GQA when < n_heads
    ffn_mult: float = 4.0
    use_swiglu: bool = False
    tie_embeddings: bool = True
    dropout: float = 0.0
    max_seq_len: int = 512
    init_std: float = 0.02

    def n_params_estimate(self, vocab_size: Optional[int] = None) -> int:
        """Analytic dense-transformer parameter count (embedding-tied)."""
        v = vocab_size or self.vocab_size
        d = self.d_model
        ffn = int(self.ffn_mult * d)
        per_layer = 4 * d * d + (3 * d * ffn if self.use_swiglu else 2 * d * ffn)
        return self.n_layers * per_layer + v * d + 2 * d * self.n_layers


@dataclass
class RefConfig:
    """Small frozen reference model theta_ref used to score tokens/documents.

    The paper uses a 0.3B model sharing the student tokenizer; scoring cost is
    a forward-only pass (2*N_ref per token) over ``corpus_inflation`` x the
    token budget (1.4x in the projections)."""

    arch: str = "tiny"                # "tiny" | "hf"
    hf_name: Optional[str] = None
    vocab_size: int = 96
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    n_kv_heads: Optional[int] = None
    ffn_mult: float = 4.0
    use_swiglu: bool = False
    max_seq_len: int = 512
    # --- accounting only (used by the FLOPs projector) ---
    params_target: float = 0.3e9
    corpus_inflation: float = 1.4
    batch_size: int = 8
    score_max_docs: int = 0           # 0 = score the whole corpus


@dataclass
class DataConfig:
    corpus: str = "synthetic"         # path to .jsonl/.txt/.npy docs, or "synthetic"
    val_corpus: Optional[str] = None
    tokenizer: str = "char"           # "char" | HF tokenizer name/path
    val_fraction: float = 0.02
    seed: int = 0
    # synthetic-corpus knobs (used only when corpus == "synthetic")
    synthetic_docs: int = 2048
    synthetic_doc_len: int = 192
    synthetic_vocab_words: int = 512
    cache_dir: str = ""               # where doc-score caches are written


@dataclass
class CROPSettings:
    """The CROP curriculum. Defaults follow the paper's projections.

    Stage A: keep documents with mean token score >= the ``doc_drop_quantile``
    percentile (30th in the paper); full next-token loss at ``context_start``.
    Stage B: replay the upper score tiers with the masked loss of Eq. (5);
    the easiest-token share decays linearly from ``rho_start`` to ``rho_end``;
    context ramps linearly to ``context_end``. Stage C: anneal on the top
    ``(1 - stage_c_doc_quantile)`` slice of documents."""

    enabled: bool = True
    doc_drop_quantile: float = 0.30   # theta_doc: drop bottom 30% of documents
    stage_b_doc_quantile: float = 0.50  # Stage B replays the upper half of tiers
    stage_c_doc_quantile: float = 0.80  # Stage C anneals on the top 20% slice
    rho_start: float = 0.40           # rho_0: easiest-token share masked at B start
    rho_end: float = 0.10             # rho_1: easiest-token share masked at B end
    # Adaptive stage gating (Eq. 6): advance when the val excess-loss slope
    # drops below gamma for ``patience`` consecutive eval windows.
    gate_enabled: bool = True
    gate_gamma: float = 1.0e-5        # loss slope per optimizer step
    gate_patience: int = 2
    min_stage_fraction: float = 0.25  # never advance before this share of a stage's budget


@dataclass
class ScheduleConfig:
    """WSD schedule, stage shares, context ramp and the token budget D*."""

    total_tokens: int = 1_200_000     # D*: token budget of the run
    stage_shares: Tuple[float, float, float] = (0.55, 0.30, 0.15)  # A / B / C
    warmup_fraction: float = 0.05     # share of the *total* budget spent warming up
    peak_lr: float = 3.0e-3
    end_lr_fraction: float = 0.1      # WSD decays to 10% of peak inside Stage C
    context_start: int = 64           # paper: 2K
    context_end: int = 128            # paper: 4K
    batch_size: int = 32              # packed sequences per optimizer step
    grad_accum: int = 1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    optimizer: str = "adamw"          # "adamw" | "muon"
    muon_momentum: float = 0.95
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    amp_dtype: str = "none"           # "bf16" | "fp16" | "none"
    shuffle_seed: int = 1234


@dataclass
class EngineConfig:
    out_dir: str = "runs/demo"
    eval_every: int = 25              # optimizer steps between evaluations
    eval_batches: int = 4
    log_every: int = 10
    seed: int = 0
    device: str = "auto"              # "auto" | "cpu" | "cuda"
    save_stable_ckpt: bool = True     # Stage-B end checkpoint (recycling asset)
    save_last_ckpt: bool = True
    resume: bool = False
    num_workers: int = 0


@dataclass
class HubConfig:
    """Hugging Face Hub targets. Model cards reference the ShimbaLLM org."""

    org: str = "ShimbaLLM"
    repo_name: Optional[str] = None   # defaults to out_dir basename
    push_after_train: bool = False
    private: bool = False


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    data: DataConfig = field(default_factory=DataConfig)
    crop: CROPSettings = field(default_factory=CROPSettings)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    hub: HubConfig = field(default_factory=HubConfig)

    # ------------------------------------------------------------------ IO
    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        known = {f for f in cls.__dataclass_fields__}
        top = {k: v for k, v in raw.items() if k in known}
        cfg = cls()
        for section, value in top.items():
            sub = getattr(cfg, section)
            fields = sub.__dataclass_fields__
            filtered = {}
            for k, v in (value or {}).items():
                if k not in fields:
                    raise KeyError(f"Unknown key '{k}' in section '{section}' of {path}")
                # YAML gives lists for tuples
                if isinstance(fields[k].type, str) and "Tuple" in fields[k].type and isinstance(v, list):
                    v = tuple(v)
                filtered[k] = v
            for k, v in filtered.items():
                setattr(sub, k, v)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        a, b, c = self.schedule.stage_shares
        total = a + b + c
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"stage_shares must sum to 1.0 (got {total:.6f})")
        if not (0.0 <= self.crop.rho_end <= self.crop.rho_start <= 1.0):
            raise ValueError("require 0 <= rho_end <= rho_start <= 1")
        if self.schedule.context_start > self.schedule.context_end:
            raise ValueError("context_start must be <= context_end")
        if self.crop.min_stage_fraction >= 1.0:
            raise ValueError("min_stage_fraction must be < 1.0")
        if self.schedule.context_end > self.model.max_seq_len and self.model.arch == "tiny":
            raise ValueError("model.max_seq_len must cover context_end")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(json.loads(json.dumps(self.to_dict())), f, sort_keys=False)

    def repo_id(self) -> str:
        name = self.hub.repo_name or os.path.basename(os.path.normpath(self.engine.out_dir))
        return f"{self.hub.org}/{name}"

    # Paper-projection helpers -------------------------------------------------
    def stage_budgets(self) -> Tuple[int, int, int]:
        """Planned token budgets (A, B, C) from the total budget and shares."""
        t = int(self.schedule.total_tokens)
        a, b, c = self.schedule.stage_shares
        return int(t * a), int(t * b), int(t * c)
