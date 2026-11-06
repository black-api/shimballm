"""Compute recycling (Stage C fan-out).

Because the WSD schedule places the decay entirely inside Stage C, the
Stage-B stable checkpoint is a reusable asset: reloading it and re-annealing
on a different top-quality slice yields an additional model for the cost of
Stage C alone (15% of the budget in the paper's projections). This is the
H3 protocol: a re-anneal should recover within ~1 MMLU point of a full
independent WSD run at the same added compute.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from ..config import RunConfig
from .trainer import CROPTrainer


class AnnealBranchManager:
    """Fans the Stage-B stable checkpoint out into N annealed branches."""

    def __init__(self, cfg: RunConfig, stable_ckpt: Optional[str] = None):
        self.cfg = cfg
        self.stable_ckpt = stable_ckpt or os.path.join(cfg.engine.out_dir, "stable.pt")
        if not os.path.exists(self.stable_ckpt):
            raise FileNotFoundError(
                f"stable checkpoint not found at {self.stable_ckpt}. "
                "Run a CROP training first (it is written at the Stage B -> C transition)."
            )
        self.branches_dir = os.path.join(cfg.engine.out_dir, "branches")
        os.makedirs(self.branches_dir, exist_ok=True)
        self.results: List[Dict] = []

    def run_branch(self, branch_seed: int = 0, budget_tokens: Optional[int] = None,
                   tag: Optional[str] = None) -> Dict:
        tag = tag or f"seed{branch_seed}"
        cfg = self._branch_cfg(budget_tokens, branch_seed)
        out_dir = os.path.join(self.branches_dir, tag)
        cfg.engine.out_dir = out_dir
        trainer = CROPTrainer(
            cfg,
            load_model_from=self.stable_ckpt,
            branch_budget_tokens=budget_tokens,
            branch_seed=branch_seed,
            tag=tag,
        )
        summary = trainer.train()
        torch.save({"model": trainer.model.state_dict(), "summary": summary},
                   os.path.join(out_dir, "branch.pt"))
        self.results.append({"tag": tag, "seed": branch_seed,
                             "val_student_ce": summary["val"]["student_ce"]})
        return summary

    def _branch_cfg(self, budget_tokens: Optional[int], seed: int) -> RunConfig:
        import copy

        cfg = copy.deepcopy(self.cfg)
        cfg.engine.out_dir = self.cfg.engine.out_dir  # corrected per-branch in run_branch
        cfg.engine.resume = False
        cfg.engine.seed = seed
        cfg.data.seed = seed
        # reuse the parent run's Phase-0 doc-score cache (scoring is paid once)
        cfg.data.cache_dir = cfg.data.cache_dir or os.path.join(
            self.cfg.engine.out_dir, "cache")
        # the branch only anneals; disable gate (Stage C always runs to budget)
        cfg.crop.gate_enabled = False
        if budget_tokens is not None:
            cfg.schedule.total_tokens = budget_tokens
        return cfg

    def best(self) -> Optional[Dict]:
        if not self.results:
            return None
        return min(self.results, key=lambda r: r["val_student_ce"])
