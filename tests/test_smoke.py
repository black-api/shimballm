"""End-to-end smoke test: the full CROP procedure on a tiny CPU run.

Phase 0 scoring -> Stage A (doc gate) -> Stage B (token replay + context
ramp) -> Stage C (anneal) -> stable checkpoint -> re-anneal branch.
"""

import json
import os

import numpy as np
import pytest
import torch


from shimballm.config import RunConfig
from shimballm.engine.recycling import AnnealBranchManager
from shimballm.engine.trainer import CROPTrainer


def _tiny_cfg(tmp_path, **over) -> RunConfig:
    cfg = RunConfig()
    cfg.model.d_model = 48
    cfg.model.n_layers = 3
    cfg.model.n_heads = 3
    cfg.model.vocab_size = 96
    cfg.model.max_seq_len = 128
    cfg.ref.d_model = 24
    cfg.ref.n_layers = 2
    cfg.ref.n_heads = 2
    cfg.data.synthetic_docs = 512
    cfg.data.synthetic_doc_len = 128
    cfg.engine.out_dir = str(tmp_path / "run")
    cfg.engine.eval_every = 10
    cfg.engine.eval_batches = 2
    cfg.engine.log_every = 5
    cfg.schedule.total_tokens = 400_000
    cfg.schedule.batch_size = 16
    cfg.schedule.context_start = 32
    cfg.schedule.context_end = 64
    cfg.schedule.peak_lr = 3e-3
    cfg.crop.gate_gamma = -1.0  # effectively disable early firing; run to budget
    for k, v in over.items():
        section, key = k.split("__")
        setattr(getattr(cfg, section), key, v)
    cfg.validate()
    return cfg


def test_full_crop_run(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    trainer = CROPTrainer(cfg)
    summary = trainer.train()

    # artifacts
    out = cfg.engine.out_dir
    assert os.path.exists(f"{out}/final.pt")
    assert os.path.exists(f"{out}/stable.pt")
    assert os.path.exists(f"{out}/summary.json")
    assert os.path.exists(f"{out}/log.jsonl")
    assert os.path.exists(f"{out}/cache/doc_scores.npy")

    with open(f"{out}/summary.json") as f:
        s = json.load(f)
    assert s["val"]["student_ce"] == pytest.approx(summary["val"]["student_ce"])
    assert np.isfinite(s["val"]["student_ce"])
    assert s["tokens_total"] > 0
    assert s["flops"]["flops_total"] > 0
    # Phase 0 ran and was cached
    assert s["phase0"]["scoring_tokens"] > 0
    # all three stages executed with sensible budgets
    stages = {e["stage"]: e for e in s["stage_log"]}
    assert set(stages) == {"A", "B", "C"}
    assert stages["A"]["consumed_tokens"] > 0
    assert stages["C"]["consumed_tokens"] > 0
    # FLOPs vs uniform baseline is reported
    assert 0.1 < s["flops_vs_baseline"] < 2.0
    # masked replay actually masked something in stage B
    assert s["flops"]["tokens_masked"] > 0


def test_scoring_is_deterministic_given_seed(tmp_path):
    """Phase 0 doc scores are a deterministic function of (corpus, ref)."""
    cfg = _tiny_cfg(tmp_path / "sc")
    t1 = CROPTrainer(cfg)
    s1 = t1.doc_scores.copy()
    t2 = CROPTrainer(_tiny_cfg(tmp_path / "sc2"))
    s2 = t2.doc_scores.copy()
    assert np.allclose(s1, s2)


def test_reanneal_branch(tmp_path):
    cfg = _tiny_cfg(tmp_path / "recycle")
    trainer = CROPTrainer(cfg)
    trainer.train()
    mgr = AnnealBranchManager(cfg)
    summary = mgr.run_branch(branch_seed=1, tag="b1")
    assert os.path.exists(f"{cfg.engine.out_dir}/branches/b1/branch.pt")
    assert np.isfinite(summary["val"]["student_ce"])
    # branch ran Stage C only
    assert all(e["stage"] == "C" for e in summary["stage_log"])


def test_baseline_mode_degenerates_to_uniform_wsd(tmp_path):
    """B1 baseline: no gating, no ramp, no replay -- shares (0.85, 0, 0.15)."""
    cfg = _tiny_cfg(
        tmp_path / "baseline",
        crop__enabled=False,
        crop__doc_drop_quantile=0.0,
        crop__stage_b_doc_quantile=0.0,
        crop__stage_c_doc_quantile=0.0,
        crop__rho_start=0.0,
        crop__rho_end=0.0,
        crop__gate_enabled=False,
        schedule__context_start=64,   # == context_end: no ramp
    )
    trainer = CROPTrainer(cfg)
    summary = trainer.train()
    assert summary["flops"]["tokens_masked"] == 0
    ctxs = [e["ctx_end"] for e in summary["stage_log"]]
    assert all(c == 64 for c in ctxs)
