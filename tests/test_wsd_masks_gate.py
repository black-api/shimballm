"""WSD schedule + replay masks + adaptive gate unit tests."""

import pytest
import torch

from shimballm.optim.lr import WSDLRSchedule
from shimballm.engine.masks import ReplayMaskScheduler, masked_ce_loss
from shimballm.engine.gates import ExcessLossGate


# --------------------------------------------------------------------------- #
# WSD schedule
# --------------------------------------------------------------------------- #
def test_wsd_phases():
    s = WSDLRSchedule(peak_lr=1.0, end_lr_fraction=0.1, warmup_fraction=0.1,
                      total_tokens=1000)
    s.set_stage_boundaries(a_end=550, b_end=850, total=1000)
    assert s.lr_at(0) < s.peak_lr                      # warming up
    assert s.lr_at(100) == pytest.approx(1.0)          # warmup done at 10% = 100
    assert s.lr_at(500) == pytest.approx(1.0)          # stable through B
    mid = s.lr_at(925)                                  # halfway through decay
    assert 0.1 < mid < 1.0
    assert s.lr_at(1000) == pytest.approx(0.1)         # decays to end fraction
    # monotone inside decay
    assert s.lr_at(860) > s.lr_at(950)


def test_wsd_boundaries_shift_with_gate():
    s = WSDLRSchedule(peak_lr=1.0, end_lr_fraction=0.1, warmup_fraction=0.05,
                      total_tokens=1000)
    s.set_stage_boundaries(a_end=300, b_end=700, total=1000)
    assert s._decay_start == 700
    s.set_stage_boundaries(a_end=150, b_end=500, total=1000)  # gate fired early
    assert s._decay_start == 500
    assert s._decay_end == 1000


# --------------------------------------------------------------------------- #
# Replay masking (Eq. 5)
# --------------------------------------------------------------------------- #
def test_rho_decays_linearly():
    r = ReplayMaskScheduler(rho_start=0.4, rho_end=0.1, stage_budget_tokens=1000)
    assert r.rho_at(0) == 0.4
    assert r.rho_at(500) == pytest.approx(0.25)
    assert r.rho_at(1000) == pytest.approx(0.1)
    assert r.rho_at(5000) == pytest.approx(0.1)  # clamped


def test_mask_keeps_top_scores():
    scores = torch.arange(100, dtype=torch.float32).reshape(1, 100)
    w = ReplayMaskScheduler(0.4, 0.1, 1000).mask_from_scores(scores, rho=0.4)
    kept = w.sum().item()
    assert kept == pytest.approx(60, abs=3)      # ~1 - rho fraction kept
    assert w[0, 0].item() == 0.0                 # easiest token masked
    assert w[0, -1].item() == 1.0                # hardest token kept
    # kept tokens are exactly the top scores
    threshold = torch.quantile(scores.reshape(-1), 0.4)
    assert torch.equal(w.bool(), scores > threshold)


def test_masked_ce_loss_weights():
    torch.manual_seed(0)
    V = 7
    logits = torch.randn(2, 5, V)
    targets = torch.randint(0, V, (2, 5))
    w = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 1, 1]], dtype=torch.float32)
    loss, ce, kept = masked_ce_loss(logits, targets, w)
    assert kept == 4
    manual = 0.0
    import torch.nn.functional as F

    ce_all = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none")
    manual = (ce_all.reshape(2, 5) * w).sum() / 4
    assert loss.item() == pytest.approx(manual.item(), rel=1e-5)


def test_degenerate_mask_falls_back():
    scores = torch.zeros(4, 8)  # all equal -> quantile keeps nothing
    w = ReplayMaskScheduler(0.4, 0.1, 100).mask_from_scores(scores, rho=0.4)
    assert w.sum().item() == 4 * 8  # fallback keeps everything


# --------------------------------------------------------------------------- #
# Adaptive stage gating (Eq. 6)
# --------------------------------------------------------------------------- #
def test_gate_fires_on_plateau():
    g = ExcessLossGate(gamma=1e-5, window_points=4, patience=2)
    vals = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5]  # flat after initial drop
    fired = False
    for i, v in enumerate(vals):
        g.record(i * 100, v)
        fired, reason = g.should_advance(500, 1000, min_stage_fraction=0.25)
        if fired:
            break
    assert fired


def test_gate_fires_when_excess_rises():
    g = ExcessLossGate(gamma=1e-5, window_points=4, patience=1)
    vals = [1.0, 1.2, 1.4, 1.6]  # excess growing = negative improvement
    fired = False
    for i, v in enumerate(vals):
        g.record(i * 100, v)
        fired, _ = g.should_advance(500, 1000, min_stage_fraction=0.25)
        if fired:
            break
    assert fired


def test_gate_respects_min_stage_fraction():
    g = ExcessLossGate(gamma=1e-5, window_points=4, patience=1)
    for v in [1.0, 0.5, 0.0, 0.0]:
        g.record(0, v)
    fire, reason = g.should_advance(10, 1000, min_stage_fraction=0.25)
    assert not fire and reason == "min-stage-not-reached"


def test_gate_not_enough_evals():
    g = ExcessLossGate(gamma=0.0, window_points=4, patience=1)
    g.record(0, 1.0)
    fire, reason = g.should_advance(900, 1000, min_stage_fraction=0.25)
    assert not fire and reason == "not-enough-evals"


def test_gate_improving_does_not_fire():
    g = ExcessLossGate(gamma=1e-5, window_points=4, patience=2)
    fired = False
    for i in range(8):
        g.record(i * 100, 1.0 - 0.1 * i)  # steadily improving
        fired, _ = g.should_advance(500, 1000, min_stage_fraction=0.25)
    assert not fired
