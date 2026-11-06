"""The compute accounting must stay falsifiable: these tests reproduce the
paper's Table 1 numbers from the engine's FLOPs formulas."""

import pytest

from shimballm.engine.flops import project_crop, train_flops, per_token_flops, format_flops


@pytest.fixture(scope="module")
def paper():
    return project_crop()  # paper defaults: 1.5B student, 100B tokens, 0.3B ref


def test_baseline_uniform_4k(paper):
    # paper: "the uniform baseline ... costs ~1.04e21 FLOPs"
    assert paper["baseline"] == pytest.approx(1.038e21, rel=0.01)


def test_doc_gating_saving(paper):
    # paper Table 1: -1.60e20 FLOPs, -15.4%
    assert paper["doc_gating_saving"] == pytest.approx(1.60e20, rel=0.02)
    assert paper["doc_gating_saving_frac"] == pytest.approx(0.154, abs=0.002)


def test_context_ramp_saving(paper):
    # paper Table 1: -6.9e19 FLOPs, -6.6%
    assert paper["context_ramp_saving"] == pytest.approx(6.9e19, rel=0.02)
    assert paper["context_ramp_saving_frac"] == pytest.approx(0.066, abs=0.002)


def test_token_masking_costs_no_flops(paper):
    assert paper["token_masking_saving"] == 0.0


def test_reference_scoring_overhead(paper):
    # paper: 0.3B ref, 1.4x corpus, forward-only -> +8.4e19, +8.1%
    assert paper["reference_scoring"] == pytest.approx(8.4e19, rel=0.01)
    assert paper["reference_overhead_frac"] == pytest.approx(0.081, abs=0.002)


def test_conservative_total(paper):
    # paper: 9.14e20 FLOPs = 88.0% of baseline
    assert paper["conservative_total"] == pytest.approx(9.14e20, rel=0.01)
    assert paper["conservative_vs_baseline"] == pytest.approx(0.880, abs=0.002)


def test_h2_scenario(paper):
    # paper: matched val loss with 20% effective-budget shift -> 7.48e20 = 72.1%
    assert paper["h2_total"] == pytest.approx(7.48e20, rel=0.01)
    assert paper["h2_vs_baseline"] == pytest.approx(0.721, abs=0.002)


def test_attention_term_grows_with_context():
    """At 2K the attention term is ~8% of the parameter term, ~15% at 4K
    (paper Sec. 2.1, for d=2048, L=28)."""
    N, L, d = 1.5e9, 28, 2048
    attn2k = L * 2048 * d / N
    attn4k = L * 4096 * d / N
    assert attn2k == pytest.approx(0.077, abs=0.01)
    assert attn4k == pytest.approx(0.154, abs=0.01)
    assert per_token_flops(N, L, d, 4096) == 2 * (N + attn4k * N)


def test_train_flops_formula():
    # 6ND without the attention term: 3 * D * 2N
    assert train_flops(100, 1e6, 0, 1, 1) == pytest.approx(6e8, rel=1e-9)


def test_format_flops():
    assert format_flops(1.038e21) == "1.04e21"
    assert format_flops(9.14e20) == "914.00e18"
    assert format_flops(8.4e19) == "84.00e18"
