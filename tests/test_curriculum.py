"""Curriculum sampler + streaming packer tests (Stage A/B/C tiers)."""

import numpy as np
import pytest

from shimballm.data.corpus import (
    DocStore,
    QuantileSampler,
    StreamingPacker,
    make_synthetic_corpus,
)


def _store(n_docs=200, doc_len=32):
    docs = make_synthetic_corpus(n_docs=n_docs, doc_len=doc_len, seed=3)
    from shimballm.tokenizer import CharTokenizer

    return DocStore.tokenize(docs, CharTokenizer())


def test_sampler_drops_bottom_quantile():
    scores = np.linspace(0, 1, 200)  # doc i has score i/199
    s = QuantileSampler(scores, lo=0.30, hi=1.0, seed=0)
    kept = set(s.band.tolist())
    assert len(kept) == pytest.approx(140, abs=2)  # top 70% kept
    assert min(kept) >= 60                         # bottom 30% (idx < 60) dropped


def test_sampler_stage_b_uses_upper_tiers():
    scores = np.linspace(0, 1, 200)
    s = QuantileSampler(scores, lo=0.50, hi=1.0, seed=0)
    assert min(s.band) >= 100


def test_packer_shapes_and_consumption():
    store = _store()
    sampler = QuantileSampler(np.zeros(len(store)), lo=0.0, hi=1.0, seed=0)
    p = StreamingPacker(store, sampler, seed=0)
    b = p.next_batch(batch_size=4, ctx_len=16)
    assert b.shape == (4, 16)
    assert p.tokens_served == 64
    b2 = p.next_batch(batch_size=2, ctx_len=8)
    assert b2.shape == (2, 8)


def test_packer_resizes_context_without_loss():
    """The Stage-B ramp: growing ctx keeps serving; buffer carries over."""
    store = _store()
    sampler = QuantileSampler(np.zeros(len(store)), lo=0.0, hi=1.0, seed=0)
    p = StreamingPacker(store, sampler, seed=0)
    seen = 0
    for ctx in (16, 16, 32, 32, 64):  # 2K -> 4K style ramp
        p.next_batch(2, ctx)
        seen += 2 * ctx
    assert p.tokens_served == seen


def test_packer_state_roundtrip():
    store = _store()
    sampler = QuantileSampler(np.zeros(len(store)), lo=0.0, hi=1.0, seed=0)
    p = StreamingPacker(store, sampler, seed=0)
    p.next_batch(4, 16)
    st = p.state_dict()
    sampler2 = QuantileSampler(np.zeros(len(store)), lo=0.0, hi=1.0, seed=0)
    p2 = StreamingPacker(store, sampler2, seed=0)
    p2.load_state_dict(st)
    assert p2.tokens_served == p.tokens_served
    assert np.array_equal(p2.buffer, p.buffer)


def test_docstore_split_val():
    store = _store(n_docs=100)
    train, val = store.split_val(0.1, seed=0)
    assert len(train) + len(val) == 100
    assert len(val) >= 8
    assert train.n_tokens + val.n_tokens == store.n_tokens
