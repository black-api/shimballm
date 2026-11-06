"""Corpus loading, tokenization, and resizable sequence packing.

The packer is CROP's data spine: it maintains a running token buffer fed by a
document sampler and cuts fixed-length blocks at the *current* context length.
When the context ramp grows (Stage B, 2K -> 4K in the paper) the buffer simply
gets re-cut at the new length, so no data is wasted at stage boundaries.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple
import numpy as np
import os

from ..tokenizer import BaseTokenizer, build_tokenizer

_EOS = 1  # synthetic docs always use id 1 as a document separator


# --------------------------------------------------------------------------- #
# Corpus sources
# --------------------------------------------------------------------------- #
def make_synthetic_corpus(
    n_docs: int = 2048,
    doc_len: int = 192,
    vocab_words: int = 512,
    seed: int = 7,
) -> List[str]:
    """Deterministic synthetic corpus with Zipfian word frequencies and a few
    reused sentence templates, so a small model's loss visibly decreases.

    It mimics the property CROP exploits: a mix of high-surprise content words
    and low-surprise boilerplate that a reference model separates well.
    """
    rng = np.random.default_rng(seed)
    # Zipf word distribution over numbered tokens, rendered as w0000, w0001...
    ranks = np.arange(1, vocab_words + 1)
    probs = 1.0 / ranks ** 1.1
    probs /= probs.sum()
    words = [f"w{i:04d}" for i in range(vocab_words)]
    templates = [
        "{a} {b} {c} {d} {e} {f} {g} {h}.",
        "in the {a} a {b} of {c} was {d}.",
        "the {a} said that {b} {c} {d} again.",
        "when {a} {b} the {c}, all {d} {e} {f}.",
        "{a} and {b} went to {c} {d} {e}.",
        "it is {a} that {b} {c} the {d} {e}.",
    ]
    docs: List[str] = []
    for _ in range(n_docs):
        n_sent = max(2, int(rng.integers(3, 7)))
        sents = []
        for _ in range(n_sent):
            t = templates[int(rng.integers(0, len(templates)))]
            n_fill = t.count("{")
            fills = iter(rng.choice(words, size=n_fill, p=probs))
            out = [next(fills) if tok.startswith("{") else tok for tok in t.split(" ")]
            sents.append(" ".join(out))
        docs.append(" ".join(sents))
    return docs


def read_documents(path: str, max_docs: int = 0) -> List[str]:
    """Read documents from .jsonl ({"text": ...} per line) or .txt (blank-line
    separated)."""
    docs: List[str] = []
    if path.endswith(".jsonl"):
        import json

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or ""
                if text:
                    docs.append(text)
                if max_docs and len(docs) >= max_docs:
                    break
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        docs = [d.strip() for d in raw.split("\n\n") if d.strip()]
        if max_docs:
            docs = docs[:max_docs]
    return docs


# --------------------------------------------------------------------------- #
# DocStore: tokenized documents
# --------------------------------------------------------------------------- #
class DocStore:
    """Tokenized corpus held as one int32 array plus per-document offsets."""

    def __init__(self, tokens: np.ndarray, lengths: List[int]):
        assert tokens.ndim == 1
        self.tokens = tokens.astype(np.int32, copy=False)
        self.lengths = list(lengths)
        self.offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    def __len__(self) -> int:
        return len(self.lengths)

    @property
    def n_tokens(self) -> int:
        return int(self.offsets[-1])

    def doc(self, i: int) -> np.ndarray:
        return self.tokens[self.offsets[i] : self.offsets[i + 1]]

    @classmethod
    def tokenize(cls, docs: List[str], tokenizer: BaseTokenizer, eos_id: int = _EOS) -> "DocStore":
        toks: List[np.ndarray] = []
        lens: List[int] = []
        for d in docs:
            ids = tokenizer.encode(d)
            if not ids:
                continue
            ids = ids + [eos_id]
            toks.append(np.asarray(ids, dtype=np.int32))
            lens.append(len(ids))
        if not toks:
            raise ValueError("corpus tokenized to zero documents")
        return cls(np.concatenate(toks), lens)

    def split_val(self, val_fraction: float, seed: int = 0) -> Tuple["DocStore", "DocStore"]:
        rng = np.random.default_rng(seed)
        n = len(self)
        idx = rng.permutation(n)
        n_val = max(8, int(n * val_fraction)) if val_fraction > 0 else 0
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return self._subset(train_idx), self._subset(val_idx)

    def _subset(self, idx: np.ndarray) -> "DocStore":
        toks, lens = [], []
        for i in idx:
            d = self.doc(int(i))
            toks.append(d)
            lens.append(len(d))
        return DocStore(np.concatenate(toks), lens) if toks else DocStore(np.array([], dtype=np.int32), [])


# --------------------------------------------------------------------------- #
# Curriculum samplers
# --------------------------------------------------------------------------- #
class QuantileSampler:
    """Samples documents whose mean reference score lies in a score-quantile
    band [lo, hi). This is the CROP tier mechanism:

      Stage A: band = (doc_drop_quantile, 1.0)   -> drop the bottom 30%
      Stage B: band = (stage_b_doc_quantile, 1.0) -> upper tiers replay
      Stage C: band = (stage_c_doc_quantile, 1.0) -> top slice for annealing

    Iterates without replacement through shuffled cycles of the band (so every
    kept document is seen once per epoch), drawing docs at random until the
    requested token count is produced."""

    def __init__(self, doc_scores: np.ndarray, lo: float, hi: float, seed: int = 0):
        self.scores = np.asarray(doc_scores, dtype=np.float64)
        q_lo, q_hi = np.quantile(self.scores, lo), np.quantile(self.scores, hi)
        # NOTE: lo == 0.0 keeps everything strictly above the drop quantile;
        # ties at the threshold go to the dropped side (conservative).
        self.band = np.where((self.scores > q_lo) & (self.scores <= max(q_hi, q_lo + 1e-12)))[0]
        if hi >= 1.0:
            self.band = np.where(self.scores > q_lo)[0]
        if len(self.band) == 0:  # degenerate corpus: fall back to everything
            self.band = np.arange(len(self.scores))
        self.order = np.random.default_rng(seed).permutation(self.band)
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.band)

    def next_doc(self) -> int:
        if self.cursor >= len(self.order):
            self.order = np.random.default_rng().permutation(self.band)
            self.cursor = 0
        i = int(self.order[self.cursor])
        self.cursor += 1
        return i


# --------------------------------------------------------------------------- #
# Streaming packer
# --------------------------------------------------------------------------- #
class StreamingPacker:
    """Packs sampled documents into [batch, context] blocks at the current
    context length. Leftover tokens survive context changes, so the Stage-B
    ramp 2K -> 4K loses no data."""

    def __init__(self, store: DocStore, sampler: QuantileSampler, seed: int = 0):
        self.store = store
        self.sampler = sampler
        self.rng = np.random.default_rng(seed)
        self.buffer = np.array([], dtype=np.int64)
        self.tokens_served = 0

    def _fill(self, need: int) -> None:
        while len(self.buffer) < need:
            d = self.store.doc(self.sampler.next_doc()).astype(np.int64)
            self.buffer = np.concatenate([self.buffer, d])

    def next_batch(self, batch_size: int, ctx_len: int) -> np.ndarray:
        need = batch_size * ctx_len
        self._fill(need)
        block, self.buffer = self.buffer[:need], self.buffer[need:]
        self.tokens_served += need
        return block.reshape(batch_size, ctx_len)

    def state_dict(self) -> dict:
        return {
            "buffer": self.buffer,
            "tokens_served": self.tokens_served,
            "sampler_cursor": self.sampler.cursor,
            "sampler_order": self.sampler.order,
        }

    def load_state_dict(self, st: dict) -> None:
        self.buffer = np.asarray(st["buffer"], dtype=np.int64)
        self.tokens_served = int(st["tokens_served"])
        self.sampler.cursor = int(st["sampler_cursor"])
        self.sampler.order = np.asarray(st["sampler_order"])
