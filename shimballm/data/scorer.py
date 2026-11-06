"""Phase 0: offline reference scoring with excess loss.

Scores every document once with the small frozen reference model theta_ref
(paper Eq. 2): the per-token next-token loss under theta_ref. Near-zero
scores mark boilerplate any competent model already predicts; high scores
mark knowledge-dense content. Document score = mean token score (Eq. 3 uses
these to build the Stage-A gate at the 30th score percentile).

Results are cached to ``<cache_dir>/doc_scores.npy`` (+ a small JSON header),
so the scoring pass is paid once per corpus no matter how many runs reuse it.
"""

from __future__ import annotations

from typing import Optional
import json
import os
import numpy as np
import torch

from ..models import logits_of
from .corpus import DocStore


@torch.no_grad()
def score_documents(
    ref_model: torch.nn.Module,
    store: DocStore,
    device: str = "cpu",
    batch_docs: int = 8,
    max_docs: int = 0,
    max_doc_tokens: int = 0,
    progress: bool = True,
) -> np.ndarray:
    """Per-document mean token loss under the frozen reference model.

    Returns:
        float32 array [n_docs] of mean per-token CE (lower = more predictable).
    """
    ref_model.eval()
    n = len(store) if max_docs <= 0 else min(max_docs, len(store))
    scores = np.zeros(n, dtype=np.float32)
    order = list(range(n))
    for start in range(0, n, batch_docs):
        idx = order[start : start + batch_docs]
        seqs = []
        for i in idx:
            d = store.doc(i)
            if max_doc_tokens and len(d) > max_doc_tokens:
                d = d[:max_doc_tokens]
            seqs.append(torch.from_numpy(d.astype(np.int64)))
        lens = [len(s) for s in seqs]
        L = max(lens)
        batch = torch.zeros(len(seqs), L, dtype=torch.long)
        for j, s in enumerate(seqs):
            batch[j, : len(s)] = s
        batch = batch.to(device)
        logits = logits_of(ref_model(batch)).float()
        # token t predicts t+1; positions 0..L-2 carry a loss for target 1..L-1
        ce = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
            reduction="none",
        ).view(batch.size(0), -1)
        for j, i in enumerate(idx):
            valid = max(lens[j] - 1, 0)
            scores[i] = ce[j, :valid].mean().item() if valid > 0 else 0.0
        if progress and (start // batch_docs) % 20 == 0:
            print(f"  scoring {start + len(idx)}/{n} docs", end="\r", flush=True)
    if progress:
        print()
    return scores


def save_doc_scores(cache_dir: str, scores: np.ndarray, meta: dict) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "doc_scores.npy")
    np.save(path, scores)
    with open(os.path.join(cache_dir, "doc_scores.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def load_doc_scores(cache_dir: str) -> Optional[np.ndarray]:
    path = os.path.join(cache_dir, "doc_scores.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def summarize_scores(scores: np.ndarray) -> dict:
    q = np.quantile(scores, [0.1, 0.3, 0.5, 0.7, 0.9])
    return {
        "n_docs": int(len(scores)),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "quantiles_10_30_50_70_90": [float(x) for x in q],
    }
