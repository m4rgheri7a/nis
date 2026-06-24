"""Narrative similarity matrix N(i,j) (spec 5).

N(i,j) = λ·max_sim(i,j) + (1−λ)·avg_topk_sim(i,j)
Cosine normalised to [0,1] via (cos+1)/2.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fimicyber.schema import Event
from fimicyber.nlp.embed import EmbStore

ScoreMatrix = np.ndarray  # shape (n, n), float32, NaN for missing


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalise_cosine(cos: float) -> float:
    return (cos + 1.0) / 2.0


def _pair_score(
    vecs_i: list[np.ndarray],
    vecs_j: list[np.ndarray],
    lam: float,
    top_k: int,
) -> float:
    """Compute N(i,j) for one pair."""
    sims: list[float] = []
    for vi in vecs_i:
        for vj in vecs_j:
            sims.append(_normalise_cosine(_cosine(vi, vj)))

    if not sims:
        return float("nan")

    max_sim = max(sims)
    k = min(top_k, len(sims))
    avg_topk = float(np.mean(sorted(sims, reverse=True)[:k]))

    return lam * max_sim + (1.0 - lam) * avg_topk


def narrative_matrix(
    events: list[Event],
    emb: EmbStore,
    cfg: Any,
) -> ScoreMatrix:
    """Return symmetric n×n matrix; diagonal = NaN; missing = NaN."""
    lam: float = float(cfg.narrative.get("lambda", 0.6))
    top_k: int = int(cfg.narrative.get("top_k", 3))

    n = len(events)
    mat = np.full((n, n), float("nan"), dtype=np.float64)

    vecs_list: list[list[np.ndarray] | None] = [emb.get_vecs(ev) for ev in events]

    for i in range(n):
        mat[i, i] = float("nan")  # diagonal always NaN
        for j in range(i + 1, n):
            vi = vecs_list[i]
            vj = vecs_list[j]
            if vi is None or vj is None:
                score = float("nan")
            else:
                score = _pair_score(vi, vj, lam, top_k)
            mat[i, j] = score
            mat[j, i] = score

    return mat
