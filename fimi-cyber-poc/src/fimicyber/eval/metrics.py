"""Retrieval metrics: P@k, MAP, nDCG@10, ROC-AUC (spec 10.2)."""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """P@k: fraction of top-k that are relevant."""
    top = ranked[:k]
    return sum(1 for r in top if r in relevant) / k if top else 0.0


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    """AP for a single query."""
    if not relevant:
        return 0.0
    n_rel = 0
    total = 0.0
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            n_rel += 1
            total += n_rel / rank
    return total / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """nDCG@k."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def roc_auc(
    all_ids: list[str],
    positives: set[frozenset],
    score_fn: Any,  # callable(id_i, id_j) → float|None
) -> float:
    """Pairwise ROC-AUC over all positive/negative pairs."""
    pos_scores: list[float] = []
    neg_scores: list[float] = []

    for i in range(len(all_ids)):
        for j in range(i + 1, len(all_ids)):
            s = score_fn(all_ids[i], all_ids[j])
            if s is None or math.isnan(s):
                continue
            pair = frozenset({all_ids[i], all_ids[j]})
            if pair in positives:
                pos_scores.append(s)
            else:
                neg_scores.append(s)

    if not pos_scores or not neg_scores:
        return float("nan")

    # U-statistic AUC
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    count = 0
    for p in pos_scores:
        for q in neg_scores:
            if p > q:
                count += 1
            elif p == q:
                count += 0.5
    return count / (n_pos * n_neg)


def evaluate_condition(
    query_ids: list[str],
    all_ids: list[str],
    positives: set[frozenset],
    score_fn: Any,  # callable(q_id, doc_id) → float|None
    cfg: Any,
) -> dict[str, float]:
    """
    Evaluate a retrieval condition.
    Returns: p@1, p@3, p@5, map, ndcg10, roc_auc
    """
    p_at_ks = cfg.eval.get("p_at", [1, 3, 5])
    ndcg_k = int(cfg.eval.get("ndcg_at", 10))

    aps: list[float] = []
    ndcgs: list[float] = []
    pat: dict[int, list[float]] = {k: [] for k in p_at_ks}

    for q_id in query_ids:
        # Rank all other events
        candidates = [(doc_id, score_fn(q_id, doc_id)) for doc_id in all_ids if doc_id != q_id]
        # Sort by score desc; NaN → bottom
        candidates.sort(key=lambda x: (x[1] is None or math.isnan(x[1]), -(x[1] or 0)))
        ranked = [doc_id for doc_id, _ in candidates]

        # Relevant: same campaign as q_id
        relevant = {
            eid
            for fs in positives
            if q_id in fs
            for eid in fs
            if eid != q_id
        }

        aps.append(average_precision(ranked, relevant))
        ndcgs.append(ndcg_at_k(ranked, relevant, ndcg_k))
        for k in p_at_ks:
            pat[k].append(precision_at_k(ranked, relevant, k))

    result: dict[str, float] = {
        "MAP": float(np.mean(aps)) if aps else float("nan"),
        "nDCG@10": float(np.mean(ndcgs)) if ndcgs else float("nan"),
    }
    for k in p_at_ks:
        result[f"P@{k}"] = float(np.mean(pat[k])) if pat[k] else float("nan")

    return result


def _average_precisions_by_query(
    query_ids: list[str],
    all_ids: list[str],
    positives: set[frozenset],
    score_fn: Any,
) -> list[float]:
    """Compute each query's AP once so bootstrap does not rerank repeatedly."""
    aps: list[float] = []
    for q_id in query_ids:
        candidates = [(doc_id, score_fn(q_id, doc_id)) for doc_id in all_ids if doc_id != q_id]
        candidates.sort(key=lambda x: (x[1] is None or math.isnan(x[1]), -(x[1] or 0)))
        ranked = [doc_id for doc_id, _ in candidates]
        relevant = {
            eid
            for fs in positives
            if q_id in fs
            for eid in fs
            if eid != q_id
        }
        aps.append(average_precision(ranked, relevant))
    return aps


def bootstrap_ci(
    query_ids: list[str],
    all_ids: list[str],
    positives: set[frozenset],
    score_fn: Any,
    cfg: Any,
    n_iter: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (ci_low, ci_high) for MAP via bootstrap."""
    if n_iter <= 0:
        return float("nan"), float("nan")

    query_aps = _average_precisions_by_query(query_ids, all_ids, positives, score_fn)
    if not query_aps:
        return float("nan"), float("nan")

    rng = random.Random(seed)
    maps: list[float] = []

    for _ in range(n_iter):
        sample = rng.choices(query_aps, k=len(query_aps))
        maps.append(float(np.mean(sample)))

    maps.sort()
    lo = maps[int(0.025 * n_iter)]
    hi = maps[int(0.975 * n_iter)]
    return lo, hi
