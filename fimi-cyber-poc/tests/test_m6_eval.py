"""M6 tests: T10, T13."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── T10: ζ=0 hard guard ──────────────────────────────────────────────────────

def test_T10_zeta_guard(cfg):
    """allow_actor=False + ζ>0 in weights → FCLS must still zero out A."""
    from fimicyber.scoring.fcls import fcls, _extract_weights

    # Config has zeta_A=0.05 but allow_actor=False must force ζ=0
    weights = _extract_weights(cfg, allow_actor=False)
    assert weights["A"] == 0.0, f"Expected A=0.0, got {weights['A']}"


def test_T10_zeta_guard_allows_actor_true(cfg):
    """allow_actor=True should pass zeta through."""
    from fimicyber.scoring.fcls import _extract_weights
    weights = _extract_weights(cfg, allow_actor=True)
    assert weights["A"] > 0.0, "Expected A>0 when allow_actor=True"


# ── T13: metrics sanity ───────────────────────────────────────────────────────

def test_T13_perfect_ranking_map_1():
    """Perfect ranking (relevant docs always first) → MAP=1.0, nDCG@10=1.0."""
    from fimicyber.eval.metrics import average_precision, ndcg_at_k

    # 3 relevant docs, perfect ranking: positions 1,2,3
    relevant = {"a", "b", "c"}
    ranked = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    ap = average_precision(ranked, relevant)
    ndcg = ndcg_at_k(ranked, relevant, 10)
    assert abs(ap - 1.0) < 1e-9, f"Expected AP=1.0, got {ap}"
    assert abs(ndcg - 1.0) < 1e-9, f"Expected nDCG=1.0, got {ndcg}"


def test_T13_reverse_ranking_map_less_than_half():
    """Reverse ranking (worst first) → MAP < 0.5."""
    from fimicyber.eval.metrics import average_precision

    relevant = {"a", "b", "c"}
    # All relevant docs at the very end
    ranked = ["d", "e", "f", "g", "h", "i", "j", "k", "a", "b", "c"]
    ap = average_precision(ranked, relevant)
    assert ap < 0.5, f"Expected AP < 0.5, got {ap}"


def test_T13_map_full_system(cfg):
    """Full MAP test using built GT and a perfect oracle score function."""
    from fimicyber.eval.groundtruth import build_ground_truth
    from fimicyber.eval.metrics import evaluate_condition
    from fimicyber.schema import Event
    from datetime import date

    # Create events with known campaign structure
    events = [
        Event(event_id="e1", title="t", description="d", campaign_id="camp1"),
        Event(event_id="e2", title="t", description="d", campaign_id="camp1"),
        Event(event_id="e3", title="t", description="d", campaign_id="camp2"),
        Event(event_id="e4", title="t", description="d", campaign_id="camp2"),
        Event(event_id="e5", title="t", description="d", campaign_id=None),
    ]
    gt = build_ground_truth(events)
    all_ids = [ev.event_id for ev in events]

    # Perfect oracle: same campaign → score 1.0, different → 0.0
    camp_map = {ev.event_id: ev.campaign_id for ev in events}
    def perfect_fn(q, d):
        qc = camp_map.get(q)
        dc = camp_map.get(d)
        if qc is None or dc is None:
            return 0.0
        return 1.0 if qc == dc else 0.0

    metrics = evaluate_condition(gt["query_ids"], all_ids, gt["positives"], perfect_fn, cfg)
    assert abs(metrics["MAP"] - 1.0) < 1e-9, f"Expected MAP=1.0, got {metrics['MAP']}"
    assert abs(metrics["nDCG@10"] - 1.0) < 1e-9, f"Expected nDCG=1.0, got {metrics['nDCG@10']}"
