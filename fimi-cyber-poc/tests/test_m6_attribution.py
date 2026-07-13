"""M6 attribution-support tests."""
from __future__ import annotations

from datetime import date

import pandas as pd

from fimicyber.attribution import (
    build_attribution_graph,
    build_attribution_hypotheses,
    evaluate_attribution,
)
from fimicyber.schema import Event


def _event(event_id: str, actor: str | None, year: int, source: str) -> Event:
    return Event(
        event_id=event_id,
        title=event_id,
        description=f"Narrative for {event_id}",
        reported_actor=actor,
        first_seen=date(year, 1, 1),
        last_seen=date(year, 1, 2),
        target_countries=["KR"],
        target_sectors=["media"],
        evidence_sources=[source],
    )


def _row(left: str, right: str, n: float, i: float | None, d: float, c: float, t: float):
    return {
        "event_i": left,
        "event_j": right,
        "N": n,
        "I": 0.99,  # synthetic-inclusive value must never be used for attribution
        "I_no_synthetic": i,
        "D": d,
        "C": c,
        "T": t,
    }


def test_temporal_actor_hypothesis_ranking_and_metrics(cfg, tmp_path):
    events = [
        _event("a1", "Actor Alpha", 2019, "https://source-one.example/report"),
        _event("a2", "Actor Alpha", 2020, "https://source-two.example/report"),
        _event("b1", "Actor Beta", 2019, "https://source-three.example/report"),
        _event("b2", "Actor Beta", 2020, "https://source-four.example/report"),
        _event("q", "Actor Alpha", 2022, "https://query-source.example/report"),
        _event("future", "Actor Beta", 2024, "https://future.example/report"),
    ]
    scores = pd.DataFrame([
        _row("q", "a1", 0.90, 0.80, 0.85, 0.75, 0.70),
        _row("q", "a2", 0.88, 0.78, 0.82, 0.72, 0.68),
        _row("q", "b1", 0.25, 0.05, 0.10, 0.10, 0.60),
        _row("q", "b2", 0.20, 0.04, 0.08, 0.12, 0.55),
        _row("q", "future", 1.00, 1.00, 1.00, 1.00, 1.00),
    ])

    hypotheses = build_attribution_hypotheses(events, scores, cfg, query_ids=["q"])
    assert list(hypotheses["candidate_actor"])[0] == "Actor Alpha"
    assert bool(hypotheses.iloc[0]["correct_actor"])
    assert "future" not in "|".join(hypotheses["supporting_event_ids"])
    assert hypotheses.iloc[0]["infrastructure_score"] < 0.9
    assert hypotheses["display_candidate"].sum() == min(3, len(hypotheses))

    metrics = evaluate_attribution(hypotheses, cfg).iloc[0]
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["top3_accuracy"] == 1.0
    assert metrics["MRR"] == 1.0

    graph = build_attribution_graph(events, hypotheses, tmp_path / "attribution.json")
    assert (tmp_path / "attribution.json").exists()
    assert any(data.get("etype") == "ASSESSED_AS" for *_, data in graph.edges(data=True))


def test_no_real_ioc_confidence_cap(cfg):
    events = [
        _event("a1", "Actor Alpha", 2019, "https://one.example/report"),
        _event("a2", "Actor Alpha", 2020, "https://two.example/report"),
        _event("b1", "Actor Beta", 2019, "https://three.example/report"),
        _event("b2", "Actor Beta", 2020, "https://four.example/report"),
        _event("q", None, 2022, "https://query.example/report"),
    ]
    scores = pd.DataFrame([
        _row("q", "a1", 0.95, None, 0.90, 0.90, 0.90),
        _row("q", "a2", 0.94, None, 0.90, 0.90, 0.90),
        _row("q", "b1", 0.10, None, 0.10, 0.10, 0.10),
        _row("q", "b2", 0.10, None, 0.10, 0.10, 0.10),
    ])

    hypotheses = build_attribution_hypotheses(events, scores, cfg, query_ids=["q"])
    top = hypotheses.iloc[0]
    assert not bool(top["real_ioc_support"])
    assert top["assessment_confidence"] <= cfg.attribution["confidence"]["no_real_ioc_cap"]


def test_evaluation_handles_unlabelled_query_as_nan(cfg):
    hypotheses = pd.DataFrame([{
        "query_event_id": "q",
        "candidate_actor": "Actor Alpha",
        "rank": 1,
        "candidate_probability": 1.0,
        "actual_actor": float("nan"),
        "correct_actor": False,
    }])
    metrics = evaluate_attribution(hypotheses, cfg).iloc[0]
    assert metrics["queries_total"] == 1
    assert metrics["queries_labeled"] == 0
    assert metrics["queries_evaluable"] == 0


def test_top3_evaluation_keeps_candidates_below_display_cutoff(cfg):
    hypotheses = pd.DataFrame([
        {
            "query_event_id": "q",
            "candidate_actor": actor,
            "rank": rank,
            "candidate_probability": probability,
            "actual_actor": "Actor Four",
            "correct_actor": actor == "Actor Four",
        }
        for rank, (actor, probability) in enumerate([
            ("Actor One", 0.40),
            ("Actor Two", 0.30),
            ("Actor Three", 0.20),
            ("Actor Four", 0.10),
        ], start=1)
    ])
    metrics = evaluate_attribution(hypotheses, cfg).iloc[0]
    assert metrics["queries_evaluable"] == 1
    assert metrics["top1_accuracy"] == 0.0
    assert metrics["top3_accuracy"] == 0.0
    assert metrics["MRR"] == 0.25


def test_candidate_actor_name_in_description_disables_narrative_signal(cfg):
    events = [
        _event("a1", "Actor Alpha", 2019, "https://one.example/report"),
        _event("a2", "Actor Alpha", 2020, "https://two.example/report"),
        _event("b1", "Actor Beta", 2019, "https://three.example/report"),
        _event("b2", "Actor Beta", 2020, "https://four.example/report"),
        _event("q", None, 2022, "https://query.example/report"),
    ]
    events[-1].description = "A report explicitly naming Actor Alpha"
    scores = pd.DataFrame([
        _row("q", "a1", 1.00, 0.10, 0.20, 0.20, 0.20),
        _row("q", "a2", 1.00, 0.10, 0.20, 0.20, 0.20),
        _row("q", "b1", 0.10, 0.10, 0.20, 0.20, 0.20),
        _row("q", "b2", 0.10, 0.10, 0.20, 0.20, 0.20),
    ])

    hypotheses = build_attribution_hypotheses(events, scores, cfg, query_ids=["q"])
    alpha = hypotheses[hypotheses["candidate_actor"] == "Actor Alpha"].iloc[0]
    assert pd.isna(alpha["narrative_score"])
    assert alpha["narrative_label_guarded_pairs"] == 2


def test_explicit_reference_ids_prevent_holdout_to_holdout_leakage(cfg):
    events = [
        _event("a1", "Actor Alpha", 2019, "https://one.example/report"),
        _event("a2", "Actor Alpha", 2020, "https://two.example/report"),
        _event("b1", "Actor Beta", 2019, "https://three.example/report"),
        _event("b2", "Actor Beta", 2020, "https://four.example/report"),
        _event("early-holdout", "Actor Beta", 2021, "https://holdout.example/report"),
        _event("q", "Actor Alpha", 2022, "https://query.example/report"),
    ]
    scores = pd.DataFrame([
        _row("q", "a1", 0.8, 0.1, 0.8, 0.8, 0.8),
        _row("q", "a2", 0.8, 0.1, 0.8, 0.8, 0.8),
        _row("q", "b1", 0.2, 0.1, 0.2, 0.2, 0.8),
        _row("q", "b2", 0.2, 0.1, 0.2, 0.2, 0.8),
        _row("q", "early-holdout", 1.0, 1.0, 1.0, 1.0, 1.0),
    ])
    hypotheses = build_attribution_hypotheses(
        events,
        scores,
        cfg,
        query_ids=["q"],
        reference_ids=["a1", "a2", "b1", "b2"],
    )
    assert "early-holdout" not in "|".join(hypotheses["supporting_event_ids"])


def test_evidence_graph_has_reference_actor_and_campaign_nodes(cfg):
    from fimicyber.graph.build import build_graph

    event = Event(
        event_id="e1",
        title="Event",
        description="Description",
        campaign_id="campaign-one",
        reported_actor="Actor Alpha",
    )
    graph = build_graph([event], cfg)
    assert graph.nodes["Campaign:campaign-one"]["ntype"] == "Campaign"
    assert graph.nodes["Actor:Actor Alpha"]["ntype"] == "Actor"
    assert graph.edges["Event:e1", "Actor:Actor Alpha"]["reference_only"] is True
