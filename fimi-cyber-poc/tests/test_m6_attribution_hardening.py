"""Hardening tests for taxonomy, calibration, and the external case."""
from __future__ import annotations

import pandas as pd

from fimicyber.attribution.calibration import calibrate_hypotheses, evaluate_attribution_scopes
from fimicyber.attribution.provenance import build_evidence_provenance
from fimicyber.attribution.taxonomy import load_actor_taxonomy
from fimicyber.attribution.generalization import (
    _label_isolated_scoring_events,
    validate_generalization_protocol,
)
from fimicyber.loaders.external import load_external_case, load_multiactor_benchmark


def test_actor_taxonomy_merges_only_declared_aliases(cfg):
    taxonomy = load_actor_taxonomy(cfg)
    assert taxonomy.resolve("IRA").actor_id == "internet_research_agency"
    assert taxonomy.resolve("IRA*").actor_id == "internet_research_agency"
    assert taxonomy.resolve("Russia").actor_id != taxonomy.resolve("pro-kremlin ecosystem").actor_id


def test_external_case_is_source_separated_and_has_real_iocs(cfg):
    events = load_external_case(cfg)
    assert len(events) == 5
    assert sum(event.evaluation_role == "reference" for event in events) == 2
    assert sum(event.evaluation_role == "holdout" for event in events) == 3
    assert all(not ioc.synthetic for event in events for ioc in event.iocs)
    assert any(ioc.value == "88.99.132.118" for event in events for ioc in event.iocs)


def _hypothesis_rows() -> pd.DataFrame:
    rows = []
    for index in range(1, 7):
        actual = "actor_a" if index % 2 else "actor_b"
        scores = [("actor_a", 0.8), ("actor_b", 0.2)] if actual == "actor_a" else [("actor_b", 0.7), ("actor_a", 0.3)]
        for rank, (candidate, score) in enumerate(scores, 1):
            rows.append({
                "query_event_id": f"q{index}",
                "query_date": f"202{index}-01-01",
                "candidate_actor_id": candidate,
                "candidate_actor": candidate,
                "rank": rank,
                "support_score": score,
                "candidate_probability": 0.5,
                "assessment_confidence": 0.5,
                "confidence_band": "lead",
                "margin_to_next": score - (scores[1][1] if rank == 1 else 0.0),
                "evidence_families": "narrative|infrastructure|ttp",
                "real_ioc_support": True,
                "source_org_count": 2,
                "actual_actor_id": actual,
                "actual_actor": actual,
                "correct_actor": candidate == actual,
                "supporting_event_ids": "r1|r2",
            })
    return pd.DataFrame(rows)


def test_chronological_calibration_and_abstention_outputs(cfg):
    calibrated, summary = calibrate_hypotheses(_hypothesis_rows(), cfg)
    assert not summary.empty
    assert set(calibrated["split_role"]) == {"calibration", "validation"}
    assert set(calibrated[calibrated["rank"] == 1]["decision"]) <= {"analyst_review", "abstain"}
    metrics = evaluate_attribution_scopes(calibrated, cfg)
    assert {"all_temporal", "calibration", "validation"} <= set(metrics["evaluation_scope"])
    assert "top1_ci95_low" in metrics


def test_external_provenance_has_record_hashes(cfg):
    events = load_external_case(cfg)
    provenance = build_evidence_provenance(
        events, cfg.data_dir / "external" / "provenance_manifest.csv"
    )
    assert not provenance.empty
    assert provenance["record_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert {"NKSC-152827", "NKSC-163811", "MANDIANT-GW-2021"} <= set(provenance["source_id"])


def test_multiactor_benchmark_is_balanced_and_protocol_frozen(cfg):
    events = load_multiactor_benchmark(cfg)
    assert len(events) == 28
    assert sum(event.evaluation_role == "reference" for event in events) == 8
    assert sum(event.evaluation_role == "holdout" for event in events) == 20
    assert all(not ioc.synthetic for event in events for ioc in event.iocs)
    checks = validate_generalization_protocol(events, cfg)
    assert checks["passed"].all(), checks.loc[~checks["passed"]].to_dict("records")
    scoring_events = _label_isolated_scoring_events(events)
    assert all(
        event.reported_actor is None and event.campaign_id is None
        for event in scoring_events if event.evaluation_role == "holdout"
    )
    assert all(
        event.reported_actor is not None
        for event in scoring_events if event.evaluation_role == "reference"
    )
