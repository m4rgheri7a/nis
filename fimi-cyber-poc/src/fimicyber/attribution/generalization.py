"""Frozen, source-separated multi-campaign external benchmark."""
from __future__ import annotations

import hashlib
import copy
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fimicyber.attribution.calibration import (
    apply_temperature_and_abstention,
    build_error_analysis,
    evaluate_attribution_scope,
)
from fimicyber.attribution.external_case import _condition_comparison
from fimicyber.attribution.hypotheses import build_attribution_hypotheses
from fimicyber.attribution.taxonomy import load_actor_taxonomy
from fimicyber.graph.build import build_graph
from fimicyber.graph.ioc_score import ioc_matrix
from fimicyber.loaders.external import load_multiactor_benchmark
from fimicyber.nlp.narrative import narrative_matrix
from fimicyber.scoring.components import compute_components
from fimicyber.scoring.fcls import build_pairwise_scores


def _protocol_path(cfg: Any) -> Path:
    benchmark_cfg = cfg.attribution.get("generalization_benchmark", {})
    return cfg.data_dir / "external" / benchmark_cfg.get(
        "protocol_file", "generalization_protocol.yaml"
    )


def _load_protocol(cfg: Any) -> tuple[dict[str, Any], str]:
    path = _protocol_path(cfg)
    raw = path.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def _event_actor_ids(events: list, cfg: Any) -> dict[str, str]:
    taxonomy = load_actor_taxonomy(cfg)
    actor_ids: dict[str, str] = {}
    for event in events:
        identity = taxonomy.resolve(event.reported_actor)
        if identity is None:
            raise ValueError(f"Unmapped benchmark label: {event.reported_actor}")
        actor_ids[event.event_id] = identity.actor_id
    return actor_ids


def _label_isolated_scoring_events(events: list) -> list:
    """Remove holdout truth fields before any feature or graph computation."""
    scoring_events = copy.deepcopy(events)
    for event in scoring_events:
        if event.evaluation_role == "holdout":
            event.reported_actor = None
            event.campaign_id = None
            event.campaign_id_source = "none"
    return scoring_events


def _attach_holdout_truth(
    hypotheses: pd.DataFrame,
    labelled_events: list,
    cfg: Any,
) -> pd.DataFrame:
    """Open the sealed labels only after candidate ranking is complete."""
    output = hypotheses.copy()
    taxonomy = load_actor_taxonomy(cfg)
    truth: dict[str, Any] = {}
    for event in labelled_events:
        if event.evaluation_role != "holdout":
            continue
        identity = taxonomy.resolve(event.reported_actor)
        if identity is None:
            raise ValueError(f"Unmapped holdout label: {event.reported_actor}")
        truth[event.event_id] = identity

    for query_id, indexes in output.groupby("query_event_id", sort=False).groups.items():
        identity = truth.get(str(query_id))
        if identity is None:
            continue
        output.loc[indexes, "actual_actor_id"] = identity.actor_id
        output.loc[indexes, "actual_actor"] = identity.display_name
        output.loc[indexes, "actual_actor_level"] = identity.actor_level
        output.loc[indexes, "correct_actor"] = (
            output.loc[indexes, "candidate_actor_id"].astype(str) == identity.actor_id
        )
    return output


def validate_generalization_protocol(
    events: list,
    cfg: Any,
    protocol: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Verify frozen design, source separation, and feature leakage controls."""
    if protocol is None:
        protocol, _ = _load_protocol(cfg)
    design = protocol["design"]
    actor_ids = _event_actor_ids(events, cfg)
    class_ids = set(protocol["class_ids"])
    references = [event for event in events if event.evaluation_role == "reference"]
    holdouts = [event for event in events if event.evaluation_role == "holdout"]

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append({
            "check": name,
            "passed": bool(passed),
            "observed": str(observed),
            "expected": str(expected),
        })

    ref_counts = Counter(actor_ids[event.event_id] for event in references)
    hold_counts = Counter(actor_ids[event.event_id] for event in holdouts)
    observed_classes = set(ref_counts) | set(hold_counts)
    add("class_ids", observed_classes == class_ids, sorted(observed_classes), sorted(class_ids))
    add("class_count", len(observed_classes) == int(design["classes"]), len(observed_classes), design["classes"])
    add(
        "reference_count_per_class",
        all(ref_counts[class_id] == int(design["reference_events_per_class"]) for class_id in class_ids),
        dict(sorted(ref_counts.items())),
        design["reference_events_per_class"],
    )
    add(
        "holdout_count_per_class",
        all(hold_counts[class_id] == int(design["holdout_events_per_class"]) for class_id in class_ids),
        dict(sorted(hold_counts.items())),
        design["holdout_events_per_class"],
    )
    add(
        "no_synthetic_iocs",
        not any(ioc.synthetic for event in events for ioc in event.iocs),
        sum(ioc.synthetic for event in events for ioc in event.iocs),
        0,
    )

    source_separated = True
    time_separated = True
    for class_id in class_ids:
        class_refs = [event for event in references if actor_ids[event.event_id] == class_id]
        class_holds = [event for event in holdouts if actor_ids[event.event_id] == class_id]
        ref_evidence = {value for event in class_refs for value in event.evidence_ids}
        hold_evidence = {value for event in class_holds for value in event.evidence_ids}
        source_separated &= ref_evidence.isdisjoint(hold_evidence)
        dated_refs = [event.first_seen for event in class_refs if event.first_seen is not None]
        dated_holds = [event.first_seen for event in class_holds if event.first_seen is not None]
        time_separated &= bool(dated_refs and dated_holds and max(dated_refs) < min(dated_holds))
    add("source_separation", source_separated, source_separated, True)
    add("time_separation", time_separated, time_separated, True)

    frozen = protocol["frozen_model"]
    add("embedding_backend", cfg.embedding.get("backend") == frozen["embedding_backend"], cfg.embedding.get("backend"), frozen["embedding_backend"])
    add("embedding_model", cfg.embedding.get("model") == frozen["embedding_model"], cfg.embedding.get("model"), frozen["embedding_model"])
    add(
        "attribution_weights",
        cfg.attribution.get("weights", {}) == frozen["attribution_weights"],
        cfg.attribution.get("weights", {}),
        frozen["attribution_weights"],
    )
    for key, expected in frozen["abstention"].items():
        observed = cfg.attribution.get("abstention", {}).get(key)
        add(f"abstention_{key}", observed == expected, observed, expected)
    return pd.DataFrame(checks)


def _prediction_table(hypotheses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id, group in hypotheses.groupby("query_event_id", sort=False):
        ordered = group.sort_values("rank")
        top = ordered.iloc[0]
        actual_id = str(top["actual_actor_id"])
        correct = ordered[ordered["candidate_actor_id"].astype(str) == actual_id]
        rows.append({
            "query_event_id": query_id,
            "actual_actor_id": actual_id,
            "actual_actor": top["actual_actor"],
            "predicted_actor_id": top["candidate_actor_id"],
            "predicted_actor": top["candidate_actor"],
            "actual_rank": int(correct["rank"].min()) if not correct.empty else None,
            "top1_correct": bool(top["candidate_actor_id"] == actual_id),
            "decision": top["decision"],
            "candidate_probability": float(top["candidate_probability"]),
            "assessment_confidence": float(top["assessment_confidence"]),
            "margin_to_next": float(top["margin_to_next"]),
            "evidence_families": top["evidence_families"],
            "supporting_event_ids": top["supporting_event_ids"],
        })
    return pd.DataFrame(rows)


def _class_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for actor_id, group in predictions.groupby("actual_actor_id", sort=True):
        reviewed = group[group["decision"] == "analyst_review"]
        rows.append({
            "actual_actor_id": actor_id,
            "actual_actor": group.iloc[0]["actual_actor"],
            "queries": len(group),
            "top1_accuracy": float(group["top1_correct"].mean()),
            "top3_accuracy": float((group["actual_rank"] <= 3).mean()),
            "review_coverage": len(reviewed) / len(group),
            "selective_accuracy": float(reviewed["top1_correct"].mean()) if not reviewed.empty else float("nan"),
            "false_attribution_rate": float((~reviewed["top1_correct"]).sum() / len(group)),
        })
    return pd.DataFrame(rows)


def _condition_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = comparison["query_event_id"].nunique()
    for condition, group in comparison.groupby("condition", sort=False):
        ranked = group[group["result_status"] == "ranked"]
        rows.append({
            "condition": condition,
            "queries": total,
            "signal_coverage": len(ranked) / total if total else 0.0,
            "top1_accuracy_all_queries": float(group["top1_correct"].mean()),
            "top1_accuracy_when_ranked": float(ranked["top1_correct"].mean()) if not ranked.empty else float("nan"),
        })
    return pd.DataFrame(rows)


def _acceptance_table(
    evaluation: pd.DataFrame,
    protocol: dict[str, Any],
    protocol_hash: str,
) -> pd.DataFrame:
    metrics = evaluation.iloc[0]
    criteria = protocol["acceptance_criteria"]
    mapping = {
        "minimum_queries_evaluable": ("queries_evaluable", ">="),
        "minimum_actor_labels": ("n_actor_labels", ">="),
        "minimum_top1_accuracy": ("top1_accuracy", ">="),
        "minimum_macro_top1_accuracy": ("macro_top1_accuracy", ">="),
        "minimum_top1_lift_over_majority": ("top1_lift_over_majority", ">="),
        "minimum_top3_accuracy": ("top3_accuracy", ">="),
        "minimum_review_coverage": ("review_coverage", ">="),
        "maximum_false_attribution_rate": ("false_attribution_rate", "<="),
    }
    rows: list[dict[str, Any]] = []
    for criterion, threshold in criteria.items():
        metric, operator = mapping[criterion]
        observed = float(metrics.get(metric, 0.0))
        passed = observed >= float(threshold) if operator == ">=" else observed <= float(threshold)
        rows.append({
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_hash,
            "criterion": criterion,
            "metric": metric,
            "operator": operator,
            "threshold": float(threshold),
            "observed": observed,
            "passed": bool(passed),
            "post_holdout_tuning": False,
        })
    overall = all(row["passed"] for row in rows)
    rows.append({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "criterion": "overall",
        "metric": "all_preregistered_criteria",
        "operator": "all",
        "threshold": 1.0,
        "observed": float(overall),
        "passed": overall,
        "post_holdout_tuning": False,
    })
    return pd.DataFrame(rows)


def run_multiactor_generalization(
    emb: Any,
    cfg: Any,
    calibrated_temperature: float,
) -> dict[str, Any]:
    """Run a label-isolated four-campaign external holdout evaluation."""
    events = load_multiactor_benchmark(cfg)
    if not events:
        return {}
    protocol, protocol_hash = _load_protocol(cfg)
    checks = validate_generalization_protocol(events, cfg, protocol)
    if not checks["passed"].all():
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise ValueError(f"Generalisation protocol validation failed: {failed}")

    reference_ids = {
        event.event_id for event in events if event.evaluation_role == "reference"
    }
    query_ids = {
        event.event_id for event in events if event.evaluation_role == "holdout"
    }

    scoring_events = _label_isolated_scoring_events(events)
    emb.encode_events(scoring_events)
    narrative = narrative_matrix(scoring_events, emb, cfg)
    graph = build_graph(scoring_events, cfg)
    infrastructure = ioc_matrix(scoring_events, graph, cfg)
    components = compute_components(scoring_events, cfg)
    scores = build_pairwise_scores(
        scoring_events,
        narrative,
        infrastructure,
        components,
        cfg,
        I_no_synthetic=infrastructure,
    )
    hypotheses = build_attribution_hypotheses(
        scoring_events,
        scores,
        cfg,
        query_ids=query_ids,
        reference_ids=reference_ids,
    )
    hypotheses = _attach_holdout_truth(hypotheses, events, cfg)
    hypotheses = apply_temperature_and_abstention(
        hypotheses,
        cfg,
        calibrated_temperature,
        split_role_override="generalization_holdout",
    )
    evaluation = evaluate_attribution_scope(
        hypotheses,
        cfg,
        "external_multiactor_generalization",
        {"generalization_holdout"},
    )
    predictions = _prediction_table(hypotheses)
    comparison = _condition_comparison(hypotheses)
    acceptance = _acceptance_table(evaluation, protocol, protocol_hash)
    return {
        "events": events,
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "protocol_checks": checks,
        "hypotheses": hypotheses,
        "evaluation": evaluation,
        "predictions": predictions,
        "class_metrics": _class_metrics(predictions),
        "condition_comparison": comparison,
        "condition_summary": _condition_summary(comparison),
        "error_analysis": build_error_analysis(hypotheses),
        "acceptance": acceptance,
        "pairwise": scores[
            scores["event_i"].isin(query_ids) | scores["event_j"].isin(query_ids)
        ].copy(),
    }
