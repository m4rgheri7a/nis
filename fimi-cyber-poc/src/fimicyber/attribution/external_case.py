"""Source-separated Ghostwriter case application."""
from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from fimicyber.attribution.calibration import (
    apply_temperature_and_abstention,
    build_error_analysis,
    evaluate_attribution_scope,
)
from fimicyber.attribution.hypotheses import build_attribution_hypotheses
from fimicyber.graph.build import build_graph
from fimicyber.graph.ioc_score import ioc_matrix
from fimicyber.loaders.external import load_external_case
from fimicyber.nlp.narrative import narrative_matrix
from fimicyber.scoring.components import compute_components
from fimicyber.scoring.fcls import build_pairwise_scores


def _condition_comparison(hypotheses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    conditions = {
        "content_only": "narrative_score",
        "ioc_only": "infrastructure_score",
        "integrated": "support_score",
    }
    for query_id, group in hypotheses.groupby("query_event_id", sort=False):
        actual = str(group.iloc[0].get("actual_actor_id") or "")
        for condition, column in conditions.items():
            numeric = pd.to_numeric(group[column], errors="coerce")
            valid = group[numeric > 0].copy()
            if valid.empty:
                rows.append({
                    "query_event_id": query_id,
                    "condition": condition,
                    "result_status": "no_signal",
                    "top_candidate": "",
                    "top_score": 0.0,
                    "actual_actor": group.iloc[0].get("actual_actor", ""),
                    "actual_rank": None,
                    "top1_correct": False,
                })
                continue
            valid["_condition_score"] = pd.to_numeric(valid[column], errors="coerce")
            ordered = valid.sort_values(
                ["_condition_score", "candidate_actor_id"], ascending=[False, True]
            )
            actual_rows = ordered[ordered["candidate_actor_id"].astype(str) == actual]
            actual_rank = (
                int(ordered.index.get_loc(actual_rows.index[0])) + 1
                if not actual_rows.empty else None
            )
            rows.append({
                "query_event_id": query_id,
                "condition": condition,
                "result_status": "ranked",
                "top_candidate": ordered.iloc[0]["candidate_actor"],
                "top_score": float(ordered.iloc[0]["_condition_score"]),
                "actual_actor": group.iloc[0].get("actual_actor", ""),
                "actual_rank": actual_rank,
                "top1_correct": actual_rank == 1,
            })
    return pd.DataFrame(rows)


def run_external_ghostwriter_case(
    base_events: list,
    emb: Any,
    cfg: Any,
    calibrated_temperature: float,
) -> dict[str, Any]:
    external_events = load_external_case(cfg)
    if not external_events:
        return {}

    # Keep the external case free of generated infrastructure and limit the
    # comparison pool to events that can form reported-actor histories.
    base_clean = copy.deepcopy([
        event for event in base_events
        if event.reported_actor and event.first_seen is not None
    ])
    for event in base_clean:
        event.iocs = [ioc for ioc in event.iocs if not ioc.synthetic]
    pool = base_clean + external_events

    emb.encode_events(pool)
    narrative = narrative_matrix(pool, emb, cfg)
    graph = build_graph(pool, cfg)
    infrastructure = ioc_matrix(pool, graph, cfg)
    components = compute_components(pool, cfg)
    scores = build_pairwise_scores(
        pool,
        narrative,
        infrastructure,
        components,
        cfg,
        I_no_synthetic=infrastructure,
    )
    holdout_ids = {
        event.event_id for event in external_events if event.evaluation_role == "holdout"
    }
    hypotheses = build_attribution_hypotheses(pool, scores, cfg, query_ids=holdout_ids)
    hypotheses = apply_temperature_and_abstention(
        hypotheses,
        cfg,
        calibrated_temperature,
        split_role_override="external_holdout",
    )
    comparison = _condition_comparison(hypotheses)
    pair_mask = scores["event_i"].isin(holdout_ids) | scores["event_j"].isin(holdout_ids)
    return {
        "events": external_events,
        "hypotheses": hypotheses,
        "evaluation": evaluate_attribution_scope(
            hypotheses, cfg, "external_ghostwriter", {"external_holdout"}
        ),
        "condition_comparison": comparison,
        "error_analysis": build_error_analysis(hypotheses),
        "pairwise": scores[pair_mask].copy(),
    }
