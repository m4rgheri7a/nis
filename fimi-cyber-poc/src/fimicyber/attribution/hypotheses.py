"""Explainable, public-source actor-hypothesis ranking.

The actor label is used only to build historical reference profiles and to
evaluate held-out queries. It is never used as a query feature. Synthetic IOCs
are excluded by reading only ``I_no_synthetic`` from pairwise scores.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import networkx as nx
import numpy as np
import pandas as pd

from fimicyber.schema import Event
from fimicyber.attribution.taxonomy import ActorIdentity, load_actor_taxonomy


_FAMILIES = ("narrative", "infrastructure", "ttp", "channel", "temporal", "target")
_PAIR_COLUMNS = {
    "narrative": "N",
    "infrastructure": "I_no_synthetic",
    "ttp": "D",
    "channel": "C",
    "temporal": "T",
}
_LEGAL_CAVEAT = (
    "Public-source attribution support only; not a legal finding or proof of identity."
)


def _actor_key(label: Any) -> str | None:
    if not isinstance(label, str) or not label.strip():
        return None
    normalised = " ".join(label.strip().split())
    return normalised.casefold() or None


def _is_excluded(label: str | None, cfg: Any) -> bool:
    key = _actor_key(label)
    excluded = {
        str(value).strip().casefold()
        for value in cfg.attribution.get("excluded_labels", [])
    }
    return key is None or key in excluded


def _pair_lookup(scores_df: pd.DataFrame) -> dict[frozenset[str], dict[str, float | None]]:
    lookup: dict[frozenset[str], dict[str, float | None]] = {}
    for _, row in scores_df.iterrows():
        key = frozenset((str(row["event_i"]), str(row["event_j"])))
        values: dict[str, float | None] = {}
        for family, column in _PAIR_COLUMNS.items():
            raw = row.get(column)
            values[family] = None if raw is None or pd.isna(raw) else float(raw)
        lookup[key] = values
    return lookup


def _set_similarity(left: Iterable[str], right: Iterable[str]) -> float | None:
    a = {str(value).strip().casefold() for value in left if str(value).strip()}
    b = {str(value).strip().casefold() for value in right if str(value).strip()}
    if not a or not b:
        return None
    intersection = len(a & b)
    jaccard = intersection / len(a | b)
    overlap = intersection / min(len(a), len(b))
    return 0.5 * jaccard + 0.5 * overlap


def _target_similarity(query: Event, reference: Event) -> float | None:
    countries = _set_similarity(query.target_countries, reference.target_countries)
    sectors = _set_similarity(query.target_sectors, reference.target_sectors)
    available = [value for value in (countries, sectors) if value is not None]
    return float(np.mean(available)) if available else None


def _weighted_support(components: dict[str, float | None], cfg: Any) -> float:
    weights = cfg.attribution.get("weights", {})
    numerator = 0.0
    denominator = 0.0
    for family in _FAMILIES:
        value = components.get(family)
        weight = float(weights.get(family, 0.0))
        if value is None or weight <= 0:
            continue
        numerator += weight * float(value)
        denominator += weight
    if not denominator:
        return 0.0
    total_weight = sum(max(0.0, float(weights.get(family, 0.0))) for family in _FAMILIES)
    coverage = denominator / total_weight if total_weight else 0.0
    power = float(cfg.attribution.get("coverage_penalty_power", 0.7))
    return (numerator / denominator) * (coverage ** power)


def _source_orgs(events: Iterable[Event]) -> list[str]:
    orgs: set[str] = set()
    for event in events:
        for source in event.evidence_sources:
            host = (urlparse(source).hostname or "").lower()
            if host.startswith("www."):
                host = host[4:]
            if host:
                orgs.add(host)
    return sorted(orgs)


def _confidence_band(value: float, cfg: Any) -> str:
    conf = cfg.attribution.get("confidence", {})
    moderate = float(conf.get("moderate_threshold", 0.55))
    strong = float(conf.get("strong_threshold", 0.75))
    if value >= strong:
        return "strong_support"
    if value >= moderate:
        return "moderate"
    return "lead"


def assessment_confidence(
    probability: float,
    evidence_families: str | list[str],
    real_ioc_support: bool,
    source_org_count: int,
    cfg: Any,
) -> float:
    """Apply evidence-quality caps to a calibrated candidate probability."""
    families = (
        [value for value in evidence_families.split("|") if value]
        if isinstance(evidence_families, str)
        else list(evidence_families)
    )
    conf_cfg = cfg.attribution.get("confidence", {})
    quality = min(1.0, len(families) / len(_FAMILIES))
    confidence = float(probability) * (0.5 + 0.5 * quality)
    if not real_ioc_support:
        confidence = min(confidence, float(conf_cfg.get("no_real_ioc_cap", 0.70)))
    if int(source_org_count) <= 1:
        confidence = min(confidence, float(conf_cfg.get("single_source_cap", 0.60)))
    if len(families) < int(conf_cfg.get("min_evidence_families", 2)):
        confidence = min(
            confidence,
            float(conf_cfg.get("insufficient_evidence_cap", 0.45)),
        )
    return max(0.0, min(1.0, confidence))


def _label_mentioned(text: str, identity: ActorIdentity, aliases: set[str]) -> bool:
    haystack = text.casefold()
    labels = {identity.display_name.casefold(), *aliases}
    return any(label in haystack for label in labels if len(label) >= 4)


def _softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    temp = max(float(temperature), 1e-6)
    shifted = np.asarray(values, dtype=float) - max(values)
    exps = np.exp(shifted / temp)
    return (exps / exps.sum()).tolist()


def _is_historical(reference: Event, query: Event, temporal_only: bool) -> bool:
    if reference.event_id == query.event_id:
        return False
    if not temporal_only:
        return True
    if query.first_seen is None or reference.first_seen is None:
        return False
    return reference.first_seen < query.first_seen


def build_attribution_hypotheses(
    events: list[Event],
    scores_df: pd.DataFrame,
    cfg: Any,
    query_ids: Iterable[str] | None = None,
    reference_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Rank public-report actor hypotheses for each query event.

    Historical reference events are grouped by ``reported_actor``. Query actor
    labels are copied to ``actual_actor`` only after scoring for evaluation.
    """
    columns = [
        "query_event_id", "query_date", "query_date_basis",
        "candidate_actor_id", "candidate_actor", "candidate_actor_level",
        "candidate_parent_actor_id", "rank",
        "support_score", "candidate_probability", "assessment_confidence",
        "confidence_band", "margin_to_next", "display_candidate", "evidence_families",
        "source_org_count", "source_orgs", "real_ioc_support",
        "narrative_label_guarded_pairs",
        "supporting_event_ids", "narrative_score", "infrastructure_score",
        "ttp_score", "channel_score", "temporal_score", "target_score",
        "actual_actor_id", "actual_actor", "actual_actor_level",
        "correct_actor", "temporal_only", "evidence_path",
        "caveat",
    ]
    if not cfg.attribution.get("enabled", True):
        return pd.DataFrame(columns=columns)

    selected = set(query_ids) if query_ids is not None else None
    permitted_references = set(reference_ids) if reference_ids is not None else None
    temporal_only = bool(cfg.attribution.get("temporal_only", True))
    min_history = int(cfg.attribution.get("min_actor_history", 2))
    top_k_refs = int(cfg.attribution.get("top_k_reference_events", 3))
    top_k_candidates = int(cfg.attribution.get("top_k_candidates", 3))
    support_threshold = float(cfg.attribution.get("support_threshold", 0.2))
    temperature = float(cfg.attribution.get("temperature", 0.15))
    taxonomy = load_actor_taxonomy(cfg)

    lookup = _pair_lookup(scores_df)
    identities: dict[str, ActorIdentity] = {}
    event_identities: dict[str, ActorIdentity] = {}
    labelled_events: list[Event] = []
    for event in events:
        if permitted_references is not None and event.event_id not in permitted_references:
            continue
        if _is_excluded(event.reported_actor, cfg):
            continue
        identity = taxonomy.resolve(event.reported_actor)
        if identity is None:
            continue
        identities.setdefault(identity.actor_id, identity)
        event_identities[event.event_id] = identity
        labelled_events.append(event)

    rows: list[dict[str, Any]] = []
    for query in events:
        if selected is not None and query.event_id not in selected:
            continue

        histories: dict[str, list[Event]] = defaultdict(list)
        for reference in labelled_events:
            if _is_historical(reference, query, temporal_only):
                identity = event_identities.get(reference.event_id)
                if identity is not None:
                    histories[identity.actor_id].append(reference)
        histories = {
            actor: refs for actor, refs in histories.items() if len(refs) >= min_history
        }
        if not histories:
            continue

        candidates: list[dict[str, Any]] = []
        for actor, references in histories.items():
            reference_support: list[tuple[float, Event, dict[str, float | None], bool]] = []
            for reference in references:
                components = dict(lookup.get(
                    frozenset((query.event_id, reference.event_id)), {}
                ))
                for family in _PAIR_COLUMNS:
                    components.setdefault(family, None)
                identity = identities[actor]
                aliases = taxonomy.aliases_for(actor)
                narrative_guarded = (
                    _label_mentioned(query.description, identity, aliases)
                    or _label_mentioned(reference.description, identity, aliases)
                )
                if narrative_guarded:
                    components["narrative"] = None
                components["target"] = _target_similarity(query, reference)
                support = _weighted_support(components, cfg)
                reference_support.append((support, reference, components, narrative_guarded))

            reference_support.sort(key=lambda item: item[0], reverse=True)
            top_refs = reference_support[:top_k_refs]
            if not top_refs:
                continue

            aggregates: dict[str, float | None] = {}
            for family in _FAMILIES:
                values = [
                    float(item[2][family])
                    for item in top_refs
                    if item[2].get(family) is not None
                ]
                aggregates[family] = float(np.mean(values)) if values else None

            support_score = float(np.mean([item[0] for item in top_refs]))
            evidence_families = [
                family for family, value in aggregates.items()
                if value is not None and value >= support_threshold
            ]
            real_ioc_support = bool(
                aggregates.get("infrastructure") is not None
                and float(aggregates["infrastructure"]) > 0
            )
            supporting_events = [item[1] for item in top_refs]
            orgs = _source_orgs([query, *supporting_events])
            candidates.append({
                "actor_key": actor,
                "identity": identities[actor],
                "candidate_actor": identities[actor].display_name,
                "support_score": support_score,
                "aggregates": aggregates,
                "evidence_families": evidence_families,
                "real_ioc_support": real_ioc_support,
                "supporting_events": supporting_events,
                "source_orgs": orgs,
                "narrative_label_guarded_pairs": sum(int(item[3]) for item in top_refs),
            })

        candidates.sort(key=lambda item: item["support_score"], reverse=True)
        probabilities = _softmax(
            [item["support_score"] for item in candidates], temperature
        )
        for item, probability in zip(candidates, probabilities):
            item["candidate_probability"] = probability

        # Keep the complete candidate ranking for honest Top-k evaluation.
        # ``display_candidate`` marks the compact analyst-facing shortlist.
        shown = candidates
        actual_identity = (
            None if _is_excluded(query.reported_actor, cfg)
            else taxonomy.resolve(query.reported_actor)
        )
        actual_key = actual_identity.actor_id if actual_identity else None
        for rank, item in enumerate(shown, start=1):
            next_score = (
                shown[rank]["support_score"] if rank < len(shown) else 0.0
            )
            margin = max(0.0, item["support_score"] - next_score)
            confidence = assessment_confidence(
                item["candidate_probability"],
                item["evidence_families"],
                item["real_ioc_support"],
                len(item["source_orgs"]),
                cfg,
            )

            aggregates = item["aggregates"]
            support_ids = [event.event_id for event in item["supporting_events"]]
            evidence_path = (
                f"Event:{query.event_id} -> ReferenceEvents:{'|'.join(support_ids)} "
                f"-> ActorHypothesis:{item['candidate_actor']}"
            )
            rows.append({
                "query_event_id": query.event_id,
                "query_date": str(query.first_seen or ""),
                "query_date_basis": query.date_basis,
                "candidate_actor_id": item["actor_key"],
                "candidate_actor": item["candidate_actor"],
                "candidate_actor_level": item["identity"].actor_level,
                "candidate_parent_actor_id": item["identity"].parent_actor_id,
                "rank": rank,
                "support_score": item["support_score"],
                "candidate_probability": item["candidate_probability"],
                "assessment_confidence": confidence,
                "confidence_band": _confidence_band(confidence, cfg),
                "margin_to_next": margin,
                "display_candidate": rank <= top_k_candidates,
                "evidence_families": "|".join(item["evidence_families"]),
                "source_org_count": len(item["source_orgs"]),
                "source_orgs": "|".join(item["source_orgs"]),
                "real_ioc_support": item["real_ioc_support"],
                "narrative_label_guarded_pairs": item["narrative_label_guarded_pairs"],
                "supporting_event_ids": "|".join(support_ids),
                "narrative_score": aggregates.get("narrative"),
                "infrastructure_score": aggregates.get("infrastructure"),
                "ttp_score": aggregates.get("ttp"),
                "channel_score": aggregates.get("channel"),
                "temporal_score": aggregates.get("temporal"),
                "target_score": aggregates.get("target"),
                "actual_actor_id": actual_key,
                "actual_actor": actual_identity.display_name if actual_identity else query.reported_actor,
                "actual_actor_level": actual_identity.actor_level if actual_identity else "unknown",
                "correct_actor": actual_key is not None and item["actor_key"] == actual_key,
                "temporal_only": temporal_only,
                "evidence_path": evidence_path,
                "caveat": _LEGAL_CAVEAT,
            })

    return pd.DataFrame(rows, columns=columns)


def evaluate_attribution(hypotheses: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Evaluate temporal actor-hypothesis ranking with ranking and calibration metrics."""
    columns = [
        "queries_total", "queries_labeled", "queries_evaluable", "history_coverage",
        "n_actor_labels", "majority_baseline_accuracy", "top1_accuracy",
        "macro_top1_accuracy", "top1_lift_over_majority", "top3_accuracy", "MRR",
        "multiclass_brier", "ECE", "temporal_only",
    ]
    if hypotheses.empty:
        return pd.DataFrame([{column: 0.0 for column in columns}])

    total = hypotheses["query_event_id"].nunique()
    evaluable_groups: list[pd.DataFrame] = []
    labeled_groups = 0
    for _, group in hypotheses.groupby("query_event_id", sort=False):
        actual_label = group.iloc[0].get("actual_actor")
        if _is_excluded(actual_label, cfg):
            continue
        labeled_groups += 1
        actual = group.iloc[0].get("actual_actor_id") or _actor_key(actual_label)
        candidate_keys = (
            set(group["candidate_actor_id"].astype(str))
            if "candidate_actor_id" in group
            else {_actor_key(value) for value in group["candidate_actor"]}
        )
        if actual is not None and actual in candidate_keys:
            evaluable_groups.append(group.sort_values("rank"))

    if not evaluable_groups:
        return pd.DataFrame([{
            "queries_total": total,
            "queries_labeled": labeled_groups,
            "queries_evaluable": 0,
            "history_coverage": 0.0,
            "top1_accuracy": 0.0,
            "n_actor_labels": 0,
            "majority_baseline_accuracy": 0.0,
            "macro_top1_accuracy": 0.0,
            "top1_lift_over_majority": 0.0,
            "top3_accuracy": 0.0,
            "MRR": 0.0,
            "multiclass_brier": 0.0,
            "ECE": 0.0,
            "temporal_only": bool(cfg.attribution.get("temporal_only", True)),
        }], columns=columns)

    top1_hits: list[float] = []
    top3_hits: list[float] = []
    reciprocal_ranks: list[float] = []
    brier_scores: list[float] = []
    top_confidences: list[float] = []
    top_correct: list[float] = []
    actual_labels: list[str] = []

    for group in evaluable_groups:
        correct_rows = group[group["correct_actor"].astype(bool)]
        correct_rank = int(correct_rows["rank"].min())
        actual_labels.append(str(group.iloc[0].get("actual_actor_id") or group.iloc[0]["actual_actor"]))
        top1_hits.append(float(correct_rank == 1))
        top3_hits.append(float(correct_rank <= 3))
        reciprocal_ranks.append(1.0 / correct_rank)

        probabilities = group["candidate_probability"].astype(float).to_numpy()
        labels = group["correct_actor"].astype(bool).astype(float).to_numpy()
        brier_scores.append(float(np.sum((probabilities - labels) ** 2)))
        top = group.iloc[0]
        top_confidences.append(float(top["candidate_probability"]))
        top_correct.append(float(bool(top["correct_actor"])))

    ece = _expected_calibration_error(top_confidences, top_correct)
    label_counts = pd.Series(actual_labels).value_counts()
    majority_baseline = float(label_counts.max() / len(actual_labels))
    per_label_accuracy = []
    for label in label_counts.index:
        mask = [value == label for value in actual_labels]
        hits = [hit for hit, selected in zip(top1_hits, mask) if selected]
        per_label_accuracy.append(float(np.mean(hits)))
    macro_top1 = float(np.mean(per_label_accuracy))
    top1_accuracy = float(np.mean(top1_hits))
    return pd.DataFrame([{
        "queries_total": total,
        "queries_labeled": labeled_groups,
        "queries_evaluable": len(evaluable_groups),
        "history_coverage": len(evaluable_groups) / labeled_groups if labeled_groups else 0.0,
        "n_actor_labels": len(label_counts),
        "majority_baseline_accuracy": majority_baseline,
        "top1_accuracy": top1_accuracy,
        "macro_top1_accuracy": macro_top1,
        "top1_lift_over_majority": top1_accuracy - majority_baseline,
        "top3_accuracy": float(np.mean(top3_hits)),
        "MRR": float(np.mean(reciprocal_ranks)),
        "multiclass_brier": float(np.mean(brier_scores)),
        "ECE": ece,
        "temporal_only": bool(cfg.attribution.get("temporal_only", True)),
    }], columns=columns)


def _expected_calibration_error(
    confidences: list[float], outcomes: list[float], n_bins: int = 10
) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    out = np.asarray(outcomes, dtype=float)
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, n_bins, endpoint=False):
        upper = lower + 1.0 / n_bins
        mask = (conf >= lower) & (conf < upper if upper < 1.0 else conf <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(conf[mask].mean()) - float(out[mask].mean()))
    return ece


def build_attribution_graph(
    events: list[Event], hypotheses: pd.DataFrame, output_path: Path
) -> nx.DiGraph:
    """Write a graph that separates reported references from assessed hypotheses."""
    graph = nx.DiGraph()
    event_map = {event.event_id: event for event in events}
    top = hypotheses[hypotheses["rank"] == 1] if not hypotheses.empty else hypotheses

    for _, row in top.iterrows():
        query_id = str(row["query_event_id"])
        actor = str(row["candidate_actor"])
        query_node = f"Event:{query_id}"
        actor_node = f"ActorHypothesis:{actor}"
        graph.add_node(query_node, ntype="Event", value=query_id, role="query")
        graph.add_node(actor_node, ntype="ActorHypothesis", value=actor)
        graph.add_edge(
            query_node,
            actor_node,
            etype="ASSESSED_AS",
            support_score=float(row["support_score"]),
            confidence=float(row["assessment_confidence"]),
            confidence_band=str(row["confidence_band"]),
            caveat=_LEGAL_CAVEAT,
        )

        ref_ids = [value for value in str(row["supporting_event_ids"]).split("|") if value]
        for ref_id in ref_ids:
            ref = event_map.get(ref_id)
            ref_node = f"ReferenceEvent:{ref_id}"
            graph.add_node(ref_node, ntype="ReferenceEvent", value=ref_id)
            graph.add_edge(query_node, ref_node, etype="SUPPORTED_BY")
            if ref and ref.reported_actor:
                graph.add_edge(
                    ref_node,
                    actor_node,
                    etype="REPORTED_ATTRIBUTION",
                    reference_only=True,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(nx.node_link_data(graph, edges="links"), ensure_ascii=False),
        encoding="utf-8",
    )
    return graph
