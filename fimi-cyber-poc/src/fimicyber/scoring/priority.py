"""Priority(i) computation (spec 9)."""
from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from fimicyber.schema import Event

_HIGH_IMPACT = {"elections", "military", "diplomacy", "government"}
_MED_IMPACT = {"social", "health", "society"}

_IMPACT_MAP: dict[str, float] = {
    **{s: 1.0 for s in _HIGH_IMPACT},
    **{s: 0.7 for s in _MED_IMPACT},
}

_TODAY = date(2026, 7, 2)  # frozen per session date in spec context


def _impact_score(sectors: list[str]) -> tuple[float, bool]:
    if not sectors:
        return 0.5, True
    scores = [_IMPACT_MAP.get(s.lower(), 0.5) for s in sectors]
    val = max(scores)
    defaulted = all(s.lower() not in _IMPACT_MAP for s in sectors)
    return val, defaulted


def _evidence_confidence(ev: Event) -> float:
    n_src = len(ev.evidence_sources)
    corr = min(1.0, n_src / 3.0)
    # Source trust: use presence of known trusted domains
    from fimicyber.ioc.confidence import _TRUSTED_ORGS, _MAJOR_MEDIA
    trust_scores = []
    for src in ev.evidence_sources:
        if any(org in src.lower() for org in _TRUSTED_ORGS):
            trust_scores.append(1.0)
        elif any(m in src.lower() for m in _MAJOR_MEDIA):
            trust_scores.append(0.7)
        else:
            trust_scores.append(0.4)
    src_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.4
    return corr * src_trust


def _cyber_relevance(ev: Event) -> float:
    op_iocs = [ioc for ioc in ev.iocs if ioc.category == "OperationalIOC"]
    if not op_iocs:
        return 0.0
    avg_conf = sum(ioc.confidence for ioc in op_iocs) / len(op_iocs)
    return avg_conf


def _urgency(ev: Event) -> float:
    last = ev.last_seen
    if last is None:
        return 0.5
    gap_days = (_TODAY - last).days
    return math.exp(-gap_days / 365.0)


def _renorm(components: dict[str, float | None], weights: dict[str, float]) -> float:
    num, den = 0.0, 0.0
    for k, w in weights.items():
        v = components.get(k)
        if v is None or math.isnan(v):
            continue
        num += w * v
        den += w
    return num / den if den > 0 else float("nan")


def compute_priority(
    events: list[Event],
    scores_df: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    theta = cfg.priority["theta"]
    weights = {
        "link": float(theta.get("link", 0.35)),
        "impact": float(theta.get("impact", 0.15)),
        "evidence": float(theta.get("evidence", 0.15)),
        "cyber": float(theta.get("cyber", 0.20)),
        "urgency": float(theta.get("urgency", 0.15)),
    }

    # Build event_id → top-3 FCLS_E3 average
    ev_ids = [ev.event_id for ev in events]
    link_scores: dict[str, float] = {}

    for ev in events:
        eid = ev.event_id
        # Get all E3 scores involving this event
        mask = (scores_df["event_i"] == eid) | (scores_df["event_j"] == eid)
        related = scores_df[mask]["FCLS_E3"].dropna().sort_values(ascending=False)
        top3 = related.head(3)
        link_scores[eid] = float(top3.mean()) if len(top3) > 0 else 0.0

    rows = []
    for ev in events:
        impact, defaulted = _impact_score(ev.target_sectors)
        comp = {
            "link": link_scores.get(ev.event_id, 0.0),
            "impact": impact,
            "evidence": _evidence_confidence(ev),
            "cyber": _cyber_relevance(ev),
            "urgency": _urgency(ev),
        }
        priority = _renorm(comp, weights)

        # Top-3 related events
        eid = ev.event_id
        mask = (scores_df["event_i"] == eid) | (scores_df["event_j"] == eid)
        sub = scores_df[mask].copy()
        sub["related"] = sub.apply(
            lambda r: r["event_j"] if r["event_i"] == eid else r["event_i"], axis=1
        )
        top_related = sub.nlargest(3, "FCLS_E3")["related"].tolist()

        rows.append({
            "event_id": eid,
            "priority": round(priority, 6),
            "link_strength": round(comp["link"], 4),
            "impact": round(comp["impact"], 4),
            "evidence_confidence": round(comp["evidence"], 4),
            "cyber_relevance": round(comp["cyber"], 4),
            "urgency": round(comp["urgency"], 4),
            "impact_defaulted": defaulted,
            "top_related": ";".join(top_related),
        })

    return pd.DataFrame(rows).sort_values("priority", ascending=False)
