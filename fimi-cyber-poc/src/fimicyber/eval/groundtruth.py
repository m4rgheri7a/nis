"""Ground truth construction (spec 10.1).

positive pair = same campaign_id (not None).
Query set Q = events where campaign size ≥ 2.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from fimicyber.schema import Event


def build_ground_truth(events: list[Event]) -> dict[str, Any]:
    """
    Returns:
      positives: set of frozenset({event_id_a, event_id_b})
      query_ids: list of event_ids eligible as queries (campaign size ≥ 2)
      campaign_map: {campaign_id: [event_ids]}
    """
    camp_map: dict[str, list[str]] = defaultdict(list)
    for ev in events:
        if ev.campaign_id:
            camp_map[ev.campaign_id].append(ev.event_id)

    positives: set[frozenset] = set()
    query_ids: list[str] = []

    for camp_id, ids in camp_map.items():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                positives.add(frozenset({ids[i], ids[j]}))
        query_ids.extend(ids)

    return {
        "positives": positives,
        "query_ids": list(set(query_ids)),
        "campaign_map": dict(camp_map),
    }


def gt_stats(gt: dict[str, Any]) -> dict[str, Any]:
    camps = gt["campaign_map"]
    eligible = {c: ids for c, ids in camps.items() if len(ids) >= 2}
    return {
        "n_campaigns_total": len(camps),
        "n_campaigns_eligible": len(eligible),
        "n_positive_pairs": len(gt["positives"]),
        "n_query_events": len(gt["query_ids"]),
        "campaign_sizes": {c: len(ids) for c, ids in eligible.items()},
    }
