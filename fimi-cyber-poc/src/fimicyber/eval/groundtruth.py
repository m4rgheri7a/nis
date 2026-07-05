"""Ground truth construction (spec 10.1).

positive pair = same campaign_id (not None).
Query set Q = events where campaign size ≥ 2.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from fimicyber.schema import Event


def build_ground_truth(
    events: list[Event],
    include_actor_surrogate: bool = True,
) -> dict[str, Any]:
    """
    Returns:
      positives: set of frozenset({event_id_a, event_id_b})
      query_ids: list of event_ids eligible as queries (campaign size ≥ 2)
      campaign_map: {campaign_id: [event_ids]}
    """
    camp_map: dict[str, list[str]] = defaultdict(list)
    for ev in events:
        if ev.campaign_id_source == "actor_surrogate" and not include_actor_surrogate:
            continue
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
        "include_actor_surrogate": include_actor_surrogate,
    }


def gt_stats(gt: dict[str, Any], events: list[Event] | None = None) -> dict[str, Any]:
    camps = gt["campaign_map"]
    eligible = {c: ids for c, ids in camps.items() if len(ids) >= 2}
    stats = {
        "n_campaigns_total": len(camps),
        "n_campaigns_eligible": len(eligible),
        "n_positive_pairs": len(gt["positives"]),
        "n_query_events": len(gt["query_ids"]),
        "campaign_sizes": {c: len(ids) for c, ids in eligible.items()},
        "include_actor_surrogate": gt.get("include_actor_surrogate", True),
    }
    if events is not None:
        stats.update({
            "n_events_actor_surrogate": sum(
                1 for ev in events if ev.campaign_id_source == "actor_surrogate"
            ),
            "n_events_debunk_group": sum(
                1 for ev in events if ev.campaign_id_source == "debunk_group"
            ),
            "n_events_explicit_campaign": sum(
                1 for ev in events if ev.campaign_id and ev.campaign_id_source == "explicit"
            ),
            "n_events_fixture_campaign": sum(
                1 for ev in events if ev.campaign_id and ev.campaign_id_source == "fixture"
            ),
        })
    return stats
