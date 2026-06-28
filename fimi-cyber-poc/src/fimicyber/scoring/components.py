"""Auxiliary components: D (TTP), C (Channel), T (temporal), A (actor) — spec 7."""
from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np

from fimicyber.schema import Event


def jaccard(a: set, b: set) -> float | None:
    if not a or not b:
        return None
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def overlap(a: set, b: set) -> float | None:
    if not a or not b:
        return None
    inter = len(a & b)
    return inter / min(len(a), len(b))


def sim_sets(a: set, b: set, mode: str) -> float | None:
    """Compute similarity by mode: jaccard | overlap | mix."""
    if not a or not b:
        return None
    if mode == "jaccard":
        return jaccard(a, b)
    if mode == "overlap":
        return overlap(a, b)
    # mix
    j = jaccard(a, b)
    o = overlap(a, b)
    if j is None or o is None:
        return None
    return 0.5 * j + 0.5 * o


def _date_gap_days(
    first_i: date | None, last_i: date | None,
    first_j: date | None, last_j: date | None,
) -> float | None:
    """Gap between two date intervals (0 if overlapping, else positive days)."""
    if first_i is None or first_j is None:
        return None
    li = last_i or first_i
    lj = last_j or first_j

    if first_i <= lj and first_j <= li:
        return 0.0

    if first_i > lj:
        return float((first_i - lj).days)
    return float((first_j - li).days)


def compute_components(
    events: list[Event],
    cfg: Any,
) -> dict[str, np.ndarray]:
    """
    Return dict of component matrices D, C, T, A each shape (n,n).
    Values ∈ [0,1] or NaN for missing.
    """
    n = len(events)
    ttp_mode = cfg.components.get("ttp_sim_mode", "mix")
    ch_mode = cfg.components.get("channel_sim_mode", "mix")
    tau_event = float(cfg.components.get("tau_event_days", 90))

    D = np.full((n, n), float("nan"))
    C = np.full((n, n), float("nan"))
    T = np.full((n, n), float("nan"))
    A = np.full((n, n), float("nan"))

    for i in range(n):
        for j in range(i + 1, n):
            ei, ej = events[i], events[j]

            # ── D: TTP similarity ────────────────────────────────────────
            d_val = sim_sets(set(ei.ttps), set(ej.ttps), ttp_mode)
            D[i, j] = D[j, i] = d_val if d_val is not None else float("nan")

            # ── C: Channel similarity ────────────────────────────────────
            c_val = sim_sets(set(ei.channels), set(ej.channels), ch_mode)
            C[i, j] = C[j, i] = c_val if c_val is not None else float("nan")

            # ── T: Temporal proximity ────────────────────────────────────
            gap = _date_gap_days(ei.first_seen, ei.last_seen, ej.first_seen, ej.last_seen)
            if gap is None:
                T[i, j] = T[j, i] = float("nan")
            elif gap == 0.0:
                T[i, j] = T[j, i] = 1.0
            else:
                T[i, j] = T[j, i] = math.exp(-gap / tau_event)

            # ── A: Actor context (not used in eval) ──────────────────────
            a_val = _actor_sim(ei, ej)
            A[i, j] = A[j, i] = a_val if a_val is not None else float("nan")

    return {"D": D, "C": C, "T": T, "A": A}


def _actor_sim(ei: Event, ej: Event) -> float | None:
    """0.5·actor_match + 0.5·Jaccard(target_countries). Missing if both actors None."""
    countries_j = jaccard(set(ei.target_countries), set(ej.target_countries))
    country_sim = countries_j if countries_j is not None else 0.0

    if ei.reported_actor is None and ej.reported_actor is None:
        if not ei.target_countries or not ej.target_countries:
            return None
        return country_sim

    if ei.reported_actor is None or ej.reported_actor is None:
        # Only country component
        if not ei.target_countries or not ej.target_countries:
            return None
        return country_sim

    actor_match = 1.0 if ei.reported_actor.strip().lower() == ej.reported_actor.strip().lower() else 0.0
    return 0.5 * actor_match + 0.5 * country_sim
