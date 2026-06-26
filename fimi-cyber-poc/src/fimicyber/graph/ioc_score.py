"""IOC-based similarity scores: I_direct and I_path (spec 6.2, 6.3).

I_direct(i,j) = Σ_{o∈Oi∩Oj} W(o) / Σ_{o∈Oi∪Oj} W(o)
I_path(i,j) = max_path( exp(−ρ·len) · path_conf · temporal_overlap )
I(i,j) = μ·I_direct + (1−μ)·I_path
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any

import networkx as nx
import numpy as np

from fimicyber.schema import Event
from fimicyber.graph.build import infra_subgraph, _node_id


def _type_weight(ioc_type: str, cfg: Any) -> float:
    return float(cfg.ioc_score["type_weight"].get(ioc_type, 0.3))


def _rarity(ioc_value: str, event_ioc_map: dict[str, set[str]]) -> float:
    df = sum(1 for ioc_set in event_ioc_map.values() if ioc_value in ioc_set)
    return 1.0 / max(df, 1)


def _ioc_weight(
    ioc_value: str,
    ioc_type: str,
    confidence: float,
    cfg: Any,
    event_ioc_map: dict[str, set[str]],
) -> float:
    tw = _type_weight(ioc_type, cfg)
    rarity = _rarity(ioc_value, event_ioc_map)
    return tw * confidence * rarity


def _temporal_overlap(
    ioc_first_i: str, ioc_last_i: str,
    ioc_first_j: str, ioc_last_j: str,
    ioc_type: str,
    cfg: Any,
) -> float:
    tau = float(cfg.ioc_score["tau_days"].get(ioc_type, 60))

    def _parse(s: str) -> date | None:
        if not s:
            return None
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None

    fi = _parse(ioc_first_i)
    li = _parse(ioc_last_i) or fi
    fj = _parse(ioc_first_j)
    lj = _parse(ioc_last_j) or fj

    if fi is None or fj is None:
        return 1.0  # unknown → no penalty

    # Overlap: [fi, li] ∩ [fj, lj]
    if fi <= lj and fj <= li:
        return 1.0

    # Gap
    gap = max((fi - lj).days if fi > lj else 0, (fj - li).days if fj > li else 0)
    return math.exp(-gap / tau)


def i_direct(
    ev_i: Event,
    ev_j: Event,
    cfg: Any,
    event_ioc_map: dict[str, set[str]],
    ioc_meta: dict[str, tuple[str, float]],  # value → (type, confidence)
) -> float | None:
    """Return I_direct or None if either set is empty."""
    oi = {ioc.value for ioc in ev_i.iocs if ioc.category == "OperationalIOC"}
    oj = {ioc.value for ioc in ev_j.iocs if ioc.category == "OperationalIOC"}

    if not oi or not oj:
        return None

    intersection = oi & oj
    union = oi | oj

    def w(val: str) -> float:
        itype, conf = ioc_meta.get(val, ("domain", 0.5))
        return _ioc_weight(val, itype, conf, cfg, event_ioc_map)

    num = sum(w(v) for v in intersection)
    den = sum(w(v) for v in union)

    return num / den if den > 0 else None


def i_path(
    ev_i: Event,
    ev_j: Event,
    G: nx.Graph,
    cfg: Any,
) -> float | None:
    """Return I_path or None if no valid path exists."""
    sub = infra_subgraph(G)

    oi_nodes = {_node_id("IOC", ioc.value) for ioc in ev_i.iocs if ioc.category == "OperationalIOC"}
    oj_nodes = {_node_id("IOC", ioc.value) for ioc in ev_j.iocs if ioc.category == "OperationalIOC"}

    oi_nodes = {n for n in oi_nodes if n in sub}
    oj_nodes = {n for n in oj_nodes if n in sub}

    if not oi_nodes or not oj_nodes:
        return None

    max_path_len = int(cfg.ioc_score.get("max_path_len", 4))
    rho = float(cfg.ioc_score.get("rho", 0.5))

    best = 0.0

    for src in oi_nodes:
        for dst in oj_nodes:
            if src == dst:
                continue  # direct share → I_direct handles this
            try:
                paths = list(
                    nx.all_simple_paths(sub, source=src, target=dst, cutoff=max_path_len)
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            for path in paths:
                # Verify no Event nodes in path (T14 constraint)
                if any(G.nodes[n].get("ntype") == "Event" for n in path):
                    continue

                path_len = len(path) - 1  # number of edges
                if path_len > max_path_len:
                    continue

                # path_confidence = product of edge confidences
                path_conf = 1.0
                edges_data: list[dict] = []
                for k in range(path_len):
                    edge_data = sub.edges[path[k], path[k + 1]]
                    path_conf *= float(edge_data.get("confidence", 1.0))
                    edges_data.append(edge_data)

                # temporal_overlap using endpoints' IOC data
                src_node = G.nodes.get(src, {})
                dst_node = G.nodes.get(dst, {})
                src_type = src_node.get("ioc_type", "domain")
                tau_type = min(
                    float(cfg.ioc_score["tau_days"].get(src_type, 60)),
                    float(cfg.ioc_score["tau_days"].get(dst_node.get("ioc_type", "domain"), 60)),
                )
                temp_ov = _temporal_overlap(
                    src_node.get("first_seen", ""),
                    src_node.get("last_seen", ""),
                    dst_node.get("first_seen", ""),
                    dst_node.get("last_seen", ""),
                    src_type, cfg,
                )

                score = math.exp(-rho * path_len) * path_conf * temp_ov
                if score > best:
                    best = score

    return best if best > 0 else None


def ioc_matrix(
    events: list[Event],
    G: nx.Graph,
    cfg: Any,
) -> np.ndarray:
    """Return n×n symmetric matrix of I(i,j). NaN where missing."""
    n = len(events)
    mu = float(cfg.ioc_score.get("mu", 0.6))

    # Build lookup maps
    event_ioc_map: dict[str, set[str]] = {}
    ioc_meta: dict[str, tuple[str, float]] = {}

    for ev in events:
        vals: set[str] = set()
        for ioc in ev.iocs:
            if ioc.category == "OperationalIOC":
                vals.add(ioc.value)
                if ioc.value not in ioc_meta:
                    ioc_meta[ioc.value] = (ioc.ioc_type, ioc.confidence)
        event_ioc_map[ev.event_id] = vals

    mat = np.full((n, n), float("nan"), dtype=np.float64)

    for i in range(n):
        mat[i, i] = float("nan")
        for j in range(i + 1, n):
            id_ = i_direct(events[i], events[j], cfg, event_ioc_map, ioc_meta)
            ip = i_path(events[i], events[j], G, cfg)

            if id_ is None and ip is None:
                score = float("nan")
            else:
                id_val = id_ if id_ is not None else 0.0
                ip_val = ip if ip is not None else 0.0
                # If only one component available, use it directly
                if id_ is None:
                    score = ip_val
                elif ip is None:
                    score = id_val
                else:
                    score = mu * id_val + (1.0 - mu) * ip_val

            mat[i, j] = score
            mat[j, i] = score

    return mat
