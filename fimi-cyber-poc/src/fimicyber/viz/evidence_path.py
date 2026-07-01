"""Evidence path visualisation with pyvis (spec 11)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from fimicyber.schema import Event
from fimicyber.graph.build import _node_id, infra_subgraph
from fimicyber.ioc.extract import defang

# Node colour palette by ntype
_NODE_COLORS = {
    "Event": "#4e79a7",
    "IOC": "#f28e2b",
    "TTP": "#76b7b2",
    "Channel": "#59a14f",
    "EvidenceSource": "#edc948",
    "Domain": "#b07aa1",
    "IP": "#ff9da7",
    "NS": "#9c755f",
    "ASN": "#bab0ac",
}
_DEFAULT_COLOR = "#aaaaaa"


def render_top_pairs(
    events: list[Event],
    G: nx.Graph,
    scores_df: pd.DataFrame,
    cfg: Any,
    top_n: int = 3,
) -> None:
    out_dir = cfg.results_dir / "evidence_paths"
    out_dir.mkdir(parents=True, exist_ok=True)

    top = scores_df.nlargest(top_n, "FCLS_E3")
    ev_map = {ev.event_id: ev for ev in events}

    for _, row in top.iterrows():
        ei_id = row["event_i"]
        ej_id = row["event_j"]
        ev_i = ev_map.get(ei_id)
        ev_j = ev_map.get(ej_id)
        if ev_i is None or ev_j is None:
            continue

        out_path = out_dir / f"pair_{ei_id}_{ej_id}.html"
        _render_pair(ev_i, ev_j, G, out_path)


def _render_pair(
    ev_i: Event,
    ev_j: Event,
    G: nx.Graph,
    out_path: Path,
) -> None:
    try:
        from pyvis.network import Network
    except ImportError:
        _render_pair_plain(ev_i, ev_j, G, out_path)
        return

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.set_options("""
    {
      "physics": {"stabilization": {"iterations": 100}},
      "edges": {"arrows": {"to": {"enabled": true}}},
      "interaction": {"hover": true}
    }
    """)

    sub = _extract_pair_subgraph(ev_i, ev_j, G)

    for node, data in sub.nodes(data=True):
        ntype = data.get("ntype", "Unknown")
        color = _NODE_COLORS.get(ntype, _DEFAULT_COLOR)
        label = _make_node_label(node, data)
        title = f"Type: {ntype}<br>ID: {node}"
        net.add_node(node, label=label, color=color, title=title, size=20)

    for src, dst, data in sub.edges(data=True):
        etype = data.get("etype", "")
        conf = data.get("confidence", 1.0)
        net.add_edge(src, dst, label=etype, title=f"{etype} (conf={conf:.2f})", width=2)

    net.save_graph(str(out_path))


def _render_pair_plain(ev_i: Event, ev_j: Event, G: nx.Graph, out_path: Path) -> None:
    """Plain HTML fallback when pyvis is not available."""
    sub = _extract_pair_subgraph(ev_i, ev_j, G)

    rows = []
    for src, dst, data in sub.edges(data=True):
        rows.append(f"<tr><td>{src}</td><td>{data.get('etype','')}</td><td>{dst}</td></tr>")

    html = f"""<!DOCTYPE html><html><body>
<h2>Evidence Path: {ev_i.event_id} ↔ {ev_j.event_id}</h2>
<table border='1'><tr><th>Source</th><th>Relation</th><th>Target</th></tr>
{''.join(rows)}
</table></body></html>"""
    out_path.write_text(html, encoding="utf-8")


def _extract_pair_subgraph(ev_i: Event, ev_j: Event, G: nx.Graph) -> nx.Graph:
    """Extract subgraph of nodes connecting the two events."""
    ei_node = _node_id("Event", ev_i.event_id)
    ej_node = _node_id("Event", ev_j.event_id)

    # Collect: both event nodes + their direct IOC/TTP/Channel neighbours
    # + any infra paths between their IOCs
    nodes_to_include: set[str] = {ei_node, ej_node}

    def _neighbours(ev_node: str) -> set[str]:
        if ev_node not in G:
            return set()
        return set(G.neighbors(ev_node))

    nei_i = _neighbours(ei_node)
    nei_j = _neighbours(ej_node)
    shared = nei_i & nei_j

    # Add all direct neighbours of both events
    nodes_to_include.update(nei_i)
    nodes_to_include.update(nei_j)

    # For shared IOC infra: add paths through infra subgraph
    ioc_i = {n for n in nei_i if G.nodes[n].get("ntype") == "IOC"}
    ioc_j = {n for n in nei_j if G.nodes[n].get("ntype") == "IOC"}
    sub_infra = infra_subgraph(G)

    for src in ioc_i:
        for dst in ioc_j:
            if src == dst:
                continue
            try:
                path = nx.shortest_path(sub_infra, source=src, target=dst)
                nodes_to_include.update(path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    sub = G.subgraph(nodes_to_include).copy()

    # Defang IOC labels
    for node, data in sub.nodes(data=True):
        if data.get("ntype") == "IOC":
            data["label_defang"] = defang(data.get("value", node))

    return sub


def _make_node_label(node: str, data: dict) -> str:
    ntype = data.get("ntype", "")
    val = data.get("value", "")
    if ntype == "IOC":
        # Show defanged value
        return defang(val)[:40]
    if ntype == "Event":
        return data.get("event_id", node)[:30]
    return val[:30] if val else node.split(":", 1)[-1][:30]
