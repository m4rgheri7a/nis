"""Event cluster graph: events as nodes, FCLS_E3 as edge weight (spec viz extension)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fimicyber.schema import Event

# Distinct palette for up to 25 campaigns
_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#e6ab02", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#b15928",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
]
_NO_CAMP_COLOR = "#888888"


def _campaign_colors(events: list[Event]) -> dict[str, str]:
    camps = sorted(set(ev.campaign_id for ev in events if ev.campaign_id))
    return {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(camps)}


def build_event_cluster(
    events: list[Event],
    scores_df: pd.DataFrame,
    cfg: Any,
    threshold: float = 0.60,
) -> Path:
    """
    Build interactive pyvis event cluster graph.
    Nodes = events (coloured by campaign), edges = pairs with FCLS_E3 >= threshold.
    Returns path to saved HTML.
    """
    try:
        from pyvis.network import Network
        _USE_PYVIS = True
    except ImportError:
        _USE_PYVIS = False

    out = cfg.results_dir / "event_cluster.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    camp_col = _campaign_colors(events)

    if _USE_PYVIS:
        _build_pyvis(events, scores_df, camp_col, threshold, out)
    else:
        _build_plain_html(events, scores_df, camp_col, threshold, out)

    return out


def _build_pyvis(
    events: list[Event],
    scores_df: pd.DataFrame,
    camp_col: dict[str, str],
    threshold: float,
    out: Path,
) -> None:
    from pyvis.network import Network

    net = Network(
        height="100vh", width="100%",
        bgcolor="#0d1117", font_color="#c9d1d9",
    )
    net.set_options("""{
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -6000,
          "centralGravity": 0.25,
          "springLength": 160,
          "springConstant": 0.04
        },
        "stabilization": {"iterations": 200}
      },
      "nodes": {
        "borderWidth": 1.5,
        "shadow": true
      },
      "edges": {
        "smooth": {"type": "continuous"},
        "shadow": false
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 150,
        "navigationButtons": true
      }
    }""")

    ev_map = {ev.event_id: ev for ev in events}

    # Add nodes
    for ev in events:
        n_iocs = len(ev.iocs)
        color = camp_col.get(ev.campaign_id or "", _NO_CAMP_COLOR)
        desc = (ev.description or "")[:100].replace("<", "&lt;")
        tooltip = (
            f"<b>{ev.event_id}</b><br>"
            f"캠페인: {ev.campaign_id or '없음'}<br>"
            f"IOC: {n_iocs}개<br>"
            f"{desc}"
        )
        size = max(10, min(35, 10 + n_iocs * 0.8))
        net.add_node(
            ev.event_id,
            label=ev.event_id,
            color={"background": color, "border": "#ffffff44",
                   "highlight": {"background": color, "border": "#ffffff"}},
            size=size,
            title=tooltip,
            group=ev.campaign_id or "unknown",
        )

    # Add edges
    for _, row in scores_df.iterrows():
        v = row.get("FCLS_E3")
        if v is None or (isinstance(v, float) and (math.isnan(v) or v < threshold)):
            continue
        width = max(0.5, float(v) * 4)
        alpha = int(min(255, max(60, float(v) * 200)))
        color_hex = f"#{alpha:02x}8888"
        net.add_edge(
            row["event_i"], row["event_j"],
            width=width,
            color=color_hex,
            title=f"FCLS_E3: {float(v):.3f}",
        )

    net.save_graph(str(out))

    # Inject legend into the saved HTML
    _inject_legend(out, camp_col)


def _inject_legend(html_path: Path, camp_col: dict[str, str]) -> None:
    """Append a legend overlay div into the pyvis HTML."""
    items = "".join(
        f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:5px">'
        f'<div style="width:13px;height:13px;border-radius:50%;background:{c};flex-shrink:0"></div>'
        f'<span style="font-size:11px;color:#c9d1d9">{camp}</span></div>'
        for camp, c in sorted(camp_col.items())
    )
    no_camp = (
        '<div style="display:flex;align-items:center;gap:7px;margin-bottom:5px">'
        f'<div style="width:13px;height:13px;border-radius:50%;background:{_NO_CAMP_COLOR};flex-shrink:0"></div>'
        '<span style="font-size:11px;color:#c9d1d9">캠페인 없음</span></div>'
    )
    legend = (
        f'<div style="position:fixed;top:14px;right:14px;background:#1e2330dd;'
        f'border:1px solid #3b4262;border-radius:10px;padding:12px 14px;'
        f'z-index:9999;max-height:80vh;overflow-y:auto;min-width:160px">'
        f'<div style="font-size:11px;font-weight:700;color:#7c8799;'
        f'text-transform:uppercase;letter-spacing:.5px;margin-bottom:9px">캠페인</div>'
        f'{items}{no_camp}'
        f'</div>'
    )
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("</body>", legend + "\n</body>")
    html_path.write_text(html, encoding="utf-8")


def _build_plain_html(
    events: list[Event],
    scores_df: pd.DataFrame,
    camp_col: dict[str, str],
    threshold: float,
    out: Path,
) -> None:
    """Fallback: plain table when pyvis is unavailable."""
    rows = ""
    for _, row in scores_df[scores_df["FCLS_E3"] >= threshold].iterrows():
        rows += (
            f"<tr><td>{row['event_i']}</td>"
            f"<td>{row['event_j']}</td>"
            f"<td>{row['FCLS_E3']:.3f}</td></tr>"
        )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:sans-serif;padding:20px;background:#0d1117;color:#c9d1d9}"
        "table{border-collapse:collapse;width:100%}th,td{padding:6px 10px;border:1px solid #444;font-size:12px}"
        "th{background:#1e2330}</style></head><body>"
        f"<h2>Event Cluster (FCLS_E3 ≥ {threshold})</h2>"
        "<p style='color:#888;font-size:12px'>pyvis 미설치 — pip install pyvis 로 인터랙티브 그래프 활성화</p>"
        "<table><thead><tr><th>Event i</th><th>Event j</th><th>FCLS_E3</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "</body></html>"
    )
    out.write_text(html, encoding="utf-8")


# ── Static PNG (matplotlib spring layout) ────────────────────────────────────

def plot_event_cluster_static(
    events: list[Event],
    scores_df: pd.DataFrame,
    cfg: Any,
    threshold: float = 0.65,
) -> None:
    """Static spring-layout cluster chart saved to figures/event_cluster.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    camp_col = _campaign_colors(events)

    G = nx.Graph()
    for ev in events:
        G.add_node(ev.event_id,
                   campaign=ev.campaign_id or "",
                   n_iocs=len(ev.iocs))

    for _, row in scores_df.iterrows():
        v = row.get("FCLS_E3")
        if v is None or (isinstance(v, float) and (math.isnan(v) or v < threshold)):
            continue
        G.add_edge(row["event_i"], row["event_j"], weight=float(v))

    if G.number_of_edges() == 0:
        return

    pos = nx.spring_layout(G, k=2.5, seed=42, weight="weight", iterations=80)

    node_colors = [
        camp_col.get(G.nodes[n]["campaign"], _NO_CAMP_COLOR)
        for n in G.nodes()
    ]
    node_sizes = [
        max(40, min(300, 40 + G.nodes[n]["n_iocs"] * 8))
        for n in G.nodes()
    ]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_alphas = [max(0.1, w - 0.4) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(14, 11), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=["#cc444488" for _ in edge_weights],
        width=[max(0.3, w * 2) for w in edge_weights],
        alpha=0.5,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        linewidths=0.5,
        edgecolors="#ffffff44",
    )

    # Labels only for nodes with many IOCs (reduce clutter)
    labels = {n: n for n in G.nodes() if G.nodes[n]["n_iocs"] >= 4}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=5, font_color="#ffffffaa",
    )

    # Legend
    patches = [
        mpatches.Patch(color=c, label=camp)
        for camp, c in sorted(camp_col.items())
    ]
    if patches:
        ax.legend(handles=patches, loc="lower left",
                  fontsize=7, framealpha=0.3,
                  facecolor="#1e2330", labelcolor="#c9d1d9")

    ax.set_title(
        f"Event Cluster Graph  (FCLS_E3 ≥ {threshold},  "
        f"{G.number_of_nodes()} events,  {G.number_of_edges()} edges)",
        color="#c9d1d9", fontsize=11, pad=10,
    )
    ax.axis("off")
    fig.tight_layout()

    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "event_cluster.png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close(fig)
