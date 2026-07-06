#!/usr/bin/env python3
"""Single-command pipeline runner.

Usage:
    python scripts/run_all.py           # full run
    python scripts/run_all.py --dry-run # print steps only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

STEPS = [
    ("M1-load",      "Load DISINFOX + EUvsDisinfo events → data/interim/events.jsonl"),
    ("M3-ioc",       "Extract & classify IOCs, generate synthetic IOCs → data/interim/iocs.jsonl"),
    ("M2-embed",     "Compute SBERT embeddings → data/processed/embeddings.parquet"),
    ("M2-narrative", "Compute narrative matrix N(i,j)"),
    ("M4-graph",     "Build evidence graph → data/processed/graph.json"),
    ("M4-ioc-score", "Compute I_direct and I_path scores"),
    ("M5-components","Compute D, C, T, A components"),
    ("M5-fcls",      "Compute FCLS(i,j) → results/pairwise_scores.csv"),
    ("M5-priority",  "Compute Priority(i) → results/priority_table.csv"),
    ("M6-eval",      "Evaluate combined E1/E2/E3 → results/metrics_summary.csv"),
    ("M7-ablation",  "Ablation study → results/ablation.csv"),
    ("M7-grid",      "Grid search → results/gridsearch.csv + figures/grid_heatmap.png"),
    ("M7-robust",    "Robustness experiment → results/robustness.csv + figures/robustness_lines.png"),
    ("M8-viz",       "Evidence path visualisation → results/evidence_paths/"),
    ("M8-cluster",   "Event cluster graph → results/event_cluster.html + figures/event_cluster.png"),
    ("M8-charts",    "Generate charts → figures/{metrics_bar,...,event_cluster}.png"),
    ("M8-report",    "Generate report → results/report.md"),
]


def dry_run() -> None:
    print("=== run_all.py — pipeline steps ===")
    for i, (name, desc) in enumerate(STEPS, 1):
        print(f"  [{i:02d}] {name:<20} {desc}")


def full_run(cfg_path: Path | None = None) -> None:
    from fimicyber.config import load_config

    cfg = load_config(cfg_path)

    fallbacks_used: list[str] = []

    # ── M1: load events ────────────────────────────────────────────────────
    _step("M1-load")
    from collections import Counter
    from fimicyber.loaders.combined import load_events
    events = load_events(cfg.data_dir / "raw", cfg, fallbacks_used)
    _save_jsonl(events, cfg.data_dir / "interim" / "events.jsonl")
    source_counts = Counter(e.source_dataset for e in events)
    source_text = ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items()))
    print(f"  → {len(events)} events loaded, "
          f"{sum(1 for e in events if e.campaign_id)} with campaign_id")
    print(f"  → source mix: {source_text}")

    # ── M3: IOC pipeline ───────────────────────────────────────────────────
    _step("M3-ioc")
    from fimicyber.ioc.extract import extract_iocs_from_event
    from fimicyber.ioc.synthetic import generate_synthetic_iocs
    for ev in events:
        extracted = extract_iocs_from_event(ev)
        ev.iocs.extend(extracted)
    events = generate_synthetic_iocs(events, cfg)
    _save_jsonl_iocs(events, cfg.data_dir / "interim" / "iocs.jsonl")
    print(f"  → IOCs attached")

    # ── M2: embeddings + narrative ─────────────────────────────────────────
    _step("M2-embed")
    from fimicyber.nlp.embed import EmbStore
    emb = EmbStore(cfg, fallbacks_used)
    emb.encode_events(events)

    _step("M2-narrative")
    from fimicyber.nlp.narrative import narrative_matrix
    N = narrative_matrix(events, emb, cfg)
    print(f"  → N matrix {N.shape}")

    # ── M4: graph + IOC score ──────────────────────────────────────────────
    _step("M4-graph")
    from fimicyber.graph.build import build_graph
    G = build_graph(events, cfg)
    import json, networkx as nx
    graph_path = cfg.data_dir / "processed" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(nx.node_link_data(G, edges="links"), ensure_ascii=False))
    print(f"  → graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    _step("M4-ioc-score")
    from fimicyber.graph.ioc_score import ioc_matrix
    I = ioc_matrix(events, G, cfg)
    print(f"  → I matrix {I.shape}")

    import copy
    events_no_synthetic_ioc = copy.deepcopy(events)
    for ev in events_no_synthetic_ioc:
        ev.iocs = [ioc for ioc in ev.iocs if not ioc.synthetic]
    G_no_synthetic = build_graph(events_no_synthetic_ioc, cfg)
    I_no_synthetic = ioc_matrix(events_no_synthetic_ioc, G_no_synthetic, cfg)

    # ── M5: components + FCLS + priority ──────────────────────────────────
    _step("M5-components")
    from fimicyber.scoring.components import compute_components
    comps = compute_components(events, cfg)

    _step("M5-fcls")
    from fimicyber.scoring.fcls import build_pairwise_scores
    scores_df = build_pairwise_scores(events, N, I, comps, cfg, I_no_synthetic=I_no_synthetic)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(cfg.results_dir / "pairwise_scores.csv", index=False)
    print(f"  → pairwise_scores.csv: {len(scores_df)} rows")

    _step("M5-priority")
    from fimicyber.scoring.priority import compute_priority
    prio_df = compute_priority(events, scores_df, cfg)
    prio_df.to_csv(cfg.results_dir / "priority_table.csv", index=False)

    # ── M6: evaluation ─────────────────────────────────────────────────────
    _step("M6-eval")
    from fimicyber.eval.experiments import run_all_experiments
    metrics_df = run_all_experiments(events, scores_df, cfg)
    metrics_df.to_csv(cfg.results_dir / "metrics_summary.csv", index=False)
    print(f"  → metrics_summary.csv")

    # ── M7: ablation + grid + robustness ──────────────────────────────────
    _step("M7-ablation")
    from fimicyber.eval.experiments import run_ablation
    abl_df = run_ablation(events, scores_df, cfg)
    abl_df.to_csv(cfg.results_dir / "ablation.csv", index=False)

    _step("M7-grid")
    from fimicyber.eval.experiments import run_grid
    grid_df = run_grid(events, N, I, comps, cfg)
    grid_df.to_csv(cfg.results_dir / "gridsearch.csv", index=False)

    _step("M7-robust")
    from fimicyber.eval.experiments import run_robustness
    rob_df = run_robustness(events, N, comps, cfg, fallbacks_used)
    rob_df.to_csv(cfg.results_dir / "robustness.csv", index=False)

    # ── M8: visualisation + charts + report ───────────────────────────────
    _step("M8-viz")
    from fimicyber.viz.evidence_path import render_top_pairs
    render_top_pairs(events, G, scores_df, cfg)

    _step("M8-cluster")
    from fimicyber.viz.cluster import build_event_cluster, plot_event_cluster_static
    build_event_cluster(events, scores_df, cfg)
    plot_event_cluster_static(events, scores_df, cfg)

    _step("M8-charts")
    from fimicyber.viz.charts import generate_all_charts
    generate_all_charts(metrics_df, grid_df, rob_df, cfg,
                        abl_df=abl_df, scores_df=scores_df, events=events)

    _step("M8-report")
    from fimicyber.viz.charts import generate_report
    generate_report(events, metrics_df, abl_df, rob_df, scores_df, cfg, fallbacks_used)

    print("\n=== Pipeline complete. See results/ ===")


def _step(name: str) -> None:
    desc = next(d for n, d in STEPS if n == name)
    print(f"\n[{name}] {desc}")


def _save_jsonl(events: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(ev.model_dump_json() + "\n")


def _save_jsonl_iocs(events: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            for ioc in ev.iocs:
                f.write(ioc.model_dump_json() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        full_run(args.config)
