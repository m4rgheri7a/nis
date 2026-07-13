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
    ("M6-attribution","Rank temporal actor hypotheses → results/attribution_hypotheses.csv"),
    ("M7-ablation",  "Ablation study → results/ablation.csv"),
    ("M7-grid",      "Grid search → results/gridsearch.csv + figures/grid_heatmap.png"),
    ("M7-robust",    "Robustness experiment → results/robustness.csv + figures/robustness_lines.png"),
    ("M8-viz",       "Evidence path visualisation → results/evidence_paths/"),
    ("M8-cluster",   "Event cluster graph → results/event_cluster.html + figures/event_cluster.png"),
    ("M8-charts",    "Generate charts → figures/{metrics_bar,...,event_cluster}.png"),
    ("M8-report",    "Generate report → results/report.md"),
    ("M9-conditions","Evidence-structuring conditions → results/condition_*.csv"),
]


def dry_run() -> None:
    print("=== run_all.py — pipeline steps ===")
    for i, (name, desc) in enumerate(STEPS, 1):
        print(f"  [{i:02d}] {name:<20} {desc}")


def full_run(cfg_path: Path | None = None, llm_model: str | None = None) -> None:
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
    print(f"  → embedding backend: {emb.backend}")

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

    _step("M6-attribution")
    from fimicyber.attribution import (
        build_error_analysis,
        build_attribution_graph,
        build_attribution_hypotheses,
        build_evidence_provenance,
        calibrate_hypotheses,
        evaluate_attribution_scopes,
        run_external_ghostwriter_case,
        run_multiactor_generalization,
    )
    attribution_raw_df = build_attribution_hypotheses(events, scores_df, cfg)
    attribution_df, calibration_df = calibrate_hypotheses(
        attribution_raw_df, cfg
    )
    attribution_df.to_csv(cfg.results_dir / "attribution_hypotheses.csv", index=False)
    calibration_df.to_csv(cfg.results_dir / "attribution_calibration.csv", index=False)
    build_error_analysis(attribution_df).to_csv(
        cfg.results_dir / "attribution_error_analysis.csv", index=False
    )
    attribution_metrics_df = evaluate_attribution_scopes(attribution_df, cfg)

    fitted_temperature = (
        float(calibration_df.iloc[0]["fitted_temperature"])
        if not calibration_df.empty else float(cfg.attribution.get("temperature", 0.15))
    )
    external_case = run_external_ghostwriter_case(
        events, emb, cfg, fitted_temperature
    )
    generalization = run_multiactor_generalization(
        emb, cfg, fitted_temperature
    )
    if external_case:
        import pandas as pd
        external_case["hypotheses"].to_csv(
            cfg.results_dir / "external_ghostwriter_hypotheses.csv", index=False
        )
        external_case["evaluation"].to_csv(
            cfg.results_dir / "external_ghostwriter_evaluation.csv", index=False
        )
        external_case["condition_comparison"].to_csv(
            cfg.results_dir / "external_ghostwriter_condition_comparison.csv", index=False
        )
        external_case["error_analysis"].to_csv(
            cfg.results_dir / "external_ghostwriter_error_analysis.csv", index=False
        )
        external_case["pairwise"].to_csv(
            cfg.results_dir / "external_ghostwriter_pairwise.csv", index=False
        )
        build_evidence_provenance(
            external_case["events"],
            cfg.data_dir / "external" / "provenance_manifest.csv",
        ).to_csv(cfg.results_dir / "evidence_provenance.csv", index=False)
        from fimicyber.viz.external_case import render_external_ghostwriter_paths
        render_external_ghostwriter_paths(
            external_case["hypotheses"], cfg.results_dir / "figures"
        )
        attribution_metrics_df = pd.concat(
            [attribution_metrics_df, external_case["evaluation"]], ignore_index=True
        )
    if generalization:
        generalization_outputs = {
            "hypotheses": "generalization_hypotheses.csv",
            "evaluation": "generalization_evaluation.csv",
            "predictions": "generalization_predictions.csv",
            "class_metrics": "generalization_class_metrics.csv",
            "condition_comparison": "generalization_condition_comparison.csv",
            "condition_summary": "generalization_condition_summary.csv",
            "error_analysis": "generalization_error_analysis.csv",
            "acceptance": "generalization_acceptance.csv",
            "protocol_checks": "generalization_protocol_checks.csv",
            "pairwise": "generalization_pairwise.csv",
        }
        for key, filename in generalization_outputs.items():
            generalization[key].to_csv(cfg.results_dir / filename, index=False)
        build_evidence_provenance(
            generalization["events"],
            cfg.data_dir / "external" / "provenance_manifest.csv",
        ).to_csv(cfg.results_dir / "generalization_evidence_provenance.csv", index=False)
        from fimicyber.attribution.paper_report import write_generalization_validation_report
        write_generalization_validation_report(
            generalization,
            cfg.results_dir / "paper_ready_generalization_validation.md",
        )
        from fimicyber.viz.generalization import render_generalization_benchmark
        render_generalization_benchmark(
            generalization["condition_summary"],
            generalization["class_metrics"],
            generalization["predictions"],
            cfg.results_dir / "figures",
        )
        attribution_metrics_df = pd.concat(
            [attribution_metrics_df, generalization["evaluation"]], ignore_index=True
        )
    attribution_metrics_df.to_csv(
        cfg.results_dir / "attribution_evaluation.csv", index=False
    )
    if external_case:
        from fimicyber.attribution.paper_report import write_paper_validation_report
        write_paper_validation_report(
            attribution_metrics_df,
            calibration_df,
            external_case,
            cfg.results_dir / "paper_ready_attribution_validation.md",
        )
    build_attribution_graph(
        events,
        attribution_df,
        cfg.results_dir / "attribution_graph.json",
    )
    print(
        f"  → attribution hypotheses: {len(attribution_df)} rows, "
        f"{attribution_df['query_event_id'].nunique() if not attribution_df.empty else 0} queries"
    )

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
    generate_report(
        events, metrics_df, abl_df, rob_df, scores_df, cfg, fallbacks_used,
        attribution_df=attribution_df,
        attribution_metrics_df=attribution_metrics_df,
    )

    # ── M9: evidence-structuring conditions ────────────────────────────────
    # Ranking the four campaigns from curated fields tells us nothing about how
    # the system would behave on an unstructured report. This step re-derives the
    # evidence from case text and re-ranks, so the LLM's actual contribution to
    # Top-1/Top-3/MRR is measured rather than assumed. LLM conditions are opt-in
    # because they need a local Ollama server.
    _step("M9-conditions")
    from fimicyber.eval.condition_benchmark import (
        run_condition_benchmark,
        write_condition_outputs,
    )
    conditions = ("curated_oracle", "rules_only")
    if llm_model:
        conditions += ("llm_guarded", "llm_only")
    condition_bundle = run_condition_benchmark(
        emb,
        cfg,
        fitted_temperature,
        model=llm_model or "-",
        conditions=conditions,
        temperature_source="fitted in this run",
    )
    if condition_bundle:
        write_condition_outputs(condition_bundle, cfg.results_dir)
        ranking = condition_bundle["ranking_metrics"]
        for _, row in ranking.iterrows():
            print(
                f"  → {row['condition']:<15} top1={row['top1_accuracy']:.2f} "
                f"top3={row['top3_accuracy']:.2f} MRR={row['MRR']:.3f} "
                f"false_attr={row['false_attribution_rate']:.2f}"
            )
    else:
        print("  → skipped: no benchmark events")

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
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Ollama tag (e.g. qwen3:14b). Adds the llm_guarded and llm_only "
             "conditions to M9. Omit to run M9 without a model.",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        full_run(args.config, llm_model=args.llm_model)
