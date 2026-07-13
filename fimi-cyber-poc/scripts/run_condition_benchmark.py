#!/usr/bin/env python3
"""Compare evidence-structuring conditions on the four-campaign benchmark.

Runs the full chain — case dossier → structuring → guardrails →
apply_structured_evidence → evidence graph → FCLS → candidate ranking — under
curated-oracle, rules-only, LLM-guarded, and unguarded-LLM conditions, so the
ranking numbers can be attributed to the extractor rather than assumed.

    python scripts/run_condition_benchmark.py --model qwen3:14b
    python scripts/run_condition_benchmark.py --conditions curated_oracle rules_only
    python scripts/run_condition_benchmark.py --no-annex   # summary-only input
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:14b", help="Ollama model tag.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Subset of: curated_oracle rules_only llm_guarded llm_only",
    )
    parser.add_argument(
        "--no-annex",
        action="store_true",
        help="Drop the technical annex from the dossier to measure IOC sparsity.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix for result files, e.g. no_annex, so an ablation does not "
             "overwrite the primary run.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the softmax temperature. Defaults to the value the main "
             "pipeline fitted (results/attribution_calibration.csv).",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    from fimicyber.config import load_config
    from fimicyber.eval.condition_benchmark import (
        CONDITIONS,
        resolve_calibrated_temperature,
        run_condition_benchmark,
        write_condition_outputs,
    )
    from fimicyber.nlp.embed import EmbStore

    cfg = load_config(args.config)
    conditions = tuple(args.conditions) if args.conditions else CONDITIONS
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        parser.error(f"Unknown condition(s): {sorted(unknown)}")

    if args.temperature is not None:
        temperature, temperature_source = args.temperature, "--temperature"
    else:
        try:
            temperature, temperature_source = resolve_calibrated_temperature(cfg)
        except FileNotFoundError as exc:
            parser.error(str(exc))

    fallbacks: list[str] = []
    emb = EmbStore(cfg, fallbacks)

    print(
        f"model={args.model}  conditions={list(conditions)}  "
        f"annex={not args.no_annex}  temperature={temperature:.5f} ({temperature_source})"
    )
    bundle = run_condition_benchmark(
        emb,
        cfg,
        temperature,
        model=args.model,
        conditions=conditions,
        include_annex=not args.no_annex,
        temperature_source=temperature_source,
    )
    if not bundle:
        parser.error("Benchmark produced no events — check data/external.")

    written = write_condition_outputs(bundle, cfg.results_dir, suffix=args.output_suffix)

    print(f"\nembedding backend: {emb.backend}")
    if fallbacks:
        print(f"fallbacks: {fallbacks}")
    print("\n=== candidate ranking ===")
    print(bundle["ranking_metrics"][[
        "condition", "model", "top1_accuracy", "top3_accuracy", "MRR",
        "review_coverage", "selective_accuracy", "false_attribution_rate",
    ]].to_string(index=False))

    if not bundle["extraction_metrics"].empty:
        print("\n=== evidence extraction vs curated gold ===")
        print(bundle["extraction_metrics"][[
            "condition", "ttp_f1", "channel_f1", "target_f1", "country_f1",
            "ioc_recall", "ioc_precision", "hallucinated_ioc_total",
            "extraction_seconds",
        ]].to_string(index=False))

    print("\n=== written ===")
    for key, path in written.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
