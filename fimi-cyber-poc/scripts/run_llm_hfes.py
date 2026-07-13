#!/usr/bin/env python3
"""Run the Hybrid-FIMI LLM evidence-structuring PoC."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["auto", "rules", "qwen3", "ollama"], default="ollama"
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--max-events", type=int, default=28)
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for result files, for example qwen3_8b.",
    )
    args = parser.parse_args()

    from fimicyber.config import load_config
    from fimicyber.loaders.external import load_multiactor_benchmark
    from fimicyber.llm import EvidenceCompiler, write_structuring_outputs

    cfg = load_config()
    events = load_multiactor_benchmark(cfg)
    if args.max_events:
        events = events[: args.max_events]

    compiler = EvidenceCompiler(model_name=args.model, mode=args.mode)
    evidence = [compiler.compile_event(event) for event in events]
    outputs = write_structuring_outputs(
        events, evidence, cfg.results_dir, suffix=args.output_suffix
    )
    print(f"backend={compiler.backend}")
    for key, path in outputs.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
