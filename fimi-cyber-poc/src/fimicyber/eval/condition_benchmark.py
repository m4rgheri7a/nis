"""Connect evidence structuring to the ranking pipeline and compare conditions.

Until now the LLM structuring experiment and the FCLS candidate ranking were two
separate programs: ``run_llm_hfes.py`` scored the extractor against gold fields,
while ``run_multiactor_generalization`` ranked campaigns from human-curated
event fields. Nothing joined them, so no reported ranking number could be
credited to the LLM.

This module runs the whole chain — dossier → structuring → guardrails →
``apply_structured_evidence`` → evidence graph → FCLS → candidate ranking — under
four conditions that share one data split and one downstream:

``curated_oracle``
    Human-normalised fields. The ceiling, and the condition the previously
    published Top-1/Top-3/MRR numbers were computed under.
``rules_only``
    Deterministic keyword and regex extraction from the case dossier.
``llm_guarded``
    Model structuring from the same dossier, filtered by the guardrails.
``llm_only``
    The same model output with the guardrails switched off. It exists to show
    what the guardrails are holding back, not as a serious baseline.

Every condition reads the same text and is evaluated against the same sealed
holdout labels. Only the evidence-extraction step changes.
"""
from __future__ import annotations

import copy
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fimicyber.attribution.calibration import (
    apply_temperature_and_abstention,
    build_error_analysis,
    evaluate_attribution_scope,
)
from fimicyber.attribution.generalization import (
    _attach_holdout_truth,
    _class_metrics,
    _label_isolated_scoring_events,
    _prediction_table,
)
from fimicyber.attribution.hypotheses import build_attribution_hypotheses
from fimicyber.graph.build import build_graph
from fimicyber.graph.ioc_score import ioc_matrix
from fimicyber.llm.dossier import assert_no_labels, build_dossiers
from fimicyber.llm.evidence import (
    EvidenceCompiler,
    StructuredEvidence,
    apply_structured_evidence,
    evaluate_structuring,
)
from fimicyber.loaders.external import load_multiactor_benchmark
from fimicyber.nlp.narrative import narrative_matrix
from fimicyber.schema import Event
from fimicyber.scoring.components import compute_components
from fimicyber.scoring.fcls import build_pairwise_scores

CONDITIONS: tuple[str, ...] = ("curated_oracle", "rules_only", "llm_guarded", "llm_only")

_COMPILER_MODE = {"rules_only": "rules", "llm_guarded": "ollama", "llm_only": "llm_only"}

_SPLIT_ROLE = "generalization_holdout"


@dataclass
class ConditionResult:
    condition: str
    model: str
    backend: str
    evaluation: pd.DataFrame
    predictions: pd.DataFrame
    class_metrics: pd.DataFrame
    error_analysis: pd.DataFrame
    hypotheses: pd.DataFrame
    extraction_per_event: pd.DataFrame | None = None
    evidence: list[StructuredEvidence] = field(default_factory=list)
    extraction_seconds: float = 0.0
    ranking_seconds: float = 0.0


def _score_and_rank(
    events: list[Event],
    emb: Any,
    cfg: Any,
    temperature: float,
    scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the shared downstream: graph → FCLS → candidate ranking → abstention."""
    reference_ids = {e.event_id for e in events if e.evaluation_role == "reference"}
    query_ids = {e.event_id for e in events if e.evaluation_role == "holdout"}

    scoring_events = _label_isolated_scoring_events(events)
    emb.encode_events(scoring_events)
    narrative = narrative_matrix(scoring_events, emb, cfg)
    graph = build_graph(scoring_events, cfg)
    infrastructure = ioc_matrix(scoring_events, graph, cfg)
    components = compute_components(scoring_events, cfg)
    scores = build_pairwise_scores(
        scoring_events,
        narrative,
        infrastructure,
        components,
        cfg,
        I_no_synthetic=infrastructure,
    )
    hypotheses = build_attribution_hypotheses(
        scoring_events, scores, cfg, query_ids=query_ids, reference_ids=reference_ids
    )
    hypotheses = _attach_holdout_truth(hypotheses, events, cfg)
    hypotheses = apply_temperature_and_abstention(
        hypotheses, cfg, temperature, split_role_override=_SPLIT_ROLE
    )
    evaluation = evaluate_attribution_scope(hypotheses, cfg, scope, {_SPLIT_ROLE})
    return hypotheses, evaluation


def run_condition(
    condition: str,
    gold_events: list[Event],
    dossiers: dict[str, str],
    emb: Any,
    cfg: Any,
    temperature: float,
    model: str,
) -> ConditionResult:
    """Extract evidence under one condition, then rank with the shared downstream."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")

    events = copy.deepcopy(gold_events)
    evidence: list[StructuredEvidence] = []
    extraction_per_event: pd.DataFrame | None = None
    backend = "curated"
    extraction_seconds = 0.0

    if condition != "curated_oracle":
        compiler = EvidenceCompiler(model_name=model, mode=_COMPILER_MODE[condition])
        backend = compiler.backend
        started = time.perf_counter()
        evidence = [
            compiler.compile_event(event, text=dossiers[event.event_id])
            for event in events
        ]
        extraction_seconds = time.perf_counter() - started

        # Scored against the pristine gold events — `events` is about to be
        # overwritten with the extracted fields.
        extraction_per_event = evaluate_structuring(gold_events, evidence)
        extraction_per_event.insert(0, "condition", condition)

        events = apply_structured_evidence(events, evidence, replace=True)

    started = time.perf_counter()
    hypotheses, evaluation = _score_and_rank(
        events, emb, cfg, temperature, scope=condition
    )
    ranking_seconds = time.perf_counter() - started

    evaluation = evaluation.copy()
    evaluation.insert(0, "condition", condition)
    evaluation.insert(1, "model", model if condition.startswith("llm") else "-")
    evaluation.insert(2, "backend", backend)

    predictions = _prediction_table(hypotheses)
    predictions.insert(0, "condition", condition)
    class_metrics = _class_metrics(predictions)
    class_metrics.insert(0, "condition", condition)
    errors = build_error_analysis(hypotheses)
    if not errors.empty:
        errors.insert(0, "condition", condition)

    return ConditionResult(
        condition=condition,
        model=model if condition.startswith("llm") else "-",
        backend=backend,
        evaluation=evaluation,
        predictions=predictions,
        class_metrics=class_metrics,
        error_analysis=errors,
        hypotheses=hypotheses,
        extraction_per_event=extraction_per_event,
        evidence=evidence,
        extraction_seconds=extraction_seconds,
        ranking_seconds=ranking_seconds,
    )


def summarise_extraction(results: list[ConditionResult]) -> pd.DataFrame:
    """Aggregate the per-event extraction scores into one row per condition."""
    rows: list[dict[str, Any]] = []
    for result in results:
        frame = result.extraction_per_event
        if frame is None or frame.empty:
            continue
        with_gold = frame[frame["gold_ioc_count"] > 0]
        rows.append({
            "condition": result.condition,
            "model": result.model,
            "backend": result.backend,
            "events": len(frame),
            "ttp_precision": float(frame["ttp_precision"].mean()),
            "ttp_recall": float(frame["ttp_recall"].mean()),
            "ttp_f1": float(frame["ttp_f1"].mean()),
            "channel_precision": float(frame["channel_precision"].mean()),
            "channel_recall": float(frame["channel_recall"].mean()),
            "channel_f1": float(frame["channel_f1"].mean()),
            "target_precision": float(frame["target_precision"].mean()),
            "target_recall": float(frame["target_recall"].mean()),
            "target_f1": float(frame["target_f1"].mean()),
            "country_f1": float(frame["country_f1"].mean()),
            # IOC recall averaged over the events that actually have a gold IOC;
            # the 16 events without one would otherwise dominate the mean.
            "events_with_gold_ioc": len(with_gold),
            "ioc_precision": float(with_gold["ioc_precision"].mean()) if len(with_gold) else float("nan"),
            "ioc_recall": float(with_gold["ioc_recall"].mean()) if len(with_gold) else float("nan"),
            "ioc_f1": float(with_gold["ioc_f1"].mean()) if len(with_gold) else float("nan"),
            "hallucinated_ioc_total": int(frame["hallucinated_ioc_count"].sum()),
            "events_with_hallucinated_ioc": int((frame["hallucinated_ioc_count"] > 0).sum()),
            "evidence_sentence_coverage": float((frame["evidence_sentence_count"] > 0).mean()),
            "extraction_seconds": round(result.extraction_seconds, 2),
        })
    return pd.DataFrame(rows)


def resolve_calibrated_temperature(cfg: Any) -> tuple[float, str]:
    """Return the softmax temperature the main pipeline fitted, and where it came from.

    ``run_multiactor_generalization`` is called with the temperature that
    ``calibrate_hypotheses`` fitted chronologically on the development corpus
    (0.040, not the 0.15 config default). Abstention thresholds are applied to
    the resulting probabilities, so a condition benchmark that used the config
    default would report a different review-coverage than the published run and
    would not be comparable to it.
    """
    path = cfg.results_dir / "attribution_calibration.csv"
    if path.exists():
        frame = pd.read_csv(path)
        if not frame.empty and "fitted_temperature" in frame.columns:
            return float(frame.iloc[0]["fitted_temperature"]), str(path)
    raise FileNotFoundError(
        f"{path} not found. Run `python scripts/run_all.py` first so the "
        "condition benchmark uses the same fitted temperature as the published "
        "generalisation run, or pass --temperature explicitly."
    )


def _gpu_state() -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "nvidia_smi": _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader",
        ]),
        "ollama_ps": _run(["ollama", "ps"]),
    }


def _model_digest(model: str) -> str:
    try:
        output = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in output.splitlines():
        parts = line.split()
        if parts and parts[0] == model:
            return parts[1] if len(parts) > 1 else ""
    return ""


def run_condition_benchmark(
    emb: Any,
    cfg: Any,
    temperature: float,
    model: str = "qwen3:14b",
    conditions: tuple[str, ...] = CONDITIONS,
    include_annex: bool = True,
    temperature_source: str = "explicit",
) -> dict[str, Any]:
    """Run every requested condition over the frozen four-campaign benchmark."""
    gold_events = load_multiactor_benchmark(cfg)
    if not gold_events:
        return {}

    dossiers = build_dossiers(gold_events, cfg, include_annex=include_annex)
    # Fail loudly rather than silently scoring a leaked input.
    assert_no_labels(dossiers, gold_events, cfg)

    results = [
        run_condition(condition, gold_events, dossiers, emb, cfg, temperature, model)
        for condition in conditions
    ]

    ranking = pd.concat([r.evaluation for r in results], ignore_index=True)
    ranking["ranking_seconds"] = [round(r.ranking_seconds, 2) for r in results]
    extraction = summarise_extraction(results)

    per_event_frames = [
        r.extraction_per_event for r in results if r.extraction_per_event is not None
    ]
    hallucinations = pd.DataFrame()
    if per_event_frames:
        per_event = pd.concat(per_event_frames, ignore_index=True)
        flagged = per_event[per_event["hallucinated_ioc_count"] > 0]
        hallucinations = flagged[
            ["condition", "event_id", "hallucinated_ioc_count", "hallucinated_iocs"]
        ].copy()
    else:
        per_event = pd.DataFrame()

    manifest = {
        "model": model,
        "model_digest": _model_digest(model),
        "conditions": list(conditions),
        "include_annex": include_annex,
        "seed": int(cfg.seed),
        "temperature_source": temperature_source,
        "llm_temperature": 0,
        "llm_seed": 42,
        "embedding_backend": getattr(emb, "backend", "unknown"),
        "embedding_model": cfg.embedding.get("model"),
        "attribution_temperature": float(temperature),
        "events": len(gold_events),
        "holdout_queries": sum(1 for e in gold_events if e.evaluation_role == "holdout"),
        "reference_events": sum(1 for e in gold_events if e.evaluation_role == "reference"),
        "extraction_seconds": {r.condition: round(r.extraction_seconds, 2) for r in results},
        "ranking_seconds": {r.condition: round(r.ranking_seconds, 2) for r in results},
        "hardware": _gpu_state(),
    }

    return {
        "results": results,
        "dossiers": dossiers,
        "gold_events": gold_events,
        "ranking_metrics": ranking,
        "extraction_metrics": extraction,
        "extraction_per_event": per_event,
        "hallucinated_iocs": hallucinations,
        "predictions": pd.concat([r.predictions for r in results], ignore_index=True),
        "class_metrics": pd.concat([r.class_metrics for r in results], ignore_index=True),
        "error_analysis": pd.concat(
            [r.error_analysis for r in results if not r.error_analysis.empty],
            ignore_index=True,
        ),
        "manifest": manifest,
    }


_RANKING_COLUMNS = [
    "condition", "model", "top1_accuracy", "top3_accuracy", "MRR",
    "review_coverage", "abstention_rate", "selective_accuracy",
    "false_attribution_rate",
]

_EXTRACTION_COLUMNS = [
    "condition", "ttp_f1", "channel_f1", "target_f1", "country_f1",
    "ioc_recall", "ioc_precision", "hallucinated_ioc_total",
    "evidence_sentence_coverage",
]


def write_condition_outputs(
    bundle: dict[str, Any], results_dir: Path, suffix: str = ""
) -> dict[str, Path]:
    """Persist every table plus a human-readable summary and a run manifest."""
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{suffix.strip('_')}" if suffix.strip("_") else ""
    written: dict[str, Path] = {}

    frames = {
        "condition_ranking_metrics": bundle["ranking_metrics"],
        "condition_extraction_metrics": bundle["extraction_metrics"],
        "condition_extraction_per_event": bundle["extraction_per_event"],
        "condition_predictions": bundle["predictions"],
        "condition_class_metrics": bundle["class_metrics"],
        "condition_error_analysis": bundle["error_analysis"],
        "condition_hallucinated_iocs": bundle["hallucinated_iocs"],
    }
    for name, frame in frames.items():
        path = results_dir / f"{name}{tag}.csv"
        frame.to_csv(path, index=False)
        written[name] = path

    for result in bundle["results"]:
        if not result.evidence:
            continue
        path = results_dir / f"condition_structured_evidence_{result.condition}{tag}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in result.evidence:
                handle.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
        written[f"evidence_{result.condition}"] = path

    dossier_path = results_dir / f"condition_case_dossiers{tag}.jsonl"
    with dossier_path.open("w", encoding="utf-8") as handle:
        for event_id, text in bundle["dossiers"].items():
            handle.write(
                json.dumps({"event_id": event_id, "dossier": text}, ensure_ascii=False) + "\n"
            )
    written["dossiers"] = dossier_path

    manifest_path = results_dir / f"condition_run_manifest{tag}.json"
    manifest_path.write_text(
        json.dumps(bundle["manifest"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written["manifest"] = manifest_path

    summary_path = results_dir / f"condition_benchmark_summary{tag}.md"
    summary_path.write_text(_summary_markdown(bundle), encoding="utf-8")
    written["summary"] = summary_path
    return written


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    present = [c for c in columns if c in frame.columns]
    view = frame[present].copy()
    for column in present:
        if view[column].dtype.kind == "f":
            view[column] = view[column].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    return view.to_markdown(index=False)


def _summary_markdown(bundle: dict[str, Any]) -> str:
    manifest = bundle["manifest"]
    lines = [
        "# Evidence-structuring conditions on the four-campaign benchmark",
        "",
        f"- model: `{manifest['model']}` (digest `{manifest['model_digest']}`)",
        f"- embedding backend: `{manifest['embedding_backend']}` / `{manifest['embedding_model']}`",
        f"- events: {manifest['events']} "
        f"({manifest['reference_events']} reference, {manifest['holdout_queries']} holdout)",
        f"- LLM decoding: temperature 0, seed 42, non-thinking",
        f"- technical annex included in dossier: {manifest['include_annex']}",
        "",
        "All conditions read the same label-scrubbed case dossier and share one",
        "downstream (evidence graph → FCLS → candidate ranking → abstention).",
        "Only the evidence-extraction step differs.",
        "",
        "## Candidate ranking (20 holdout queries)",
        "",
        _table(bundle["ranking_metrics"], _RANKING_COLUMNS),
        "",
        "## Evidence extraction against curated gold fields",
        "",
        _table(bundle["extraction_metrics"], _EXTRACTION_COLUMNS),
        "",
        "## Per-campaign Top-1",
        "",
        _table(
            bundle["class_metrics"].pivot_table(
                index="actual_actor", columns="condition", values="top1_accuracy"
            ).reset_index(),
            ["actual_actor", *bundle["class_metrics"]["condition"].unique().tolist()],
        ),
        "",
        "The LLM is an evidence-structuring module, not the decision maker: the",
        "ranking, the abstention policy, and the final assessment stay in the",
        "graph and scoring layer.",
        "",
    ]
    return "\n".join(lines)
