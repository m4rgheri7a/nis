"""Chronological calibration, abstention, and attribution diagnostics."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from fimicyber.attribution.hypotheses import (
    _confidence_band,
    _expected_calibration_error,
    assessment_confidence,
)


def _actor_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def _probabilities(scores: Iterable[float], temperature: float) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    if values.size == 0:
        return values
    shifted = values - values.max()
    exp = np.exp(shifted / max(float(temperature), 1e-6))
    return exp / exp.sum()


def _evaluable_query_ids(hypotheses: pd.DataFrame) -> list[str]:
    result: list[str] = []
    for query_id, group in hypotheses.groupby("query_event_id", sort=False):
        actual = _actor_id(group.iloc[0].get("actual_actor_id"))
        candidates = {str(value) for value in group["candidate_actor_id"]}
        if actual and actual in candidates:
            result.append(str(query_id))
    return result


def _nll(hypotheses: pd.DataFrame, query_ids: set[str], temperature: float) -> float:
    losses: list[float] = []
    for _, group in hypotheses[hypotheses["query_event_id"].isin(query_ids)].groupby(
        "query_event_id", sort=False
    ):
        ordered = group.sort_values("rank")
        probs = _probabilities(ordered["support_score"], temperature)
        labels = ordered["correct_actor"].astype(bool).to_numpy()
        if labels.any():
            losses.append(-float(np.log(max(probs[labels][0], 1e-12))))
    return float(np.mean(losses)) if losses else float("nan")


def _fit_temperature(hypotheses: pd.DataFrame, query_ids: set[str], cfg: Any) -> tuple[float, float, float]:
    cal_cfg = cfg.attribution.get("calibration", {})
    baseline = float(cfg.attribution.get("temperature", 0.15))
    if not query_ids or cal_cfg.get("enabled", True) is False:
        score = _nll(hypotheses, query_ids, baseline)
        return baseline, score, score
    grid = np.geomspace(
        float(cal_cfg.get("min_temperature", 0.02)),
        float(cal_cfg.get("max_temperature", 2.0)),
        int(cal_cfg.get("grid_points", 80)),
    )
    losses = np.asarray([_nll(hypotheses, query_ids, value) for value in grid])
    best = int(np.nanargmin(losses))
    return float(grid[best]), _nll(hypotheses, query_ids, baseline), float(losses[best])


def apply_temperature_and_abstention(
    hypotheses: pd.DataFrame,
    cfg: Any,
    temperature: float,
    split_roles: dict[str, str] | None = None,
    split_role_override: str | None = None,
) -> pd.DataFrame:
    output = hypotheses.copy()
    if output.empty:
        for column in ("split_role", "decision", "abstention_reason"):
            output[column] = pd.Series(dtype="object")
        return output

    abstention = cfg.attribution.get("abstention", {})
    confidence_threshold = float(abstention.get("confidence_threshold", 0.55))
    margin_threshold = float(abstention.get("margin_threshold", 0.03))
    min_families = int(abstention.get("min_evidence_families", 2))
    output["split_role"] = "not_evaluable"
    output["decision"] = "not_top_ranked"
    output["abstention_reason"] = ""

    for query_id, indexes in output.groupby("query_event_id", sort=False).groups.items():
        group = output.loc[indexes].sort_values("rank")
        probs = _probabilities(group["support_score"], temperature)
        output.loc[group.index, "candidate_probability"] = probs
        role = split_role_override or (split_roles or {}).get(str(query_id), "not_evaluable")
        output.loc[group.index, "split_role"] = role

        for index, probability in zip(group.index, probs):
            row = output.loc[index]
            confidence = assessment_confidence(
                float(probability),
                str(row.get("evidence_families") or ""),
                bool(row.get("real_ioc_support")),
                int(row.get("source_org_count") or 0),
                cfg,
            )
            output.at[index, "assessment_confidence"] = confidence
            output.at[index, "confidence_band"] = _confidence_band(confidence, cfg)

        top_index = group.index[0]
        top = output.loc[top_index]
        reasons: list[str] = []
        if float(top["assessment_confidence"]) < confidence_threshold:
            reasons.append("low_confidence")
        if float(top.get("margin_to_next") or 0.0) < margin_threshold:
            reasons.append("small_margin")
        family_count = len([value for value in str(top.get("evidence_families") or "").split("|") if value])
        if family_count < min_families:
            reasons.append("insufficient_evidence_families")
        output.at[top_index, "decision"] = "abstain" if reasons else "analyst_review"
        output.at[top_index, "abstention_reason"] = "|".join(reasons)
    return output


def calibrate_hypotheses(hypotheses: pd.DataFrame, cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hypotheses.empty:
        return apply_temperature_and_abstention(hypotheses, cfg, 0.15), pd.DataFrame()
    evaluable = set(_evaluable_query_ids(hypotheses))
    query_dates = (
        hypotheses[hypotheses["query_event_id"].isin(evaluable)]
        .groupby("query_event_id", as_index=False)["query_date"]
        .first()
        .sort_values(["query_date", "query_event_id"])
    )
    ordered_ids = query_dates["query_event_id"].astype(str).tolist()
    fraction = float(cfg.attribution.get("calibration", {}).get("fraction", 0.60))
    split_at = min(max(1, int(len(ordered_ids) * fraction)), max(1, len(ordered_ids) - 1))
    calibration_ids = set(ordered_ids[:split_at])
    validation_ids = set(ordered_ids[split_at:])
    temperature, baseline_nll, fitted_nll = _fit_temperature(hypotheses, calibration_ids, cfg)
    roles = {query_id: "calibration" for query_id in calibration_ids}
    roles.update({query_id: "validation" for query_id in validation_ids})
    calibrated = apply_temperature_and_abstention(hypotheses, cfg, temperature, roles)
    summary = pd.DataFrame([{
        "method": "chronological_temperature_scaling",
        "calibration_queries": len(calibration_ids),
        "validation_queries": len(validation_ids),
        "fitted_temperature": temperature,
        "baseline_temperature": float(cfg.attribution.get("temperature", 0.15)),
        "baseline_calibration_NLL": baseline_nll,
        "fitted_calibration_NLL": fitted_nll,
        "label_usage": "actor labels used only in chronological calibration partition",
    }])
    return calibrated, summary


def _bootstrap_ci(values: list[float], iters: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(iters)]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def evaluate_attribution_scope(
    hypotheses: pd.DataFrame,
    cfg: Any,
    scope: str,
    split_roles: set[str] | None = None,
) -> pd.DataFrame:
    frame = hypotheses
    if split_roles is not None and "split_role" in frame:
        frame = frame[frame["split_role"].isin(split_roles)]
    total = frame["query_event_id"].nunique() if not frame.empty else 0
    records: list[dict[str, Any]] = []
    labeled = 0
    for query_id, group in frame.groupby("query_event_id", sort=False):
        ordered = group.sort_values("rank")
        actual = _actor_id(ordered.iloc[0].get("actual_actor_id"))
        if not actual:
            continue
        labeled += 1
        correct = ordered[ordered["candidate_actor_id"].astype(str) == actual]
        if correct.empty:
            continue
        correct_rank = int(correct["rank"].min())
        top = ordered.iloc[0]
        probs = ordered["candidate_probability"].astype(float).to_numpy()
        labels = (ordered["candidate_actor_id"].astype(str) == actual).astype(float).to_numpy()
        records.append({
            "actual": actual,
            "top1": float(correct_rank == 1),
            "top3": float(correct_rank <= 3),
            "rr": 1.0 / correct_rank,
            "brier": float(np.sum((probs - labels) ** 2)),
            "confidence": float(top["candidate_probability"]),
            "decision": str(top.get("decision") or "analyst_review"),
        })
    if not records:
        return pd.DataFrame([{"evaluation_scope": scope, "queries_total": total, "queries_labeled": labeled, "queries_evaluable": 0}])

    top1 = [row["top1"] for row in records]
    top3 = [row["top3"] for row in records]
    labels = pd.Series([row["actual"] for row in records])
    majority = float(labels.value_counts().max() / len(labels))
    macro = float(np.mean([
        np.mean([row["top1"] for row in records if row["actual"] == actor])
        for actor in labels.unique()
    ]))
    accepted = [row for row in records if row["decision"] == "analyst_review"]
    iters = int(cfg.attribution.get("bootstrap_iters", 500))
    top1_low, top1_high = _bootstrap_ci(top1, iters, int(cfg.seed))
    top3_low, top3_high = _bootstrap_ci(top3, iters, int(cfg.seed) + 1)
    confidences = [row["confidence"] for row in records]
    return pd.DataFrame([{
        "evaluation_scope": scope,
        "queries_total": total,
        "queries_labeled": labeled,
        "queries_evaluable": len(records),
        "history_coverage": len(records) / labeled if labeled else 0.0,
        "n_actor_labels": int(labels.nunique()),
        "majority_baseline_accuracy": majority,
        "top1_accuracy": float(np.mean(top1)),
        "top1_ci95_low": top1_low,
        "top1_ci95_high": top1_high,
        "macro_top1_accuracy": macro,
        "top1_lift_over_majority": float(np.mean(top1)) - majority,
        "top3_accuracy": float(np.mean(top3)),
        "top3_ci95_low": top3_low,
        "top3_ci95_high": top3_high,
        "MRR": float(np.mean([row["rr"] for row in records])),
        "multiclass_brier": float(np.mean([row["brier"] for row in records])),
        "ECE": _expected_calibration_error(confidences, top1),
        "review_coverage": len(accepted) / len(records),
        "abstention_rate": 1.0 - len(accepted) / len(records),
        "selective_accuracy": float(np.mean([row["top1"] for row in accepted])) if accepted else float("nan"),
        "false_attribution_rate": sum(1 for row in accepted if not row["top1"]) / len(records),
        "temporal_only": bool(cfg.attribution.get("temporal_only", True)),
    }])


def evaluate_attribution_scopes(hypotheses: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    frames = [evaluate_attribution_scope(hypotheses, cfg, "all_temporal")]
    if "split_role" in hypotheses:
        for role in ("calibration", "validation"):
            if (hypotheses["split_role"] == role).any():
                frames.append(evaluate_attribution_scope(hypotheses, cfg, role, {role}))
    return pd.concat(frames, ignore_index=True)


def build_error_analysis(hypotheses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id, group in hypotheses.groupby("query_event_id", sort=False):
        ordered = group.sort_values("rank")
        actual = _actor_id(ordered.iloc[0].get("actual_actor_id"))
        correct = ordered[ordered["candidate_actor_id"].astype(str) == actual]
        if not actual or correct.empty:
            continue
        top = ordered.iloc[0]
        is_correct = bool(str(top["candidate_actor_id"]) == actual)
        decision = str(top.get("decision") or "analyst_review")
        rows.append({
            "query_event_id": query_id,
            "split_role": top.get("split_role", ""),
            "actual_actor": top.get("actual_actor", ""),
            "predicted_actor": top.get("candidate_actor", ""),
            "actual_rank": int(correct["rank"].min()),
            "decision": decision,
            "error_type": ("correct" if is_correct else "incorrect") + ("_abstained" if decision == "abstain" else "_reviewed"),
            "candidate_probability": top.get("candidate_probability"),
            "assessment_confidence": top.get("assessment_confidence"),
            "margin_to_next": top.get("margin_to_next"),
            "abstention_reason": top.get("abstention_reason", ""),
            "evidence_families": top.get("evidence_families", ""),
            "supporting_event_ids": top.get("supporting_event_ids", ""),
        })
    return pd.DataFrame(rows)
