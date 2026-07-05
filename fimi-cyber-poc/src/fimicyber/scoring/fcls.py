"""FCLS score computation with missing-value renormalisation (spec 8).

FCLS(i,j) = αN + βI + γD + δC + εT + ζA
Re-normalised: Σ_available(w_k·x_k) / Σ_available(w_k)
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from fimicyber.schema import Event

_WORD_RE = re.compile(r"[A-Za-z0-9가-힣_]+")
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")
_FCLS_KEYS = ("N", "I", "D", "C", "T", "A")


def fcls(
    components: dict[str, float | None],
    weights: dict[str, float],
) -> float:
    """
    Compute FCLS for a single pair.

    components: {'N': float|nan, 'I': float|nan, 'D': float|nan, ...}
    weights:    {'N': alpha, 'I': beta, 'D': gamma, ...}

    Missing (None or NaN) components are excluded from both numerator and denominator.
    """
    num = 0.0
    den = 0.0
    for key, w in weights.items():
        val = components.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        num += w * val
        den += w
    return num / den if den > 0 else float("nan")


def fcls_strict(
    components: dict[str, float | None],
    weights: dict[str, float],
    policy: dict[str, Any] | None = None,
) -> float:
    """
    Evidence-aware FCLS used for E3 ranking.

    The plain fcls() function intentionally preserves the paper's missing-value
    renormalisation formula. This stricter variant keeps that formula but adds
    three guards needed for sparse public FIMI data:
      - down-weight low-information narrative-only matches via N_conf;
      - penalise pairs with poor component coverage;
      - cap pairs where N is the only available evidence axis.
    """
    policy = policy or {}
    adjusted = dict(components)
    n_conf = _valid_float(adjusted.get("N_conf"))
    if n_conf is not None and _valid_float(adjusted.get("N")) is not None:
        adjusted["N"] = float(adjusted["N"]) * n_conf

    score = fcls(adjusted, weights)
    if math.isnan(score):
        return score

    active_weight_sum = sum(w for key, w in weights.items() if key in _FCLS_KEYS and w > 0)
    available = _available_keys(adjusted, weights)
    available_weight_sum = sum(weights.get(key, 0.0) for key in available)
    coverage = available_weight_sum / active_weight_sum if active_weight_sum > 0 else 0.0

    power = float(policy.get("coverage_penalty_power", 0.7))
    score *= coverage ** power if coverage > 0 else 0.0

    min_components = int(policy.get("min_evidence_components", 2))
    if len(available) < min_components:
        score *= float(policy.get("insufficient_evidence_penalty", 0.35))

    if available == ["N"]:
        score = min(score, float(policy.get("n_only_cap", 0.45)))

    return max(0.0, min(1.0, score))


def _valid_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _available_keys(
    components: dict[str, float | None],
    weights: dict[str, float],
) -> list[str]:
    keys: list[str] = []
    for key in _FCLS_KEYS:
        if weights.get(key, 0.0) <= 0:
            continue
        if _valid_float(components.get(key)) is not None:
            keys.append(key)
    return keys


def narrative_confidence(event: Event, cfg: Any | None = None) -> float:
    """Return a conservative quality weight for event.description in [0, 1]."""
    policy = cfg.fcls if cfg is not None else {}
    min_tokens = int(policy.get("n_conf_min_tokens", 12))
    full_tokens = int(policy.get("n_conf_full_tokens", 45))

    text = _URL_RE.sub(" ", event.description or "").strip()
    if not text or text.lower() in {"no description", "none", "unknown", "n/a"}:
        return 0.0

    tokens = _WORD_RE.findall(text)
    if not tokens:
        return 0.0

    span = max(1, full_tokens - min_tokens)
    length_score = max(0.0, min(1.0, (len(tokens) - min_tokens) / span))
    unique_score = max(0.35, min(1.0, len(set(t.lower() for t in tokens)) / len(tokens)))
    return length_score * unique_score


def pair_narrative_confidence(ei: Event, ej: Event, cfg: Any | None = None) -> float:
    return min(narrative_confidence(ei, cfg), narrative_confidence(ej, cfg))


def _extract_weights(cfg: Any, allow_actor: bool = False) -> dict[str, float]:
    fc = cfg.fcls
    weights = {
        "N": float(fc.get("alpha_N", 0.30)),
        "I": float(fc.get("beta_I", 0.30)),
        "D": float(fc.get("gamma_D", 0.15)),
        "C": float(fc.get("delta_C", 0.10)),
        "T": float(fc.get("epsilon_T", 0.10)),
        "A": float(fc.get("zeta_A", 0.05)),
    }
    if not allow_actor:
        weights["A"] = 0.0  # hard guard (spec 10.3)
    return weights


def build_pairwise_scores(
    events: list[Event],
    N: np.ndarray,
    I: np.ndarray,
    comps: dict[str, np.ndarray],
    cfg: Any,
    allow_actor: bool = False,
    I_no_synthetic: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build pairwise_scores.csv with component diagnostics and score columns."""
    n = len(events)
    weights_e3 = _extract_weights(cfg, allow_actor=False)
    fcls_policy = cfg.fcls

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            n_val = N[i, j]
            i_val = I[i, j]
            d_val = comps["D"][i, j]
            c_val = comps["C"][i, j]
            t_val = comps["T"][i, j]
            a_val = comps["A"][i, j]

            def _v(x):
                return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

            comp_dict = {
                "N": _v(n_val), "I": _v(i_val),
                "D": _v(d_val), "C": _v(c_val),
                "T": _v(t_val), "A": _v(a_val),
            }
            n_conf = pair_narrative_confidence(events[i], events[j], cfg)
            comp_strict = dict(comp_dict)
            comp_strict["N_conf"] = n_conf

            fcls_e1 = fcls(comp_dict, {"N": 1.0})
            fcls_e2 = fcls(comp_dict, {"I": 1.0})
            fcls_e3_raw = fcls(comp_dict, weights_e3)
            fcls_e3 = fcls_strict(comp_strict, weights_e3, fcls_policy)

            i_no_synth_val = None
            fcls_e2_no_synth = float("nan")
            fcls_e3_no_synth = float("nan")
            if I_no_synthetic is not None:
                i_no_synth_val = _v(I_no_synthetic[i, j])
                comp_real = dict(comp_strict)
                comp_real["I"] = i_no_synth_val
                fcls_e2_no_synth = fcls(comp_real, {"I": 1.0})
                fcls_e3_no_synth = fcls_strict(comp_real, weights_e3, fcls_policy)

            available = _available_keys(comp_strict, weights_e3)
            active_weight_sum = sum(w for key, w in weights_e3.items() if key in _FCLS_KEYS and w > 0)
            available_weight_sum = sum(weights_e3.get(key, 0.0) for key in available)
            coverage = available_weight_sum / active_weight_sum if active_weight_sum > 0 else float("nan")

            rows.append({
                "event_i": events[i].event_id,
                "event_j": events[j].event_id,
                "N": _v(n_val), "I": _v(i_val),
                "I_no_synthetic": i_no_synth_val,
                "D": _v(d_val), "C": _v(c_val),
                "T": _v(t_val), "A": _v(a_val),
                "N_conf": n_conf,
                "evidence_components": len(available),
                "evidence_coverage": coverage,
                "FCLS_E1": fcls_e1 if not math.isnan(fcls_e1) else None,
                "FCLS_E2": fcls_e2 if not math.isnan(fcls_e2) else None,
                "FCLS_E2_no_synthetic_ioc": fcls_e2_no_synth if not math.isnan(fcls_e2_no_synth) else None,
                "FCLS_E3_raw": fcls_e3_raw if not math.isnan(fcls_e3_raw) else None,
                "FCLS_E3": fcls_e3 if not math.isnan(fcls_e3) else None,
                "FCLS_E3_no_synthetic_ioc": fcls_e3_no_synth if not math.isnan(fcls_e3_no_synth) else None,
            })

    return pd.DataFrame(rows)
