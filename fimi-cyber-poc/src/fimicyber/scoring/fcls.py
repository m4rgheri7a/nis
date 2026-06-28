"""FCLS score computation with missing-value renormalisation (spec 8).

FCLS(i,j) = αN + βI + γD + δC + εT + ζA
Re-normalised: Σ_available(w_k·x_k) / Σ_available(w_k)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from fimicyber.schema import Event


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
) -> pd.DataFrame:
    """Build pairwise_scores.csv with columns i,j,N,I,D,C,T,A,FCLS_E1,FCLS_E2,FCLS_E3."""
    n = len(events)
    weights_e3 = _extract_weights(cfg, allow_actor=False)

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

            fcls_e1 = fcls(comp_dict, {"N": 1.0})
            fcls_e2 = fcls(comp_dict, {"I": 1.0})
            fcls_e3 = fcls(comp_dict, weights_e3)

            rows.append({
                "event_i": events[i].event_id,
                "event_j": events[j].event_id,
                "N": _v(n_val), "I": _v(i_val),
                "D": _v(d_val), "C": _v(c_val),
                "T": _v(t_val), "A": _v(a_val),
                "FCLS_E1": fcls_e1 if not math.isnan(fcls_e1) else None,
                "FCLS_E2": fcls_e2 if not math.isnan(fcls_e2) else None,
                "FCLS_E3": fcls_e3 if not math.isnan(fcls_e3) else None,
            })

    return pd.DataFrame(rows)
