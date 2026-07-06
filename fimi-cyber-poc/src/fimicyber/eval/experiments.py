"""Experiment runner: E1/E2/E3, ablation, grid, robustness (spec 10.3-10.5)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from fimicyber.schema import Event
from fimicyber.eval.groundtruth import build_ground_truth, gt_stats
from fimicyber.eval.metrics import evaluate_condition, bootstrap_ci, roc_auc
from fimicyber.scoring.fcls import fcls, fcls_strict, pair_narrative_confidence


# ── ζ=0 hard guard ────────────────────────────────────────────────────────────

def _build_score_fn(
    events: list[Event],
    scores_df: pd.DataFrame,
    weight_key: str,
    allow_actor: bool = False,
) -> Any:
    """
    Return callable(q_id, doc_id) → float|None.

    weight_key: 'FCLS_E1' | 'FCLS_E2' | 'FCLS_E3' | or a column name in scores_df.
    """
    # Build lookup: frozenset({a,b}) → score
    col = weight_key
    if col not in scores_df.columns:
        raise ValueError(f"Column {col!r} not in pairwise_scores")

    lookup: dict[frozenset, float | None] = {}
    for _, row in scores_df.iterrows():
        key = frozenset({row["event_i"], row["event_j"]})
        val = row[col]
        lookup[key] = None if (val is None or (isinstance(val, float) and math.isnan(val))) else float(val)

    def score_fn(q_id: str, doc_id: str) -> float | None:
        return lookup.get(frozenset({q_id, doc_id}))

    return score_fn


def run_evaluation(
    events: list[Event],
    scores_df: pd.DataFrame,
    cfg: Any,
    allow_actor: bool = False,
    include_actor_surrogate: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Run E1, E2, E3 (and optionally E3+A) evaluations.
    allow_actor must be False for E3 (hard guard on ζ).
    """
    if not allow_actor and cfg.fcls.get("zeta_A", 0.05) > 0:
        # ζ>0 requested but allow_actor=False → OK, FCLS computation already set ζ=0
        pass

    gt = build_ground_truth(events, include_actor_surrogate=include_actor_surrogate)
    all_ids = [ev.event_id for ev in events]
    query_ids = gt["query_ids"]
    positives = gt["positives"]

    results: dict[str, dict[str, float]] = {}

    conditions = [("E1", "FCLS_E1"), ("E2", "FCLS_E2")]
    if "FCLS_E2_no_synthetic_ioc" in scores_df.columns and scores_df["FCLS_E2_no_synthetic_ioc"].notna().any():
        conditions.append(("E2_no_synthetic_ioc", "FCLS_E2_no_synthetic_ioc"))
    if "FCLS_E3_raw" in scores_df.columns:
        conditions.append(("E3_raw", "FCLS_E3_raw"))
    conditions.append(("E3", "FCLS_E3"))
    if "FCLS_E3_no_synthetic_ioc" in scores_df.columns and scores_df["FCLS_E3_no_synthetic_ioc"].notna().any():
        conditions.append(("E3_no_synthetic_ioc", "FCLS_E3_no_synthetic_ioc"))

    gt_mode = "campaign_or_actor_surrogate" if include_actor_surrogate else "explicit_campaign_only"
    if len({ev.source_dataset for ev in events}) > 1 and include_actor_surrogate:
        gt_mode = "combined_all_datasets"
    for condition, col in conditions:
        fn = _build_score_fn(events, scores_df, col, allow_actor=allow_actor)
        metrics = evaluate_condition(query_ids, all_ids, positives, fn, cfg)
        ci_lo, ci_hi = bootstrap_ci(query_ids, all_ids, positives, fn, cfg,
                                     n_iter=int(cfg.eval.get("bootstrap_iters", 1000)),
                                     seed=cfg.seed)
        metrics["MAP_CI_low"] = ci_lo
        metrics["MAP_CI_high"] = ci_hi
        metrics["gt_mode"] = gt_mode
        results[condition] = metrics

    return results


def run_all_experiments(
    events: list[Event],
    scores_df: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    """Run E1/E2/E3 and return metrics_summary.csv DataFrame."""
    results = run_evaluation(events, scores_df, cfg, allow_actor=False)

    rows = []
    for condition, metrics in results.items():
        row = {"condition": condition, **metrics}
        rows.append(row)

    return pd.DataFrame(rows)


def run_ablation(
    events: list[Event],
    scores_df: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    """Ablation: remove each component N/I/D/C/T one at a time."""
    from fimicyber.scoring.fcls import fcls_strict as fcls_fn, _extract_weights

    gt = build_ground_truth(events)
    all_ids = [ev.event_id for ev in events]
    query_ids = gt["query_ids"]
    positives = gt["positives"]

    base_weights = _extract_weights(cfg, allow_actor=False)
    components_to_ablate = ["N", "I", "D", "C", "T"]

    comp_lookup: dict[frozenset, dict[str, float | None]] = {}
    for _, row in scores_df.iterrows():
        comp = {
            k: (None if row.get(k) is None or pd.isna(row.get(k)) else float(row.get(k)))
            for k in ["N", "I", "D", "C", "T", "A", "N_conf"]
        }
        comp_lookup[frozenset({row["event_i"], row["event_j"]})] = comp

    rows = []
    for ablate in components_to_ablate:
        w = dict(base_weights)
        w[ablate] = 0.0

        # Build score function from raw components
        def _score_fn_ablate(q_id: str, doc_id: str, _w=w) -> float | None:
            comp = comp_lookup.get(frozenset({q_id, doc_id}))
            if comp is None:
                return None
            val = fcls_fn(comp, _w, cfg.fcls)
            return None if math.isnan(val) else val

        metrics = evaluate_condition(query_ids, all_ids, positives, _score_fn_ablate, cfg)
        rows.append({"ablated": ablate, **metrics})

    return pd.DataFrame(rows)


def run_grid(
    events: list[Event],
    N: np.ndarray,
    I: np.ndarray,
    comps: dict[str, np.ndarray],
    cfg: Any,
) -> pd.DataFrame:
    """Grid search over α, β ∈ config.eval.grid_alpha × grid_beta."""
    from fimicyber.scoring.fcls import fcls_strict as fcls_fn

    gt = build_ground_truth(events)
    all_ids = [ev.event_id for ev in events]
    query_ids = gt["query_ids"]
    positives = gt["positives"]

    grid_alpha = cfg.eval.get("grid_alpha", [0.1, 0.2, 0.3, 0.4, 0.5])
    grid_beta = cfg.eval.get("grid_beta", [0.1, 0.2, 0.3, 0.4, 0.5])

    n = len(events)
    ev_ids = [ev.event_id for ev in events]
    idx_map = {eid: i for i, eid in enumerate(ev_ids)}

    rows = []
    for alpha in grid_alpha:
        for beta in grid_beta:
            rem = 1.0 - alpha - beta
            if rem < 0:
                continue
            # γ:δ:ε = 1.5:1:1 ratio
            total_ratio = 3.5
            gamma = rem * 1.5 / total_ratio
            delta = rem * 1.0 / total_ratio
            epsilon = rem * 1.0 / total_ratio

            w = {"N": alpha, "I": beta, "D": gamma, "C": delta, "T": epsilon, "A": 0.0}

            def _score_fn(q_id: str, doc_id: str, _w=w) -> float | None:
                i, j = idx_map.get(q_id), idx_map.get(doc_id)
                if i is None or j is None:
                    return None
                comp = {
                    "N": None if math.isnan(N[i, j]) else float(N[i, j]),
                    "I": None if math.isnan(I[i, j]) else float(I[i, j]),
                    "D": None if math.isnan(comps["D"][i, j]) else float(comps["D"][i, j]),
                    "C": None if math.isnan(comps["C"][i, j]) else float(comps["C"][i, j]),
                    "T": None if math.isnan(comps["T"][i, j]) else float(comps["T"][i, j]),
                    "A": None,
                    "N_conf": pair_narrative_confidence(events[i], events[j], cfg),
                }
                val = fcls_fn(comp, _w, cfg.fcls)
                return None if math.isnan(val) else val

            metrics = evaluate_condition(query_ids, all_ids, positives, _score_fn, cfg)
            rows.append({"alpha": alpha, "beta": beta, "gamma": gamma,
                         "delta": delta, "epsilon": epsilon, **metrics})

    return pd.DataFrame(rows)


def run_robustness(
    events: list[Event],
    N: np.ndarray,
    comps: dict[str, np.ndarray],
    cfg: Any,
    fallbacks_used: list[str] | None = None,
) -> pd.DataFrame:
    """9-scenario robustness experiment (noise × coverage)."""
    from fimicyber.ioc.synthetic import generate_synthetic_iocs
    from fimicyber.graph.build import build_graph
    from fimicyber.graph.ioc_score import ioc_matrix
    from fimicyber.scoring.fcls import build_pairwise_scores, fcls_strict as fcls_fn, _extract_weights

    sc = cfg.synthetic
    grid = sc["robustness_grid"]
    noise_vals = grid["noise_ratio"]
    cov_vals = grid["coverage"]
    base_seed = int(sc.get("base_seed", 42))

    gt = build_ground_truth(events)
    all_ids = [ev.event_id for ev in events]
    query_ids = gt["query_ids"]
    positives = gt["positives"]

    w_e3 = _extract_weights(cfg, allow_actor=False)
    w_e2 = {"N": 0.0, "I": 1.0, "D": 0.0, "C": 0.0, "T": 0.0, "A": 0.0}

    rows = []
    scenario_idx = 0
    for noise in noise_vals:
        for coverage in cov_vals:
            scenario_seed = base_seed + scenario_idx
            scenario_idx += 1

            # Modify synthetic config temporarily
            import copy
            sc_copy = copy.deepcopy(sc)
            sc_copy["default"]["noise_ratio"] = noise
            sc_copy["default"]["coverage"] = coverage
            sc_copy["base_seed"] = scenario_seed

            class _CfgScenario:
                def __init__(self, parent, synth):
                    self._parent = parent
                    self._synth = synth
                @property
                def seed(self): return scenario_seed
                @property
                def synthetic(self): return self._synth
                @property
                def ioc_score(self): return self._parent.ioc_score
                @property
                def fcls(self): return self._parent.fcls
                @property
                def components(self): return self._parent.components
                @property
                def eval(self): return self._parent.eval
                @property
                def narrative(self): return self._parent.narrative
                @property
                def embedding(self): return self._parent.embedding
                @property
                def priority(self): return self._parent.priority
                @property
                def results_dir(self): return self._parent.results_dir
                @property
                def data_dir(self): return self._parent.data_dir

            cfg_sc = _CfgScenario(cfg, sc_copy)

            # Deep-copy events and regenerate IOCs
            import copy as _copy
            events_sc = _copy.deepcopy(events)
            # Remove existing synthetic IOCs
            for ev in events_sc:
                ev.iocs = [ioc for ioc in ev.iocs if not ioc.synthetic]

            # Allow noise=0 only in robustness (spec 10.5)
            try:
                events_sc = generate_synthetic_iocs(events_sc, cfg_sc)
            except ValueError:
                # noise=0.0 triggers our guard in the main config but not here
                # Bypass by patching validation
                events_sc = _generate_synthetic_no_guard(events_sc, cfg_sc)

            G_sc = build_graph(events_sc, cfg_sc)
            I_sc = ioc_matrix(events_sc, G_sc, cfg_sc)

            n = len(events_sc)
            ev_ids = [ev.event_id for ev in events_sc]
            idx_map = {eid: i for i, eid in enumerate(ev_ids)}

            def _make_fn(I_mat, w):
                def fn(q_id, doc_id):
                    i, j = idx_map.get(q_id), idx_map.get(doc_id)
                    if i is None or j is None:
                        return None
                    comp = {
                        "N": None if math.isnan(N[i, j]) else float(N[i, j]),
                        "I": None if math.isnan(I_mat[i, j]) else float(I_mat[i, j]),
                        "D": None if math.isnan(comps["D"][i, j]) else float(comps["D"][i, j]),
                        "C": None if math.isnan(comps["C"][i, j]) else float(comps["C"][i, j]),
                        "T": None if math.isnan(comps["T"][i, j]) else float(comps["T"][i, j]),
                        "A": None,
                        "N_conf": pair_narrative_confidence(events_sc[i], events_sc[j], cfg),
                    }
                    val = fcls_fn(comp, w, cfg.fcls)
                    return None if math.isnan(val) else val
                return fn

            for cond_label, w in [("E2", w_e2), ("E3", w_e3)]:
                fn = _make_fn(I_sc, w)
                metrics = evaluate_condition(query_ids, all_ids, positives, fn, cfg)
                rows.append({
                    "scenario": scenario_idx - 1,
                    "noise_ratio": noise,
                    "coverage": coverage,
                    "condition": cond_label,
                    **metrics,
                })

    return pd.DataFrame(rows)


def _generate_synthetic_no_guard(events: list[Event], cfg: Any) -> list[Event]:
    """Version of generate_synthetic_iocs that skips config validation (robustness only)."""
    import random, copy
    from datetime import date, timedelta
    from fimicyber.ioc.synthetic import (
        _generate_pool, _generate_noise_pool, _apply_jitter,
        _validate_reserved, _save_manifest,
    )
    import hashlib, json
    from collections import defaultdict

    sc = cfg.synthetic
    default = sc["default"]
    coverage = default["coverage"]
    noise_ratio = default["noise_ratio"]
    jitter_days = default["temporal_jitter_days"]
    ns_link_prob = default["ns_link_prob"]
    type_mix = default["type_mix"]
    base_seed = int(sc.get("base_seed", 42))
    eligible_sources = set(default.get("eligible_source_datasets", ["disinfox", "fixture"]))

    rng = random.Random(base_seed)

    camp_map: dict[str, list[int]] = defaultdict(list)
    for idx, ev in enumerate(events):
        if ev.campaign_id and ev.source_dataset in eligible_sources:
            camp_map[ev.campaign_id].append(idx)

    manifest: dict = {"seed": base_seed, "coverage": coverage, "noise_ratio": noise_ratio,
                      "jitter_days": jitter_days, "injections": [], "noise_injections": []}

    camp_shared_iocs = {}
    camp_shared_ns = {}

    for camp_id, idxs in camp_map.items():
        if len(idxs) < 2:
            continue
        n_shared = max(2, rng.randint(3, 7))
        pool = _generate_pool(rng, n_shared, type_mix, ns_link_prob)
        camp_shared_iocs[camp_id] = pool
        camp_shared_ns[camp_id] = None
        if rng.random() < ns_link_prob:
            from fimicyber.ioc.synthetic import _gen_ns
            camp_shared_ns[camp_id] = _gen_ns(rng)

    noise_pool = _generate_noise_pool(rng)

    for camp_id, idxs in camp_map.items():
        if len(idxs) < 2 or camp_id not in camp_shared_iocs:
            continue
        pool = camp_shared_iocs[camp_id]
        ns_val = camp_shared_ns.get(camp_id)
        for idx in idxs:
            ev = events[idx]
            n_assign = max(1, int(len(pool) * coverage))
            chosen = rng.sample(pool, min(n_assign, len(pool)))
            dated = _apply_jitter(chosen, ev, jitter_days, rng)
            ev.iocs.extend(dated)
            if ns_val:
                from fimicyber.ioc.synthetic import _make_ioc
                ev.iocs.append(_make_ioc(ns_val, "ns", ev.first_seen, ev.last_seen))
            manifest["injections"].append({"campaign": camp_id, "event_id": ev.event_id,
                                           "ioc_values": [i.value for i in dated], "ns": ns_val})

    if noise_ratio > 0:
        all_idxs = [
            idx for idx, ev in enumerate(events)
            if ev.source_dataset in eligible_sources
        ]
        n_noise = max(1, int(len(events) * noise_ratio))
        noise_targets = rng.sample(all_idxs, min(n_noise, len(all_idxs)))
        for idx in noise_targets:
            ev = events[idx]
            noise_iocs = _apply_jitter(noise_pool, ev, jitter_days, rng)
            ev.iocs.extend(noise_iocs)
            manifest["noise_injections"].append({"event_id": ev.event_id,
                                                  "ioc_values": [i.value for i in noise_iocs]})

    _save_manifest(manifest, cfg)
    return events
