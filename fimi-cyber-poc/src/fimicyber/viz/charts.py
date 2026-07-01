"""Matplotlib chart generation (spec 11)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_all_charts(
    metrics_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    rob_df: pd.DataFrame,
    cfg: Any,
    abl_df: pd.DataFrame | None = None,
    scores_df: pd.DataFrame | None = None,
    events: list | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _metrics_bar(metrics_df, fig_dir / "metrics_bar.png")
    _robustness_lines(rob_df, fig_dir / "robustness_lines.png")
    _grid_heatmap(grid_df, fig_dir / "grid_heatmap.png")

    if abl_df is not None:
        e3_rows = metrics_df[metrics_df["condition"] == "E3"]
        baseline_map = float(e3_rows["MAP"].iloc[0]) if not e3_rows.empty else 0.0
        _ablation_bar(abl_df, baseline_map, fig_dir / "ablation_bar.png")

    if scores_df is not None:
        _score_distribution(scores_df, fig_dir / "score_distribution.png")
        _component_radar(scores_df, fig_dir / "component_radar.png")

    if scores_df is not None and events is not None:
        _pairwise_heatmap(scores_df, events, fig_dir / "pairwise_heatmap.png")


def _metrics_bar(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    conditions = df["condition"].tolist()
    map_vals = df["MAP"].tolist()
    ndcg_vals = df["nDCG@10"].tolist()

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, map_vals, width, label="MAP", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, ndcg_vals, width, label="nDCG@10", color="#dd8452")

    ax.set_xlabel("Condition")
    ax.set_ylabel("Score")
    ax.set_title("E1 / E2 / E3 Retrieval Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _robustness_lines(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))

    for condition, group in df.groupby("condition"):
        # Average across coverage for each noise level
        avg = group.groupby("noise_ratio")["MAP"].mean().reset_index()
        ax.plot(avg["noise_ratio"], avg["MAP"], marker="o", label=condition)

    ax.set_xlabel("Noise ratio")
    ax.set_ylabel("MAP (avg over coverage levels)")
    ax.set_title("Robustness: MAP vs. Noise Ratio (E2 vs E3)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _grid_heatmap(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(index="alpha", columns="beta", values="MAP", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="MAP")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
    ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
    ax.set_xlabel("β (I weight)")
    ax.set_ylabel("α (N weight)")
    ax.set_title("Grid Search: MAP by α × β")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _ablation_bar(abl_df: pd.DataFrame, baseline_map: float, out: Path) -> None:
    import matplotlib.pyplot as plt

    components = abl_df["ablated"].tolist()
    map_vals = abl_df["MAP"].tolist()
    colors = ["#c44e52" if v < baseline_map * 0.95 else "#4c72b0" for v in map_vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(components, map_vals, color=colors, width=0.5)
    ax.axhline(baseline_map, color="#2ca02c", linestyle="--", linewidth=1.5,
               label=f"E3 baseline ({baseline_map:.3f})")
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xlabel("Ablated component")
    ax.set_ylabel("MAP")
    ax.set_title("Ablation Study: MAP when each component is removed")
    ax.set_ylim(0, min(1.05, max(map_vals + [baseline_map]) * 1.15))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _score_distribution(scores_df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    palette = {"FCLS_E1": "#4c72b0", "FCLS_E2": "#dd8452", "FCLS_E3": "#55a868"}
    for col, color in palette.items():
        if col in scores_df.columns:
            vals = scores_df[col].dropna()
            ax.hist(vals, bins=30, alpha=0.5, color=color, label=col, density=True)

    ax.set_xlabel("FCLS Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: E1 / E2 / E3")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _component_radar(scores_df: pd.DataFrame, out: Path, top_k: int = 20) -> None:
    import matplotlib.pyplot as plt

    comp_cols = ["N", "I", "D", "C", "T"]
    available = [c for c in comp_cols if c in scores_df.columns]
    if not available or "FCLS_E3" not in scores_df.columns:
        return

    top = scores_df.nlargest(top_k, "FCLS_E3")[available]
    means = top.mean().fillna(0).tolist()

    n_cats = len(available)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    values = means + means[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, color="#4c72b0")
    ax.fill(angles, values, alpha=0.25, color="#4c72b0")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title(f"Avg component values\n(top {top_k} pairs by FCLS_E3)", pad=15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _pairwise_heatmap(scores_df: pd.DataFrame, events: list, out: Path,
                      max_events: int = 40) -> None:
    import matplotlib.pyplot as plt

    ev_camp = [(ev.event_id, ev.campaign_id) for ev in events if ev.campaign_id]
    ev_camp.sort(key=lambda x: x[1])
    selected = ev_camp[:max_events]
    if len(selected) < 2:
        return

    sel_ids = [eid for eid, _ in selected]
    sel_camps = [c for _, c in selected]
    n = len(sel_ids)
    idx_map = {eid: i for i, eid in enumerate(sel_ids)}

    mat = np.full((n, n), np.nan)
    for _, row in scores_df.iterrows():
        i = idx_map.get(row["event_i"])
        j = idx_map.get(row["event_j"])
        if i is not None and j is not None:
            v = row.get("FCLS_E3")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                mat[i, j] = float(v)
                mat[j, i] = float(v)
    np.fill_diagonal(mat, 1.0)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="FCLS_E3")

    short_ids = [eid[-4:] if len(eid) > 4 else eid for eid in sel_ids]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_ids, rotation=90, fontsize=6)
    ax.set_yticklabels(short_ids, fontsize=6)
    ax.set_title(f"Pairwise FCLS_E3 heatmap (top {n} campaign events, sorted by campaign)")

    prev_camp = sel_camps[0]
    for k in range(1, n):
        if sel_camps[k] != prev_camp:
            ax.axhline(k - 0.5, color="#1f77b4", linewidth=0.8, alpha=0.7)
            ax.axvline(k - 0.5, color="#1f77b4", linewidth=0.8, alpha=0.7)
            prev_camp = sel_camps[k]

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report(
    events: list,
    metrics_df: pd.DataFrame,
    abl_df: pd.DataFrame,
    rob_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    cfg: Any,
    fallbacks_used: list[str],
) -> None:
    from fimicyber.eval.groundtruth import build_ground_truth, gt_stats

    gt = build_ground_truth(events)
    stats = gt_stats(gt)

    n_real_iocs = sum(1 for ev in events for ioc in ev.iocs if not ioc.synthetic and ioc.category == "OperationalIOC")
    n_synth_iocs = sum(1 for ev in events for ioc in ev.iocs if ioc.synthetic)

    # Top 3 pairs
    top3 = scores_df.nlargest(3, "FCLS_E3")[["event_i", "event_j", "N", "I", "D", "C", "T", "FCLS_E3"]]

    lines = [
        "# FIMI-Cyber Link Score PoC — Results Report",
        "",
        "## ① 실행 환경 및 폴백",
        "",
        f"- **Python**: 3.11+",
        f"- **Events**: {len(events)}",
    ]

    if fallbacks_used:
        lines.append("- **폴백 사용**:")
        for fb in fallbacks_used:
            lines.append(f"  - {fb}")
    else:
        lines.append("- **폴백**: 없음 (모든 컴포넌트 정상 동작)")

    lines += [
        "",
        "## ② 데이터 통계",
        "",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| 총 사건 수 | {len(events)} |",
        f"| Campaign 보유 사건 | {sum(1 for e in events if e.campaign_id)} |",
        f"| 실제 OperationalIOC | {n_real_iocs} |",
        f"| 합성 IOC | {n_synth_iocs} |",
        "",
        "## ③ Ground Truth 통계",
        "",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| 총 캠페인 수 | {stats['n_campaigns_total']} |",
        f"| 평가 가능 캠페인 수 (크기≥2) | {stats['n_campaigns_eligible']} |",
        f"| Positive pair 수 | {stats['n_positive_pairs']} |",
        f"| 쿼리 사건 수 (|Q|) | {stats['n_query_events']} |",
        "",
        "## ④ 조건별 성능",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## ⑤ Ablation",
        "",
        abl_df.to_markdown(index=False),
        "",
        "## ⑥ 강건성",
        "",
        rob_df.to_markdown(index=False),
        "",
        "## ⑦ 상위 3쌍 점수 분해",
        "",
        top3.to_markdown(index=False),
        "",
        "증거 경로 HTML: `results/evidence_paths/`",
        "",
        "## ⑧ Synthetic Manifest 요약",
        "",
        "→ `results/synthetic_manifest.json` 참조",
        "",
        "## ⑨ 한계",
        "",
        "- **소표본**: 사건 수가 수십~백여 건으로, 평가 지표의 95% CI가 넓습니다. "
          "Bootstrap CI를 metrics_summary.csv에 기재하였으며, 결과는 탐색적 참고치로 한정합니다.",
        "- **합성 IOC 의존**: 실제 OperationalIOC 수가 적어 I 성분의 상당 부분이 합성 데이터에 의존합니다. "
          "이는 결과 해석 시 명시적으로 제한 사항으로 고려해야 합니다.",
        "- **ζ 제외 사유**: A항(reported_actor)은 Ground Truth 생성에 사용된 campaign_id와 "
          "상관관계가 있어 평가에 포함 시 순환논리가 발생합니다. E3 평가에서 ζ=0이 강제됩니다.",
    ]

    out = cfg.results_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
