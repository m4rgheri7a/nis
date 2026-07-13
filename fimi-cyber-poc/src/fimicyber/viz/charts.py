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
    attribution_df: pd.DataFrame | None = None,
    attribution_metrics_df: pd.DataFrame | None = None,
) -> None:
    from fimicyber.eval.groundtruth import build_ground_truth, gt_stats
    from collections import Counter

    gt = build_ground_truth(events)
    stats = gt_stats(gt, events)
    source_counts = Counter(ev.source_dataset for ev in events)

    n_real_iocs = sum(1 for ev in events for ioc in ev.iocs if not ioc.synthetic and ioc.category == "OperationalIOC")
    n_synth_iocs = sum(1 for ev in events for ioc in ev.iocs if ioc.synthetic)

    curated_events = [ev for ev in events if str(ev.source_dataset).startswith("curated")]
    curated_ids = {ev.event_id for ev in curated_events}
    curated_real_iocs = sum(
        1
        for ev in curated_events
        for ioc in ev.iocs
        if not ioc.synthetic and ioc.category == "OperationalIOC"
    )
    relation_path = cfg.data_dir / "curated" / "ioc_relations.csv"
    curated_relations = 0
    if relation_path.exists():
        curated_relations = max(0, len(relation_path.read_text(encoding="utf-8").splitlines()) - 1)


    # Top 3 pairs
    top_cols = [
        "event_i", "event_j", "N", "N_conf", "I", "I_no_synthetic", "D", "C", "T",
        "evidence_components", "evidence_coverage", "FCLS_E3_raw", "FCLS_E3",
    ]
    top_cols = [c for c in top_cols if c in scores_df.columns]
    top3 = scores_df.nlargest(3, "FCLS_E3")[top_cols]
    comp_cov = {
        col: float(scores_df[col].notna().mean())
        for col in ["N", "I", "I_no_synthetic", "D", "C", "T"]
        if col in scores_df.columns
    }

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
        f"| Actor-surrogate campaign 사건 | {stats.get('n_events_actor_surrogate', 0)} |",
        f"| EUvsDisinfo debunk-group 사건 | {stats.get('n_events_debunk_group', 0)} |",
        f"| Explicit campaign 사건 | {stats.get('n_events_explicit_campaign', 0)} |",
        f"| 실제 OperationalIOC | {n_real_iocs} |",
        f"| 합성 IOC | {n_synth_iocs} |",
        "",
        "### 데이터 출처 구성",
        "",
        f"| source_dataset | events |",
        f"|---|---:|",
        *[f"| {k} | {v} |" for k, v in sorted(source_counts.items())],
        "",
        "## ③ Ground Truth 통계",
        "",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| 총 캠페인 수 | {stats['n_campaigns_total']} |",
        f"| 평가 가능 캠페인 수 (크기≥2) | {stats['n_campaigns_eligible']} |",
        f"| Positive pair 수 | {stats['n_positive_pairs']} |",
        f"| 쿼리 사건 수 (|Q|) | {stats['n_query_events']} |",
        f"| Actor-surrogate 포함 | {stats['include_actor_surrogate']} |",
        "",
        "## ③-1 성분 커버리지",
        "",
        f"| 성분 | non-null 비율 |",
        f"|---|---:|",
        *[f"| {k} | {v:.6f} |" for k, v in comp_cov.items()],
        "",
    ]

    if curated_events:
        curated_pair_scores = scores_df[
            scores_df["event_i"].isin(curated_ids) & scores_df["event_j"].isin(curated_ids)
        ].copy()
        curated_i_nonnull = (
            int(curated_pair_scores["I_no_synthetic"].notna().sum())
            if "I_no_synthetic" in curated_pair_scores.columns
            else 0
        )
        curated_i_positive = (
            int((curated_pair_scores["I_no_synthetic"].fillna(0) > 0).sum())
            if "I_no_synthetic" in curated_pair_scores.columns
            else 0
        )
        top_curated_cols = [
            c for c in [
                "event_i", "event_j", "I_no_synthetic",
                "FCLS_E3", "FCLS_E3_no_synthetic_ioc"
            ]
            if c in curated_pair_scores.columns
        ]
        curated_top = curated_pair_scores.nlargest(5, "FCLS_E3_no_synthetic_ioc")[top_curated_cols]
        lines += [
            "## ③-2 Curated 실제 IOC 케이스",
            "",
            "| 항목 | 값 |",
            "|---|---:|",
            f"| Curated events | {len(curated_events)} |",
            f"| Curated real OperationalIOC | {curated_real_iocs} |",
            f"| Curated IOC relations | {curated_relations} |",
            f"| Curated event pairs | {len(curated_pair_scores)} |",
            f"| I_no_synthetic non-null pairs | {curated_i_nonnull} |",
            f"| I_no_synthetic positive pairs | {curated_i_positive} |",
            "",
            "이 섹션은 전체 성능 검증이 아니라 공개 보고서 기반 실제 IOC가 증거 그래프와 `I_no_synthetic` 경로에 연결되는지 확인하기 위한 케이스 진단입니다.",
            "",
            curated_top.to_markdown(index=False),
            "",
        ]

    lines += [
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
    ]

    if attribution_df is not None and attribution_metrics_df is not None:
        top_attribution = (
            attribution_df[attribution_df["rank"] == 1]
            .nlargest(10, "assessment_confidence")
        )
        attr_cols = [
            "query_event_id", "candidate_actor", "support_score",
            "candidate_probability", "assessment_confidence", "confidence_band",
            "evidence_families", "real_ioc_support", "source_org_count",
        ]
        attr_cols = [column for column in attr_cols if column in top_attribution.columns]
        lines += [
            "## ⑦-1 공개자료 기반 행위자 귀속지원",
            "",
            "귀속지원 평가는 query보다 먼저 관측된 사건만 reference profile로 사용하고, "
            "`reported_actor`를 입력 특징에서 제외하며, 인프라 성분은 실제 IOC "
            "`I_no_synthetic`만 사용합니다.",
            "",
            attribution_metrics_df.to_markdown(index=False),
            "",
            top_attribution[attr_cols].to_markdown(index=False),
            "",
            "전체 후보와 경쟁 가설: `results/attribution_hypotheses.csv`",
            "",
            "귀속지원 그래프: `results/attribution_graph.json`",
            "",
        ]

    lines += [
        "## ⑧ Synthetic Manifest 요약",
        "",
        "→ `results/synthetic_manifest.json` 참조",
        "",
        "## ⑨ 한계",
        "",
        "- **소표본**: 사건 수가 수십~백여 건으로, 평가 지표의 95% CI가 넓습니다. "
          "Bootstrap CI를 metrics_summary.csv에 기재하였으며, 결과는 탐색적 참고치로 한정합니다.",
        "- **합성 IOC 의존**: 실제 OperationalIOC 수가 적어 I 성분의 상당 부분이 합성 데이터에 의존합니다. "
          "통합 실행에서는 EUvsDisinfo metadata row와 curated real-IOC event에는 synthetic IOC를 주입하지 않고, 기존 DISINFOX/fixture "
          "branch에만 주입합니다.",
        "- **E3 결측 보정**: `FCLS_E3_raw`는 원래 결측 재정규화 점수이며, `FCLS_E3`는 "
          "N-only collapse를 줄이기 위해 narrative confidence, evidence coverage, 최소 증거축 "
          "조건을 적용한 metadata-level integrated linkage 점수입니다.",
        "- **No-synthetic IOC 조건**: `E2_no_synthetic_ioc`와 `E3_no_synthetic_ioc`는 "
          "합성 IOC를 제거한 상태의 보조 진단 지표입니다. 실제 OperationalIOC가 0건이면 "
          "실제 IOC 기반 성능으로 해석하지 않으며, 현재 실행에서는 신호가 curated Doppelganger "
          "소규모 케이스에 집중되어 있으므로 전체 benchmark 성능으로 과대 해석하지 않습니다.",
        "- **라벨 누수 방지**: EUvsDisinfo의 `debunk_id`는 `campaign_id` 평가 라벨로만 사용하고, "
          "description 임베딩 입력에는 포함하지 않습니다.",
        "- **Ground Truth 주의**: 통합 평가의 `campaign_id`는 DISINFOX에서는 actor-surrogate, "
          "EUvsDisinfo에서는 debunk-group을 포함하는 unified link group입니다. 두 데이터셋의 "
          "ground truth granularity가 다르므로 최종 성능 입증보다 구조 검증으로 해석합니다.",
        "- **ζ 제외 사유**: A항(reported_actor)은 Ground Truth 생성에 사용된 campaign_id와 "
          "상관관계가 있어 평가에 포함 시 순환논리가 발생합니다. E3 평가에서 ζ=0이 강제됩니다.",
        "- **귀속지원의 법적 한계**: 행위자 후보는 공개 보고서의 과거 라벨과 비합성 증거를 "
          "이용한 분석 가설입니다. 가입자 정보, 서버 로그, 압수물 또는 적법한 디지털 포렌식 "
          "증거를 대체하지 않으며 범죄나 신원을 확정하지 않습니다.",
    ]

    out = cfg.results_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
