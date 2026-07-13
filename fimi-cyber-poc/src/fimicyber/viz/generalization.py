"""Paper-ready chart for the frozen multi-campaign benchmark."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_generalization_benchmark(
    condition_summary: pd.DataFrame,
    class_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "content_only": "#2B6CB0",
        "ioc_only": "#C05621",
        "integrated": "#147D64",
    }
    actor_order = class_metrics["actual_actor_id"].tolist()
    actor_names = {
        row["actual_actor_id"]: row["actual_actor"]
        for _, row in class_metrics.iterrows()
    }
    short_names = {
        "doppelganger": "Doppelganger",
        "ghostwriter_unc1151": "Ghostwriter",
        "spamouflage_dragonbridge": "Dragonbridge",
        "storm_1516": "Storm-1516",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ink = "#172033"
    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(colors=ink)

    conditions = condition_summary.set_index("condition").loc[
        ["content_only", "ioc_only", "integrated"]
    ]
    x = np.arange(len(conditions))
    axes[0].bar(
        x,
        conditions["top1_accuracy_all_queries"],
        color=[colors[index] for index in conditions.index],
        width=0.62,
    )
    axes[0].axhline(0.25, color="#555B66", linestyle="--", linewidth=1.1, label="Majority baseline")
    axes[0].set_xticks(x, ["Content", "IOC", "Integrated"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Top-1 accuracy", color=ink)
    axes[0].set_title("A. Analysis condition", loc="left", fontweight="bold", color=ink)
    legend = axes[0].legend(frameon=False, fontsize=8, loc="upper center")
    plt.setp(legend.get_texts(), color=ink)
    for index, value in enumerate(conditions["top1_accuracy_all_queries"]):
        axes[0].text(index, value + 0.025, f"{value:.2f}", ha="center", fontsize=9, color=ink)

    metrics = class_metrics.set_index("actual_actor_id").loc[actor_order]
    x = np.arange(len(metrics))
    width = 0.36
    axes[1].bar(x - width / 2, metrics["top1_accuracy"], width, color="#2B6CB0", label="Top-1")
    axes[1].bar(x + width / 2, metrics["review_coverage"], width, color="#D69E2E", label="Review coverage")
    axes[1].set_xticks(x, [short_names.get(value, value) for value in actor_order], rotation=24, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("B. Per-class behavior", loc="left", fontweight="bold", color=ink)
    legend = axes[1].legend(frameon=False, fontsize=8, loc="upper center")
    plt.setp(legend.get_texts(), color=ink)
    for index, (top1, coverage) in enumerate(zip(metrics["top1_accuracy"], metrics["review_coverage"])):
        axes[1].text(index - width / 2, top1 + 0.025, f"{top1:.1f}", ha="center", fontsize=8, color=ink)
        axes[1].text(index + width / 2, coverage + 0.025, f"{coverage:.1f}", ha="center", fontsize=8, color=ink)

    confusion = pd.crosstab(
        predictions["actual_actor_id"], predictions["predicted_actor_id"]
    ).reindex(index=actor_order, columns=actor_order, fill_value=0)
    image = axes[2].imshow(confusion.to_numpy(), cmap="Blues", vmin=0, vmax=5)
    axes[2].set_xticks(np.arange(len(actor_order)), [short_names.get(value, value) for value in actor_order], rotation=35, ha="right")
    axes[2].set_yticks(np.arange(len(actor_order)), [short_names.get(value, value) for value in actor_order])
    axes[2].set_xlabel("Predicted", color=ink)
    axes[2].set_ylabel("Actual", color=ink)
    axes[2].set_title("C. Closed-set confusion", loc="left", fontweight="bold", color=ink)
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            value = int(confusion.iloc[row, column])
            axes[2].text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value >= 3 else "#172033",
                fontsize=10,
                fontweight="bold" if value else "normal",
            )
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes[:2]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color(ink)
        ax.grid(axis="y", color="#D8DEE8", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Frozen external multi-campaign generalisation pilot (20 holdout events)",
        fontsize=14,
        fontweight="bold",
        color=ink,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "generalization_benchmark.png", bbox_inches="tight", facecolor="white", dpi=300)
    fig.savefig(output_dir / "generalization_benchmark.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
