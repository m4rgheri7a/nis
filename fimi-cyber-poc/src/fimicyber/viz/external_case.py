"""Static evidence-path figure for the Ghostwriter external case."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


def _box(ax, x: float, y: float, width: float, height: float, text: str, edge: str, fill: str) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.5, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=9, color="#172033")


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str, label: str = "") -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=color)
    ax.add_patch(arrow)
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.035, label,
                ha="center", va="bottom", fontsize=7.5, color=color)


def render_external_ghostwriter_paths(hypotheses: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    top = hypotheses[hypotheses["rank"] == 1].set_index("query_event_id")

    fig, ax = plt.subplots(figsize=(13.2, 6.8), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(0.5, 0.96, "Ghostwriter external-case evidence paths", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#172033")
    ax.text(0.5, 0.915, "Independent NCSC reference incidents and Mandiant 2021 holdout cases",
            ha="center", va="center", fontsize=9.5, color="#58657a")

    ax.text(0.16, 0.855, "Historical references", ha="center", fontsize=10, fontweight="bold", color="#2457a6")
    ax.text(0.48, 0.855, "Observed links", ha="center", fontsize=10, fontweight="bold", color="#a84b13")
    ax.text(0.76, 0.855, "Held-out queries", ha="center", fontsize=10, fontweight="bold", color="#126b63")
    ax.text(0.93, 0.855, "System action", ha="center", fontsize=10, fontweight="bold", color="#5b3f91")

    _box(ax, 0.16, 0.67, 0.24, 0.12, "2018 Karoblis allegation\nNCSC incident 152827", "#2b67b2", "#eef4ff")
    _box(ax, 0.16, 0.31, 0.24, 0.12, "2019 corruption allegation\nNCSC incident 163811", "#2b67b2", "#eef4ff")

    _box(ax, 0.48, 0.69, 0.18, 0.09, "88.99.132[.]118\nshared C2 IP", "#c94b4b", "#fff1f1")
    _box(ax, 0.48, 0.50, 0.20, 0.09, "Malicious document /\nPowerShell TTP family", "#b36b18", "#fff7e8")
    _box(ax, 0.48, 0.29, 0.18, 0.09, "94.103.82[.]136\nshared sender IP", "#c94b4b", "#fff1f1")

    query_specs = [
        ("gw-test-2021-polskie-radio", 0.70, "Polskie Radio\nRADIOSTAR lure"),
        ("gw-test-2021-gift", 0.50, "Gift lure\nVIDEOKILLER"),
        ("gw-test-2021-socis", 0.28, "SOCIS\nweaponised release"),
    ]
    for query_id, y, label in query_specs:
        _box(ax, 0.76, y, 0.19, 0.10, label, "#138079", "#eefaf8")
        if query_id in top.index:
            row = top.loc[query_id]
            decision = "REVIEW" if row["decision"] == "analyst_review" else "ABSTAIN"
            confidence = float(row["assessment_confidence"])
            color = "#39734f" if decision == "REVIEW" else "#9b4b40"
            fill = "#eef8f1" if decision == "REVIEW" else "#fff2ef"
            _box(ax, 0.93, y, 0.11, 0.085, f"{decision}\n{confidence:.2f}", color, fill)
            _arrow(ax, (0.855, y), (0.872, y), "#5b3f91")

    _arrow(ax, (0.28, 0.67), (0.39, 0.69), "#c94b4b", "IOC")
    _arrow(ax, (0.57, 0.69), (0.665, 0.70), "#c94b4b", "IOC")
    _arrow(ax, (0.28, 0.31), (0.39, 0.29), "#c94b4b", "IOC")
    _arrow(ax, (0.57, 0.29), (0.665, 0.28), "#c94b4b", "IOC")
    _arrow(ax, (0.28, 0.64), (0.38, 0.53), "#b36b18", "TTP")
    _arrow(ax, (0.28, 0.34), (0.38, 0.47), "#b36b18", "TTP")
    _arrow(ax, (0.58, 0.50), (0.665, 0.50), "#b36b18", "TTP")

    ax.text(0.5, 0.075,
            "Solid red paths are normalized public IOC matches. The amber path is behavioral support only.\n"
            "These paths prioritize analyst review; they do not establish legal or state attribution.",
            ha="center", va="center", fontsize=8.5, color="#58657a")

    fig.tight_layout(pad=0.7)
    fig.savefig(output_dir / "external_ghostwriter_evidence.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "external_ghostwriter_evidence.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
