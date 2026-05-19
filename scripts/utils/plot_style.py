from __future__ import annotations

from pathlib import Path
from typing import Iterable


PALETTE = {
    "primary": "#147D64",
    "primary_dark": "#0B4F45",
    "secondary": "#2F80ED",
    "accent": "#F2A541",
    "danger": "#D95D39",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "text": "#1F2937",
}

SERIES_COLORS = ["#147D64", "#2F80ED", "#F2A541", "#D95D39", "#7C3AED", "#0891B2"]


def apply_plot_style() -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette=SERIES_COLORS,
        rc={
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D1D5DB",
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["primary_dark"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        },
    )
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300


def polish_axis(
    ax: object,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = False,
) -> None:
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=12)  # type: ignore[attr-defined]
    if xlabel is not None:
        ax.set_xlabel(xlabel)  # type: ignore[attr-defined]
    if ylabel is not None:
        ax.set_ylabel(ylabel)  # type: ignore[attr-defined]
    ax.grid(True, axis="y", alpha=0.9)  # type: ignore[attr-defined]
    ax.grid(True, axis="x", alpha=0.25)  # type: ignore[attr-defined]
    if legend:
        ax.legend(frameon=False, loc="best")  # type: ignore[attr-defined]


def save_figure(fig: object, base_path: Path, make_png: bool = True, make_svg: bool = True) -> list[Path]:
    saved: list[Path] = []
    if make_png:
        png = base_path.with_suffix(".png")
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")  # type: ignore[attr-defined]
        saved.append(png)
    if make_svg:
        svg = base_path.with_suffix(".svg")
        fig.savefig(svg, bbox_inches="tight", facecolor="white")  # type: ignore[attr-defined]
        saved.append(svg)
    return saved


def save_figure_no_return(fig: object, base_path: Path, make_png: bool = True, make_svg: bool = True) -> None:
    save_figure(fig, base_path, make_png, make_svg)


def soften_spines(axes: Iterable[object]) -> None:
    for ax in axes:
        for spine in ax.spines.values():  # type: ignore[attr-defined]
            spine.set_color("#D1D5DB")
