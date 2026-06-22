from __future__ import annotations

import matplotlib.pyplot as plt

from tp_paper_config import (
    FIG_DPI,
    FONT_FAMILY,
    FONT_SIZE,
    LABEL_SIZE,
    LEGEND_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
)


PALETTE = {
    "observed": "#25364a",
    "simulated": "#d1495b",
    "full_model": "#d1495b",
    "simple_model": "#2f6c8f",
    "ml_best": "#4c9f70",
    "cool": "#4c78a8",
    "warm": "#e07a5f",
    "accent": "#8c5e58",
    "grey": "#7b8794",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#34495e",
            "axes.linewidth": 0.9,
            "grid.color": "#d3dce6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "savefig.dpi": FIG_DPI,
        }
    )


def style_axes(ax, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#425466")
    ax.spines["bottom"].set_color("#425466")
    ax.tick_params(colors="#334e68")
    if grid:
        ax.grid(True, alpha=0.65, zorder=0)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#243b53",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
    )
