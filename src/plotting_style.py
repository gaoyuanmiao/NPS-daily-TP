from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


COLORS = {
    "Observed": "#8A8F94",
    "Differentiable model": "#1F5A78",
    "Best machine-learning model": "#B9804E",
    "Non-differentiable daily model": "#6F9A82",
    "Stop-gradient source-generation ablation": "#927DB5",
    "Band90": "#BFD6E4",
    "Band50": "#7DA6C2",
    "Grid": "#C6CDD3",
    "Axis": "#333333",
    "Text": "#222222",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.0,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["Axis"])
    ax.spines["bottom"].set_color(COLORS["Axis"])
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(axis="both", colors=COLORS["Axis"], width=0.75, length=3.0, direction="out")
    ax.grid(True, axis=grid_axis, linestyle="-", linewidth=0.45, alpha=0.18, color=COLORS["Grid"])


def format_date_axis(ax) -> None:
    locator = mdates.MonthLocator(interval=2)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = ["%Y", "%b", "%b", "%H:%M", "%H:%M", "%S.%f"]
    formatter.zero_formats = ["", "%Y", "%b", "%b", "%H:%M", "%H:%M"]
    formatter.offset_formats = ["", "", "", "", "", ""]
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def metric_box() -> dict:
    return {"facecolor": "white", "edgecolor": "#D6D6D6", "linewidth": 0.45, "alpha": 0.92, "boxstyle": "round,pad=0.25"}
