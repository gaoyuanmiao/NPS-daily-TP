#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import ticker


ROOT = Path(__file__).resolve().parent
PRED_CSV = ROOT / "tp_direct_predictions_direct_physical_90kg_tp.csv"
COMPARE_PNG = ROOT / "tp_current_obs_vs_sim.png"
LOAD_UNIT = "kg"


class MetricBundle:
    def __init__(self, nse_value: float, r2_value: float, rmse_value: float, bias_value: float):
        self.nse = float(nse_value)
        self.r2 = float(r2_value)
        self.rmse = float(rmse_value)
        self.bias = float(bias_value)


def compute_metrics_np(obs: np.ndarray, sim: np.ndarray) -> MetricBundle:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    den = float(np.sum((obs - np.mean(obs)) ** 2))
    nse_value = float(1.0 - np.sum((sim - obs) ** 2) / den) if den > 1e-12 else float("nan")
    corr = np.corrcoef(obs, sim)[0, 1] if len(obs) > 1 else np.nan
    r2_value = float(corr * corr) if np.isfinite(corr) else float("nan")
    rmse_value = float(np.sqrt(np.mean((sim - obs) ** 2)))
    bias_value = float(np.mean(sim - obs))
    return MetricBundle(
        nse_value=nse_value,
        r2_value=r2_value,
        rmse_value=rmse_value,
        bias_value=bias_value,
    )


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11.5,
            "axes.titlesize": 15,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.facecolor": "#fcfcfb",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#36485c",
            "axes.linewidth": 0.9,
            "grid.color": "#d7dee7",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "axes.titlepad": 12,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )


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
        color="#1f2933",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.92),
    )


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#45556c")
    ax.spines["bottom"].set_color("#45556c")
    ax.tick_params(colors="#334155")


def plot_compare(dates: pd.Series, obs: np.ndarray, sim_raw: np.ndarray) -> None:
    apply_publication_style()
    metrics = compute_metrics_np(obs, sim_raw)
    fig, ax = plt.subplots(figsize=(15.2, 6.2), dpi=260)
    style_axes(ax)
    observed_color = "#243b53"
    simulated_color = "#c44536"
    residual_fill = "#dfe7f1"
    ax.fill_between(dates, obs, sim_raw, color=residual_fill, alpha=0.42, linewidth=0, zorder=1)
    ax.plot(
        dates,
        obs,
        color=observed_color,
        lw=2.8,
        label="Observed",
        zorder=3,
        solid_capstyle="round",
    )
    ax.plot(
        dates,
        sim_raw,
        color=simulated_color,
        lw=2.2,
        label="Simulated (physical core)",
        zorder=4,
        solid_capstyle="round",
    )
    ax.set_title("Daily TP Outlet Load", loc="left")
    ax.set_ylabel(f"Daily TP outlet load ({LOAD_UNIT} d$^{{-1}}$)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.72, axis="both")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.985, 0.995))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.margins(x=0.01)
    ax.text(
        0.018,
        0.945,
        f"NSE  {metrics.nse:.3f}\nR$^2$  {metrics.r2:.3f}\nRMSE  {metrics.rmse:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.5,
        color="#1f2937",
        bbox=dict(
            boxstyle="round,pad=0.42",
            facecolor="white",
            alpha=0.97,
            edgecolor="#d8dee8",
            linewidth=0.9,
        ),
    )
    ax.text(
        0.018,
        1.035,
        "Observed and modeled daily outlet signal",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#64748b",
    )
    add_panel_label(ax, "(a)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(COMPARE_PNG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(PRED_CSV, parse_dates=["date"])
    plot_compare(
        dates=df["date"],
        obs=df["obs_tp"].to_numpy(dtype=float),
        sim_raw=df["sim_tp"].to_numpy(dtype=float),
    )
    print(f"Saved TP comparison figure to: {COMPARE_PNG}")


if __name__ == "__main__":
    main()
