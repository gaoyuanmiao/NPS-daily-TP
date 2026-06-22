from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from tp_paper_config import FIGSIZE_GRID
from tp_paper_utils import (
    calculate_metrics,
    identify_events,
    observed_label,
    read_prediction_file,
    residual_label,
    rmse_label,
    save_figure,
    save_table,
    simulated_label,
    split_train_val,
    value_label,
)
from tp_plotting_style import PALETTE, add_panel_label, apply_style, style_axes


def build_metrics_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, val_df = split_train_val(df)
    train_val_rows = []
    for period_name, subset in [("calibration", train_df), ("validation", val_df), ("full", df)]:
        train_val_rows.append({"period": period_name, **calculate_metrics(subset["obs"], subset["simulated"])})
    train_val_metrics = pd.DataFrame(train_val_rows)

    seasonal_rows = []
    for season, subset in df.groupby("season"):
        seasonal_rows.append({"season": season, **calculate_metrics(subset["obs"], subset["simulated"]), "n_days": len(subset)})
    seasonal_metrics = pd.DataFrame(seasonal_rows)

    event_rows = []
    for event_name in [
        "all_days",
        "rainfall_days",
        "non_rainfall_days",
        "high_runoff_days",
        "peak_TP_days",
        "low_runoff_high_TP_days",
    ]:
        subset = df[df[event_name]].copy()
        if subset.empty:
            continue
        event_rows.append({"event_type": event_name, **calculate_metrics(subset["obs"], subset["simulated"]), "n_days": len(subset)})
    event_metrics = pd.DataFrame(event_rows)
    return train_val_metrics, seasonal_metrics, event_metrics


def plot_train_val_diagnostics(df: pd.DataFrame) -> None:
    train_df, val_df = split_train_val(df)
    split_date = val_df["date"].iloc[0]
    resid = df["simulated"] - df["obs"]
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRID, dpi=300)
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(df["date"], df["obs"], color=PALETTE["observed"], linewidth=2.0, label="Observed")
    ax1.plot(df["date"], df["simulated"], color=PALETTE["simulated"], linewidth=1.8, label="Simulated TP")
    ax1.axvline(split_date, color=PALETTE["grey"], linestyle="--", linewidth=1.2)
    ax1.set_title("Observed and simulated TP across calibration and validation periods")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(value_label())
    ax1.legend(frameon=False, ncol=2, loc="upper right")
    style_axes(ax1)
    add_panel_label(ax1, "(a)")

    ax2.plot(df["date"], resid, color=PALETTE["accent"], linewidth=1.4)
    ax2.axhline(0.0, color=PALETTE["grey"], linestyle="--", linewidth=1.0)
    ax2.axvline(split_date, color=PALETTE["grey"], linestyle="--", linewidth=1.2)
    ax2.set_title("Residual time series")
    ax2.set_xlabel("Date")
    ax2.set_ylabel(residual_label())
    style_axes(ax2)
    add_panel_label(ax2, "(b)")

    ax3.scatter(train_df["obs"], train_df["simulated"], marker="o", s=24, color=PALETTE["cool"], alpha=0.72, label="Calibration")
    ax3.scatter(val_df["obs"], val_df["simulated"], marker="^", s=32, color=PALETTE["warm"], alpha=0.72, label="Validation")
    xy = np.linspace(df["obs"].min(), df["obs"].max(), 100)
    ax3.plot(xy, xy, color=PALETTE["grey"], linestyle="--", linewidth=1.0)
    ax3.set_title("Observed versus simulated TP")
    ax3.set_xlabel(observed_label())
    ax3.set_ylabel(simulated_label())
    ax3.legend(frameon=False, loc="lower right")
    style_axes(ax3)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(5))
    add_panel_label(ax3, "(c)")

    ax4.hist(resid, bins=24, color=PALETTE["cool"], alpha=0.78, edgecolor="white")
    ax4.axvline(np.mean(resid), color=PALETTE["warm"], linewidth=1.5, label=f"Mean residual = {np.mean(resid):.3f}")
    ax4.set_title("Residual distribution")
    ax4.set_xlabel(residual_label())
    ax4.set_ylabel("Frequency")
    ax4.legend(frameon=False)
    style_axes(ax4)
    add_panel_label(ax4, "(d)")

    fig.tight_layout()
    save_figure(fig, "figure_tp_train_val_timeseries_residuals.png")
    plt.close(fig)


def plot_season_event_validation(df: pd.DataFrame, seasonal_metrics: pd.DataFrame, event_metrics: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRID, dpi=300)
    ax1, ax2, ax3, ax4 = axes.ravel()

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    season_plot = seasonal_metrics.set_index("season").reindex(seasons).reset_index()
    x = np.arange(len(season_plot))
    width = 0.36
    ax1.bar(x - width / 2, season_plot["NSE"], width=width, color=PALETTE["cool"], label="NSE")
    ax1.bar(x + width / 2, season_plot["RMSE"], width=width, color=PALETTE["warm"], label="RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(season_plot["season"])
    ax1.set_title("Seasonal validation metrics")
    ax1.set_ylabel(f"NSE (-) / {rmse_label()}")
    ax1.legend(frameon=False)
    style_axes(ax1)
    add_panel_label(ax1, "(a)")

    event_order = [
        "all_days",
        "rainfall_days",
        "non_rainfall_days",
        "high_runoff_days",
        "peak_TP_days",
        "low_runoff_high_TP_days",
    ]
    event_plot = event_metrics.set_index("event_type").reindex(event_order).reset_index()
    x2 = np.arange(len(event_plot))
    ax2.bar(x2 - width / 2, event_plot["NSE"], width=width, color=PALETTE["cool"], label="NSE")
    ax2.bar(x2 + width / 2, event_plot["RMSE"], width=width, color=PALETTE["warm"], label="RMSE")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(event_plot["event_type"], rotation=28, ha="right")
    ax2.set_title("Event-based validation metrics")
    ax2.set_ylabel(f"NSE (-) / {rmse_label()}")
    ax2.legend(frameon=False)
    style_axes(ax2)
    add_panel_label(ax2, "(b)")

    peak_df = df[df["peak_TP_days"]].copy()
    ax3.scatter(peak_df["obs"], peak_df["simulated"], s=26, color=PALETTE["simulated"], alpha=0.75, edgecolors="white", linewidths=0.35)
    xy = np.linspace(max(peak_df["obs"].min(), 0.0), peak_df["obs"].max(), 100)
    ax3.plot(xy, xy, color=PALETTE["grey"], linestyle="--", linewidth=1.0)
    ax3.set_title("Peak-day observed versus simulated TP")
    ax3.set_xlabel(observed_label())
    ax3.set_ylabel(simulated_label())
    style_axes(ax3)
    add_panel_label(ax3, "(c)")

    err = df["simulated"] - df["obs"]
    ax4.scatter(df["runoff"], err, s=18, color=PALETTE["accent"], alpha=0.55, edgecolors="white", linewidths=0.25)
    ax4.axhline(0.0, color=PALETTE["grey"], linestyle="--", linewidth=1.0)
    ax4.set_title("Residual error versus runoff")
    ax4.set_xlabel("Runoff")
    ax4.set_ylabel(residual_label())
    style_axes(ax4)
    add_panel_label(ax4, "(d)")

    fig.tight_layout()
    save_figure(fig, "figure_tp_season_event_validation.png")
    plt.close(fig)


def main() -> None:
    df = identify_events(read_prediction_file())
    train_val_metrics, seasonal_metrics, event_metrics = build_metrics_tables(df)
    save_table(train_val_metrics, "tp_full_model_train_val_metrics.csv")
    save_table(seasonal_metrics, "tp_full_model_seasonal_metrics.csv")
    save_table(event_metrics, "tp_full_model_event_metrics.csv")
    plot_train_val_diagnostics(df)
    plot_season_event_validation(df, seasonal_metrics, event_metrics)


if __name__ == "__main__":
    main()
