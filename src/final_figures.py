from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting_style import COLORS, format_date_axis, metric_box, setup_style, style_axis


ROOT = Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, path_png: Path, path_pdf: Path) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_final_figures() -> list[Path]:
    setup_style()
    fig_dir = ROOT / "figures"
    pred_dir = ROOT / "results" / "predictions"
    metric_dir = ROOT / "results" / "metrics"
    ensemble_path = ROOT / "results" / "ensemble" / "tp_ensemble_predictions.csv"
    sens_path = ROOT / "results" / "sensitivity" / "tp_parameter_sensitivity.csv"

    diff = pd.read_csv(pred_dir / "tp_differentiable_predictions.csv", parse_dates=["date"])
    ml = pd.read_csv(pred_dir / "tp_ml_best_predictions.csv", parse_dates=["date"])
    nondiff = pd.read_csv(pred_dir / "tp_nondiff_daily_predictions.csv", parse_dates=["date"])
    stopg = pd.read_csv(pred_dir / "tp_stop_gradient_predictions.csv", parse_dates=["date"])
    ensemble = pd.read_csv(ensemble_path, parse_dates=["date"])
    sens = pd.read_csv(sens_path)

    diff_metrics = _read_json(metric_dir / "tp_differentiable_metrics.json")["metrics"]
    ml_metrics = _read_json(metric_dir / "tp_ml_best_metrics.json")["metrics"]
    nondiff_metrics = _read_json(metric_dir / "tp_nondiff_daily_metrics.json")["metrics"]
    stopg_metrics = _read_json(metric_dir / "tp_stop_gradient_metrics.json")["metrics"]
    ens_metrics = _read_json(metric_dir / "tp_ensemble_median_metrics.json")["metrics"]

    split_date = ensemble.loc[ensemble["period"] == "Validation", "date"].min()
    train_mid = ensemble.loc[ensemble["period"] == "Calibration", "date"].iloc[len(ensemble.loc[ensemble["period"] == "Calibration"]) // 2]
    val_mid = ensemble.loc[ensemble["period"] == "Validation", "date"].iloc[len(ensemble.loc[ensemble["period"] == "Validation"]) // 2]

    created: list[Path] = []

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.fill_between(ensemble["date"], ensemble["q05_tp"], ensemble["q95_tp"], color=COLORS["Band90"], alpha=0.65, linewidth=0.0, label="90% prediction interval")
    ax.fill_between(ensemble["date"], ensemble["q25_tp"], ensemble["q75_tp"], color=COLORS["Band50"], alpha=0.80, linewidth=0.0, label="50% prediction interval")
    ax.plot(ensemble["date"], ensemble["observed_tp"], color=COLORS["Observed"], linewidth=1.2, label="Observed")
    ax.plot(ensemble["date"], ensemble["median_tp"], color=COLORS["Differentiable model"], linewidth=1.6, label="Differentiable median")
    ax.axvline(split_date, color="#9A9A9A", lw=1.0, ls="--")
    ax.text(train_mid, ax.get_ylim()[1] * 0.95, "Calibration", ha="center", va="top", fontsize=8.2, color="#666666")
    ax.text(val_mid, ax.get_ylim()[1] * 0.95, "Validation", ha="center", va="top", fontsize=8.2, color="#666666")
    ax.text(
        0.985,
        0.90,
        f"Calibration: NSE = {ens_metrics['calibration']['nse']:.2f}, R² = {ens_metrics['calibration']['r2']:.2f}\n"
        f"Validation: NSE = {ens_metrics['validation']['nse']:.2f}, R² = {ens_metrics['validation']['r2']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color=COLORS["Text"],
        bbox=metric_box(),
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("TP load (kg d⁻¹)")
    style_axis(ax, "y")
    format_date_axis(ax)
    ax.legend(loc="upper left", frameon=False)
    p1 = fig_dir / "figure_tp_timeseries_interval.png"
    _save(fig, p1, fig_dir / "figure_tp_timeseries_interval.pdf")
    created.append(p1)

    model_frames = [
        ("Differentiable model", diff, diff_metrics),
        ("Best machine-learning model", ml, ml_metrics),
        ("Non-differentiable daily model", nondiff, nondiff_metrics),
        ("Stop-gradient source-generation ablation", stopg, stopg_metrics),
    ]
    max_xy = max(
        float(
            max(
                frame["observed_tp"].max(),
                frame["simulated_tp"].max(),
            )
        )
        for _, frame, _ in model_frames
    ) * 1.05
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.4), sharex=True, sharey=True)
    for ax, (title, frame, metrics), label in zip(axes.flatten(), model_frames, ["(a)", "(b)", "(c)", "(d)"]):
        cal = frame["period"] == "Calibration"
        val = frame["period"] == "Validation"
        ax.scatter(frame.loc[cal, "observed_tp"], frame.loc[cal, "simulated_tp"], s=18, marker="o", color=COLORS[title], alpha=0.75)
        ax.scatter(frame.loc[val, "observed_tp"], frame.loc[val, "simulated_tp"], s=22, marker="^", color=COLORS[title], alpha=0.88)
        ax.plot([0, max_xy], [0, max_xy], ls="--", lw=0.9, color="#999999")
        ax.set_xlim(0, max_xy)
        ax.set_ylim(0, max_xy)
        style_axis(ax, "both")
        ax.set_title(f"{label} {title}", loc="left")
        ax.text(
            0.03,
            0.97,
            f"Cal NSE = {metrics['calibration']['nse']:.2f}\n"
            f"Val NSE = {metrics['validation']['nse']:.2f}\n"
            f"Cal R² = {metrics['calibration']['r2']:.2f}\n"
            f"Val R² = {metrics['validation']['r2']:.2f}\n"
            f"RMSE = {metrics['all']['rmse']:.3f}\n"
            f"PBIAS = {metrics['all']['pbias']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            bbox=metric_box(),
        )
    axes[1, 0].set_xlabel("Observed TP load (kg d⁻¹)")
    axes[1, 1].set_xlabel("Observed TP load (kg d⁻¹)")
    axes[0, 0].set_ylabel("Simulated TP load (kg d⁻¹)")
    axes[1, 0].set_ylabel("Simulated TP load (kg d⁻¹)")
    p2 = fig_dir / "figure_tp_scatter_model_comparison.png"
    _save(fig, p2, fig_dir / "figure_tp_scatter_model_comparison.pdf")
    created.append(p2)

    top10 = sens.head(10).sort_values("relative_sensitivity", ascending=True)
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    ax.barh(top10["parameter"], top10["relative_sensitivity"], color=COLORS["Differentiable model"], alpha=0.88)
    for y, x in zip(top10["parameter"], top10["relative_sensitivity"]):
        ax.text(float(x) + 1.0, y, f"{x:.1f}", va="center", ha="left", fontsize=7.8)
    ax.set_xlabel("Relative sensitivity (%)")
    ax.set_ylabel("Parameter")
    ax.set_title("Relative parameter sensitivity of raw physical TP output")
    style_axis(ax, "x")
    p3 = fig_dir / "figure_tp_parameter_sensitivity.png"
    _save(fig, p3, fig_dir / "figure_tp_parameter_sensitivity.pdf")
    created.append(p3)

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.plot(diff["date"], diff["observed_tp"], color=COLORS["Observed"], linewidth=1.2, label="Observed")
    ax.plot(diff["date"], diff["simulated_tp"], color=COLORS["Differentiable model"], linewidth=1.6, label="Differentiable model")
    ax.axvline(split_date, color="#9A9A9A", lw=1.0, ls="--")
    ax.text(train_mid, ax.get_ylim()[1] * 0.95, "Calibration", ha="center", va="top", fontsize=8.2, color="#666666")
    ax.text(val_mid, ax.get_ylim()[1] * 0.95, "Validation", ha="center", va="top", fontsize=8.2, color="#666666")
    ax.text(
        0.985,
        0.90,
        f"Calibration: NSE = {diff_metrics['calibration']['nse']:.2f}, R² = {diff_metrics['calibration']['r2']:.2f}\n"
        f"Validation: NSE = {diff_metrics['validation']['nse']:.2f}, R² = {diff_metrics['validation']['r2']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color=COLORS["Text"],
        bbox=metric_box(),
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("TP load (kg d⁻¹)")
    style_axis(ax, "y")
    format_date_axis(ax)
    ax.legend(loc="upper left", frameon=False)
    p4 = fig_dir / "figure_tp_timeseries_final_model.png"
    _save(fig, p4, fig_dir / "figure_tp_timeseries_final_model.pdf")
    created.append(p4)
    return created
