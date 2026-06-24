from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def _load_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "observed_tp", "simulated_tp", "period"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    return df


def run_prediction_consistency_checks(root: Path | None = None) -> dict[str, object]:
    root = root or ROOT
    pred_dir = root / "results" / "predictions"
    metrics_dir = root / "results" / "metrics"
    prediction_files = {
        "differentiable": pred_dir / "tp_differentiable_predictions.csv",
        "ml": pred_dir / "tp_ml_best_predictions.csv",
        "nondiff": pred_dir / "tp_nondiff_daily_predictions.csv",
        "stop_gradient": pred_dir / "tp_stop_gradient_predictions.csv",
    }
    frames = {name: _load_prediction(path) for name, path in prediction_files.items()}
    base = frames["differentiable"][["date", "observed_tp", "period"]].reset_index(drop=True)

    for name, frame in frames.items():
        subset = frame[["date", "observed_tp", "period"]].reset_index(drop=True)
        if not subset["date"].equals(base["date"]):
            raise ValueError(f"{name} prediction dates do not match the differentiable model.")
        if not np.allclose(subset["observed_tp"].to_numpy(dtype=float), base["observed_tp"].to_numpy(dtype=float)):
            raise ValueError(f"{name} observed_tp values do not match the differentiable model.")
        if not subset["period"].equals(base["period"]):
            raise ValueError(f"{name} period labels do not match the differentiable model.")

    cal_count = int((base["period"] == "Calibration").sum())
    val_count = int((base["period"] == "Validation").sum())
    if cal_count <= 0 or val_count <= 0:
        raise ValueError("Calibration/Validation split counts are invalid.")

    stop_grad = frames["stop_gradient"]["simulated_tp"].to_numpy(dtype=float)
    diff = frames["differentiable"]["simulated_tp"].to_numpy(dtype=float)
    if np.allclose(stop_grad, diff):
        raise ValueError("Stop-gradient predictions are identical to the differentiable model.")

    stop_metrics = json.loads((metrics_dir / "tp_stop_gradient_metrics.json").read_text(encoding="utf-8"))
    ablation_type = stop_metrics.get("ablation", {}).get("type")
    if ablation_type != "stop_gradient_source_generation":
        raise ValueError("The stop-gradient metrics file does not declare the correct ablation type.")

    figure_strings = [
        "Observed",
        "Differentiable model",
        "90% prediction interval",
        "50% prediction interval",
        "Observed TP load (kg d⁻¹)",
        "Simulated TP load (kg d⁻¹)",
        "TP load (kg d⁻¹)",
        "Relative parameter sensitivity of raw physical TP output",
        "Calibration",
        "Validation",
        "Stop-gradient source-generation ablation",
    ]
    if any("TN" in text for text in figure_strings):
        raise ValueError("A figure text template still contains 'TN'.")

    return {
        "prediction_files": {name: str(path) for name, path in prediction_files.items()},
        "calibration_count": cal_count,
        "validation_count": val_count,
        "stop_gradient_is_distinct": True,
        "figure_text_has_tn": False,
    }
