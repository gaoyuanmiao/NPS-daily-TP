from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import feature_columns, load_tp_dataset
from .metrics import compute_metrics, metrics_by_period


def _optional_model(name: str):
    lowered = name.lower()
    if lowered == "xgboost":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=240,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=2026,
            )
        except Exception:
            return None
    if lowered == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=280,
                learning_rate=0.04,
                max_depth=-1,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=2026,
                verbose=-1,
            )
        except Exception:
            return None
    return None


def candidate_models() -> dict[str, object]:
    models: dict[str, object] = {
        "Random Forest": RandomForestRegressor(n_estimators=400, max_depth=7, random_state=2026),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, max_depth=8, random_state=2026),
        "GradientBoosting": GradientBoostingRegressor(random_state=2026, n_estimators=260, learning_rate=0.04, max_depth=2),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(hidden_layer_sizes=(48, 24), alpha=1e-3, learning_rate_init=2e-3, max_iter=5000, random_state=2026)),
            ]
        ),
    }
    for name in ("XGBoost", "LightGBM"):
        model = _optional_model(name)
        if model is not None:
            models[name] = model
    return models


def train_ml_baselines(
    output_prediction_csv: Path,
    output_all_metrics_csv: Path,
    output_best_metrics_json: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], str]:
    dataset = load_tp_dataset()
    df = dataset.frame.copy()
    features = feature_columns()
    X = df[features].to_numpy(dtype=float)
    y = df["TP"].to_numpy(dtype=float)
    train_mask = (df["period"] == "Calibration").to_numpy()
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_all = X
    metric_rows: list[dict[str, object]] = []
    best_name = ""
    best_score = -np.inf
    best_pred: np.ndarray | None = None
    best_metrics: dict[str, dict[str, float]] | None = None

    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_all), dtype=float)
        pred = np.clip(pred, 0.0, None)
        metrics = metrics_by_period(y, pred, df["period"].to_numpy())
        metric_rows.append(
            {
                "model_name": name,
                "calibration_nse": metrics["calibration"]["nse"],
                "calibration_r2": metrics["calibration"]["r2"],
                "calibration_rmse": metrics["calibration"]["rmse"],
                "calibration_pbias": metrics["calibration"]["pbias"],
                "validation_nse": metrics["validation"]["nse"],
                "validation_r2": metrics["validation"]["r2"],
                "validation_rmse": metrics["validation"]["rmse"],
                "validation_pbias": metrics["validation"]["pbias"],
                "all_nse": metrics["all"]["nse"],
                "all_r2": metrics["all"]["r2"],
                "all_rmse": metrics["all"]["rmse"],
                "all_pbias": metrics["all"]["pbias"],
            }
        )
        score = (
            3.0 * np.nan_to_num(metrics["validation"]["nse"], nan=-10.0)
            + 1.5 * np.nan_to_num(metrics["validation"]["r2"], nan=-10.0)
            + 0.5 * np.nan_to_num(metrics["calibration"]["nse"], nan=-10.0)
            - 0.2 * metrics["validation"]["rmse"]
        )
        if score > best_score:
            best_score = score
            best_name = name
            best_pred = pred
            best_metrics = metrics

    if best_pred is None or best_metrics is None:
        raise RuntimeError("No machine-learning baseline produced predictions.")

    pred_df = pd.DataFrame(
        {
            "date": df["date"],
            "observed_tp": y,
            "simulated_tp": best_pred,
            "period": df["period"],
            "model_name": best_name,
        }
    )
    output_prediction_csv.parent.mkdir(parents=True, exist_ok=True)
    output_all_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    output_best_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_prediction_csv, index=False)
    pd.DataFrame(metric_rows).sort_values(["validation_nse", "validation_r2"], ascending=[False, False]).to_csv(output_all_metrics_csv, index=False)
    output_best_metrics_json.write_text(
        json.dumps({"best_model": best_name, "metrics": best_metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pred_df, best_metrics, best_name

