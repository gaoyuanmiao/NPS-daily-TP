from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from tp_paper_config import INTER_DIR, RANDOM_SEED
from tp_paper_utils import calculate_metrics, make_fert_factor, read_prediction_file, save_table, split_train_val


PREDICTION_CSV = INTER_DIR / "tp_simple_physical_predictions.csv"


def minmax_scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if xmax - xmin <= 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def build_api(rain: np.ndarray, lam: float) -> np.ndarray:
    out = np.zeros_like(rain, dtype=float)
    state = 0.0
    for i, val in enumerate(rain):
        state = float(val) + lam * state
        out[i] = state
    return out


def build_simple_features(df: pd.DataFrame, lambda_api: float, k_release: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rain = df["rain"].to_numpy(dtype=float)
    runoff = df["runoff"].to_numpy(dtype=float)
    fert = make_fert_factor(df["date"])
    api = build_api(rain, lambda_api)
    rain_n = minmax_scale(rain)
    runoff_n = minmax_scale(runoff)
    api_n = minmax_scale(api)
    fert_n = fert.astype(float)
    doy = pd.to_datetime(df["date"]).dt.dayofyear.to_numpy(dtype=float)
    sin1 = np.sin(2.0 * np.pi * doy / 365.0)
    cos1 = np.cos(2.0 * np.pi * doy / 365.0)

    source_index = 0.10 + 0.34 * runoff_n + 0.22 * rain_n + 0.18 * api_n + 0.16 * fert_n + 0.08 * sin1 + 0.05 * cos1
    runoff_response = 0.20 + 0.80 * runoff_n
    storage = np.zeros(len(df), dtype=float)
    release = np.zeros(len(df), dtype=float)
    state = 0.0
    for i in range(len(df)):
        release[i] = k_release * state + source_index[i] * runoff_response[i]
        state = (1.0 - k_release) * state + source_index[i]
        storage[i] = state
    features = np.column_stack([release, storage, rain_n, runoff_n, api_n, fert_n, sin1, cos1])
    return features, source_index, release


def fit_simple_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = split_train_val(df)
    best = None
    results = []
    for lambda_api in [0.30, 0.50, 0.70, 0.85, 0.95]:
        for k_release in [0.03, 0.05, 0.10, 0.20, 0.35, 0.50]:
            features, source_index, release = build_simple_features(df, lambda_api, k_release)
            n_train = len(train_df)
            model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
            model.fit(features[:n_train], train_df["obs"].to_numpy(dtype=float))
            pred = np.maximum(model.predict(features), 0.0)
            train_metrics = calculate_metrics(train_df["obs"], pred[:n_train])
            val_metrics = calculate_metrics(val_df["obs"], pred[n_train:])
            results.append(
                {
                    "lambda_api": lambda_api,
                    "k_release": k_release,
                    "train_NSE": train_metrics["NSE"],
                    "train_R2": train_metrics["R2"],
                    "train_RMSE": train_metrics["RMSE"],
                    "val_NSE": val_metrics["NSE"],
                    "val_R2": val_metrics["R2"],
                    "val_RMSE": val_metrics["RMSE"],
                }
            )
            score = (val_metrics["NSE"], -val_metrics["RMSE"])
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "pred": pred,
                    "source_index": source_index,
                    "release": release,
                    "lambda_api": lambda_api,
                    "k_release": k_release,
                }

    pred_df = df.copy()
    pred_df["simple_physical_simulated"] = best["pred"]
    pred_df["source_index"] = best["source_index"]
    pred_df["release_index"] = best["release"]
    pred_df["period"] = ["train"] * len(train_df) + ["validation"] * len(val_df)
    pred_df.to_csv(PREDICTION_CSV, index=False, encoding="utf-8-sig")

    metrics_rows = []
    for period_name, subset, pred_subset in [
        ("train", train_df, best["pred"][: len(train_df)]),
        ("validation", val_df, best["pred"][len(train_df) :]),
        ("full", df, best["pred"]),
    ]:
        metrics_rows.append(
            {
                "model": "Simple empirical-physical model",
                "period": period_name,
                **calculate_metrics(subset["obs"], pred_subset),
                "lambda_api": best["lambda_api"],
                "k_release": best["k_release"],
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    save_table(metrics_df, "tp_simple_physical_metrics.csv")
    save_table(pd.DataFrame(results).sort_values(["val_NSE", "val_RMSE"], ascending=[False, True]), "tp_simple_physical_grid_search.csv")
    return pred_df, metrics_df


def main() -> None:
    df = read_prediction_file()
    fit_simple_baseline(df)


if __name__ == "__main__":
    main()
