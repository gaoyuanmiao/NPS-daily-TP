from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from tp_paper_config import FIGSIZE_GRID, INTER_DIR, RANDOM_SEED
from tp_paper_utils import (
    calculate_metrics,
    make_fert_factor,
    observed_label,
    predicted_label,
    read_prediction_file,
    rmse_label,
    save_figure,
    save_table,
    simulated_label,
    split_train_val,
    value_label,
)
from tp_plotting_style import PALETTE, add_panel_label, apply_style, style_axes


PREDICTION_CSV = INTER_DIR / "tp_ml_baseline_predictions.csv"


def build_api(rain: np.ndarray, lam: float = 0.85) -> np.ndarray:
    out = np.zeros_like(rain, dtype=float)
    state = 0.0
    for i, val in enumerate(rain):
        state = float(val) + lam * state
        out[i] = state
    return out


def add_lag(x: pd.Series, lag: int) -> pd.Series:
    return x.shift(lag).bfill().fillna(0.0)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["api"] = build_api(out["rain"].to_numpy(dtype=float))
    out["fert_factor"] = make_fert_factor(out["date"])
    for lag in [1, 3, 7]:
        out[f"rain_lag{lag}"] = add_lag(out["rain"], lag)
        out[f"runoff_lag{lag}"] = add_lag(out["runoff"], lag)
    out["rain_roll3"] = out["rain"].rolling(3, min_periods=1).mean()
    out["rain_roll7"] = out["rain"].rolling(7, min_periods=1).mean()
    out["runoff_roll3"] = out["runoff"].rolling(3, min_periods=1).mean()
    out["runoff_roll7"] = out["runoff"].rolling(7, min_periods=1).mean()
    doy = pd.to_datetime(out["date"]).dt.dayofyear.to_numpy(dtype=float)
    month = pd.to_datetime(out["date"]).dt.month.to_numpy(dtype=float)
    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.0)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.0)
    out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    return out


def fit_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    feat_df = build_feature_frame(df)
    features = [
        "rain", "runoff", "api", "rain_lag1", "rain_lag3", "rain_lag7", "runoff_lag1", "runoff_lag3", "runoff_lag7",
        "rain_roll3", "rain_roll7", "runoff_roll3", "runoff_roll7", "doy_sin", "doy_cos", "month_sin", "month_cos", "fert_factor",
    ]
    train_df, val_df = split_train_val(feat_df)
    n_train = len(train_df)
    X = feat_df[features].to_numpy(dtype=float)
    y = feat_df["obs"].to_numpy(dtype=float)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    models = {
        "RF": RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=2, random_state=RANDOM_SEED),
        "GBRT": GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=300, learning_rate=0.03, max_depth=2),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    }

    pred_df = df.copy()
    metrics_rows = []
    val_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.maximum(model.predict(X), 0.0)
        pred_df[name] = pred
        for period_name, obs_vec, sim_vec in [("train", y_train, pred[:n_train]), ("validation", y_val, pred[n_train:]), ("full", y, pred)]:
            metrics = calculate_metrics(obs_vec, sim_vec)
            metrics_rows.append({"model": name, "period": period_name, **metrics})
            if period_name == "validation":
                val_scores[name] = metrics["NSE"]
    pred_df.to_csv(PREDICTION_CSV, index=False, encoding="utf-8-sig")
    metrics_df = pd.DataFrame(metrics_rows)
    save_table(metrics_df, "tp_ml_baseline_metrics.csv")
    best_model = max(val_scores, key=val_scores.get)
    return pred_df, metrics_df, best_model


def plot_baseline_comparison(full_df: pd.DataFrame, ml_pred_df: pd.DataFrame, ml_metrics_df: pd.DataFrame, best_model: str) -> None:
    simple_df = pd.read_csv(INTER_DIR / "tp_simple_physical_predictions.csv", parse_dates=["date"])
    train_df, val_df = split_train_val(full_df)
    n_train = len(train_df)
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRID, dpi=300)
    ax1, ax2, ax3, ax4 = axes.ravel()

    val_metrics = ml_metrics_df[ml_metrics_df["period"] == "validation"].copy()
    val_metrics = val_metrics.set_index("model").loc[["RF", "GBRT", "Ridge"]].reset_index()
    x = np.arange(len(val_metrics))
    width = 0.36
    ax1.bar(x - width / 2, val_metrics["NSE"], width=width, color=PALETTE["cool"], label="NSE")
    ax1.bar(x + width / 2, val_metrics["RMSE"], width=width, color=PALETTE["warm"], label=rmse_label())
    ax1.set_xticks(x)
    ax1.set_xticklabels(val_metrics["model"])
    ax1.set_ylabel(f"NSE (-) / {rmse_label()}")
    ax1.set_title("Validation performance of ML baselines")
    ax1.legend(frameon=False)
    style_axes(ax1)
    add_panel_label(ax1, "(a)")

    ax2.scatter(val_df["obs"], ml_pred_df.loc[n_train:, best_model], s=22, color=PALETTE["ml_best"], alpha=0.78, edgecolors="white", linewidths=0.35)
    xy = np.linspace(val_df["obs"].min(), val_df["obs"].max(), 100)
    ax2.plot(xy, xy, color=PALETTE["grey"], linestyle="--", linewidth=1.0)
    ax2.set_xlabel(observed_label())
    ax2.set_ylabel(predicted_label())
    ax2.set_title("Validation scatter of best ML baseline")
    style_axes(ax2)
    add_panel_label(ax2, "(b)")

    ax3.plot(val_df["date"], val_df["obs"], color=PALETTE["observed"], linewidth=2.0, label="Observed")
    ax3.plot(val_df["date"], val_df["simulated"], color=PALETTE["full_model"], linewidth=1.9, label="Full process-informed model")
    ax3.plot(val_df["date"], simple_df.loc[n_train:, "simple_physical_simulated"], color=PALETTE["simple_model"], linewidth=1.7, label="Simple empirical-physical model")
    ax3.plot(val_df["date"], ml_pred_df.loc[n_train:, best_model], color=PALETTE["ml_best"], linewidth=1.7, label="Pure ML baseline")
    ax3.set_title("Validation time-series comparison")
    ax3.set_xlabel("Date")
    ax3.set_ylabel(value_label())
    ax3.legend(frameon=False, ncol=2, loc="upper right")
    style_axes(ax3)
    add_panel_label(ax3, "(c)")

    summary_models = {
        "Full process-informed model": val_df["simulated"].to_numpy(dtype=float),
        "Simple empirical-physical model": simple_df.loc[n_train:, "simple_physical_simulated"].to_numpy(dtype=float),
        "Pure ML baseline": ml_pred_df.loc[n_train:, best_model].to_numpy(dtype=float),
    }
    for label, values in summary_models.items():
        ax4.scatter(val_df["obs"], values, s=18, alpha=0.55, label=label)
    ax4.plot(xy, xy, color=PALETTE["grey"], linestyle="--", linewidth=1.0)
    ax4.set_xlabel(observed_label())
    ax4.set_ylabel(simulated_label())
    ax4.set_title("Validation observed-simulated comparison")
    ax4.legend(frameon=False, loc="lower right")
    style_axes(ax4)
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax4.yaxis.set_major_locator(ticker.MaxNLocator(5))
    add_panel_label(ax4, "(d)")

    fig.tight_layout()
    save_figure(fig, "figure_tp_baseline_model_comparison.png")
    plt.close(fig)


def main() -> None:
    full_df = read_prediction_file()
    pred_df, metrics_df, best_model = fit_models(full_df)
    plot_baseline_comparison(full_df, pred_df, metrics_df, best_model)


if __name__ == "__main__":
    main()
