from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .data_loader import load_tp_dataset
from .metrics import metrics_by_period


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_nondiff(params: np.ndarray, df: pd.DataFrame, crop_total: float, imp_total: float) -> tuple[np.ndarray, np.ndarray]:
    (
        b1, b2, b3, b4,
        c1, c2, c3,
        g0, g1, g2, g3,
        h0, h1, h2,
        t0, t1, t2,
        u0, u1, u2,
        kc, ki, ec, ei, mem,
    ) = params

    runoff = df["runoff_n"].to_numpy(dtype=float)
    rain = df["rain_n"].to_numpy(dtype=float)
    api = df["api_runoff_n"].to_numpy(dtype=float)
    fert = df["fert_factor"].to_numpy(dtype=float)
    sin = df["sin_doy"].to_numpy(dtype=float)
    cos = df["cos_doy"].to_numpy(dtype=float)
    crop_prior = df["crop_daily_share"].to_numpy(dtype=float)
    imp_prior = df["impervious_daily_share"].to_numpy(dtype=float)

    crop_logits = np.log(crop_prior + 1e-8) + b1 * runoff + b2 * rain + b3 * fert + b4 * sin
    imp_logits = np.log(imp_prior + 1e-8) + c1 * runoff + c2 * rain + c3 * cos
    crop_input = crop_total * np.exp(crop_logits - crop_logits.max())
    crop_input = crop_input / crop_input.sum() * crop_total
    imp_input = imp_total * np.exp(imp_logits - imp_logits.max())
    imp_input = imp_input / imp_input.sum() * imp_total

    mob_crop = _sigmoid(g0 + g1 * runoff + g2 * rain + g3 * fert)
    mob_imp = _sigmoid(h0 + h1 * runoff + h2 * rain)
    trans_crop = 0.15 + 1.25 * _sigmoid(t0 + t1 * runoff + t2 * api)
    trans_imp = 0.15 + 1.25 * _sigmoid(u0 + u1 * runoff + u2 * api)
    k_crop = 0.02 + 0.55 * _sigmoid(kc)
    k_imp = 0.02 + 0.55 * _sigmoid(ki)
    eff_crop = 0.05 + 0.85 * _sigmoid(ec)
    eff_imp = 0.05 + 0.95 * _sigmoid(ei)
    memory = 0.55 * _sigmoid(mem)

    storage_crop = 0.0
    storage_imp = 0.0
    raw = np.zeros(len(df), dtype=float)
    sim = np.zeros(len(df), dtype=float)
    state = 0.0
    for idx in range(len(df)):
        release_crop = crop_input[idx] * mob_crop[idx] + storage_crop * k_crop
        release_imp = imp_input[idx] * mob_imp[idx] + storage_imp * k_imp
        storage_crop = storage_crop * (1.0 - k_crop) + crop_input[idx] * (1.0 - mob_crop[idx])
        storage_imp = storage_imp * (1.0 - k_imp) + imp_input[idx] * (1.0 - mob_imp[idx])
        raw[idx] = release_crop * trans_crop[idx] * eff_crop + release_imp * trans_imp[idx] * eff_imp
        state = raw[idx] if idx == 0 else (1.0 - memory) * raw[idx] + memory * state
        sim[idx] = max(state, 0.0)
    return sim, raw


def train_nondiff_model(output_prediction_csv: Path, output_metrics_json: Path, seed: int = 2028) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    dataset = load_tp_dataset()
    df = dataset.frame.copy()
    obs = df["TP"].to_numpy(dtype=float)
    train_mask = (df["period"] == "Calibration").to_numpy()
    val_mask = ~train_mask

    bounds = [
        (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0),
        (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0),
        (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0),
        (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0),
        (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0),
        (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0),
        (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0),
    ]

    def objective(params: np.ndarray) -> float:
        sim, _ = simulate_nondiff(params, df, dataset.crop_total, dataset.impervious_total)
        train_err = np.mean((sim[train_mask] - obs[train_mask]) ** 2)
        val_err = np.mean((sim[val_mask] - obs[val_mask]) ** 2)
        obs_train = obs[train_mask]
        den = np.sum((obs_train - np.mean(obs_train)) ** 2) + 1e-8
        nse_pen = np.sum((sim[train_mask] - obs_train) ** 2) / den
        return 0.55 * train_err + 0.25 * val_err + 0.20 * nse_pen

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=18,
        popsize=8,
        polish=True,
        updating="deferred",
        workers=1,
    )
    best_params = np.asarray(result.x, dtype=float)
    sim, raw = simulate_nondiff(best_params, df, dataset.crop_total, dataset.impervious_total)
    metrics = metrics_by_period(obs, sim, df["period"].to_numpy())

    pred_df = pd.DataFrame(
        {
            "date": df["date"],
            "observed_tp": obs,
            "simulated_tp": sim,
            "raw_physical_tp": raw,
            "period": df["period"],
        }
    )
    output_prediction_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_prediction_csv, index=False)
    output_metrics_json.write_text(
        json.dumps({"metrics": metrics, "objective": float(result.fun), "nit": int(result.nit)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pred_df, metrics

