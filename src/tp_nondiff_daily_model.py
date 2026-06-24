from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .data_loader import load_tp_dataset
from .metrics import metrics_by_period


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_nondiff(params: np.ndarray, df: pd.DataFrame, crop_total: float, imp_total: float) -> tuple[np.ndarray, np.ndarray]:
    (
        b1,
        b2,
        b3,
        b4,
        c1,
        c2,
        c3,
        g0,
        g1,
        g2,
        g3,
        h0,
        h1,
        h2,
        t0,
        t1,
        t2,
        u0,
        u1,
        u2,
        kc,
        ki,
        ec,
        ei,
        mem,
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


def _objective(params: np.ndarray, df: pd.DataFrame, obs: np.ndarray, train_mask: np.ndarray, crop_total: float, imp_total: float) -> float:
    sim, _ = simulate_nondiff(params, df, crop_total, imp_total)
    sim_train = sim[train_mask]
    obs_train = obs[train_mask]
    mse = np.mean((sim_train - obs_train) ** 2)
    den = np.sum((obs_train - np.mean(obs_train)) ** 2) + 1e-8
    nse_penalty = np.sum((sim_train - obs_train) ** 2) / den
    log_mse = np.mean((np.log1p(np.clip(sim_train, 0.0, None)) - np.log1p(np.clip(obs_train, 0.0, None))) ** 2)
    diff_loss = np.mean((np.diff(sim_train) - np.diff(obs_train)) ** 2)
    peak_thr = np.quantile(obs_train, 0.9)
    peak_mask = obs_train >= peak_thr
    peak_loss = float(np.mean((sim_train[peak_mask] - obs_train[peak_mask]) ** 2)) if np.any(peak_mask) else 0.0
    bias_penalty = abs(np.mean(sim_train - obs_train))
    volume_penalty = abs(np.sum(sim_train) - np.sum(obs_train)) / max(abs(np.sum(obs_train)), 1e-8)
    return float(
        0.34 * nse_penalty
        + 0.16 * mse
        + 0.12 * log_mse
        + 0.12 * diff_loss
        + 0.12 * peak_loss
        + 0.07 * bias_penalty
        + 0.07 * volume_penalty
    )


def train_nondiff_model(output_prediction_csv: Path, output_metrics_json: Path, seed: int = 2028) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    dataset = load_tp_dataset()
    df = dataset.frame.copy()
    obs = df["TP"].to_numpy(dtype=float)
    train_mask = (df["period"] == "Calibration").to_numpy()

    bounds = [(-4.0, 4.0)] * 25
    objective = lambda params: _objective(np.asarray(params, dtype=float), df, obs, train_mask, dataset.crop_total, dataset.impervious_total)

    global_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=22,
        popsize=10,
        polish=False,
        updating="deferred",
        workers=1,
    )
    local_result = minimize(
        objective,
        x0=np.asarray(global_result.x, dtype=float),
        method="Powell",
        options={"maxiter": 180, "xtol": 1e-3, "ftol": 1e-3},
    )
    best_params = np.asarray(local_result.x if local_result.success else global_result.x, dtype=float)
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
        json.dumps(
            {
                "metrics": metrics,
                "objective": float(objective(best_params)),
                "differential_evolution": {"fun": float(global_result.fun), "nit": int(global_result.nit)},
                "powell_refine": {"success": bool(local_result.success), "fun": float(local_result.fun)},
                "training": {
                    "calibration_only_objective": True,
                    "validation_used_for_training": False,
                    "validation_used_for_parameter_selection": False,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return pred_df, metrics
