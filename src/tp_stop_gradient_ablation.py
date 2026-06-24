from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data_loader import load_tp_dataset
from .metrics import metrics_by_period
from .tp_differentiable_model import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_SOURCE_GENERATION_PATH,
    DirectPhysicalTPModel,
    load_differentiable_checkpoint,
)


def _calibration_loss(obs: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    den = torch.sum((obs - torch.mean(obs)) ** 2).clamp_min(1e-8)
    nse_penalty = torch.sum((sim - obs) ** 2) / den
    mse = torch.mean((sim - obs) ** 2)
    log_mse = torch.mean((torch.log1p(torch.clamp_min(sim, 0.0)) - torch.log1p(torch.clamp_min(obs, 0.0))) ** 2)
    diff_loss = torch.mean((torch.diff(sim) - torch.diff(obs)) ** 2)
    peak_thr = torch.quantile(obs.detach(), 0.9)
    peak_mask = obs >= peak_thr
    peak_loss = torch.mean((sim[peak_mask] - obs[peak_mask]) ** 2) if torch.any(peak_mask) else torch.tensor(0.0, device=obs.device)
    bias_penalty = torch.abs(torch.mean(sim - obs))
    volume_penalty = torch.abs(torch.sum(sim) - torch.sum(obs)) / torch.clamp_min(torch.abs(torch.sum(obs)), 1e-8)
    return (
        0.34 * nse_penalty
        + 0.16 * mse
        + 0.12 * log_mse
        + 0.12 * diff_loss
        + 0.12 * peak_loss
        + 0.07 * bias_penalty
        + 0.07 * volume_penalty
    )


def train_stop_gradient_ablation(
    output_prediction_csv: Path,
    output_metrics_json: Path,
    *,
    source_generation_csv: Path = DEFAULT_SOURCE_GENERATION_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    epochs: int = 1600,
    seed: int = 2027,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_tp_dataset()
    frame = dataset.frame.copy()
    source_df = pd.read_csv(source_generation_csv, parse_dates=["date"])
    expected_cols = {"crop_daily_source", "impervious_daily_source", "total_daily_source", "period"}
    if not expected_cols.issubset(source_df.columns):
        raise ValueError(f"Stop-gradient source generation file is missing required columns: {sorted(expected_cols)}")

    fixed_source_inputs = {
        "crop_daily_source": source_df["crop_daily_source"].to_numpy(dtype=np.float32),
        "impervious_daily_source": source_df["impervious_daily_source"].to_numpy(dtype=np.float32),
    }
    model = DirectPhysicalTPModel(dataset=dataset, fixed_source_inputs=fixed_source_inputs)
    model.freeze_source_generation()

    full_state = load_differentiable_checkpoint(checkpoint_path)
    model_state = model.state_dict()
    for name, tensor in full_state.items():
        if name in model_state and name not in {"fixed_crop_input", "fixed_imp_input"}:
            model_state[name] = tensor.clone()
    model.load_state_dict(model_state, strict=False)
    model.freeze_source_generation()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=0.018)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=120, min_lr=6e-4)

    obs_t = torch.as_tensor(frame["TP"].to_numpy(dtype=np.float32))
    train_mask = torch.as_tensor((frame["period"] == "Calibration").to_numpy().copy())

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model()
        loss = _calibration_loss(obs_t[train_mask], out["sim"][train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
        optimizer.step()
        scheduler.step(loss.detach())
        current_loss = float(loss.detach().cpu())
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Stop-gradient ablation failed to produce a checkpoint.")

    model.load_state_dict(best_state, strict=False)
    with torch.no_grad():
        out = model()
    sim = out["sim"].detach().cpu().numpy()
    raw = out["raw"].detach().cpu().numpy()
    metrics = metrics_by_period(frame["TP"].to_numpy(dtype=float), sim, frame["period"].to_numpy())
    pred_df = pd.DataFrame(
        {
            "date": frame["date"],
            "observed_tp": frame["TP"],
            "simulated_tp": sim,
            "raw_physical_tp": raw,
            "period": frame["period"],
        }
    )

    output_prediction_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_prediction_csv, index=False)
    output_metrics_json.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "ablation": {
                    "type": "stop_gradient_source_generation",
                    "source_generation_file": str(source_generation_csv),
                    "source_generation_trainable": False,
                    "downstream_parameters_trainable": True,
                    "validation_used_for_training": False,
                    "best_checkpoint_selected_on": "calibration_loss_only",
                    "best_calibration_loss": best_loss,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return pred_df, metrics
