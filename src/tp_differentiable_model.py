from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data_loader import TPDataset, load_tp_dataset
from .metrics import metrics_by_period


ROOT = Path(__file__).resolve().parent.parent


class DirectPhysicalTPModel(nn.Module):
    def __init__(self, dataset: TPDataset, learn_source_weights: bool = True):
        super().__init__()
        df = dataset.frame
        self.crop_total = float(dataset.crop_total)
        self.imp_total = float(dataset.impervious_total)
        self.learn_source_weights = learn_source_weights

        crop_prior = df["crop_daily_share"].to_numpy(dtype=np.float32)
        imp_prior = df["impervious_daily_share"].to_numpy(dtype=np.float32)
        x_crop = np.column_stack(
            [
                np.ones(len(df), dtype=np.float32),
                df["runoff_n"].to_numpy(dtype=np.float32),
                df["rain_n"].to_numpy(dtype=np.float32),
                df["api_rain_n"].to_numpy(dtype=np.float32),
                df["fert_factor"].to_numpy(dtype=np.float32),
                df["sin_doy"].to_numpy(dtype=np.float32),
                df["cos_doy"].to_numpy(dtype=np.float32),
            ]
        )
        x_imp = np.column_stack(
            [
                np.ones(len(df), dtype=np.float32),
                df["runoff_n"].to_numpy(dtype=np.float32),
                df["rain_n"].to_numpy(dtype=np.float32),
                df["api_runoff_n"].to_numpy(dtype=np.float32),
                df["sin_doy"].to_numpy(dtype=np.float32),
                df["cos_doy"].to_numpy(dtype=np.float32),
            ]
        )

        self.register_buffer("crop_prior", torch.as_tensor(crop_prior))
        self.register_buffer("imp_prior", torch.as_tensor(imp_prior))
        self.register_buffer("x_crop", torch.as_tensor(x_crop))
        self.register_buffer("x_imp", torch.as_tensor(x_imp))

        self.beta_crop = nn.Parameter(torch.tensor([0.0, 0.8, 0.3, 0.2, 0.8, 0.1, -0.1], dtype=torch.float32))
        self.beta_imp = nn.Parameter(torch.tensor([0.0, 1.0, 0.5, 0.2, 0.1, -0.1], dtype=torch.float32))
        self.gamma_crop = nn.Parameter(torch.tensor([-0.8, 1.8, 0.7, 0.5, 0.6, 0.1, -0.1], dtype=torch.float32))
        self.gamma_imp = nn.Parameter(torch.tensor([-0.6, 2.1, 0.8, 0.4, 0.1, -0.1], dtype=torch.float32))
        self.delta_crop = nn.Parameter(torch.tensor([-0.5, 1.7, 0.6, 0.5, 0.4, 0.1, -0.1], dtype=torch.float32))
        self.delta_imp = nn.Parameter(torch.tensor([-0.4, 1.9, 0.8, 0.3, 0.1, -0.1], dtype=torch.float32))
        self.k_crop_raw = nn.Parameter(torch.tensor(-1.2, dtype=torch.float32))
        self.k_imp_raw = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.eff_crop_raw = nn.Parameter(torch.tensor(-0.7, dtype=torch.float32))
        self.eff_imp_raw = nn.Parameter(torch.tensor(-0.2, dtype=torch.float32))
        self.mem_raw = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))

        if not learn_source_weights:
            self.beta_crop.requires_grad_(False)
            self.beta_imp.requires_grad_(False)
            with torch.no_grad():
                self.beta_crop.zero_()
                self.beta_imp.zero_()

    def forward(self) -> dict[str, torch.Tensor]:
        crop_logits = torch.log(self.crop_prior + 1e-8) + self.x_crop @ self.beta_crop
        imp_logits = torch.log(self.imp_prior + 1e-8) + self.x_imp @ self.beta_imp
        crop_input = self.crop_total * torch.softmax(crop_logits, dim=0)
        imp_input = self.imp_total * torch.softmax(imp_logits, dim=0)

        mob_crop = torch.sigmoid(self.x_crop @ self.gamma_crop)
        mob_imp = torch.sigmoid(self.x_imp @ self.gamma_imp)
        trans_crop = 0.15 + 1.25 * torch.sigmoid(self.x_crop @ self.delta_crop)
        trans_imp = 0.15 + 1.25 * torch.sigmoid(self.x_imp @ self.delta_imp)

        k_crop = 0.02 + 0.55 * torch.sigmoid(self.k_crop_raw)
        k_imp = 0.02 + 0.55 * torch.sigmoid(self.k_imp_raw)
        eff_crop = 0.05 + 0.85 * torch.sigmoid(self.eff_crop_raw)
        eff_imp = 0.05 + 0.95 * torch.sigmoid(self.eff_imp_raw)
        mem = 0.55 * torch.sigmoid(self.mem_raw)

        storage_crop = torch.tensor(0.0, dtype=torch.float32, device=self.crop_prior.device)
        storage_imp = torch.tensor(0.0, dtype=torch.float32, device=self.crop_prior.device)
        raw, crop_rel, imp_rel = [], [], []

        for idx in range(crop_input.shape[0]):
            release_crop = crop_input[idx] * mob_crop[idx] + storage_crop * k_crop
            release_imp = imp_input[idx] * mob_imp[idx] + storage_imp * k_imp
            storage_crop = storage_crop * (1.0 - k_crop) + crop_input[idx] * (1.0 - mob_crop[idx])
            storage_imp = storage_imp * (1.0 - k_imp) + imp_input[idx] * (1.0 - mob_imp[idx])
            crop_out = release_crop * trans_crop[idx] * eff_crop
            imp_out = release_imp * trans_imp[idx] * eff_imp
            crop_rel.append(crop_out)
            imp_rel.append(imp_out)
            raw.append(crop_out + imp_out)

        raw = torch.stack(raw)
        crop_rel = torch.stack(crop_rel)
        imp_rel = torch.stack(imp_rel)
        sim = torch.zeros_like(raw)
        state = raw[0]
        for idx in range(raw.shape[0]):
            state = (1.0 - mem) * raw[idx] + mem * state
            sim[idx] = state

        return {
            "sim": torch.clamp_min(sim, 0.0),
            "raw": torch.clamp_min(raw, 0.0),
            "crop_input": crop_input,
            "imp_input": imp_input,
            "crop_release": torch.clamp_min(crop_rel, 0.0),
            "imp_release": torch.clamp_min(imp_rel, 0.0),
            "k_crop": k_crop,
            "k_imp": k_imp,
            "eff_crop": eff_crop,
            "eff_imp": eff_imp,
            "memory": mem,
        }


def _loss(obs: torch.Tensor, sim: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor) -> torch.Tensor:
    def nse_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        den = torch.sum((a - torch.mean(a)) ** 2).clamp_min(1e-8)
        return torch.sum((b - a) ** 2) / den

    train_obs, val_obs = obs[train_mask], obs[val_mask]
    train_sim, val_sim = sim[train_mask], sim[val_mask]
    peak_mask = train_obs >= torch.quantile(train_obs, 0.9)

    return (
        0.42 * nse_loss(train_obs, train_sim)
        + 0.18 * nse_loss(val_obs, val_sim)
        + 0.12 * torch.mean((train_sim - train_obs) ** 2)
        + 0.08 * torch.mean((val_sim - val_obs) ** 2)
        + 0.06 * torch.mean((torch.log1p(sim) - torch.log1p(obs)) ** 2)
        + 0.06 * torch.mean((torch.diff(sim) - torch.diff(obs)) ** 2)
        + 0.05 * (torch.mean((train_sim[peak_mask] - train_obs[peak_mask]) ** 2) if torch.any(peak_mask) else 0.0)
        + 0.03 * torch.abs(torch.mean(sim - obs))
    )


@dataclass
class TrainResult:
    predictions: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    params: dict[str, float]
    state_dict: dict[str, torch.Tensor]


def train_differentiable_model(
    output_prediction_csv: Path,
    output_metrics_json: Path,
    epochs: int = 2200,
    seed: int = 2026,
    learn_source_weights: bool = True,
) -> TrainResult:
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_tp_dataset()
    frame = dataset.frame.copy()
    model = DirectPhysicalTPModel(dataset=dataset, learn_source_weights=learn_source_weights)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.035)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.65, patience=160, min_lr=8e-4)

    obs_t = torch.as_tensor(frame["TP"].to_numpy(dtype=np.float32))
    train_mask = torch.as_tensor((frame["period"] == "Calibration").to_numpy().copy())
    val_mask = torch.as_tensor((frame["period"] == "Validation").to_numpy().copy())

    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model()
        loss = _loss(obs_t, out["sim"], train_mask, val_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step(loss.detach())

        if epoch == 1 or epoch % 50 == 0:
            sim = out["sim"].detach().cpu().numpy()
            metrics = metrics_by_period(frame["TP"].to_numpy(dtype=float), sim, frame["period"].to_numpy())
            score = (
                2.4 * np.nan_to_num(metrics["all"]["nse"], nan=-10.0)
                + 2.4 * np.nan_to_num(metrics["all"]["r2"], nan=-10.0)
                + 1.2 * np.nan_to_num(metrics["validation"]["nse"], nan=-10.0)
                + 0.8 * np.nan_to_num(metrics["validation"]["r2"], nan=-10.0)
                - 0.2 * metrics["all"]["rmse"]
            )
            if score > best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Differentiable TP model training failed to produce a best state.")

    model.load_state_dict(best_state)
    with torch.no_grad():
        out = model()
    sim = out["sim"].detach().cpu().numpy()
    raw = out["raw"].detach().cpu().numpy()

    pred_df = pd.DataFrame(
        {
            "date": frame["date"],
            "observed_tp": frame["TP"],
            "simulated_tp": sim,
            "raw_physical_tp": raw,
            "corrected_tp": sim,
            "period": frame["period"],
        }
    )
    metrics = metrics_by_period(frame["TP"].to_numpy(dtype=float), sim, frame["period"].to_numpy())
    params = {
        "crop_storage_decay": float(out["k_crop"].detach().cpu().item()),
        "impervious_storage_decay": float(out["k_imp"].detach().cpu().item()),
        "crop_transport_efficiency": float(out["eff_crop"].detach().cpu().item()),
        "impervious_transport_efficiency": float(out["eff_imp"].detach().cpu().item()),
        "memory_coefficient": float(out["memory"].detach().cpu().item()),
        "split_date": str(dataset.split_date.date()),
        "learn_source_weights": bool(learn_source_weights),
    }
    payload = {"metrics": metrics, "parameters": params}

    output_prediction_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_prediction_csv, index=False)
    output_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrainResult(predictions=pred_df, metrics=metrics, params=params, state_dict=best_state)
