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
DEFAULT_CHECKPOINT_PATH = ROOT / "results" / "checkpoints" / "tp_differentiable_model.pt"
DEFAULT_SOURCE_GENERATION_PATH = ROOT / "results" / "predictions" / "tp_differentiable_source_generation.csv"


class DirectPhysicalTPModel(nn.Module):
    def __init__(
        self,
        dataset: TPDataset,
        *,
        fixed_source_inputs: dict[str, np.ndarray] | None = None,
    ):
        super().__init__()
        df = dataset.frame
        self.crop_total = float(dataset.crop_total)
        self.imp_total = float(dataset.impervious_total)
        self.uses_fixed_source_inputs = fixed_source_inputs is not None

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

        if fixed_source_inputs is not None:
            crop_fixed = np.asarray(fixed_source_inputs["crop_daily_source"], dtype=np.float32)
            imp_fixed = np.asarray(fixed_source_inputs["impervious_daily_source"], dtype=np.float32)
            self.register_buffer("fixed_crop_input", torch.as_tensor(crop_fixed))
            self.register_buffer("fixed_imp_input", torch.as_tensor(imp_fixed))

    def freeze_source_generation(self) -> None:
        self.beta_crop.requires_grad_(False)
        self.beta_imp.requires_grad_(False)

    def generate_source(self) -> dict[str, torch.Tensor]:
        if self.uses_fixed_source_inputs:
            crop_input = self.fixed_crop_input
            imp_input = self.fixed_imp_input
            crop_weight = crop_input / max(self.crop_total, 1e-8)
            imp_weight = imp_input / max(self.imp_total, 1e-8)
            crop_logits = torch.log(torch.clamp_min(crop_weight, 1e-8))
            imp_logits = torch.log(torch.clamp_min(imp_weight, 1e-8))
        else:
            crop_logits = torch.log(self.crop_prior + 1e-8) + self.x_crop @ self.beta_crop
            imp_logits = torch.log(self.imp_prior + 1e-8) + self.x_imp @ self.beta_imp
            crop_input = self.crop_total * torch.softmax(crop_logits, dim=0)
            imp_input = self.imp_total * torch.softmax(imp_logits, dim=0)
            crop_weight = crop_input / max(self.crop_total, 1e-8)
            imp_weight = imp_input / max(self.imp_total, 1e-8)
        return {
            "crop_input": crop_input,
            "imp_input": imp_input,
            "total_input": crop_input + imp_input,
            "crop_weight": crop_weight,
            "imp_weight": imp_weight,
            "crop_logits": crop_logits,
            "imp_logits": imp_logits,
        }

    def route_sources(self, crop_input: torch.Tensor, imp_input: torch.Tensor) -> dict[str, torch.Tensor]:
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
            "crop_release": torch.clamp_min(crop_rel, 0.0),
            "imp_release": torch.clamp_min(imp_rel, 0.0),
            "k_crop": k_crop,
            "k_imp": k_imp,
            "eff_crop": eff_crop,
            "eff_imp": eff_imp,
            "memory": mem,
        }

    def forward(self) -> dict[str, torch.Tensor]:
        source = self.generate_source()
        downstream = self.route_sources(source["crop_input"], source["imp_input"])
        return {**source, **downstream}


def _calibration_loss(obs: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    def nse_penalty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        den = torch.sum((a - torch.mean(a)) ** 2).clamp_min(1e-8)
        return torch.sum((b - a) ** 2) / den

    mse = torch.mean((sim - obs) ** 2)
    huber = torch.nn.functional.smooth_l1_loss(sim, obs, beta=0.10)
    log_mse = torch.mean((torch.log1p(torch.clamp_min(sim, 0.0)) - torch.log1p(torch.clamp_min(obs, 0.0))) ** 2)
    diff_loss = torch.mean((torch.diff(sim) - torch.diff(obs)) ** 2)
    peak_thr = torch.quantile(obs.detach(), 0.9)
    peak_mask = obs >= peak_thr
    peak_loss = torch.mean((sim[peak_mask] - obs[peak_mask]) ** 2) if torch.any(peak_mask) else torch.tensor(0.0, device=obs.device)
    bias_penalty = torch.abs(torch.mean(sim - obs))
    volume_penalty = torch.abs(torch.sum(sim) - torch.sum(obs)) / torch.clamp_min(torch.abs(torch.sum(obs)), 1e-8)
    return (
        0.34 * nse_penalty(obs, sim)
        + 0.16 * mse
        + 0.12 * huber
        + 0.10 * log_mse
        + 0.10 * diff_loss
        + 0.10 * peak_loss
        + 0.04 * bias_penalty
        + 0.04 * volume_penalty
    )


@dataclass
class TrainResult:
    predictions: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    params: dict[str, float]
    state_dict: dict[str, torch.Tensor]
    source_generation: pd.DataFrame
    checkpoint_path: Path


def _build_source_generation_frame(frame: pd.DataFrame, out: dict[str, torch.Tensor]) -> pd.DataFrame:
    crop_source = out["crop_input"].detach().cpu().numpy()
    imp_source = out["imp_input"].detach().cpu().numpy()
    crop_weight = out["crop_weight"].detach().cpu().numpy()
    imp_weight = out["imp_weight"].detach().cpu().numpy()
    return pd.DataFrame(
        {
            "date": frame["date"],
            "crop_daily_source": crop_source,
            "impervious_daily_source": imp_source,
            "total_daily_source": crop_source + imp_source,
            "crop_source_weight": crop_weight,
            "impervious_source_weight": imp_weight,
            "period": frame["period"],
        }
    )


def _save_checkpoint(path: Path, state_dict: dict[str, torch.Tensor], params: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "parameters": params}, path)


def load_differentiable_checkpoint(path: Path = DEFAULT_CHECKPOINT_PATH) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return payload["state_dict"]


def train_differentiable_model(
    output_prediction_csv: Path,
    output_metrics_json: Path,
    *,
    source_generation_csv: Path = DEFAULT_SOURCE_GENERATION_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    epochs: int = 2200,
    seed: int = 2026,
) -> TrainResult:
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_tp_dataset()
    frame = dataset.frame.copy()
    model = DirectPhysicalTPModel(dataset=dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.035)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.65, patience=160, min_lr=8e-4)

    obs_t = torch.as_tensor(frame["TP"].to_numpy(dtype=np.float32))
    train_mask = torch.as_tensor((frame["period"] == "Calibration").to_numpy().copy())
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for _epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model()
        train_loss = _calibration_loss(obs_t[train_mask], out["sim"][train_mask])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step(train_loss.detach())

        current_loss = float(train_loss.detach().cpu())
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Differentiable TP model training failed to produce a checkpoint.")

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
        "best_calibration_loss": best_loss,
    }
    source_df = _build_source_generation_frame(frame, out)
    payload = {
        "metrics": metrics,
        "parameters": params,
        "training": {
            "calibration_only_loss": True,
            "validation_used_for_training": False,
            "validation_used_for_checkpoint_selection": False,
            "seed": seed,
            "epochs": epochs,
        },
    }

    output_prediction_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    source_generation_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_prediction_csv, index=False)
    source_df.to_csv(source_generation_csv, index=False)
    output_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_checkpoint(checkpoint_path, best_state, params)
    return TrainResult(
        predictions=pred_df,
        metrics=metrics,
        params=params,
        state_dict=best_state,
        source_generation=source_df,
        checkpoint_path=checkpoint_path,
    )
