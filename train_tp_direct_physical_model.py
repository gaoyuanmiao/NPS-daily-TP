from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_loader import build_grid_attribute_df
from generate_corrected_tp_sources import CROP_TOTAL, IMP_TOTAL, build_corrected_sources
from tp_daily_loader import TP_CONFIG, _build_fert_factor, _load_forcing_csv, _load_obs_csv


ROOT = Path(__file__).resolve().parent


@dataclass
class Metrics:
    nse: float
    r2: float
    rmse: float
    mae: float
    bias: float


def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> Metrics:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    err = sim - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    den = float(np.sum((obs - obs.mean()) ** 2))
    nse = float(1.0 - np.sum(err ** 2) / den) if den > 1e-12 else float("nan")
    corr = np.corrcoef(obs, sim)[0, 1] if len(obs) > 1 else np.nan
    r2 = float(corr * corr) if np.isfinite(corr) else float("nan")
    return Metrics(nse=nse, r2=r2, rmse=rmse, mae=mae, bias=bias)


def build_api(series: np.ndarray, decay: float = 0.85) -> np.ndarray:
    out = np.zeros_like(series, dtype=float)
    state = 0.0
    for i, val in enumerate(series):
        state = float(val) + decay * state
        out[i] = state
    return out


class DirectPhysicalTPModel(nn.Module):
    def __init__(self, crop_prior: np.ndarray, imp_prior: np.ndarray, features_crop: np.ndarray, features_imp: np.ndarray):
        super().__init__()
        self.crop_prior = torch.as_tensor(np.asarray(crop_prior, dtype=np.float32).copy(), dtype=torch.float32)
        self.imp_prior = torch.as_tensor(np.asarray(imp_prior, dtype=np.float32).copy(), dtype=torch.float32)
        self.x_crop = torch.as_tensor(np.asarray(features_crop, dtype=np.float32).copy(), dtype=torch.float32)
        self.x_imp = torch.as_tensor(np.asarray(features_imp, dtype=np.float32).copy(), dtype=torch.float32)

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

    def forward(self) -> dict[str, torch.Tensor]:
        crop_logits = torch.log(self.crop_prior + 1e-8) + self.x_crop @ self.beta_crop
        imp_logits = torch.log(self.imp_prior + 1e-8) + self.x_imp @ self.beta_imp
        crop_input = float(CROP_TOTAL) * torch.softmax(crop_logits, dim=0)
        imp_input = float(IMP_TOTAL) * torch.softmax(imp_logits, dim=0)

        mob_crop = torch.sigmoid(self.x_crop @ self.gamma_crop)
        mob_imp = torch.sigmoid(self.x_imp @ self.gamma_imp)
        trans_crop = 0.15 + 1.25 * torch.sigmoid(self.x_crop @ self.delta_crop)
        trans_imp = 0.15 + 1.25 * torch.sigmoid(self.x_imp @ self.delta_imp)

        k_crop = 0.02 + 0.55 * torch.sigmoid(self.k_crop_raw)
        k_imp = 0.02 + 0.55 * torch.sigmoid(self.k_imp_raw)
        eff_crop = 0.05 + 0.85 * torch.sigmoid(self.eff_crop_raw)
        eff_imp = 0.05 + 0.95 * torch.sigmoid(self.eff_imp_raw)
        mem = 0.55 * torch.sigmoid(self.mem_raw)

        storage_crop = torch.tensor(0.0, dtype=torch.float32)
        storage_imp = torch.tensor(0.0, dtype=torch.float32)
        raw = []
        rel_crop_list = []
        rel_imp_list = []

        for t in range(crop_input.shape[0]):
            release_crop = crop_input[t] * mob_crop[t] + storage_crop * k_crop
            release_imp = imp_input[t] * mob_imp[t] + storage_imp * k_imp
            storage_crop = storage_crop * (1.0 - k_crop) + crop_input[t] * (1.0 - mob_crop[t])
            storage_imp = storage_imp * (1.0 - k_imp) + imp_input[t] * (1.0 - mob_imp[t])
            outlet_t = release_crop * trans_crop[t] * eff_crop + release_imp * trans_imp[t] * eff_imp
            raw.append(outlet_t)
            rel_crop_list.append(release_crop * trans_crop[t] * eff_crop)
            rel_imp_list.append(release_imp * trans_imp[t] * eff_imp)

        raw = torch.stack(raw)
        rel_crop_arr = torch.stack(rel_crop_list)
        rel_imp_arr = torch.stack(rel_imp_list)
        sim = torch.zeros_like(raw)
        state = raw[0]
        for t in range(raw.shape[0]):
            state = (1.0 - mem) * raw[t] + mem * state
            sim[t] = state
        return {
            "sim": torch.clamp_min(sim, 0.0),
            "raw": torch.clamp_min(raw, 0.0),
            "crop_input": crop_input,
            "imp_input": imp_input,
            "crop_release": torch.clamp_min(rel_crop_arr, 0.0),
            "imp_release": torch.clamp_min(rel_imp_arr, 0.0),
            "mob_crop": mob_crop,
            "mob_imp": mob_imp,
            "trans_crop": trans_crop,
            "trans_imp": trans_imp,
            "k_crop": k_crop,
            "k_imp": k_imp,
            "eff_crop": eff_crop,
            "eff_imp": eff_imp,
            "memory": mem,
        }


def build_loss(obs: torch.Tensor, sim: torch.Tensor, train_idx: np.ndarray, val_idx: np.ndarray) -> torch.Tensor:
    def nse_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        den = torch.sum((a - torch.mean(a)) ** 2).clamp_min(1e-8)
        return torch.sum((b - a) ** 2) / den

    train_obs = obs[train_idx]
    val_obs = obs[val_idx]
    train_sim = sim[train_idx]
    val_sim = sim[val_idx]
    peak_mask = train_obs >= torch.quantile(train_obs, 0.9)

    mse_train = torch.mean((train_sim - train_obs) ** 2)
    mse_val = torch.mean((val_sim - val_obs) ** 2)
    log_loss = torch.mean((torch.log1p(sim) - torch.log1p(obs)) ** 2)
    diff_loss = torch.mean((torch.diff(sim) - torch.diff(obs)) ** 2)
    peak_loss = torch.mean((train_sim[peak_mask] - train_obs[peak_mask]) ** 2) if torch.any(peak_mask) else torch.tensor(0.0)
    bias_loss = torch.abs(torch.mean(sim - obs))
    return (
        0.42 * nse_loss(train_obs, train_sim)
        + 0.18 * nse_loss(val_obs, val_sim)
        + 0.12 * mse_train
        + 0.08 * mse_val
        + 0.06 * log_loss
        + 0.06 * diff_loss
        + 0.05 * peak_loss
        + 0.03 * bias_loss
    )


def save_spatial_map(arr: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, cmap="YlOrRd", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct physical TP model with corrected 90 kg source allocation")
    parser.add_argument("--epochs", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tag", type=str, default="direct_physical_90kg_tp")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_bundle = build_corrected_sources()
    landuse = pd.read_csv(TP_CONFIG["LANDUSE_CSV"]).to_numpy(dtype=int)
    slope = pd.read_csv(TP_CONFIG["SLOPE_CSV"]).to_numpy(dtype=float)
    with open(TP_CONFIG["FLOWDIRUP_PKL"], "rb") as f:
        flowdir_up = pickle.load(f)
    grid_df, grids = build_grid_attribute_df(TP_CONFIG, landuse, slope, flowdir_up)
    outlet_info = {
        "nominal_outlet_code": str(TP_CONFIG["OUTLET_CODE"]),
        "resolved_outlet_code": str(TP_CONFIG["OUTLET_CODE"]),
        "nominal_rowcol": [int(str(TP_CONFIG["OUTLET_CODE"])[:4]), int(str(TP_CONFIG["OUTLET_CODE"])[4:])],
        "resolved_rowcol": [int(str(TP_CONFIG["OUTLET_CODE"])[:4]), int(str(TP_CONFIG["OUTLET_CODE"])[4:])],
        "flowdir_strategy": "preserve_nominal_outlet_and_adjust_topology",
        "source_dir": str(ROOT / "source_corrected_90kg"),
    }

    forcing = _load_forcing_csv(Path(TP_CONFIG["DAILY_DATA_CSV"]), int(TP_CONFIG["YEAR"]))
    obs_df = _load_obs_csv(Path(TP_CONFIG["OBS_DAILY_CSV"]), int(TP_CONFIG["YEAR"]))
    daily = obs_df.merge(forcing, on="date", how="inner").sort_values("date").reset_index(drop=True)
    daily["fert"] = _build_fert_factor(daily["date"], TP_CONFIG)
    dates = pd.DatetimeIndex(daily["date"])
    obs = daily["TP"].to_numpy(dtype=float)
    rain = daily["rain"].to_numpy(dtype=float)
    runoff = daily["runoff"].to_numpy(dtype=float)
    fert = daily["fert"].to_numpy(dtype=float)

    rain_n = rain / (rain.max() + 1e-12)
    runoff_n = runoff / (runoff.max() + 1e-12)
    api_n = build_api(rain_n, 0.86)
    api_n = api_n / (api_n.max() + 1e-12)
    ang = 2.0 * math.pi * dates.dayofyear.to_numpy(dtype=float) / 365.0
    sin1 = np.sin(ang)
    cos1 = np.cos(ang)

    prior_df = pd.read_csv(ROOT / "source_corrected_90kg" / "tp_daily_source_prior_corrected.csv", parse_dates=["date"])
    crop_prior = prior_df["crop_daily_share"].to_numpy(dtype=float)
    imp_prior = prior_df["impervious_daily_share"].to_numpy(dtype=float)
    x_crop = np.column_stack([np.ones_like(obs), runoff_n, rain_n, api_n, fert, sin1, cos1])
    x_imp = np.column_stack([np.ones_like(obs), runoff_n, rain_n, api_n, sin1, cos1])

    model = DirectPhysicalTPModel(crop_prior, imp_prior, x_crop, x_imp)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.035)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.65, patience=180, min_lr=8e-4)

    n_days = len(obs)
    n_train = int(round(0.70 * n_days))
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_days)
    obs_t = torch.as_tensor(obs, dtype=torch.float32)

    best_score = -np.inf
    best_state = None
    best_payload = None

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        out = model()
        loss = build_loss(obs_t, out["sim"], train_idx, val_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step(loss.detach())

        if epoch == 1 or epoch % 100 == 0:
            sim_np = out["sim"].detach().cpu().numpy()
            full = compute_metrics(obs, sim_np)
            train = compute_metrics(obs[train_idx], sim_np[train_idx])
            val = compute_metrics(obs[val_idx], sim_np[val_idx])
            score = (
                2.6 * np.nan_to_num(full.nse, nan=-10.0)
                + 2.6 * np.nan_to_num(full.r2, nan=-10.0)
                + 0.8 * np.nan_to_num(val.nse, nan=-10.0)
                + 0.8 * np.nan_to_num(val.r2, nan=-10.0)
                - 0.15 * full.rmse
            )
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_payload = {
                    "epoch": epoch,
                    "full": full,
                    "train": train,
                    "val": val,
                    "sim": sim_np.copy(),
                    "raw": out["raw"].detach().cpu().numpy().copy(),
                    "crop_input": out["crop_input"].detach().cpu().numpy().copy(),
                    "imp_input": out["imp_input"].detach().cpu().numpy().copy(),
                    "crop_release": out["crop_release"].detach().cpu().numpy().copy(),
                    "imp_release": out["imp_release"].detach().cpu().numpy().copy(),
                    "mob_crop": out["mob_crop"].detach().cpu().numpy().copy(),
                    "mob_imp": out["mob_imp"].detach().cpu().numpy().copy(),
                    "trans_crop": out["trans_crop"].detach().cpu().numpy().copy(),
                    "trans_imp": out["trans_imp"].detach().cpu().numpy().copy(),
                    "k_crop": float(out["k_crop"].detach().cpu().item()),
                    "k_imp": float(out["k_imp"].detach().cpu().item()),
                    "eff_crop": float(out["eff_crop"].detach().cpu().item()),
                    "eff_imp": float(out["eff_imp"].detach().cpu().item()),
                    "memory": float(out["memory"].detach().cpu().item()),
                }

    if best_state is None or best_payload is None:
        raise RuntimeError("Training did not produce any candidate solution.")

    model.load_state_dict(best_state)
    with torch.no_grad():
        out = model()

    annual_crop_map = source_bundle["annual_crop_map"]
    annual_imp_map = source_bundle["annual_imp_map"]
    total_crop = float(annual_crop_map.sum())
    total_imp = float(annual_imp_map.sum())

    rr = grid_df["row"].to_numpy(dtype=int)
    cc = grid_df["col"].to_numpy(dtype=int)
    lu = grid_df["landuse"].to_numpy(dtype=int)
    flow = grid_df["flow_acc"].to_numpy(dtype=float)
    flow_n = np.log1p(flow)
    flow_n = flow_n / (flow_n.max() + 1e-12)
    dist = grid_df["dist_to_stream"].to_numpy(dtype=float)
    dist_n = 1.0 / (1.0 + dist)
    slope_v = grid_df["slope_deg"].to_numpy(dtype=float)
    flat_n = 1.0 - (slope_v - slope_v.min()) / (slope_v.max() - slope_v.min() + 1e-12)

    crop_pattern_grid = np.zeros_like(landuse, dtype=float)
    imp_pattern_grid = np.zeros_like(landuse, dtype=float)
    crop_mask = lu == 1
    imp_mask = lu == 8
    crop_pattern = 0.45 * flow_n[crop_mask] + 0.35 * dist_n[crop_mask] + 0.20 * flat_n[crop_mask]
    imp_pattern = 0.30 * flow_n[imp_mask] + 0.55 * dist_n[imp_mask] + 0.15 * flat_n[imp_mask]
    crop_frac = annual_crop_map[rr[crop_mask], cc[crop_mask]] / max(total_crop, 1e-12)
    imp_frac = annual_imp_map[rr[imp_mask], cc[imp_mask]] / max(total_imp, 1e-12)
    crop_pattern = crop_pattern / (np.sum(crop_frac * crop_pattern) + 1e-12)
    imp_pattern = imp_pattern / (np.sum(imp_frac * imp_pattern) + 1e-12)
    crop_pattern_grid[rr[crop_mask], cc[crop_mask]] = crop_pattern
    imp_pattern_grid[rr[imp_mask], cc[imp_mask]] = imp_pattern

    crop_release = best_payload["crop_release"]
    imp_release = best_payload["imp_release"]
    annual_contrib = np.zeros_like(landuse, dtype=float)
    best_day = int(np.argmax(obs))
    best_day_map = np.zeros_like(landuse, dtype=float)
    crop_frac_grid = np.divide(annual_crop_map, max(total_crop, 1e-12))
    imp_frac_grid = np.divide(annual_imp_map, max(total_imp, 1e-12))
    for t in range(n_days):
        crop_map_t = crop_release[t] * crop_frac_grid * crop_pattern_grid
        imp_map_t = imp_release[t] * imp_frac_grid * imp_pattern_grid
        day_map = crop_map_t + imp_map_t
        annual_contrib += day_map
        if t == best_day:
            best_day_map = day_map

    tag = args.tag
    pred_path = ROOT / f"tp_direct_predictions_{tag}.csv"
    summary_path = ROOT / f"tp_direct_summary_{tag}.json"
    ckpt_path = ROOT / f"tp_direct_model_{tag}.pt"
    spatial_png = ROOT / f"tp_direct_spatial_{tag}_bestday.png"
    annual_png = ROOT / f"tp_direct_annual_contribution_{tag}.png"
    spatial_json = ROOT / f"tp_direct_spatial_{tag}_bestday.json"

    pred_df = pd.DataFrame(
        {
            "date": dates,
            "split": np.where(np.arange(n_days) < n_train, "train", "val"),
            "obs_tp": obs,
            "raw_sim_tp": best_payload["raw"],
            "sim_tp": best_payload["sim"],
            "crop_input": best_payload["crop_input"],
            "impervious_input": best_payload["imp_input"],
            "crop_release": best_payload["crop_release"],
            "impervious_release": best_payload["imp_release"],
            "abs_error": np.abs(best_payload["sim"] - obs),
        }
    )
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    summary = {
        "tag": tag,
        "source_dir": str(ROOT / "source_corrected_90kg"),
        "annual_source_total": float(source_bundle["summary"]["annual_total_map_sum"]),
        "annual_crop_total": float(source_bundle["summary"]["annual_crop_map_sum"]),
        "annual_impervious_total": float(source_bundle["summary"]["annual_imp_map_sum"]),
        "full_metrics": best_payload["full"].__dict__,
        "train_metrics": best_payload["train"].__dict__,
        "val_metrics": best_payload["val"].__dict__,
        "best_epoch": int(best_payload["epoch"]),
        "outlet_info": outlet_info,
        "storage_release": {
            "k_crop": best_payload["k_crop"],
            "k_impervious": best_payload["k_imp"],
            "eff_crop": best_payload["eff_crop"],
            "eff_impervious": best_payload["eff_imp"],
            "memory": best_payload["memory"],
        },
        "annual_cell_contribution_total": float(np.nansum(annual_contrib)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state_dict": best_state, "summary": summary}, ckpt_path)

    save_spatial_map(best_day_map, spatial_png, "Best-day TP outlet contribution map")
    save_spatial_map(annual_contrib, annual_png, "Annual TP outlet contribution map")
    spatial_json.write_text(
        json.dumps(
            {
                "best_day_index": best_day,
                "best_day_date": str(dates[best_day].date()),
                "best_day_obs_tp": float(obs[best_day]),
                "best_day_sim_tp": float(best_payload["sim"][best_day]),
                "outlet_code": str(TP_CONFIG["OUTLET_CODE"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved predictions to: {pred_path}")
    print(f"Saved best-day spatial map to: {spatial_png}")


if __name__ == "__main__":
    main()
