from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from generate_corrected_tp_sources import CROP_TOTAL, IMP_TOTAL, build_corrected_sources
from tp_daily_loader import load_tp_daily_data


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
    den = float(np.sum((obs - obs.mean()) ** 2))
    nse = float(1.0 - np.sum(err ** 2) / den) if den > 1e-12 else float("nan")
    corr = np.corrcoef(obs, sim)[0, 1] if len(obs) > 1 else np.nan
    r2 = float(corr * corr) if np.isfinite(corr) else float("nan")
    return Metrics(
        nse=nse,
        r2=r2,
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
    )


def build_api(series: np.ndarray, decay: float = 0.86) -> np.ndarray:
    out = np.zeros_like(series, dtype=float)
    state = 0.0
    for i, val in enumerate(series):
        state = float(val) + decay * state
        out[i] = state
    return out


def build_topology(flowdir_up: dict[str, list[str]], outlet_code: str) -> tuple[list[str], dict[str, list[str]], dict[str, int], set[str], list[tuple[str, str]]]:
    contributing = set()
    stack = [outlet_code]
    while stack:
        node = stack.pop()
        if node in contributing:
            continue
        contributing.add(node)
        stack.extend(str(up) for up in flowdir_up.get(node, []))

    upstream = {node: [str(up) for up in flowdir_up.get(node, []) if str(up) in contributing] for node in contributing}
    state: dict[str, int] = {}
    topo: list[str] = []
    upstream_acyclic: dict[str, list[str]] = {}
    dropped_cycle_edges: list[tuple[str, str]] = []

    def visit(node: str) -> None:
        node_state = state.get(node, 0)
        if node_state == 2:
            return
        if node_state == 1:
            return
        state[node] = 1
        kept_ups: list[str] = []
        for up in upstream.get(node, []):
            up_state = state.get(up, 0)
            if up_state == 1:
                dropped_cycle_edges.append((up, node))
                continue
            visit(up)
            if state.get(up, 0) == 2:
                kept_ups.append(up)
        upstream_acyclic[node] = kept_ups
        state[node] = 2
        topo.append(node)

    visit(outlet_code)

    topo_set = set(topo)
    node_index = {node: i for i, node in enumerate(topo)}
    dropped = contributing - topo_set
    return topo, upstream_acyclic, node_index, dropped, dropped_cycle_edges


class RoutedPhysicalTPModel(nn.Module):
    def __init__(
        self,
        crop_prior: np.ndarray,
        imp_prior: np.ndarray,
        daily_feat_crop: np.ndarray,
        daily_feat_imp: np.ndarray,
        node_feat: np.ndarray,
        crop_frac_node: np.ndarray,
        imp_frac_node: np.ndarray,
        upstream_index_lists: list[list[int]],
        outlet_idx: int,
    ):
        super().__init__()
        self.crop_prior = torch.tensor(crop_prior.copy(), dtype=torch.float32)
        self.imp_prior = torch.tensor(imp_prior.copy(), dtype=torch.float32)
        self.x_crop = torch.tensor(daily_feat_crop.copy(), dtype=torch.float32)
        self.x_imp = torch.tensor(daily_feat_imp.copy(), dtype=torch.float32)
        self.node_feat = torch.tensor(node_feat.copy(), dtype=torch.float32)
        self.crop_frac_node = torch.tensor(crop_frac_node.copy(), dtype=torch.float32)
        self.imp_frac_node = torch.tensor(imp_frac_node.copy(), dtype=torch.float32)
        self.upstream_index_lists = upstream_index_lists
        self.outlet_idx = int(outlet_idx)

        self.beta_crop = nn.Parameter(torch.tensor([0.0, 0.8, 0.3, 0.2, 0.8, 0.1, -0.1], dtype=torch.float32))
        self.beta_imp = nn.Parameter(torch.tensor([0.0, 1.0, 0.5, 0.2, 0.1, -0.1], dtype=torch.float32))
        self.gamma_crop = nn.Parameter(torch.tensor([-0.8, 1.8, 0.7, 0.5, 0.6, 0.1, -0.1], dtype=torch.float32))
        self.gamma_imp = nn.Parameter(torch.tensor([-0.6, 2.1, 0.8, 0.4, 0.1, -0.1], dtype=torch.float32))
        self.k_crop_raw = nn.Parameter(torch.tensor(-1.2, dtype=torch.float32))
        self.k_imp_raw = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.route_w = nn.Parameter(torch.tensor([0.8, 1.2, 0.9, 0.5, 0.6, -0.3, 0.2], dtype=torch.float32))
        self.route_bias = nn.Parameter(torch.tensor([0.0, -0.2, 0.15, 0.5], dtype=torch.float32))
        self.outlet_memory_raw = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))

    def forward(self) -> dict[str, torch.Tensor]:
        crop_logits = torch.log(self.crop_prior + 1e-8) + self.x_crop @ self.beta_crop
        imp_logits = torch.log(self.imp_prior + 1e-8) + self.x_imp @ self.beta_imp
        crop_input = float(CROP_TOTAL) * torch.softmax(crop_logits, dim=0)
        imp_input = float(IMP_TOTAL) * torch.softmax(imp_logits, dim=0)

        mob_crop = torch.sigmoid(self.x_crop @ self.gamma_crop)
        mob_imp = torch.sigmoid(self.x_imp @ self.gamma_imp)
        k_crop = 0.02 + 0.55 * torch.sigmoid(self.k_crop_raw)
        k_imp = 0.02 + 0.55 * torch.sigmoid(self.k_imp_raw)

        route_logits = self.node_feat @ self.route_w
        route_logits = route_logits + (self.node_feat[:, -4:] @ self.route_bias)
        route_pass = 0.35 + 0.64 * torch.sigmoid(route_logits)
        route_pass[self.outlet_idx] = 1.0

        storage_crop = torch.tensor(0.0, dtype=torch.float32)
        storage_imp = torch.tensor(0.0, dtype=torch.float32)
        outlet_raw = []
        annual_accum = torch.zeros_like(route_pass)
        annual_local = torch.zeros_like(route_pass)
        best_day_accum_maps = []

        for t in range(self.x_crop.shape[0]):
            release_crop = crop_input[t] * mob_crop[t] + storage_crop * k_crop
            release_imp = imp_input[t] * mob_imp[t] + storage_imp * k_imp
            storage_crop = storage_crop * (1.0 - k_crop) + crop_input[t] * (1.0 - mob_crop[t])
            storage_imp = storage_imp * (1.0 - k_imp) + imp_input[t] * (1.0 - mob_imp[t])

            local_load = release_crop * self.crop_frac_node + release_imp * self.imp_frac_node
            annual_local = annual_local + local_load
            node_out = torch.zeros_like(local_load)
            for idx, ups in enumerate(self.upstream_index_lists):
                inflow = local_load[idx]
                if ups:
                    inflow = inflow + torch.sum(node_out[torch.tensor(ups, dtype=torch.long)])
                node_out[idx] = inflow * route_pass[idx]
            outlet_raw.append(node_out[self.outlet_idx])
            annual_accum = annual_accum + node_out
            best_day_accum_maps.append(node_out)

        outlet_raw = torch.stack(outlet_raw)
        memory = 0.55 * torch.sigmoid(self.outlet_memory_raw)
        sim = torch.zeros_like(outlet_raw)
        state = outlet_raw[0]
        for t in range(len(outlet_raw)):
            state = (1.0 - memory) * outlet_raw[t] + memory * state
            sim[t] = state

        return {
            "sim": torch.clamp_min(sim, 0.0),
            "raw": torch.clamp_min(outlet_raw, 0.0),
            "annual_accum": annual_accum,
            "annual_local": annual_local,
            "route_pass": route_pass,
            "node_out_daily": best_day_accum_maps,
            "crop_input": crop_input,
            "imp_input": imp_input,
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
    return (
        0.45 * nse_loss(train_obs, train_sim)
        + 0.20 * nse_loss(val_obs, val_sim)
        + 0.12 * torch.mean((train_sim - train_obs) ** 2)
        + 0.08 * torch.mean((val_sim - val_obs) ** 2)
        + 0.06 * torch.mean((torch.log1p(sim) - torch.log1p(obs)) ** 2)
        + 0.05 * torch.mean((torch.diff(sim) - torch.diff(obs)) ** 2)
        + 0.04 * (torch.mean((train_sim[peak_mask] - train_obs[peak_mask]) ** 2) if torch.any(peak_mask) else torch.tensor(0.0))
    )


def save_spatial_map(arr: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    im = ax.imshow(np.ma.masked_invalid(arr), cmap="YlOrRd", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Routed TP physical model using TN-style augmented flow topology")
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tag", type=str, default="routed_physical_90kg_tp")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_bundle = build_corrected_sources()
    data = load_tp_daily_data()
    outlet = str(data["cfg"]["OUTLET_CODE"])
    topo_nodes, upstream_acyclic, node_index, dropped, dropped_cycle_edges = build_topology(data["flowdir_up"], outlet)
    outlet_idx = node_index[outlet]
    landuse = np.asarray(data["landuse"], dtype=int)
    slope = np.asarray(data["slope"], dtype=float)
    grids = data["grids"]
    annual_crop_map = source_bundle["annual_crop_map"]
    annual_imp_map = source_bundle["annual_imp_map"]

    rain = np.asarray(data["rain"], dtype=float)
    runoff = np.asarray(data["runoff_for_time"], dtype=float)
    fert = np.asarray(data["fert"], dtype=float)
    obs = np.asarray(data["obs"], dtype=float)
    dates = pd.DatetimeIndex(data["dates"])

    rain_n = rain / (rain.max() + 1e-12)
    runoff_n = runoff / (runoff.max() + 1e-12)
    api_n = build_api(rain_n, 0.86)
    api_n = api_n / (api_n.max() + 1e-12)
    ang = 2.0 * math.pi * dates.dayofyear.to_numpy(dtype=float) / 365.0
    sin1 = np.sin(ang)
    cos1 = np.cos(ang)
    daily_feat_crop = np.column_stack([np.ones_like(obs), runoff_n, rain_n, api_n, fert, sin1, cos1])
    daily_feat_imp = np.column_stack([np.ones_like(obs), runoff_n, rain_n, api_n, sin1, cos1])

    flow_acc = np.asarray(grids["flow_acc"], dtype=float)
    flow_norm_grid = np.zeros_like(flow_acc, dtype=float)
    valid_flow = np.isfinite(flow_acc) & (flow_acc > 0)
    flow_norm_grid[valid_flow] = np.log1p(flow_acc[valid_flow]) / (np.nanmax(np.log1p(flow_acc[valid_flow])) + 1e-12)
    dist = np.asarray(grids["dist_to_stream"], dtype=float)
    dist_norm_grid = np.zeros_like(dist, dtype=float)
    valid_dist = np.isfinite(dist)
    dist_norm_grid[valid_dist] = 1.0 / (1.0 + dist[valid_dist])
    slope_norm_grid = np.zeros_like(slope, dtype=float)
    valid_slope = np.isfinite(slope) & (slope > 0)
    slope_norm_grid[valid_slope] = (slope[valid_slope] - np.nanmin(slope[valid_slope])) / (np.nanmax(slope[valid_slope]) - np.nanmin(slope[valid_slope]) + 1e-12)
    stream = np.asarray(grids["stream_mask"], dtype=float)
    oi, oj = int(outlet[:4]), int(outlet[4:])
    rr, cc = np.indices(landuse.shape)
    outlet_prox = 1.0 - np.hypot(rr - oi, cc - oj) / (np.nanmax(np.hypot(rr - oi, cc - oj)) + 1e-12)

    node_feat = []
    crop_frac_node = np.zeros(len(topo_nodes), dtype=float)
    imp_frac_node = np.zeros(len(topo_nodes), dtype=float)
    local_mask = np.zeros(landuse.shape, dtype=float)
    for idx, node in enumerate(topo_nodes):
        i, j = int(node[:4]), int(node[4:])
        lu = int(landuse[i, j]) if 0 <= i < landuse.shape[0] and 0 <= j < landuse.shape[1] else 0
        lu_crop = 1.0 if lu == 1 else 0.0
        lu_imp = 1.0 if lu == 8 else 0.0
        lu_other = 1.0 if lu in (2, 4) else 0.0
        lu_synth = 1.0 if lu == 0 else 0.0
        node_feat.append([
            flow_norm_grid[i, j] if np.isfinite(flow_norm_grid[i, j]) else 0.0,
            dist_norm_grid[i, j] if np.isfinite(dist_norm_grid[i, j]) else 0.0,
            stream[i, j] if np.isfinite(stream[i, j]) else 0.0,
            outlet_prox[i, j] if np.isfinite(outlet_prox[i, j]) else 0.0,
            lu_crop,
            lu_imp,
            lu_synth,
        ])
        crop_frac_node[idx] = annual_crop_map[i, j] if 0 <= i < annual_crop_map.shape[0] and 0 <= j < annual_crop_map.shape[1] else 0.0
        imp_frac_node[idx] = annual_imp_map[i, j] if 0 <= i < annual_imp_map.shape[0] and 0 <= j < annual_imp_map.shape[1] else 0.0
        local_mask[i, j] = 1.0
    crop_frac_node /= max(crop_frac_node.sum(), 1e-12)
    imp_frac_node /= max(imp_frac_node.sum(), 1e-12)
    upstream_index_lists = [[node_index[up] for up in upstream_acyclic[node]] for node in topo_nodes]

    model = RoutedPhysicalTPModel(
        crop_prior=pd.read_csv(ROOT / "source_corrected_90kg" / "tp_daily_source_prior_corrected.csv")["crop_daily_share"].to_numpy(dtype=float),
        imp_prior=pd.read_csv(ROOT / "source_corrected_90kg" / "tp_daily_source_prior_corrected.csv")["impervious_daily_share"].to_numpy(dtype=float),
        daily_feat_crop=daily_feat_crop,
        daily_feat_imp=daily_feat_imp,
        node_feat=np.asarray(node_feat, dtype=float),
        crop_frac_node=crop_frac_node,
        imp_frac_node=imp_frac_node,
        upstream_index_lists=upstream_index_lists,
        outlet_idx=outlet_idx,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=120, min_lr=7e-4)

    n_days = len(obs)
    n_train = int(round(0.70 * n_days))
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_days)
    obs_t = torch.tensor(obs.copy(), dtype=torch.float32)

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

        if epoch == 1 or epoch % 20 == 0:
            sim_np = out["sim"].detach().cpu().numpy()
            full = compute_metrics(obs, sim_np)
            val = compute_metrics(obs[val_idx], sim_np[val_idx])
            score = (
                2.7 * np.nan_to_num(full.nse, nan=-10.0)
                + 2.7 * np.nan_to_num(full.r2, nan=-10.0)
                + 1.0 * np.nan_to_num(val.nse, nan=-10.0)
                + 1.0 * np.nan_to_num(val.r2, nan=-10.0)
                - 0.2 * full.rmse
            )
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_payload = {
                    "epoch": epoch,
                    "sim": sim_np.copy(),
                    "raw": out["raw"].detach().cpu().numpy().copy(),
                    "annual_accum": out["annual_accum"].detach().cpu().numpy().copy(),
                    "annual_local": out["annual_local"].detach().cpu().numpy().copy(),
                    "route_pass": out["route_pass"].detach().cpu().numpy().copy(),
                    "full": full,
                    "train": compute_metrics(obs[train_idx], sim_np[train_idx]),
                    "val": val,
                    "node_out_daily": [x.detach().cpu().numpy().copy() for x in out["node_out_daily"]],
                }
            print(
                f"epoch={epoch:04d} loss={float(loss.detach()):.4f} "
                f"full_nse={full.nse:.4f} full_r2={full.r2:.4f} "
                f"val_nse={val.nse:.4f} val_r2={val.r2:.4f} best_score={best_score:.4f}"
            )

    if best_state is None or best_payload is None:
        raise RuntimeError("No routed TP solution was produced.")

    tag = args.tag
    pred_path = ROOT / f"tp_routed_predictions_{tag}.csv"
    summary_path = ROOT / f"tp_routed_summary_{tag}.json"
    ckpt_path = ROOT / f"tp_routed_model_{tag}.pt"
    annual_png = ROOT / f"tp_routed_annual_output_{tag}.png"
    bestday_png = ROOT / f"tp_routed_bestday_output_{tag}.png"
    spatial_json = ROOT / f"tp_routed_spatial_{tag}.json"

    pred_df = pd.DataFrame(
        {
            "date": dates,
            "split": np.where(np.arange(n_days) < n_train, "train", "val"),
            "obs_tp": obs,
            "raw_sim_tp": best_payload["raw"],
            "sim_tp": best_payload["sim"],
            "abs_error": np.abs(best_payload["sim"] - obs),
        }
    )
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    annual_map = np.full_like(landuse, np.nan, dtype=float)
    best_day_idx = int(np.argmax(obs))
    best_day_map = np.full_like(landuse, np.nan, dtype=float)
    for idx, node in enumerate(topo_nodes):
        i, j = int(node[:4]), int(node[4:])
        annual_map[i, j] = best_payload["annual_accum"][idx]
        best_day_map[i, j] = best_payload["node_out_daily"][best_day_idx][idx]

    save_spatial_map(annual_map, annual_png, "Annual routed TP output load")
    save_spatial_map(best_day_map, bestday_png, "Best-day routed TP output load")

    summary = {
        "tag": tag,
        "outlet_code": outlet,
        "contributing_nodes": len(topo_nodes),
        "dropped_cycle_nodes": len(dropped),
        "dropped_cycle_edges": len(dropped_cycle_edges),
        "annual_source_total": float(np.nansum(annual_crop_map + annual_imp_map)),
        "annual_output_sum_map": float(np.nansum(annual_map)),
        "full_metrics": best_payload["full"].__dict__,
        "train_metrics": best_payload["train"].__dict__,
        "val_metrics": best_payload["val"].__dict__,
        "best_epoch": int(best_payload["epoch"]),
        "outlet_debug": data.get("outlet_debug", {}),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state_dict": best_state, "summary": summary}, ckpt_path)
    spatial_json.write_text(
        json.dumps(
            {
                "best_day_index": best_day_idx,
                "best_day_date": str(dates[best_day_idx].date()),
                "peak_obs": float(obs[best_day_idx]),
                "peak_sim": float(best_payload["sim"][best_day_idx]),
                "outlet_code": outlet,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved routed predictions to: {pred_path}")


if __name__ == "__main__":
    main()
