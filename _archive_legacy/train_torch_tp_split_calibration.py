from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")

from model_components_numpy import DailyNPSCore
from model_components_torch import (
    LearnableDailyTotalTorch,
    build_source_maps_with_ml_torch,
    run_single_day_with_trace_torch,
)
from tp_daily_loader import export_tp_daily_inputs, load_tp_daily_data


ROOT = Path(__file__).resolve().parent


@dataclass
class MetricBundle:
    nse: float
    r2: float
    rmse: float
    mae: float
    bias: float


class InitialPoolState(nn.Module):
    def __init__(self, init_value: float = 0.001):
        super().__init__()
        init = torch.log(torch.expm1(torch.tensor(init_value, dtype=torch.float32)))
        self.bgc_dis = nn.Parameter(init.clone())
        self.bgc_ads = nn.Parameter(init.clone())
        self.hyd_dis = nn.Parameter(init.clone())
        self.hyd_ads = nn.Parameter(init.clone())

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.nn.functional.softplus(self.bgc_dis),
            torch.nn.functional.softplus(self.bgc_ads),
            torch.nn.functional.softplus(self.hyd_dis),
            torch.nn.functional.softplus(self.hyd_ads),
        )


class PhysicalObservationOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(10, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        nn.init.normal_(self.mlp[0].weight, mean=0.0, std=0.08)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, mean=0.0, std=0.08)
        nn.init.zeros_(self.mlp[2].bias)
        nn.init.normal_(self.mlp[4].weight, mean=0.0, std=0.05)
        nn.init.constant_(self.mlp[4].bias, -2.7)

    def forward(
        self,
        routed_out: torch.Tensor,
        hyd_out: torch.Tensor,
        runoff: torch.Tensor,
        rain: torch.Tensor,
        wetness: torch.Tensor,
        l_daily: torch.Tensor,
        f_surface: torch.Tensor,
        doy: torch.Tensor,
    ) -> torch.Tensor:
        memory = 0.25 * torch.sigmoid(self.memory)
        ang = 2.0 * torch.pi * doy / 365.0
        features = torch.stack(
            [
                torch.log1p(torch.clamp_min(raw_load := routed_out + hyd_out, 0.0)),
                torch.log1p(torch.clamp_min(routed_out, 0.0)),
                torch.log1p(torch.clamp_min(hyd_out, 0.0)),
                torch.log1p(torch.clamp_min(runoff, 0.0)),
                torch.log1p(torch.clamp_min(rain, 0.0)),
                torch.log1p(torch.clamp_min(wetness, 0.0)),
                torch.log1p(torch.clamp_min(l_daily, 0.0)),
                f_surface,
                torch.sin(ang),
                torch.cos(ang),
            ],
            dim=1,
        )
        inst = torch.nn.functional.softplus(self.mlp(features).squeeze(-1))
        out = torch.zeros_like(inst)
        state = inst[0]
        for i in range(len(inst)):
            state = (1.0 - memory) * inst[i] + memory * state
            out[i] = state
        return torch.clamp_min(out, 0.0)


def classify_landuse(landuse_np: np.ndarray) -> np.ndarray:
    sink = np.zeros_like(landuse_np, dtype=np.int32)
    sink[landuse_np == 1] = 1
    sink[(landuse_np == 2) | (landuse_np == 4)] = 2
    sink[landuse_np == 8] = 3
    return sink


def nse_torch(obs: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    den = torch.sum((obs - torch.mean(obs)) ** 2)
    return 1.0 - torch.sum((obs - sim) ** 2) / torch.clamp_min(den, 1e-12)


def compute_metrics_np(obs: np.ndarray, sim: np.ndarray) -> MetricBundle:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    mae = float(np.mean(np.abs(sim - obs)))
    bias = float(np.mean(sim - obs))
    den = float(np.sum((obs - np.mean(obs)) ** 2))
    nse = float(1.0 - np.sum((sim - obs) ** 2) / den) if den > 1e-12 else float("nan")
    corr = np.corrcoef(obs, sim)[0, 1] if len(obs) > 1 else np.nan
    r2 = float(corr * corr) if np.isfinite(corr) else float("nan")
    return MetricBundle(nse=nse, r2=r2, rmse=rmse, mae=mae, bias=bias)


def build_loss(
    sim: torch.Tensor,
    obs: torch.Tensor,
    dates: pd.DatetimeIndex,
    l_daily: torch.Tensor,
    f_surface: torch.Tensor,
    prior_l: np.ndarray | None,
    fs_prior: np.ndarray | None,
) -> torch.Tensor:
    months = torch.as_tensor(dates.month.to_numpy().copy(), dtype=torch.float32, device=sim.device)
    wet_mask = ((months >= 4.0) & (months <= 9.0)).to(sim.dtype)
    mse = torch.mean((sim - obs) ** 2)
    mae = torch.mean(torch.abs(sim - obs))
    huber = torch.nn.functional.smooth_l1_loss(sim, obs, beta=0.03)
    log_mse = torch.mean((torch.log1p(sim) - torch.log1p(obs)) ** 2)
    nse_penalty = 1.0 - torch.clamp(nse_torch(obs, sim), min=-10.0, max=1.0)
    obs_mean = torch.mean(obs).clamp_min(1e-6)
    peak_weights = 1.0 + 2.5 * (obs / obs_mean)
    weighted_mse = torch.mean(peak_weights * (sim - obs) ** 2)
    wet_mse = torch.sum((1.0 + wet_mask) * (sim - obs) ** 2) / torch.sum(1.0 + wet_mask)
    diff_loss = torch.mean((torch.diff(sim) - torch.diff(obs)) ** 2)
    bias_penalty = torch.abs(torch.mean(sim - obs))
    reg_smooth = torch.mean(torch.diff(torch.diff(l_daily)) ** 2)
    reg_fs_smooth = torch.mean(torch.diff(torch.diff(f_surface)) ** 2)

    reg_prior = torch.tensor(0.0, device=sim.device)
    if prior_l is not None:
        prior = torch.as_tensor(prior_l, dtype=torch.float32, device=sim.device)
        reg_prior = torch.mean((l_daily / torch.sum(l_daily) - prior / torch.sum(prior)) ** 2)

    reg_fs_prior = torch.tensor(0.0, device=sim.device)
    if fs_prior is not None:
        reg_fs_prior = torch.mean((f_surface - torch.as_tensor(fs_prior, dtype=torch.float32, device=sim.device)) ** 2)

    return (
        0.34 * nse_penalty
        + 0.14 * mse
        + 0.08 * mae
        + 0.08 * huber
        + 0.08 * log_mse
        + 0.14 * weighted_mse
        + 0.07 * wet_mse
        + 0.03 * diff_loss
        + 0.02 * bias_penalty
        + 0.01 * reg_smooth
        + 0.01 * reg_prior
        + 0.01 * reg_fs_smooth
        + 0.01 * reg_fs_prior
    )


def run_sequence(
    data: dict,
    postorder_nodes,
    xy,
    sink_classified: torch.Tensor,
    par: torch.Tensor,
    theta_time: torch.Tensor,
    ml_w: torch.Tensor,
    ml_b: torch.Tensor,
    init_pool_state: InitialPoolState,
    device: str,
    keep_maps: bool = False,
):
    rows, cols = data["landuse"].shape
    time_gen = LearnableDailyTotalTorch(
        cfg=data["cfg"],
        dates=data["dates"],
        rain=data["rain"],
        runoff=data["runoff_for_time"],
        fert_factor=data["fert"],
        year_total=data["year_total"],
        prior_indicator=data.get("prior_L"),
        fs_prior=data.get("fs_prior"),
        device=device,
    )
    l_daily, f_surface, _ = time_gen.forward(theta_time)
    source_maps = build_source_maps_with_ml_torch(
        X=data["X"],
        groups=data["groups"],
        idx_g=data["idx_g"],
        pix_r=data["pix_r"],
        pix_c=data["pix_c"],
        group_ratio=data["group_ratio"],
        landuse_shape=(rows, cols),
        L_daily=l_daily,
        ml_w=ml_w,
        ml_b=ml_b,
        device=device,
    )

    template = torch.mean(torch.stack(source_maps, dim=0).detach(), dim=0) + 1e-6
    template = template / torch.sum(template)
    init_bgc_dis, init_bgc_ads, init_hyd_dis, init_hyd_ads = init_pool_state()
    bgc_dis_pool = init_bgc_dis * template
    bgc_ads_pool = init_bgc_ads * template
    hyd_dis_pool = init_hyd_dis * template
    hyd_ads_pool = init_hyd_ads * template

    raw_load = []
    routed_out = []
    hyd_out = []
    routed_maps = []
    wetness = []
    wet_state = torch.tensor(0.0, dtype=torch.float32, device=device)

    for day in range(len(source_maps)):
        pcp_day = torch.tensor(data["rain"][day], dtype=torch.float32, device=device)
        surface_flow_day = torch.tensor(data["surface_flow"][day], dtype=torch.float32, device=device)
        wet_state = 0.82 * wet_state + surface_flow_day + 0.15 * pcp_day
        (
            day_output,
            out_dis,
            out_ads,
            hyd_scalar,
            out_dis_map,
            out_ads_map,
            bgc_dis_pool,
            bgc_ads_pool,
            hyd_dis_pool,
            hyd_ads_pool,
        ) = run_single_day_with_trace_torch(
            source=source_maps[day],
            par=par,
            f_surface=f_surface[day],
            sink_classified=sink_classified,
            sink=data["landuse"],
            slope=data["slope"],
            dir_up=data["flowdir_up"],
            postorder_nodes=postorder_nodes,
            xy=xy,
            bgc_dis_pool=bgc_dis_pool,
            bgc_ads_pool=bgc_ads_pool,
            hyd_dis_pool=hyd_dis_pool,
            hyd_ads_pool=hyd_ads_pool,
            pcp=pcp_day,
            surface_flow=surface_flow_day,
            outlet_code=data["cfg"]["OUTLET_CODE"],
        )
        raw_load.append(day_output)
        routed_out.append(out_dis + out_ads)
        hyd_out.append(hyd_scalar)
        wetness.append(wet_state)
        if keep_maps:
            routed_maps.append((out_dis_map + out_ads_map).detach().cpu().numpy())

    return {
        "raw_load": torch.stack(raw_load),
        "routed_out": torch.stack(routed_out),
        "hyd_out": torch.stack(hyd_out),
        "wetness": torch.stack(wetness),
        "l_daily": l_daily,
        "f_surface": f_surface,
        "source_maps": source_maps if keep_maps else None,
        "routed_maps": routed_maps if keep_maps else None,
    }


def project_in_bounds(par: torch.Tensor, theta_time: torch.Tensor, ml_w: torch.Tensor, ml_b: torch.Tensor, data: dict) -> None:
    with torch.no_grad():
        par.clamp_(
            min=torch.as_tensor(data["par_lb_all"], dtype=par.dtype, device=par.device),
            max=torch.as_tensor(data["par_ub_all"], dtype=par.dtype, device=par.device),
        )
        theta_time.clamp_(
            min=torch.as_tensor(data["time_lb"], dtype=theta_time.dtype, device=theta_time.device),
            max=torch.as_tensor(data["time_ub"], dtype=theta_time.dtype, device=theta_time.device),
        )
        ml_lb = torch.as_tensor(data["ml_lb"], dtype=ml_w.dtype, device=ml_w.device)
        ml_ub = torch.as_tensor(data["ml_ub"], dtype=ml_w.dtype, device=ml_w.device)
        ml_w.clamp_(min=ml_lb[:-1], max=ml_ub[:-1])
        ml_b.clamp_(min=ml_lb[-1], max=ml_ub[-1])


def save_spatial_map(map_arr: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    im = ax.imshow(map_arr, cmap="YlOrRd")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="TP daily physics-first split calibration")
    parser.add_argument("--max-epochs", type=int, default=260)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-days", type=int, default=73)
    parser.add_argument("--tag", type=str, default="physics_first_tp")
    parser.add_argument("--head-inner-steps", type=int, default=12)
    args = parser.parse_args()

    device = "cpu"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_tp_daily_data()
    export_tp_daily_inputs(
        data,
        ROOT / "tp_daily_forcing_2023.csv",
        ROOT / "tp_daily_obs_2023.csv",
        ROOT / "tp_daily_meta_2023.json",
    )

    n_days = len(data["obs"])
    val_days = min(max(30, int(args.val_days)), n_days - 60)
    n_train = n_days - val_days
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_days)

    core = DailyNPSCore(
        cfg=data["cfg"],
        landuse=data["landuse"],
        slope=data["slope"],
        flowdir_up=data["flowdir_up"],
        X=data["X"],
        groups=data["groups"],
        idx_g=data["idx_g"],
        pix_r=data["pix_r"],
        pix_c=data["pix_c"],
        group_ratio=data["group_ratio"],
        dates=data["dates"],
        rain=data["rain"],
        surface_flow=data["surface_flow"],
        fert=data["fert"],
        runoff_for_time=data["runoff_for_time"],
        year_total=data["year_total"],
        prior_L=data.get("prior_L"),
        fs_prior=data.get("fs_prior"),
    )
    postorder_nodes = core.phys.postorder_nodes
    xy = core.phys.xy
    sink_classified = torch.as_tensor(classify_landuse(data["landuse"]), dtype=torch.float32, device=device)
    obs = torch.tensor(np.asarray(data["obs"], dtype=np.float32).copy(), dtype=torch.float32, device=device)
    runoff = torch.tensor(np.asarray(data["runoff_for_time"], dtype=np.float32).copy(), dtype=torch.float32, device=device)
    rain = torch.tensor(np.asarray(data["rain"], dtype=np.float32).copy(), dtype=torch.float32, device=device)
    doy = torch.as_tensor(data["dates"].dayofyear.to_numpy().copy(), dtype=torch.float32, device=device)

    par = torch.tensor(data["par0"], dtype=torch.float32, device=device, requires_grad=True)
    theta_time = torch.tensor(data["time0"], dtype=torch.float32, device=device, requires_grad=True)
    ml_w = torch.tensor(data["ml_w0"], dtype=torch.float32, device=device, requires_grad=True)
    ml_b = torch.tensor(float(data["ml_b0"]), dtype=torch.float32, device=device, requires_grad=True)
    init_pool_state = InitialPoolState().to(device)
    obs_operator = PhysicalObservationOperator().to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": [par], "lr": 1.5e-3},
            {"params": [theta_time], "lr": 1.0e-3},
            {"params": [ml_w, ml_b], "lr": 5.0e-3},
            {"params": init_pool_state.parameters(), "lr": 8.0e-4},
            {"params": obs_operator.parameters(), "lr": 1.0e-2},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=20, min_lr=1e-5)

    best_score = -np.inf
    best_payload = None
    best_epoch = -1

    tag = args.tag
    summary_path = ROOT / f"tp_split_calibration_summary_{tag}.json"
    pred_path = ROOT / f"tp_split_predictions_{tag}.csv"
    ckpt_path = ROOT / f"best_torch_tp_split_calibration_{tag}.pt"
    plot_path = ROOT / f"tp_split_obs_vs_sim_{tag}.png"
    spatial_plot_path = ROOT / f"tp_spatial_{tag}_bestday.png"
    spatial_json_path = ROOT / f"tp_spatial_{tag}_bestday.json"

    for epoch in range(1, int(args.max_epochs) + 1):
        optimizer.zero_grad()
        seq = run_sequence(data, postorder_nodes, xy, sink_classified, par, theta_time, ml_w, ml_b, init_pool_state, device)
        for _ in range(int(args.head_inner_steps)):
            optimizer.zero_grad()
            sim_head = obs_operator(
                seq["routed_out"].detach(),
                seq["hyd_out"].detach(),
                runoff.detach(),
                rain.detach(),
                seq["wetness"].detach(),
                seq["l_daily"].detach(),
                seq["f_surface"].detach(),
                doy.detach(),
            )
            head_loss = build_loss(
                sim_head[train_idx],
                obs[train_idx],
                data["dates"][train_idx],
                seq["l_daily"][train_idx].detach(),
                seq["f_surface"][train_idx].detach(),
                None,
                None,
            )
            head_loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        sim = obs_operator(seq["routed_out"], seq["hyd_out"], runoff, rain, seq["wetness"], seq["l_daily"], seq["f_surface"], doy)

        train_loss = build_loss(
            sim[train_idx],
            obs[train_idx],
            data["dates"][train_idx],
            seq["l_daily"],
            seq["f_surface"],
            data.get("prior_L"),
            data.get("fs_prior"),
        )
        val_loss = build_loss(
            sim[val_idx],
            obs[val_idx],
            data["dates"][val_idx],
            seq["l_daily"][val_idx],
            seq["f_surface"][val_idx],
            None,
            None,
        )
        loss = 0.8 * train_loss + 0.2 * val_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([par, theta_time, ml_w, ml_b], max_norm=5.0)
        optimizer.step()
        project_in_bounds(par, theta_time, ml_w, ml_b, data)
        scheduler.step(loss.detach())

        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                sim_np = sim.detach().cpu().numpy()
                load_np = seq["raw_load"].detach().cpu().numpy()
                full_metrics = compute_metrics_np(data["obs"], sim_np)
                train_metrics = compute_metrics_np(data["obs"][train_idx], sim_np[train_idx])
                val_metrics = compute_metrics_np(data["obs"][val_idx], sim_np[val_idx])
                load_train_metrics = compute_metrics_np(data["obs"][train_idx], load_np[train_idx])
                score = (
                    2.2 * float(np.nan_to_num(full_metrics.nse, nan=-10.0))
                    + 2.0 * float(np.nan_to_num(full_metrics.r2, nan=-10.0))
                    + 0.6 * float(np.nan_to_num(train_metrics.nse, nan=-10.0))
                    + 0.6 * float(np.nan_to_num(train_metrics.r2, nan=-10.0))
                    + 0.8 * float(np.nan_to_num(val_metrics.nse, nan=-10.0))
                    + 0.8 * float(np.nan_to_num(val_metrics.r2, nan=-10.0))
                    + 0.3 * float(np.nan_to_num(load_train_metrics.nse, nan=-10.0))
                )
                if score > best_score:
                    seq_maps = run_sequence(
                        data, postorder_nodes, xy, sink_classified, par, theta_time, ml_w, ml_b, init_pool_state, device, keep_maps=True
                    )
                    sim_best = obs_operator(
                        seq_maps["routed_out"],
                        seq_maps["hyd_out"],
                        runoff,
                        rain,
                        seq_maps["wetness"],
                        seq_maps["l_daily"],
                        seq_maps["f_surface"],
                        doy,
                    ).detach().cpu().numpy()
                    best_day = int(np.argmax(data["obs"]))
                    best_payload = {
                        "epoch": epoch,
                        "score": score,
                        "par": par.detach().cpu(),
                        "theta_time": theta_time.detach().cpu(),
                        "ml_w": ml_w.detach().cpu(),
                        "ml_b": ml_b.detach().cpu(),
                        "obs_operator": obs_operator.state_dict(),
                        "init_pool_state": init_pool_state.state_dict(),
                        "raw_load": seq_maps["raw_load"].detach().cpu().numpy(),
                        "sim": sim_best,
                        "routed_out": seq_maps["routed_out"].detach().cpu().numpy(),
                        "hyd_out": seq_maps["hyd_out"].detach().cpu().numpy(),
                        "l_daily": seq_maps["l_daily"].detach().cpu().numpy(),
                        "f_surface": seq_maps["f_surface"].detach().cpu().numpy(),
                        "full_metrics": full_metrics.__dict__,
                        "train_metrics": train_metrics.__dict__,
                        "val_metrics": val_metrics.__dict__,
                        "best_day_index": best_day,
                        "best_day_routed_map": seq_maps["routed_maps"][best_day],
                        "best_day_source_map": seq_maps["source_maps"][best_day].detach().cpu().numpy(),
                    }
                    best_score = score
                    best_epoch = epoch
                    torch.save(best_payload, ckpt_path)

                print(
                    f"[epoch {epoch:03d}] loss={float(loss.detach().cpu()):.4f} "
                    f"full NSE={full_metrics.nse:.4f} R2={full_metrics.r2:.4f} "
                    f"train NSE={train_metrics.nse:.4f} R2={train_metrics.r2:.4f} "
                    f"val NSE={val_metrics.nse:.4f} R2={val_metrics.r2:.4f}"
                )

    if best_payload is None:
        raise RuntimeError("Calibration did not produce a checkpoint.")

    summary = {
        "tag": tag,
        "best_epoch": int(best_epoch),
        "score": float(best_score),
        "split": {"total_days": int(n_days), "train_days": int(len(train_idx)), "val_days": int(len(val_idx))},
        "full_metrics": best_payload["full_metrics"],
        "train_metrics": best_payload["train_metrics"],
        "val_metrics": best_payload["val_metrics"],
        "outlet_info": data.get("outlet_info", {}),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(data["dates"]),
            "split": np.where(np.arange(n_days) < n_train, "train", "val"),
            "obs_tp": data["obs"],
            "raw_sim_tp": best_payload["raw_load"],
            "sim_tp": best_payload["sim"],
            "routed_out": best_payload["routed_out"],
            "hyd_out": best_payload["hyd_out"],
            "l_daily": best_payload["l_daily"],
            "f_surface": best_payload["f_surface"],
            "abs_error": np.abs(best_payload["sim"] - data["obs"]),
        }
    )
    pred_df.to_csv(pred_path, index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(pred_df["date"], pred_df["obs_tp"], label="Observed TP", color="#c0392b", linewidth=1.8)
    ax.plot(pred_df["date"], pred_df["sim_tp"], label="Physics-first TP", color="#1f77b4", linewidth=1.6)
    split_date = pred_df["date"].iloc[n_train]
    ax.axvline(split_date, color="gray", linestyle="--", linewidth=1.0)
    ax.text(split_date, ax.get_ylim()[1] * 0.95, "train/val", ha="left", va="top", fontsize=9, color="gray")
    ax.set_title(f"TP Daily Simulation ({tag})")
    ax.set_ylabel("TP")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    best_day = int(best_payload["best_day_index"])
    save_spatial_map(
        best_payload["best_day_routed_map"],
        spatial_plot_path,
        f"Best-day routed TP load ({pd.to_datetime(data['dates'][best_day]).date()})",
    )
    spatial_json_path.write_text(
        json.dumps(
            {
                "date": str(pd.to_datetime(data["dates"][best_day]).date()),
                "obs": float(data["obs"][best_day]),
                "sim": float(best_payload["sim"][best_day]),
                "outlet_info": data.get("outlet_info", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("TP calibration finished")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
