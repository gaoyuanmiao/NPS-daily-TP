from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_components_numpy import DailyNPSCore
from tp_daily_loader import load_tp_daily_data
from train_torch_tp_split_calibration import (
    InitialPoolState,
    PhysicalObservationOperator,
    classify_landuse,
    compute_metrics_np,
    run_sequence,
)


ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "best_torch_tp_split_calibration_physics_first_tp.pt"


@dataclass
class ModelBundle:
    data: dict
    postorder_nodes: list[str]
    xy: dict
    sink_classified: torch.Tensor
    par: torch.Tensor
    theta_time: torch.Tensor
    ml_w: torch.Tensor
    ml_b: torch.Tensor
    init_pool_state: InitialPoolState
    obs_operator: PhysicalObservationOperator


def load_bundle() -> ModelBundle:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    data = load_tp_daily_data()
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
    init_pool_state = InitialPoolState()
    init_pool_state.load_state_dict(ckpt["init_pool_state"])
    obs_operator = PhysicalObservationOperator()
    obs_operator.load_state_dict(ckpt["obs_operator"])
    return ModelBundle(
        data=data,
        postorder_nodes=core.phys.postorder_nodes,
        xy=core.phys.xy,
        sink_classified=torch.as_tensor(classify_landuse(data["landuse"]), dtype=torch.float32),
        par=ckpt["par"].clone().float(),
        theta_time=ckpt["theta_time"].clone().float(),
        ml_w=ckpt["ml_w"].clone().float(),
        ml_b=ckpt["ml_b"].clone().float(),
        init_pool_state=init_pool_state,
        obs_operator=obs_operator,
    )


def simulate(bundle: ModelBundle, par=None, theta_time=None, ml_w=None, ml_b=None, rain=None, runoff=None):
    data = dict(bundle.data)
    if rain is not None:
        data["rain"] = np.asarray(rain, dtype=float).copy()
    if runoff is not None:
        runoff = np.asarray(runoff, dtype=float).copy()
        data["runoff_for_time"] = runoff
        data["surface_flow"] = runoff / (np.nanmax(runoff) + 1e-6)

    par = bundle.par if par is None else par
    theta_time = bundle.theta_time if theta_time is None else theta_time
    ml_w = bundle.ml_w if ml_w is None else ml_w
    ml_b = bundle.ml_b if ml_b is None else ml_b

    seq = run_sequence(
        data,
        bundle.postorder_nodes,
        bundle.xy,
        bundle.sink_classified,
        par,
        theta_time,
        ml_w,
        ml_b,
        bundle.init_pool_state,
        "cpu",
        keep_maps=False,
    )
    runoff_t = torch.as_tensor(np.asarray(data["runoff_for_time"], dtype=np.float32).copy(), dtype=torch.float32)
    rain_t = torch.as_tensor(np.asarray(data["rain"], dtype=np.float32).copy(), dtype=torch.float32)
    doy = torch.as_tensor(data["dates"].dayofyear.to_numpy().copy(), dtype=torch.float32)
    sim = bundle.obs_operator(
        seq["routed_out"],
        seq["hyd_out"],
        runoff_t,
        rain_t,
        seq["wetness"],
        seq["l_daily"],
        seq["f_surface"],
        doy,
    )
    sim_np = sim.detach().cpu().numpy()
    return {
        "sim": sim_np,
        "metrics": compute_metrics_np(data["obs"], sim_np),
        "seq": seq,
    }


def parameter_labels() -> list[str]:
    base_names = [
        "Input TP dissolution coeff",
        "Soil dissolved leaching coeff",
        "Groundwater dissolved leaching coeff",
        "Soil adsorbed leaching coeff",
        "Groundwater adsorbed leaching coeff",
        "BGC dissolved current release",
        "BGC dissolved legacy release",
        "BGC dissolved leaching coeff",
        "BGC adsorbed current release",
        "BGC adsorbed legacy release",
        "BGC adsorbed leaching coeff",
        "HYD dissolved current release",
        "HYD dissolved legacy release",
        "HYD dissolved leaching coeff",
        "HYD adsorbed current release",
        "HYD adsorbed legacy release",
        "HYD adsorbed leaching coeff",
    ]
    labels = []
    for idx in range(57):
        if idx <= 16:
            labels.append(f"A{idx+1:02d} {base_names[idx]}")
        elif idx <= 33:
            labels.append(f"N{idx-16:02d} {base_names[idx-17]}")
        elif idx <= 50:
            labels.append(f"U{idx-33:02d} {base_names[idx-34]}")
        elif idx == 51:
            labels.append("T52 Reserved threshold parameter")
        elif idx == 52:
            labels.append("T53 Reserved threshold parameter")
        elif idx == 53:
            labels.append("T54 Forest attenuation threshold")
        elif idx == 54:
            labels.append("T55 Reserved threshold parameter")
        elif idx == 55:
            labels.append("T56 Grassland attenuation threshold")
        else:
            labels.append("T57 Water attenuation threshold")
    return labels


def top_parameter_indices() -> list[int]:
    return [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 53, 55, 56]


def clipped_normal(rng: np.random.Generator, center: np.ndarray, low: np.ndarray, high: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = center + rng.normal(0.0, scale, size=center.shape)
    return np.clip(out, low, high)


def run_uncertainty(bundle: ModelBundle, n: int = 24) -> dict:
    rng = np.random.default_rng(20260512)
    base = simulate(bundle)
    sims = [base["sim"]]
    p_idx = np.array(top_parameter_indices(), dtype=int)
    par0 = bundle.par.detach().cpu().numpy().copy()
    par_lb = np.asarray(bundle.data["par_lb_all"], dtype=float)
    par_ub = np.asarray(bundle.data["par_ub_all"], dtype=float)
    theta0 = bundle.theta_time.detach().cpu().numpy().copy()
    theta_lb = np.asarray(bundle.data["time_lb"], dtype=float)
    theta_ub = np.asarray(bundle.data["time_ub"], dtype=float)
    ml_w0 = bundle.ml_w.detach().cpu().numpy().copy()
    ml_lb = np.asarray(bundle.data["ml_lb"], dtype=float)
    ml_ub = np.asarray(bundle.data["ml_ub"], dtype=float)
    ml_b0 = float(bundle.ml_b.detach().cpu())

    for _ in range(n):
        par_new = par0.copy()
        p_scale = 0.06 * (par_ub[p_idx] - par_lb[p_idx])
        par_new[p_idx] = clipped_normal(rng, par0[p_idx], par_lb[p_idx], par_ub[p_idx], p_scale)

        theta_new = clipped_normal(rng, theta0, theta_lb, theta_ub, 0.05 * (theta_ub - theta_lb))
        ml_w_new = clipped_normal(rng, ml_w0, ml_lb[:-1], ml_ub[:-1], 0.08 * (ml_ub[:-1] - ml_lb[:-1]))
        ml_b_new = float(np.clip(ml_b0 + rng.normal(0.0, 0.08 * (ml_ub[-1] - ml_lb[-1])), ml_lb[-1], ml_ub[-1]))

        result = simulate(
            bundle,
            par=torch.as_tensor(par_new, dtype=torch.float32),
            theta_time=torch.as_tensor(theta_new, dtype=torch.float32),
            ml_w=torch.as_tensor(ml_w_new, dtype=torch.float32),
            ml_b=torch.tensor(ml_b_new, dtype=torch.float32),
        )
        sims.append(result["sim"])

    arr = np.vstack(sims)
    q05 = np.quantile(arr, 0.05, axis=0)
    q50 = np.quantile(arr, 0.50, axis=0)
    q95 = np.quantile(arr, 0.95, axis=0)
    uncertainty_width = float(np.mean(q95 - q05))
    coverage = float(np.mean((np.asarray(bundle.data["obs"]) >= q05) & (np.asarray(bundle.data["obs"]) <= q95)))
    return {"q05": q05, "q50": q50, "q95": q95, "width": uncertainty_width, "coverage": coverage, "n": len(sims)}


def run_stability(bundle: ModelBundle, levels=(0.0, 0.05, 0.10, 0.15), reps: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(20260513)
    p_idx = np.array(top_parameter_indices(), dtype=int)
    par0 = bundle.par.detach().cpu().numpy().copy()
    par_lb = np.asarray(bundle.data["par_lb_all"], dtype=float)
    par_ub = np.asarray(bundle.data["par_ub_all"], dtype=float)
    theta0 = bundle.theta_time.detach().cpu().numpy().copy()
    theta_lb = np.asarray(bundle.data["time_lb"], dtype=float)
    theta_ub = np.asarray(bundle.data["time_ub"], dtype=float)
    rows = []

    for level in levels:
        for rep in range(reps):
            if level == 0.0:
                result = simulate(bundle)
            else:
                par_new = par0.copy()
                par_range = par_ub[p_idx] - par_lb[p_idx]
                par_new[p_idx] = np.clip(par0[p_idx] + rng.normal(0.0, level * par_range, size=len(p_idx)), par_lb[p_idx], par_ub[p_idx])
                theta_new = np.clip(theta0 + rng.normal(0.0, level * (theta_ub - theta_lb), size=theta0.shape), theta_lb, theta_ub)
                rain_new = np.clip(np.asarray(bundle.data["rain"], dtype=float) * (1.0 + rng.normal(0.0, level, size=len(bundle.data["rain"]))), 0.0, None)
                runoff_new = np.clip(np.asarray(bundle.data["runoff_for_time"], dtype=float) * (1.0 + rng.normal(0.0, level, size=len(bundle.data["runoff_for_time"]))), 0.0, None)
                result = simulate(
                    bundle,
                    par=torch.as_tensor(par_new, dtype=torch.float32),
                    theta_time=torch.as_tensor(theta_new, dtype=torch.float32),
                    rain=rain_new,
                    runoff=runoff_new,
                )
            rows.append(
                {
                    "perturbation": level,
                    "rep": rep,
                    "nse": result["metrics"].nse,
                    "r2": result["metrics"].r2,
                    "rmse": result["metrics"].rmse,
                }
            )
    return pd.DataFrame(rows)


def run_parameter_sensitivity(bundle: ModelBundle, rel_step: float = 0.10, candidate_idx: list[int] | None = None) -> pd.DataFrame:
    labels = parameter_labels()
    par0 = bundle.par.detach().cpu().numpy().copy()
    lb = np.asarray(bundle.data["par_lb_all"], dtype=float)
    ub = np.asarray(bundle.data["par_ub_all"], dtype=float)
    base_nse = simulate(bundle)["metrics"].nse
    rows = []
    use_idx = top_parameter_indices() if candidate_idx is None else candidate_idx

    for idx in use_idx:
        center = par0[idx]
        span = ub[idx] - lb[idx]
        step = max(rel_step * abs(center), 0.03 * span)
        low = max(lb[idx], center - step)
        high = min(ub[idx], center + step)
        if np.isclose(low, high):
            rows.append({"index": idx, "label": labels[idx], "nse_sensitivity": 0.0})
            continue
        deltas = []
        for trial in (low, high):
            par_new = par0.copy()
            par_new[idx] = trial
            metric = simulate(bundle, par=torch.as_tensor(par_new, dtype=torch.float32))["metrics"].nse
            rel = abs((trial - center) / (abs(center) + 1e-6))
            deltas.append(abs(metric - base_nse) / max(rel, 1e-6))
        rows.append({"index": idx, "label": labels[idx], "nse_sensitivity": float(np.mean(deltas))})
    df = pd.DataFrame(rows).sort_values("nse_sensitivity", ascending=False).reset_index(drop=True)
    return df


def plot_uncertainty(bundle: ModelBundle, out: dict, path: Path) -> None:
    dates = pd.to_datetime(bundle.data["dates"])
    obs = np.asarray(bundle.data["obs"], dtype=float)
    base = simulate(bundle)["sim"]
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.fill_between(dates, out["q05"], out["q95"], color="#9ecae1", alpha=0.45, label="90% uncertainty band")
    ax.plot(dates, out["q50"], color="#2c7fb8", linewidth=1.7, label="Ensemble median")
    ax.plot(dates, base, color="#08306b", linewidth=1.4, alpha=0.95, label="Best simulation")
    ax.plot(dates, obs, color="#cb181d", linewidth=1.2, label="Observed TP")
    ax.set_ylabel("TP")
    ax.set_title("Prediction Uncertainty of Daily TP Simulation")
    ax.grid(alpha=0.20)
    ax.legend(ncol=4, frameon=False, loc="upper right")
    ax.text(
        0.01,
        0.95,
        f"Mean band width = {out['width']:.4f}\nCoverage = {out['coverage']:.2%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_stability(df: pd.DataFrame, path: Path) -> None:
    levels = sorted(df["perturbation"].unique())
    data = [df.loc[df["perturbation"] == level, "nse"].to_numpy() for level in levels]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    colors = ["#d0d1e6", "#a6bddb", "#74a9cf", "#2b8cbe", "#045a8d"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set(facecolor=color, alpha=0.8, edgecolor="#1f1f1f")
    for median in bp["medians"]:
        median.set(color="#8c2d04", linewidth=1.6)
    ax.axhline(0.60, color="#cb181d", linestyle="--", linewidth=1.0, label="Target NSE = 0.60")
    ax.set_xticks(np.arange(1, len(levels) + 1))
    ax.set_xticklabels([f"{int(level * 100)}%" for level in levels])
    ax.set_xlabel("Combined perturbation level")
    ax.set_ylabel("Full-period NSE")
    ax.set_title("Model Stability Under Input and Parameter Perturbations")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_sensitivity(df: pd.DataFrame, path: Path, topn: int = 10) -> None:
    use = df.head(topn).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.barh(use["label"], use["nse_sensitivity"], color="#3182bd", alpha=0.9)
    ax.set_xlabel("Local NSE sensitivity index")
    ax.set_title("Key Parameter Identification by Local Sensitivity")
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="TP model uncertainty, stability, and key-parameter diagnostics")
    parser.add_argument("--uncertainty-runs", type=int, default=18)
    parser.add_argument("--stability-reps", type=int, default=6)
    parser.add_argument("--sensitivity-count", type=int, default=10)
    args = parser.parse_args()

    bundle = load_bundle()
    candidate_idx = top_parameter_indices()[: int(args.sensitivity_count)]
    uncertainty = run_uncertainty(bundle, n=int(args.uncertainty_runs))
    stability = run_stability(bundle, levels=(0.0, 0.05, 0.10, 0.15), reps=int(args.stability_reps))
    sensitivity = run_parameter_sensitivity(bundle, rel_step=0.10, candidate_idx=candidate_idx)

    uncertainty_path = ROOT / "tp_uncertainty_band.png"
    stability_path = ROOT / "tp_stability_boxplot.png"
    sensitivity_path = ROOT / "tp_key_parameter_sensitivity.png"
    plot_uncertainty(bundle, uncertainty, uncertainty_path)
    plot_stability(stability, stability_path)
    plot_sensitivity(sensitivity, sensitivity_path)

    summary = {
        "uncertainty": {
            "ensemble_size": int(uncertainty["n"]),
            "mean_band_width": float(uncertainty["width"]),
            "coverage_90": float(uncertainty["coverage"]),
        },
        "stability": [
            {
                "perturbation": float(level),
                "nse_mean": float(group["nse"].mean()),
                "nse_std": float(group["nse"].std(ddof=0)),
                "nse_min": float(group["nse"].min()),
                "nse_max": float(group["nse"].max()),
                "r2_mean": float(group["r2"].mean()),
                "rmse_mean": float(group["rmse"].mean()),
            }
            for level, group in stability.groupby("perturbation")
        ],
        "key_parameters": sensitivity.head(10).to_dict(orient="records"),
        "figures": {
            "uncertainty": str(uncertainty_path),
            "stability": str(stability_path),
            "sensitivity": str(sensitivity_path),
        },
    }
    (ROOT / "tp_model_diagnostics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
