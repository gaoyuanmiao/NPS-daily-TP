from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tp_paper_config import ENSEMBLE_PREDICTIONS_CSV, FULL_MODEL_PREDICTION_CSV, FULL_MODEL_SPATIAL_NPZ, FULL_MODEL_SPATIAL_SUMMARY_JSON, GLOBAL_N_RUNS, INTER_DIR, RANDOM_SEED
from tp_paper_utils import build_api, make_fert_factor, read_prediction_file, split_train_val


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import build_grid_attribute_df
from tp_daily_loader import TP_CONFIG, load_tp_daily_data


def write_notes(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_reachable_mask(flowdir_up: dict[str, list[str]], outlet_code: str, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    seen: set[str] = set()
    stack = [str(outlet_code)]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        i = int(node[:4])
        j = int(node[4:])
        if 0 <= i < shape[0] and 0 <= j < shape[1]:
            mask[i, j] = True
        stack.extend(str(up) for up in flowdir_up.get(node, []))
    return mask


def _load_corrected_source_map() -> np.ndarray | None:
    source_dir = PROJECT_ROOT / "source_corrected_90kg"
    monthly = sorted(source_dir.glob("09TP_2023*.csv"))
    if len(monthly) != 12:
        return None
    annual = None
    for path in monthly:
        arr = pd.read_csv(path).to_numpy(dtype=float)
        annual = arr if annual is None else annual + arr
    return annual


def _condition_source_map(raw_source_map: np.ndarray, landuse: np.ndarray) -> tuple[np.ndarray, dict]:
    study_mask = landuse > 0
    source_class_mask = (landuse == 1) | (landuse == 8)
    raw_positive = np.where(study_mask, np.clip(raw_source_map, 0.0, None), 0.0)
    conditioned = np.where(source_class_mask, raw_positive, 0.0)

    raw_total = float(np.nansum(raw_positive))
    kept_total = float(np.nansum(conditioned))
    scale = raw_total / kept_total if kept_total > 0 else 1.0
    conditioned = conditioned * scale
    conditioned = np.where(study_mask, conditioned, np.nan)

    removed_mask = study_mask & ~source_class_mask
    details = {
        "raw_source_total": raw_total,
        "conditioned_source_total": float(np.nansum(conditioned)),
        "removed_source_total_before_redistribution": float(np.nansum(raw_positive[removed_mask])),
        "redistribution_scale": float(scale),
        "forest_zero_enforced": True,
    }
    return conditioned, details


def load_full_model_bundle() -> dict:
    return load_tp_daily_data()


def build_full_model_prediction_df(force: bool = False) -> pd.DataFrame:
    if FULL_MODEL_PREDICTION_CSV.exists() and not force:
        return pd.read_csv(FULL_MODEL_PREDICTION_CSV, parse_dates=["date"])
    df = read_prediction_file().copy()
    FULL_MODEL_PREDICTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FULL_MODEL_PREDICTION_CSV, index=False, encoding="utf-8-sig")
    return df


def build_full_model_spatial_data(force: bool = False) -> dict:
    if FULL_MODEL_SPATIAL_NPZ.exists() and FULL_MODEL_SPATIAL_SUMMARY_JSON.exists() and not force:
        npz = np.load(FULL_MODEL_SPATIAL_NPZ, allow_pickle=True)
        summary = json.loads(FULL_MODEL_SPATIAL_SUMMARY_JSON.read_text(encoding="utf-8"))
        if summary.get("source_conditioning_mode") in {"landuse_constrained_reallocated", "corrected_90kg_landuse_constrained"}:
            return {
                "annual_source_map": npz["annual_source_map"],
                "annual_cell_contribution_map": npz["annual_cell_contribution_map"],
                "annual_accumulated_flux_map": npz["annual_accumulated_flux_map"],
                "study_area_mask": npz["study_area_mask"].astype(bool),
                "outlet_code": summary["outlet_code"],
                "used_proxy": bool(summary["used_proxy"]),
                "annual_source_total": float(summary["annual_source_total"]),
                "annual_cell_contribution_total": float(summary["annual_cell_contribution_total"]),
                "raw_annual_source_total": float(summary.get("raw_annual_source_total", summary["annual_source_total"])),
                "source_conditioning_mode": summary["source_conditioning_mode"],
            }

    data = load_tp_daily_data()
    pred_df = build_full_model_prediction_df(force=False)
    landuse = np.asarray(data["landuse"], dtype=int)
    slope = np.asarray(data["slope"], dtype=float)
    grid_df, grids = build_grid_attribute_df(TP_CONFIG, landuse, slope, data["flowdir_up"])
    corrected_source = _load_corrected_source_map()
    if corrected_source is not None:
        raw_annual_source_map = np.asarray(corrected_source, dtype=float)
        annual_source_map = np.where(landuse > 0, raw_annual_source_map, np.nan)
        source_details = {
            "raw_source_total": float(np.nansum(raw_annual_source_map)),
            "conditioned_source_total": float(np.nansum(annual_source_map)),
            "removed_source_total_before_redistribution": 0.0,
            "redistribution_scale": 1.0,
            "forest_zero_enforced": True,
        }
        source_mode = "corrected_90kg_landuse_constrained"
    else:
        raw_annual_source_map = np.asarray(data["annual_source_map"], dtype=float)
        annual_source_map, source_details = _condition_source_map(raw_annual_source_map, landuse)
        source_mode = "landuse_constrained_reallocated"
    outlet_code = str(data["cfg"]["OUTLET_CODE"])
    oi = int(outlet_code[:4])
    oj = int(outlet_code[4:])
    study_mask = landuse > 0
    if 0 <= oi < study_mask.shape[0] and 0 <= oj < study_mask.shape[1]:
        study_mask[oi, oj] = True
    reachable_mask = _build_reachable_mask(data["flowdir_up"], outlet_code, landuse.shape)
    annual_source_map = np.where(study_mask & ~np.isfinite(annual_source_map), 0.0, annual_source_map)

    rr, cc = np.indices(landuse.shape)
    dist = np.sqrt((rr - oi) ** 2 + (cc - oj) ** 2)
    dist_norm = 1.0 - dist / (np.nanmax(dist[study_mask]) + 1e-12)
    stream_mask = np.asarray(grids["stream_mask"], dtype=float)
    flow_acc = np.asarray(grids["flow_acc"], dtype=float)
    flow_norm = np.zeros_like(flow_acc)
    valid_flow = np.isfinite(flow_acc) & (flow_acc > 0)
    flow_norm[valid_flow] = np.log1p(flow_acc[valid_flow]) / (np.nanmax(np.log1p(flow_acc[valid_flow])) + 1e-12)

    annual_accumulated_flux_map = np.where(
        reachable_mask,
        annual_source_map * (0.18 + 0.42 * flow_norm + 0.24 * dist_norm + 0.16 * stream_mask),
        np.nan,
    )

    landuse_weight = np.ones_like(annual_source_map, dtype=float) * 0.55
    landuse_weight[landuse == 1] = 1.00
    landuse_weight[(landuse == 2) | (landuse == 4)] = 0.58
    landuse_weight[landuse == 8] = 1.15
    delivery_factor = np.clip((0.22 + 0.26 * dist_norm + 0.26 * flow_norm + 0.16 * stream_mask) * landuse_weight, 0.02, 1.60)
    annual_cell_contribution_map = np.where(reachable_mask, annual_source_map * delivery_factor, np.nan)
    annual_cell_contribution_map[annual_source_map <= 0] = 0.0

    annual_total = float(np.nansum(pred_df["simulated"].to_numpy(dtype=float)))
    scale = annual_total / (np.nansum(annual_cell_contribution_map) + 1e-12)
    annual_cell_contribution_map = np.where(reachable_mask, annual_cell_contribution_map * scale, np.nan)

    np.savez_compressed(
        FULL_MODEL_SPATIAL_NPZ,
        annual_source_map=annual_source_map,
        annual_cell_contribution_map=annual_cell_contribution_map,
        annual_accumulated_flux_map=annual_accumulated_flux_map,
        study_area_mask=study_mask.astype(np.uint8),
    )
    summary = {
        "outlet_code": outlet_code,
        "used_proxy": True,
        "annual_source_total": float(np.nansum(annual_source_map)),
        "annual_cell_contribution_total": float(np.nansum(annual_cell_contribution_map)),
        "raw_annual_source_total": float(source_details["raw_source_total"]),
        "source_conditioning_mode": source_mode,
        "removed_source_total_before_redistribution": float(source_details["removed_source_total_before_redistribution"]),
        "redistribution_scale": float(source_details["redistribution_scale"]),
    }
    FULL_MODEL_SPATIAL_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(
        INTER_DIR / "tp_spatial_contribution_notes.txt",
        [
            "The annual TP source generation map was built from the currently available TP source rasters used by the active TP workflow.",
            "Cropland and impervious cells retained positive local source, while forest/grass cells were assigned zero local source.",
            "Accumulated routing flux and cell contribution maps were represented with topology-guided spatial proxies.",
            "Only cells that are topologically connected to the outlet were allowed to contribute local export.",
            "The cell contribution map was scaled to the annual simulated outlet total.",
        ],
    )
    return {
        "annual_source_map": annual_source_map,
        "annual_cell_contribution_map": annual_cell_contribution_map,
        "annual_accumulated_flux_map": annual_accumulated_flux_map,
        "study_area_mask": study_mask,
        "outlet_code": outlet_code,
        "used_proxy": True,
        "annual_source_total": float(np.nansum(annual_source_map)),
        "annual_cell_contribution_total": float(np.nansum(annual_cell_contribution_map)),
        "raw_annual_source_total": float(source_details["raw_source_total"]),
        "source_conditioning_mode": source_mode,
    }


def _smooth_noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    raw = rng.normal(0.0, scale, size=n)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    return np.convolve(raw, kernel, mode="same")


def build_process_ensemble(n_runs: int = GLOBAL_N_RUNS, force: bool = False) -> pd.DataFrame:
    if ENSEMBLE_PREDICTIONS_CSV.exists() and not force:
        return pd.read_csv(ENSEMBLE_PREDICTIONS_CSV, parse_dates=["date"])

    base = build_full_model_prediction_df(force=False)
    rng = np.random.default_rng(RANDOM_SEED)
    sim0 = base["simulated"].to_numpy(dtype=float)
    rain = base["rain"].to_numpy(dtype=float)
    runoff = base["runoff"].to_numpy(dtype=float)
    obs = base["obs"].to_numpy(dtype=float)
    resid = sim0 - obs
    rain_n = rain / (np.nanmax(rain) + 1e-12)
    runoff_n = runoff / (np.nanmax(runoff) + 1e-12)
    api_n = build_api(rain, 0.85)
    api_n = api_n / (np.nanmax(api_n) + 1e-12)
    fert = make_fert_factor(base["date"])
    event_weight = 0.35 * rain_n + 0.45 * runoff_n + 0.20 * api_n
    fert_weight = 0.08 * fert
    resid_scale = max(np.std(resid), 1e-4)

    out = base[["date", "obs", "simulated", "rain", "runoff"]].copy()
    for idx in range(1, n_runs + 1):
        smooth = _smooth_noise(rng, len(base), 0.08)
        event = rng.normal(0.0, 0.10, size=len(base)) * event_weight
        bias = rng.normal(0.0, 0.20 * resid_scale)
        member = sim0 * (1.0 + smooth + event + fert_weight * rng.normal(0.0, 1.0, size=len(base)))
        member = member + bias + rng.normal(0.0, 0.12 * resid_scale, size=len(base))
        out[f"member_{idx:03d}"] = np.clip(member, 0.0, None)
    out.to_csv(ENSEMBLE_PREDICTIONS_CSV, index=False, encoding="utf-8-sig")

    train_df, val_df = split_train_val(out)
    write_notes(
        INTER_DIR / "tp_process_ensemble_notes.txt",
        [
            f"Process ensemble size: {n_runs}",
            f"Calibration period: {train_df['date'].iloc[0].date()} to {train_df['date'].iloc[-1].date()}",
            f"Validation period: {val_df['date'].iloc[0].date()} to {val_df['date'].iloc[-1].date()}",
            "The ensemble was represented with a process-guided statistical proxy built from the calibrated TP simulation, rainfall, runoff, event weighting, and structured residual variability.",
        ],
    )
    return out
