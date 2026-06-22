from __future__ import annotations

import json
import pickle
from pathlib import Path
from heapq import heappop, heappush

import numpy as np
import pandas as pd

from data_loader import build_grid_attribute_df, landuse_to_group


ROOT = Path(__file__).resolve().parent
TN_PROJECT_ROOT = ROOT.parent / "TN_daily_model_uniform_from_v9_clean_20260420"
DEFAULT_TP_SOURCE_DIR = ROOT / "source_corrected_90kg"
LEGACY_TP_SOURCE_DIR = ROOT / "source"


TP_CONFIG = {
    "LANDUSE_CSV": ROOT / "input" / "landuse" / "mini_land_use_data.csv",
    "SLOPE_CSV": ROOT / "input" / "slope" / "slope_data1.csv",
    "FLOWDIRUP_PKL": ROOT / "input" / "flow" / "flowdirup.pkl",
    "DAILY_DATA_CSV": ROOT / "input" / "forcing" / "daily_data.csv",
    "OBS_DAILY_CSV": ROOT / "input" / "forcing" / "obs.csv",
    "TP_SOURCE_DIR": DEFAULT_TP_SOURCE_DIR,
    "YEAR": 2023,
    "OUTLET_CODE": "00940067",
    "STREAM_PERCENTILE": 90,
    "DIST_TO_STREAM_IN_METERS": False,
    "CELL_SIZE_M": 30.0,
    "FEAT_COLS": ("slope_deg_z", "log_flow_acc_z", "dist_to_stream_z", "annual_source_z"),
    "FERT_MONTHS": [3, 5, 7],
    "SPRING_EQUINOX_MONTH": 3,
    "SPRING_EQUINOX_DAY": 20,
    "SPRING_EQUINOX_WINDOW": 7,
    "SURFACE_FLOW_MODE": "runoff_fraction",
    "ML_W_INIT": [0.2, 0.9, -0.5, 1.6],
    "ML_B_INIT": 0.0,
    "ML_W_LOWER": [-4.0, -4.0, -4.0, -4.0],
    "ML_W_UPPER": [4.0, 4.0, 4.0, 4.0],
    "ML_B_LOWER": -4.0,
    "ML_B_UPPER": 4.0,
    "TIME_INIT": [
        1.0, 5.0, 0.85, -1.0, 0.3, 0.2, 0.1, 0.05, -1.0, 0.6, 0.3, 0.2, 0.8, 1.0,
        -1.0, 0.6, 0.8, 0.4, 0.6, 0.2, 0.2
    ],
    "TIME_LB": [
        0.2, 0.0, 0.0, -5.0, -3.0, -3.0, -3.0, -3.0, -8.0, -3.0, -3.0, -3.0, -3.0, 0.05,
        -8.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0
    ],
    "TIME_UB": [
        5.0, 30.0, 0.99, 5.0, 3.0, 3.0, 3.0, 3.0, 5.0, 3.0, 3.0, 3.0, 5.0, 10.0,
        8.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
    ],
    "CALIBRATE_PHYS": True,
    "CALIB_PAR_INDEXES": list(range(57)),
}


def _read_grid(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy()


def _load_tp_monthly_sources(source_dir: Path, year: int) -> tuple[dict[str, np.ndarray], pd.Series]:
    monthly_maps: dict[str, np.ndarray] = {}
    totals: dict[str, float] = {}
    for month in range(1, 13):
        key = f"{year}{month:02d}"
        path = source_dir / f"09TP_{key}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing TP source raster: {path}")
        arr = pd.read_csv(path).to_numpy(dtype=float)
        monthly_maps[key] = arr
        totals[key] = float(np.nansum(arr))
    return monthly_maps, pd.Series(totals)


def _load_obs_csv(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["TP"] = pd.to_numeric(df["TP"], errors="coerce")
    df["TN"] = pd.to_numeric(df["TN"], errors="coerce")
    df = df[df["date"].dt.year == year].copy()
    return df.sort_values("date").reset_index(drop=True)


def _load_forcing_csv(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rain"] = pd.to_numeric(df["rain"], errors="coerce")
    df["runoff"] = pd.to_numeric(df["runoff"], errors="coerce")
    df = df[df["date"].dt.year == year].copy()
    df["rain"] = df["rain"].fillna(0.0).clip(lower=0.0)
    df["runoff"] = df["runoff"].fillna(0.0).clip(lower=0.0)
    return df.sort_values("date").reset_index(drop=True)


def _build_fert_factor(dates: pd.Series, cfg: dict) -> np.ndarray:
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    df["month"] = df["date"].dt.month
    df["f_month"] = df["month"].isin(cfg["FERT_MONTHS"]).astype(float)
    spring_dates = pd.to_datetime(
        {
            "year": df["date"].dt.year,
            "month": cfg["SPRING_EQUINOX_MONTH"],
            "day": cfg["SPRING_EQUINOX_DAY"],
        }
    )
    df["days_from_se"] = (df["date"] - spring_dates).dt.days
    df["f_spring"] = (df["days_from_se"].abs() <= cfg["SPRING_EQUINOX_WINDOW"]).astype(float)
    df["fert"] = (df["f_month"] + df["f_spring"]).clip(upper=1.0)
    return df["fert"].rolling(window=5, center=True, min_periods=1).max().to_numpy(dtype=float)


def _compute_group_ratio_from_map(annual_source_map: np.ndarray, landuse: np.ndarray) -> dict[int, float]:
    ratio = {1: 0.0, 2: 0.0, 3: 0.0}
    lu_group = np.vectorize(landuse_to_group)(landuse)
    for group in ratio:
        ratio[group] = float(np.nansum(np.where(lu_group == group, annual_source_map, 0.0)))
    total = sum(ratio.values())
    if total <= 0.0:
        return ratio
    return {k: v / total for k, v in ratio.items()}


def _build_monthly_daily_prior(dates: pd.DatetimeIndex, monthly_totals: pd.Series) -> np.ndarray:
    prior = np.zeros(len(dates), dtype=float)
    for idx, date in enumerate(dates):
        key = f"{date.year}{date.month:02d}"
        prior[idx] = float(monthly_totals.get(key, 0.0)) / float(pd.Period(date, freq="M").days_in_month)
    total = prior.sum()
    if total <= 0.0:
        return np.full(len(dates), 1.0 / max(len(dates), 1), dtype=float)
    return prior / total


def _load_tn_forced_terminals() -> list[str] | None:
    candidate = TN_PROJECT_ROOT / "tn_physical_target_over_06_candidate.json"
    if not candidate.exists():
        return None
    try:
        obj = json.loads(candidate.read_text(encoding="utf-8"))
        topo = obj.get("topology", {})
        terms = topo.get("forced_outlet_terminals", [])
        return [str(t) for t in terms] if terms else None
    except Exception:
        return None


def _cell_code(i: int, j: int) -> str:
    return f"{int(i):04d}{int(j):04d}"


def _decode_cell_code(code: str) -> tuple[int, int]:
    return int(str(code)[:4]), int(str(code)[4:])


def filter_flowdir_to_study_area(flowdir_up: dict, study_area_mask: np.ndarray) -> dict:
    rows, cols = study_area_mask.shape
    filtered = {}
    for dn, ups in flowdir_up.items():
        i, j = _decode_cell_code(str(dn))
        if not (0 <= i < rows and 0 <= j < cols and study_area_mask[i, j]):
            continue
        kept = []
        for up in ups:
            ui, uj = _decode_cell_code(str(up))
            if 0 <= ui < rows and 0 <= uj < cols and study_area_mask[ui, uj]:
                kept.append(str(up))
        filtered[str(dn)] = kept
    return filtered


def _astar_path_to_connected(start_code: str, connected_mask: np.ndarray, study_area_mask: np.ndarray, outlet_code: str):
    rows, cols = study_area_mask.shape
    si, sj = _decode_cell_code(start_code)
    if not (0 <= si < rows and 0 <= sj < cols and study_area_mask[si, sj]):
        return None
    oi, oj = _decode_cell_code(outlet_code)
    outlet_dist = lambda i, j: float(np.hypot(i - oi, j - oj))
    moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    pq = []
    heappush(pq, (0.35 * outlet_dist(si, sj), 0.0, (si, sj)))
    parent = {(si, sj): None}
    best = {(si, sj): 0.0}
    goal = None

    while pq:
        _, g, (i, j) = heappop(pq)
        if connected_mask[i, j] and (i, j) != (si, sj):
            goal = (i, j)
            break
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if not (0 <= ni < rows and 0 <= nj < cols and study_area_mask[ni, nj]):
                continue
            step = 1.4142 if di and dj else 1.0
            trend_penalty = max(0.0, outlet_dist(ni, nj) - outlet_dist(i, j)) * 0.35
            ng = g + step + trend_penalty
            key = (ni, nj)
            if ng < best.get(key, 1e18):
                best[key] = ng
                parent[key] = (i, j)
                heuristic = 0.0 if connected_mask[ni, nj] else 0.35 * outlet_dist(ni, nj)
                heappush(pq, (ng + heuristic, ng, key))

    if goal is None:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return [_cell_code(i, j) for i, j in path[::-1]]


def augment_flowdir_to_outlet(flowdir_up: dict, outlet_code: str, rows: int, cols: int, study_area_mask: np.ndarray | None = None, forced_terminals=None):
    oi, oj = _decode_cell_code(outlet_code)
    if not (0 <= oi < rows and 0 <= oj < cols):
        raise ValueError(f"OUTLET_CODE {outlet_code} out of bounds for grid {rows}x{cols}")

    all_nodes = set(flowdir_up.keys())
    upstream_nodes = set()
    for ups in flowdir_up.values():
        upstream_nodes.update(ups)
        all_nodes.update(ups)

    if study_area_mask is None:
        study_area_mask = np.ones((rows, cols), dtype=bool)

    if outlet_code in all_nodes:
        return flowdir_up, {"outlet_was_missing": False, "stitched_terminal_count": 0, "synthetic_nodes": []}

    terminal_nodes = sorted([node for node in flowdir_up.keys() if node not in upstream_nodes])
    augmented = {str(node): [str(up) for up in ups] for node, ups in flowdir_up.items()}
    augmented.setdefault(outlet_code, [])
    connected_nodes = set(all_nodes)
    synthetic_nodes = set()
    synthetic_edges = 0

    def upstream_size(root_code: str) -> int:
        seen = set()
        stack = [root_code]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(flowdir_up.get(node, []))
        return len(seen)

    terminal_meta = []
    for terminal in terminal_nodes:
        ti, tj = _decode_cell_code(terminal)
        dist = float(np.hypot(ti - oi, tj - oj))
        size = upstream_size(terminal)
        score = size / ((dist + 1.0) ** 1.5)
        terminal_meta.append((score, dist, size, terminal))

    if forced_terminals:
        selected_terminals = [str(t) for t in forced_terminals]
    else:
        terminal_meta.sort(reverse=True)
        best_score = terminal_meta[0][0] if terminal_meta else 0.0
        selected_terminals = [
            terminal for score, dist, size, terminal in terminal_meta
            if score >= max(1.0, 0.35 * best_score)
        ][:5]
        if not selected_terminals and terminal_meta:
            selected_terminals = [terminal_meta[0][3]]

    connected_mask = np.zeros((rows, cols), dtype=bool)
    connected_mask[oi, oj] = True
    ordered_terminals = sorted(
        selected_terminals,
        key=lambda code: np.hypot(_decode_cell_code(code)[0] - oi, _decode_cell_code(code)[1] - oj),
        reverse=True,
    )
    for terminal in ordered_terminals:
        path = _astar_path_to_connected(terminal, connected_mask, study_area_mask, outlet_code)
        if not path:
            continue
        for idx in range(1, len(path)):
            up = path[idx - 1]
            dn = path[idx]
            augmented.setdefault(dn, [])
            if up not in augmented[dn]:
                augmented[dn].append(up)
                synthetic_edges += 1
            if dn not in connected_nodes:
                synthetic_nodes.add(dn)
                connected_nodes.add(dn)
            if up not in connected_nodes:
                synthetic_nodes.add(up)
                connected_nodes.add(up)
            di, dj = _decode_cell_code(dn)
            ui, uj = _decode_cell_code(up)
            connected_mask[di, dj] = True
            connected_mask[ui, uj] = True

    return augmented, {
        "outlet_was_missing": True,
        "stitched_terminal_count": len(terminal_nodes),
        "selected_terminal_count": len(selected_terminals),
        "selected_terminals": selected_terminals,
        "synthetic_node_count": len(synthetic_nodes),
        "synthetic_edge_count": synthetic_edges,
        "synthetic_nodes": sorted(synthetic_nodes),
    }


def _resolve_valid_outlet(cfg: dict, grid_df: pd.DataFrame) -> tuple[str, dict]:
    nominal = str(cfg["OUTLET_CODE"])
    ni, nj = int(nominal[:4]), int(nominal[4:])
    work = grid_df.copy()
    work["dist_nominal"] = (work["row"] - ni).abs() + (work["col"] - nj).abs()
    work["score"] = work["flow_acc"].astype(float) / (work["dist_nominal"].astype(float) + 1.0)

    stream = work[work["is_stream"] == 1].copy()
    if not stream.empty:
        best = stream.sort_values(["dist_nominal", "flow_acc"], ascending=[True, False]).iloc[0]
        strongest = stream.sort_values(["score", "flow_acc"], ascending=[False, False]).iloc[0]
        chosen = strongest if float(strongest["flow_acc"]) >= 1.4 * float(best["flow_acc"]) else best
    else:
        chosen = work.sort_values(["dist_nominal", "flow_acc"], ascending=[True, False]).iloc[0]

    resolved = str(chosen["pixel_id"])
    info = {
        "nominal_outlet_code": nominal,
        "resolved_outlet_code": resolved,
        "nominal_rowcol": [int(ni), int(nj)],
        "resolved_rowcol": [int(chosen["row"]), int(chosen["col"])],
        "resolved_flow_acc": float(chosen["flow_acc"]),
        "resolved_distance_to_nominal": int(chosen["dist_nominal"]),
    }
    return resolved, info


def _build_surface_flow(rain: np.ndarray, runoff_raw: np.ndarray, mode: str) -> np.ndarray:
    if mode == "runoff_fraction":
        return runoff_raw / (np.nanmax(runoff_raw) + 1e-6)
    if mode == "normalize_to_rain":
        return rain * (runoff_raw / (np.nanmax(runoff_raw) + 1e-6))
    return runoff_raw.copy()


def load_tp_daily_data(cfg: dict | None = None) -> dict:
    cfg = dict(TP_CONFIG if cfg is None else cfg)
    cfg["TP_SOURCE_DIR"] = Path(cfg.get("TP_SOURCE_DIR", DEFAULT_TP_SOURCE_DIR))
    if not cfg["TP_SOURCE_DIR"].exists():
        cfg["TP_SOURCE_DIR"] = LEGACY_TP_SOURCE_DIR

    landuse = _read_grid(Path(cfg["LANDUSE_CSV"])).astype(int)
    slope = _read_grid(Path(cfg["SLOPE_CSV"])).astype(float)
    if landuse.shape != slope.shape:
        raise ValueError(f"landuse shape {landuse.shape} != slope shape {slope.shape}")

    with open(cfg["FLOWDIRUP_PKL"], "rb") as f:
        raw_flowdir_up = pickle.load(f)
    study_area_mask = landuse > 0
    oi, oj = _decode_cell_code(str(cfg["OUTLET_CODE"]))
    if 0 <= oi < study_area_mask.shape[0] and 0 <= oj < study_area_mask.shape[1]:
        study_area_mask[oi, oj] = True
    flowdir_up = filter_flowdir_to_study_area(raw_flowdir_up, study_area_mask)
    forced_terminals = _load_tn_forced_terminals()
    flowdir_up, outlet_debug = augment_flowdir_to_outlet(
        flowdir_up,
        str(cfg["OUTLET_CODE"]),
        rows=landuse.shape[0],
        cols=landuse.shape[1],
        study_area_mask=study_area_mask,
        forced_terminals=forced_terminals,
    )

    obs_df = _load_obs_csv(Path(cfg["OBS_DAILY_CSV"]), int(cfg["YEAR"]))
    forcing_df = _load_forcing_csv(Path(cfg["DAILY_DATA_CSV"]), int(cfg["YEAR"]))
    daily_df = obs_df.merge(forcing_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if daily_df.empty:
        raise ValueError("No overlapping daily observations and forcing records were found.")

    monthly_maps, monthly_totals = _load_tp_monthly_sources(Path(cfg["TP_SOURCE_DIR"]), int(cfg["YEAR"]))
    annual_source_map = np.zeros_like(next(iter(monthly_maps.values())), dtype=float)
    for arr in monthly_maps.values():
        annual_source_map = annual_source_map + arr

    grid_df, grids = build_grid_attribute_df(cfg, landuse, slope, flowdir_up)
    outlet_info = {
        "nominal_outlet_code": str(cfg["OUTLET_CODE"]),
        "resolved_outlet_code": str(cfg["OUTLET_CODE"]),
        "nominal_rowcol": [int(str(cfg["OUTLET_CODE"])[:4]), int(str(cfg["OUTLET_CODE"])[4:])],
        "resolved_rowcol": [int(str(cfg["OUTLET_CODE"])[:4]), int(str(cfg["OUTLET_CODE"])[4:])],
        "flowdir_mode": "tn_style_augmented_topology",
        "source_dir": str(cfg["TP_SOURCE_DIR"]),
    }

    source_vals = annual_source_map[grid_df["row"].to_numpy(dtype=int), grid_df["col"].to_numpy(dtype=int)]
    grid_df["annual_source"] = source_vals.astype(float)
    grid_df["annual_source_z"] = (
        (grid_df["annual_source"] - float(grid_df["annual_source"].mean()))
        / (float(grid_df["annual_source"].std()) + 1e-6)
    )

    feat_cols = cfg["FEAT_COLS"]
    X = grid_df[list(feat_cols)].to_numpy(dtype=float)
    groups = grid_df["lu_group"].to_numpy(dtype=int)
    idx_g = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    pix_r = grid_df["row"].to_numpy(dtype=int)
    pix_c = grid_df["col"].to_numpy(dtype=int)

    dates = pd.DatetimeIndex(daily_df["date"])
    fert = _build_fert_factor(daily_df["date"], cfg)
    rain = daily_df["rain"].to_numpy(dtype=float)
    runoff_raw = daily_df["runoff"].to_numpy(dtype=float)
    surface_flow = _build_surface_flow(rain, runoff_raw, str(cfg["SURFACE_FLOW_MODE"]))

    prior_month = _build_monthly_daily_prior(dates, monthly_totals)
    event_prior = runoff_raw / (np.sum(runoff_raw) + 1e-12)
    prior_l = 0.55 * prior_month + 0.45 * event_prior
    fs_prior = np.clip(surface_flow, 0.0, 1.0)
    group_ratio = _compute_group_ratio_from_map(annual_source_map, landuse)
    year_total = float(np.nansum(annual_source_map))

    return {
        "cfg": cfg,
        "landuse": landuse,
        "slope": slope,
        "flowdir_up": flowdir_up,
        "par0": pd.read_excel(ROOT / "tp_parameter_raw.xlsx").iloc[64:121]["value"].to_numpy(dtype=float),
        "par_lb_all": pd.read_excel(ROOT / "tp_parameter_raw.xlsx").iloc[64:121]["min"].to_numpy(dtype=float),
        "par_ub_all": pd.read_excel(ROOT / "tp_parameter_raw.xlsx").iloc[64:121]["max"].to_numpy(dtype=float),
        "calib_par_idx": np.array(cfg["CALIB_PAR_INDEXES"], dtype=int),
        "dates": dates,
        "rain": rain,
        "runoff_for_time": runoff_raw,
        "surface_flow": surface_flow,
        "fert": fert,
        "obs": daily_df["TP"].to_numpy(dtype=float),
        "obs_tn": daily_df["TN"].to_numpy(dtype=float),
        "year_total": year_total,
        "group_ratio": group_ratio,
        "grid_df": grid_df,
        "grids": grids,
        "outlet_info": outlet_info,
        "outlet_debug": outlet_debug,
        "X": X,
        "groups": groups,
        "idx_g": idx_g,
        "pix_r": pix_r,
        "pix_c": pix_c,
        "ml_w0": np.array(cfg["ML_W_INIT"], dtype=float),
        "ml_b0": float(cfg["ML_B_INIT"]),
        "ml_lb": np.array(cfg["ML_W_LOWER"] + [cfg["ML_B_LOWER"]], dtype=float),
        "ml_ub": np.array(cfg["ML_W_UPPER"] + [cfg["ML_B_UPPER"]], dtype=float),
        "time0": np.array(cfg["TIME_INIT"], dtype=float),
        "time_lb": np.array(cfg["TIME_LB"], dtype=float),
        "time_ub": np.array(cfg["TIME_UB"], dtype=float),
        "prior_L": prior_l.astype(float),
        "fs_prior": fs_prior.astype(float),
        "annual_source_map": annual_source_map.astype(float),
        "monthly_source_totals": monthly_totals,
        "monthly_source_maps": monthly_maps,
    }


def export_tp_daily_inputs(data: dict, forcing_csv: Path, obs_csv: Path, meta_json: Path) -> None:
    forcing = pd.DataFrame(
        {
            "date": pd.to_datetime(data["dates"]),
            "rain": data["rain"],
            "runoff": data["runoff_for_time"],
            "surface_flow": data["surface_flow"],
            "fert_factor": data["fert"],
        }
    )
    obs = pd.DataFrame(
        {
            "date": pd.to_datetime(data["dates"]),
            "TP": data["obs"],
            "TN": data["obs_tn"],
        }
    )
    forcing.to_csv(forcing_csv, index=False)
    obs.to_csv(obs_csv, index=False)
    meta = {
        "year_total": float(data["year_total"]),
        "group_ratio": {str(k): float(v) for k, v in data["group_ratio"].items()},
        "n_days": int(len(data["dates"])),
        "outlet_info": data.get("outlet_info", {}),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
