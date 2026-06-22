# -*- coding: utf-8 -*-
"""
数据加载模块 - 负责读取所有输入文件并进行基本预处理
为TN日尺度模型提供统一的数据接口

主要功能：
1. 读取土地利用、坡度、流向拓扑等栅格数据
2. 读取物理参数表
3. 读取日尺度驱动数据和观测数据
4. 计算施肥因子等时间序列特征
5. 构建网格属性（特征工程）
6. 返回统一格式的数据字典

注意：本模块不包含任何模型计算逻辑
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
from scipy.ndimage import distance_transform_edt


# =============================================================================
# 辅助函数（仅数据预处理用）
# =============================================================================

def landuse_to_group(lu: int) -> int:
    """土地利用类型转组别编码"""
    if lu == 1:
        return 1
    if lu in (2, 4):
        return 2
    if lu == 8:
        return 3
    return 0


def compute_group_ratio(year_source_dict, source_to_group_map):
    """计算各土地利用组的年源项比例"""
    group_year = {1: 0.0, 2: 0.0, 3: 0.0}
    for src, gmap in source_to_group_map.items():
        if src not in year_source_dict:
            continue
        val = float(year_source_dict[src])
        for g, frac in gmap.items():
            group_year[int(g)] += val * float(frac)
    tot = sum(group_year.values())
    if tot <= 0:
        return {1: 0.0, 2: 0.0, 3: 0.0}
    ratio = {g: group_year[g] / tot for g in group_year}
    s = sum(ratio.values())
    ratio = {g: ratio[g] / (s + 1e-12) for g in ratio}
    return ratio


def build_basin_mask_and_valid(flowdir_up: dict, rows: int, cols: int):
    """构建流域掩码和有效单元集合（固定顺序）"""
    valid = set(flowdir_up.keys())
    for ups in flowdir_up.values():
        valid.update(ups)

    # 固定顺序：按单元ID排序（格式为"00000000"）
    valid = sorted(valid)

    basin = np.zeros((rows, cols), dtype=bool)
    for cid in valid:
        i = int(cid[:4]); j = int(cid[4:])
        if 0 <= i < rows and 0 <= j < cols:
            basin[i, j] = True
    return basin, valid


def compute_flow_acc_with_cycle_flag(flowdir_up: dict, valid_cells):
    """计算汇流累积量并检测循环（valid_cells应为有序列表）"""
    # 确保valid_cells是列表（已排序）
    if isinstance(valid_cells, set):
        valid_cells = sorted(valid_cells)

    flowdir_down = defaultdict(list)
    upstream_count = defaultdict(int)

    for cell, ups in flowdir_up.items():
        upstream_count[cell] = len(ups)
        for up in ups:
            flowdir_down[up].append(cell)

    for cell in valid_cells:
        upstream_count.setdefault(cell, 0)

    flow_acc = {cell: 1 for cell in valid_cells}
    q = deque([c for c, n in upstream_count.items() if n == 0])

    visited = set()
    while q:
        c = q.popleft()
        visited.add(c)
        for dn in flowdir_down.get(c, []):
            flow_acc[dn] += flow_acc[c]
            upstream_count[dn] -= 1
            if upstream_count[dn] == 0:
                q.append(dn)

    is_cycle = {c: (0 if c in visited else 1) for c in valid_cells}
    return flow_acc, is_cycle, visited


def extract_stream_and_distance(flow_acc_grid, basin_mask, valid_mask, percentile,
                                dist_in_m=False, cellsize=30.0):
    """提取河网并计算到河流距离"""
    vals = flow_acc_grid[valid_mask]
    vals = vals[np.isfinite(vals)]
    thr = np.percentile(vals, percentile) if len(vals) else np.nan
    stream = (flow_acc_grid >= thr) & basin_mask

    dist_pix = distance_transform_edt(~stream)
    dist = np.full_like(flow_acc_grid, np.nan, dtype=np.float32)
    dist[basin_mask] = dist_pix[basin_mask].astype(np.float32)
    if dist_in_m:
        dist = dist * float(cellsize)
    return stream, dist


def build_grid_attribute_df(cfg, landuse: np.ndarray, slope: np.ndarray, flowdir_up: dict):
    """构建网格属性DataFrame（特征工程）"""
    rows, cols = landuse.shape
    basin_mask, valid_cells = build_basin_mask_and_valid(flowdir_up, rows, cols)
    flow_acc_dict, is_cycle_dict, visited = compute_flow_acc_with_cycle_flag(flowdir_up, valid_cells)

    flow_acc = np.full((rows, cols), np.nan, dtype=np.float32)
    is_cycle_grid = np.zeros((rows, cols), dtype=np.int8)
    for cid in valid_cells:
        i = int(cid[:4]); j = int(cid[4:])
        if 0 <= i < rows and 0 <= j < cols:
            flow_acc[i, j] = float(flow_acc_dict.get(cid, 1))
            is_cycle_grid[i, j] = 1 if is_cycle_dict.get(cid, 0) else 0

    land_ok = np.isfinite(landuse) & (landuse != 0)
    slope_ok = np.isfinite(slope)
    not_cycle = (is_cycle_grid == 0)
    valid_mask = basin_mask & land_ok & slope_ok & not_cycle

    stream, dist_to_stream = extract_stream_and_distance(
        flow_acc, basin_mask, valid_mask,
        percentile=float(cfg["STREAM_PERCENTILE"]),
        dist_in_m=bool(cfg["DIST_TO_STREAM_IN_METERS"]),
        cellsize=float(cfg["CELL_SIZE_M"])
    )

    rr, cc = np.where(valid_mask)
    df = pd.DataFrame({
        "pixel_id": [f"{r:04d}{c:04d}" for r, c in zip(rr, cc)],
        "row": rr.astype(int),
        "col": cc.astype(int),
        "landuse": landuse[rr, cc].astype(int),
        "slope_deg": slope[rr, cc].astype(float),
        "flow_acc": flow_acc[rr, cc].astype(float),
        "dist_to_stream": dist_to_stream[rr, cc].astype(float),
        "is_stream": stream[rr, cc].astype(int),
        "is_cycle_node": is_cycle_grid[rr, cc].astype(int),
    })
    df["lu_group"] = df["landuse"].apply(lambda x: landuse_to_group(int(x)))
    df = df[df["lu_group"] != 0].copy()

    df["log_flow_acc"] = np.log(df["flow_acc"].astype(float) + 1.0)

    for col in ("slope_deg", "log_flow_acc", "dist_to_stream"):
        mu = df[col].mean()
        sd = df[col].std()
        df[col + "_z"] = (df[col] - mu) / (sd + 1e-6)

    grids = dict(
        basin_mask=basin_mask,
        valid_mask=valid_mask,
        flow_acc=flow_acc,
        dist_to_stream=dist_to_stream,
        stream_mask=stream,
        is_cycle=is_cycle_grid,
    )
    return df, grids


# =============================================================================
# 配置检查函数
# =============================================================================

def check_config_keys(cfg, required_keys):
    """检查配置字典中必需键是否存在"""
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"配置字典缺少以下必需键: {missing}")
    return True


# =============================================================================
# 主加载函数
# =============================================================================

def load_all_data(cfg):
    """
    加载所有输入数据并返回统一格式的数据字典

    参数:
        cfg: 配置字典，包含所有文件路径和参数

    返回:
        data: 包含所有加载数据的字典，键包括:
            - landuse, slope, flowdir_up: 栅格和拓扑数据
            - par0, par_lb_all, par_ub_all, calib_par_idx: 物理参数
            - dates, rain, runoff, surface_flow, fert, obs: 时间序列数据
            - grid_df, X, groups, pix_r, pix_c, idx_g: 网格属性
            - pollutant, year_total, group_ratio: 污染物配置
            - ml_w0, ml_b0, ml_lb, ml_ub: ML参数初始化
            - time0, time_lb, time_ub: 时间生成器参数初始化
            - prior_L, fs_prior: 先验信息
    """
    data = {}

    # 检查关键配置键
    required_cfg_keys = [
        "LANDUSE_CSV", "SLOPE_CSV", "FLOWDIRUP_PKL", "PARAM_XLSX",
        "DAILY_DATA_CSV", "OBS_DAILY_CSV", "OUTLET_CODE", "POLLUTANT",
        "TN_YEAR_TOTAL", "TP_YEAR_TOTAL", "TN_year_source", "TP_year_source",
        "source_to_group", "STREAM_PERCENTILE", "DIST_TO_STREAM_IN_METERS",
        "CELL_SIZE_M", "FEAT_COLS", "FERT_MONTHS", "SPRING_EQUINOX_MONTH",
        "SPRING_EQUINOX_DAY", "SPRING_EQUINOX_WINDOW", "SURFACE_FLOW_MODE",
        "ML_W_INIT", "ML_B_INIT", "ML_W_LOWER", "ML_W_UPPER", "ML_B_LOWER",
        "ML_B_UPPER", "TIME_INIT", "TIME_LB", "TIME_UB", "CALIBRATE_PHYS",
        "CALIB_PAR_INDEXES"
    ]
    check_config_keys(cfg, required_cfg_keys)

    # ---- 1. 读取栅格和拓扑数据 ----
    print("Loading landuse...")
    data["landuse"] = np.array(pd.read_csv(cfg["LANDUSE_CSV"], header=None)).astype(int)
    print("Loading slope...")
    data["slope"] = np.array(pd.read_csv(cfg["SLOPE_CSV"], header=None)).astype(float)

    if data["landuse"].shape != data["slope"].shape:
        raise ValueError(f"landuse shape {data['landuse'].shape} != slope shape {data['slope'].shape}")

    print("Loading flow topology...")
    with open(cfg["FLOWDIRUP_PKL"], "rb") as f:
        data["flowdir_up"] = pickle.load(f)

    # ---- 2. 读取物理参数 ----
    print("Loading physical parameters...")
    par_df = pd.read_excel(cfg["PARAM_XLSX"])
    for col in ("value", "min", "max"):
        if col not in par_df.columns:
            raise ValueError(f"parameter.xlsx missing column '{col}'. Found: {list(par_df.columns)}")

    data["par0"] = par_df["value"].to_numpy(dtype=float)
    data["par_lb_all"] = par_df["min"].to_numpy(dtype=float)
    data["par_ub_all"] = par_df["max"].to_numpy(dtype=float)

    if cfg["CALIBRATE_PHYS"]:
        idx = cfg["CALIB_PAR_INDEXES"]
        if idx is None:
            data["calib_par_idx"] = np.arange(len(data["par0"]), dtype=int)
        else:
            data["calib_par_idx"] = np.array(idx, dtype=int)
    else:
        data["calib_par_idx"] = np.array([], dtype=int)

    # ---- 3. 读取日尺度驱动数据 ----
    print("Loading daily forcing data...")
    dd = pd.read_csv(cfg["DAILY_DATA_CSV"], encoding="gbk")
    if not {"date", "rain", "runoff"}.issubset(dd.columns):
        raise ValueError("daily_data.csv must contain columns: date, rain, runoff")

    dd["date"] = pd.to_datetime(dd["date"])
    dd = dd.sort_values("date").reset_index(drop=True)

    # 计算施肥因子 - 修复问题1：按每个日期自己的年份计算春分日期
    df = dd.copy()
    df["month"] = df["date"].dt.month
    df["F_month"] = df["month"].isin(cfg["FERT_MONTHS"]).astype(float)

    # 向量化计算每个日期到当年春分日的距离
    # 为每个日期创建春分日（当年3月20日）
    def spring_equinox_for_date(date_series):
        """为每个日期返回当年春分日（3月20日）"""
        years = date_series.dt.year
        return pd.to_datetime({
            'year': years,
            'month': cfg["SPRING_EQUINOX_MONTH"],
            'day': cfg["SPRING_EQUINOX_DAY"]
        })

    spring_dates = spring_equinox_for_date(df["date"])
    df["days_from_se"] = (df["date"] - spring_dates).dt.days
    df["F_spring"] = (df["days_from_se"].abs() <= cfg["SPRING_EQUINOX_WINDOW"]).astype(float)
    df["F_raw"] = (df["F_month"] + df["F_spring"]).clip(upper=1.0)
    df["fert_factor"] = df["F_raw"].rolling(window=5, center=True, min_periods=1).max()

    daily_df = df[["date", "rain", "runoff", "fert_factor"]].copy()
    daily_df["rain"] = np.maximum(daily_df["rain"].astype(float), 0.0)
    daily_df["runoff"] = np.maximum(daily_df["runoff"].astype(float), 0.0)

    # ---- 4. 读取观测数据 ----
    print("Loading observation data...")
    obs = pd.read_csv(cfg["OBS_DAILY_CSV"])

    # 检查必需列 - 修复问题2
    required_obs_cols = ["date", "TN"]
    missing_obs_cols = [col for col in required_obs_cols if col not in obs.columns]
    if missing_obs_cols:
        raise ValueError(f"obs.csv missing required columns: {missing_obs_cols}")

    obs["date"] = pd.to_datetime(obs["date"])
    obs = obs.sort_values("date").set_index("date")

    daily_df = daily_df.set_index("date")
    common = daily_df.index.intersection(obs.index)
    if len(common) == 0:
        raise ValueError("No common dates between daily forcing and obs")

    data["dates"] = common
    data["rain"] = daily_df.loc[common, "rain"].to_numpy(dtype=float)
    runoff_raw = daily_df.loc[common, "runoff"].to_numpy(dtype=float)
    data["fert"] = daily_df.loc[common, "fert_factor"].to_numpy(dtype=float)

    # 计算地表流（用于先验）
    if cfg["SURFACE_FLOW_MODE"] == "raw":
        data["surface_flow"] = runoff_raw.copy()
    elif cfg["SURFACE_FLOW_MODE"] == "normalize_to_rain":
        rmax = np.max(runoff_raw) + 1e-12
        data["surface_flow"] = data["rain"] * (runoff_raw / rmax)
    else:
        raise ValueError("SURFACE_FLOW_MODE must be 'raw' or 'normalize_to_rain'")

    data["runoff_for_time"] = runoff_raw

    # ---- 5. 污染物配置 ----
    pollutant = cfg["POLLUTANT"].upper()
    if pollutant not in ("TN", "TP"):
        raise ValueError("POLLUTANT must be TN or TP")
    data["pollutant"] = pollutant

    if pollutant == "TN":
        data["obs"] = obs.loc[common, "TN"].to_numpy(dtype=float)
        data["year_total"] = float(cfg["TN_YEAR_TOTAL"])
        data["group_ratio"] = compute_group_ratio(cfg["TN_year_source"], cfg["source_to_group"])
    else:
        data["obs"] = obs.loc[common, "TP"].to_numpy(dtype=float)
        data["year_total"] = float(cfg["TP_YEAR_TOTAL"])
        data["group_ratio"] = compute_group_ratio(cfg["TP_year_source"], cfg["source_to_group"])

    # ---- 6. 构建网格属性 ----
    print("Building grid attributes...")
    grid_df, grids = build_grid_attribute_df(cfg, data["landuse"], data["slope"], data["flowdir_up"])
    data["grid_df"] = grid_df
    data["grids"] = grids

    feat_cols = cfg["FEAT_COLS"]
    for c in feat_cols:
        if c not in grid_df.columns:
            raise ValueError(f"Feature {c} not found in grid_df columns")

    data["X"] = grid_df[list(feat_cols)].to_numpy(dtype=float)
    data["groups"] = grid_df["lu_group"].to_numpy(dtype=int)
    data["pix_r"] = grid_df["row"].to_numpy(dtype=int)
    data["pix_c"] = grid_df["col"].to_numpy(dtype=int)

    unique_groups = np.unique(data["groups"])
    data["idx_g"] = {g: np.where(data["groups"] == g)[0] for g in unique_groups}

    # ---- 7. 初始化参数（从配置读取） ----
    data["ml_w0"] = np.array(cfg["ML_W_INIT"], dtype=float)
    data["ml_b0"] = float(cfg["ML_B_INIT"])
    data["ml_lb"] = np.array(cfg["ML_W_LOWER"] + [cfg["ML_B_LOWER"]], dtype=float)
    data["ml_ub"] = np.array(cfg["ML_W_UPPER"] + [cfg["ML_B_UPPER"]], dtype=float)

    data["time0"] = np.array(cfg["TIME_INIT"], dtype=float)
    data["time_lb"] = np.array(cfg["TIME_LB"], dtype=float)
    data["time_ub"] = np.array(cfg["TIME_UB"], dtype=float)

    # ---- 8. 计算先验信息 ----
    data["prior_L"] = np.maximum(data["rain"] - 5.0, 0.0) + 0.5 * data["fert"]

    # 地表比例先验：经验性估计，基于地表流与降雨的比值，不是严格的物理定义量
    # 主要用于正则化约束，避免f_surface学习过程中出现不合理值
    fs_prior = data["surface_flow"] / (data["rain"] + 1e-6)
    data["fs_prior"] = np.clip(fs_prior, 0.0, 1.0)

    print("Data loading completed.")
    return data


# =============================================================================
# 快速检查函数
# =============================================================================

def check_loaded_data(data):
    """检查加载的数据完整性"""
    required_keys = [
        "landuse", "slope", "flowdir_up", "par0", "dates", "rain",
        "obs", "grid_df", "X", "groups"
    ]

    missing = [k for k in required_keys if k not in data]
    if missing:
        print(f"Warning: Missing keys in data: {missing}")
        return False

    print(f"Landuse shape: {data['landuse'].shape}")
    print(f"Slope shape: {data['slope'].shape}")
    print(f"Flowdir_up keys: {len(data['flowdir_up'])}")
    print(f"Parameter count: {len(data['par0'])}")
    print(f"Time steps: {len(data['dates'])}")
    print(f"Grid cells: {len(data['grid_df'])}")
    print(f"Observed dates: {data['dates'].min()} to {data['dates'].max()}")

    return True