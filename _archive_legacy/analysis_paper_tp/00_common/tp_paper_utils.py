from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tp_paper_config import FIG_DIR, TABLE_DIR, TP_PLOT_UNIT, TRAIN_RATIO


OBS_CANDIDATES = ["obs_tp", "TP", "obs", "observed"]
SIM_CANDIDATES = ["sim_tp", "simulated", "sim", "prediction", "pred"]
DATE_CANDIDATES = ["date", "Date", "datetime"]


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception:
            continue
    return pd.read_csv(path, **kwargs)


def find_existing_file(patterns: list[str]) -> Path | None:
    roots = [Path(__file__).resolve().parents[2], Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for root in roots:
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file() and path not in seen:
                    seen.add(path)
                    return path
    return None


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def make_fert_factor(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    dates = pd.to_datetime(dates)
    if isinstance(dates, pd.Series):
        months = dates.dt.month
        years = dates.dt.year
    else:
        months = dates.month
        years = dates.year
    fert = pd.Series(months).isin([3, 5, 7]).astype(float).to_numpy(dtype=float)
    spring_ref = pd.to_datetime({"year": years, "month": 3, "day": 20})
    if isinstance(dates, pd.Series):
        spring_boost = (dates - spring_ref).dt.days.to_numpy(dtype=float)
    else:
        spring_boost = (dates - spring_ref).days.astype(float)
    spring_mask = (np.abs(spring_boost) <= 7).astype(float)
    raw = np.clip(fert + spring_mask, 0.0, 1.0)
    return pd.Series(raw).rolling(window=5, center=True, min_periods=1).max().to_numpy(dtype=float)


def build_api(rain: np.ndarray, lam: float = 0.85) -> np.ndarray:
    out = np.zeros_like(np.asarray(rain, dtype=float), dtype=float)
    state = 0.0
    for i, val in enumerate(np.asarray(rain, dtype=float)):
        state = float(val) + lam * state
        out[i] = state
    return out


def calculate_metrics(obs: np.ndarray | pd.Series, sim: np.ndarray | pd.Series) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    resid = sim - obs
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    pbias = float(100.0 * np.sum(resid) / (np.sum(obs) + 1e-12))
    den = float(np.sum((obs - np.mean(obs)) ** 2))
    nse = float(1.0 - np.sum(resid**2) / den) if den > 1e-12 else float("nan")
    corr = np.corrcoef(obs, sim)[0, 1] if len(obs) > 1 else np.nan
    r2 = float(corr * corr) if np.isfinite(corr) else float("nan")
    return {"NSE": nse, "R2": r2, "RMSE": rmse, "MAE": mae, "Bias": bias, "PBIAS": pbias}


def split_train_val(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = max(1, min(len(df) - 1, int(round(len(df) * train_ratio))))
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def assign_season(date: pd.Timestamp) -> str:
    month = pd.Timestamp(date).month
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def identify_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["season"] = pd.to_datetime(out["date"]).map(assign_season)
    out["all_days"] = True
    out["rainfall_days"] = np.asarray(out["rain"], dtype=float) > 1.0e-12
    out["non_rainfall_days"] = ~out["rainfall_days"]
    runoff = np.asarray(out["runoff"], dtype=float)
    obs = np.asarray(out["obs"], dtype=float)
    out["high_runoff_days"] = runoff >= np.nanquantile(runoff, 0.75)
    out["peak_TP_days"] = obs >= np.nanquantile(obs, 0.90)
    out["low_runoff_high_TP_days"] = (runoff <= np.nanquantile(runoff, 0.25)) & (obs >= np.nanquantile(obs, 0.75))
    return out


def save_table(df: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_figure(fig, filename: str) -> Path:
    path = FIG_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def value_label() -> str:
    return f"Daily TP load ({TP_PLOT_UNIT})"


def observed_label() -> str:
    return f"Observed TP load ({TP_PLOT_UNIT})"


def predicted_label() -> str:
    return f"Predicted TP load ({TP_PLOT_UNIT})"


def simulated_label() -> str:
    return f"Simulated TP load ({TP_PLOT_UNIT})"


def residual_label() -> str:
    return f"Residual ({TP_PLOT_UNIT})"


def rmse_label() -> str:
    return f"RMSE ({TP_PLOT_UNIT})"


def _prediction_priority(path: Path) -> int:
    name = path.name.lower()
    if name == "tp_direct_predictions_direct_physical_90kg_tp.csv":
        return 500
    if name.startswith("tp_direct_predictions_"):
        return 450
    if name == "tp_split_predictions_physics_first_tp.csv":
        return 400
    if name == "tp_split_predictions_rollback_uniform_tp.csv":
        return 300
    if "tp_split_predictions" in name:
        return 200
    return 100


def _score_prediction_file(path: Path) -> tuple[int, int, int]:
    try:
        df = safe_read_csv(path)
    except Exception:
        return (-1, -1, -1)
    cols = {c.lower() for c in df.columns}
    completeness = sum(int(c in cols) for c in ("date", "obs_tp", "sim_tp", "routed_out", "hyd_out", "raw_sim_tp"))
    val_len = int((df.get("split", pd.Series(dtype=str)).astype(str).str.lower() == "val").sum()) if "split" in df.columns else 0
    return (_prediction_priority(path), val_len, completeness)


def _load_forcing_frame() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "tp_daily_forcing_2023.csv",
        root / "input" / "forcing" / "daily_data.csv",
    ]
    for path in candidates:
        if path.exists():
            df = safe_read_csv(path)
            date_col = _pick_column(list(df.columns), DATE_CANDIDATES)
            if date_col is None:
                continue
            df = df.rename(columns={date_col: "date"})
            if "rain" not in df.columns or "runoff" not in df.columns:
                continue
            out = df[["date", "rain", "runoff"]].copy()
            out["date"] = pd.to_datetime(out["date"])
            return out
    raise FileNotFoundError("No TP forcing file could be found.")


def read_prediction_file() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "tp_direct_predictions_direct_physical_90kg_tp.csv",
        root / "tp_split_predictions_physics_first_tp.csv",
        root / "tp_split_predictions_rollback_uniform_tp.csv",
    ]
    candidates.extend(sorted(root.glob("*tp_direct_predictions*.csv")))
    candidates.extend(sorted(root.glob("*tp_split_predictions*.csv")))
    best = None
    best_score = (-1, -1, -1)
    for path in candidates:
        if path.exists():
            score = _score_prediction_file(path)
            if score > best_score:
                best = path
                best_score = score
    if best is None:
        raise FileNotFoundError("No TP prediction csv could be located.")

    df = safe_read_csv(best)
    date_col = _pick_column(list(df.columns), DATE_CANDIDATES)
    obs_col = _pick_column(list(df.columns), OBS_CANDIDATES)
    sim_col = _pick_column(list(df.columns), SIM_CANDIDATES)
    if date_col is None or obs_col is None or sim_col is None:
        raise ValueError(f"Prediction file lacks required columns: {best}")

    rename_map = {date_col: "date", obs_col: "obs", sim_col: "simulated"}
    extra_cols = {}
    for cand in ["raw_sim_tp", "routed_out", "hyd_out", "l_daily", "f_surface", "split"]:
        if cand in df.columns:
            extra_cols[cand] = cand
    out = df.rename(columns=rename_map)[["date", "obs", "simulated", *extra_cols.keys()]].copy()
    out["date"] = pd.to_datetime(out["date"])

    forcing = _load_forcing_frame()
    if "rain" not in out.columns or "runoff" not in out.columns:
        out = out.merge(forcing, on="date", how="left")
    out["rain"] = pd.to_numeric(out["rain"], errors="coerce").fillna(0.0)
    out["runoff"] = pd.to_numeric(out["runoff"], errors="coerce").fillna(0.0)
    out["obs"] = pd.to_numeric(out["obs"], errors="coerce")
    out["simulated"] = pd.to_numeric(out["simulated"], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    out.attrs["source_path"] = str(best)
    return out
