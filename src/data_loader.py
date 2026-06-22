from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
SOURCE_DIR = ROOT / "source_corrected_90kg"
DEFAULT_YEAR = 2023
FERT_MONTHS = {3, 5, 7}
SPRING_EQUINOX_MONTH = 3
SPRING_EQUINOX_DAY = 20
SPRING_WINDOW = 7


@dataclass
class TPDataset:
    frame: pd.DataFrame
    split_date: pd.Timestamp
    crop_total: float
    impervious_total: float


def _build_fert_factor(dates: pd.Series) -> np.ndarray:
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    df["month"] = df["date"].dt.month
    df["fert_month"] = df["month"].isin(FERT_MONTHS).astype(float)
    spring_dates = pd.to_datetime(
        {"year": df["date"].dt.year, "month": SPRING_EQUINOX_MONTH, "day": SPRING_EQUINOX_DAY}
    )
    df["days_from_se"] = (df["date"] - spring_dates).dt.days
    df["fert_spring"] = (df["days_from_se"].abs() <= SPRING_WINDOW).astype(float)
    fert = (df["fert_month"] + df["fert_spring"]).clip(upper=1.0)
    return fert.rolling(window=5, center=True, min_periods=1).max().to_numpy(dtype=float)


def _api(series: np.ndarray, decay: float) -> np.ndarray:
    state = 0.0
    out = np.zeros_like(series, dtype=float)
    for idx, value in enumerate(series):
        state = float(value) + decay * state
        out[idx] = state
    return out


def _norm(values: np.ndarray) -> np.ndarray:
    vmax = float(np.nanmax(values))
    if vmax <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return values / vmax


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window=window, min_periods=1).sum().to_numpy(dtype=float)


def load_tp_dataset(year: int = DEFAULT_YEAR) -> TPDataset:
    forcing = pd.read_csv(INPUT_DIR / "forcing" / "daily_data.csv")
    forcing["date"] = pd.to_datetime(forcing["date"], errors="coerce")
    forcing["rain"] = pd.to_numeric(forcing["rain"], errors="coerce").fillna(0.0).clip(lower=0.0)
    forcing["runoff"] = pd.to_numeric(forcing["runoff"], errors="coerce").fillna(0.0).clip(lower=0.0)

    obs = pd.read_csv(INPUT_DIR / "forcing" / "obs.csv")
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    obs["TP"] = pd.to_numeric(obs["TP"], errors="coerce")
    obs["TN"] = pd.to_numeric(obs["TN"], errors="coerce")

    prior = pd.read_csv(SOURCE_DIR / "tp_daily_source_prior_corrected.csv")
    prior["date"] = pd.to_datetime(prior["date"], errors="coerce")

    df = (
        obs.merge(forcing, on="date", how="inner")
        .merge(prior, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    df = df[df["date"].dt.year == year].copy()
    if df.empty:
        raise ValueError(f"No TP daily records found for year {year}.")

    df["fert_factor"] = _build_fert_factor(df["date"])
    df["month"] = df["date"].dt.month.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear.astype(int)
    theta = 2.0 * np.pi * df["dayofyear"].to_numpy(dtype=float) / 365.0
    df["sin_doy"] = np.sin(theta)
    df["cos_doy"] = np.cos(theta)
    df["rain_roll3"] = _rolling(df["rain"].to_numpy(dtype=float), 3)
    df["rain_roll7"] = _rolling(df["rain"].to_numpy(dtype=float), 7)
    df["runoff_roll3"] = _rolling(df["runoff"].to_numpy(dtype=float), 3)
    df["runoff_roll7"] = _rolling(df["runoff"].to_numpy(dtype=float), 7)
    df["api_rain"] = _api(df["rain"].to_numpy(dtype=float), 0.86)
    df["api_runoff"] = _api(df["runoff"].to_numpy(dtype=float), 0.82)

    for col in ("rain", "runoff", "rain_roll3", "rain_roll7", "runoff_roll3", "runoff_roll7", "api_rain", "api_runoff"):
        df[f"{col}_n"] = _norm(df[col].to_numpy(dtype=float))

    n_days = len(df)
    n_train = int(round(0.70 * n_days))
    df["period"] = np.where(np.arange(n_days) < n_train, "Calibration", "Validation")
    split_date = pd.Timestamp(df.loc[n_train, "date"]) if n_train < n_days else pd.Timestamp(df.loc[n_train - 1, "date"])

    crop_total = float(df["crop_daily_total_prior"].sum())
    impervious_total = float(df["impervious_daily_total_prior"].sum())
    return TPDataset(frame=df, split_date=split_date, crop_total=crop_total, impervious_total=impervious_total)


def feature_columns() -> list[str]:
    return [
        "rain_n",
        "runoff_n",
        "api_rain_n",
        "api_runoff_n",
        "rain_roll3_n",
        "rain_roll7_n",
        "runoff_roll3_n",
        "runoff_roll7_n",
        "fert_factor",
        "crop_daily_total_prior",
        "impervious_daily_total_prior",
        "daily_total_prior",
        "sin_doy",
        "cos_doy",
        "month",
        "dayofyear",
    ]

