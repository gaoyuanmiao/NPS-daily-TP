from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import build_grid_attribute_df
from tp_daily_loader import TP_CONFIG, _build_fert_factor, _load_forcing_csv


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "source_corrected_90kg"
SUMMARY_JSON = OUT_DIR / "tp_source_corrected_summary.json"
DAILY_PRIOR_CSV = OUT_DIR / "tp_daily_source_prior_corrected.csv"

ANNUAL_TOTALS = {
    "rural_life": 10.49,
    "livestock": 8.21,
    "aquaculture": 26.84,
    "agriculture": 44.00,
}
IMP_TOTAL = ANNUAL_TOTALS["rural_life"] + ANNUAL_TOTALS["livestock"]
CROP_TOTAL = ANNUAL_TOTALS["aquaculture"] + ANNUAL_TOTALS["agriculture"]
YEAR_TOTAL = IMP_TOTAL + CROP_TOTAL


def _normalize_positive(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full_like(arr, 1.0 / max(arr.size, 1), dtype=float)
    return arr / total


def build_corrected_sources() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    landuse = pd.read_csv(TP_CONFIG["LANDUSE_CSV"]).to_numpy(dtype=int)
    slope = pd.read_csv(TP_CONFIG["SLOPE_CSV"]).to_numpy(dtype=float)
    with open(TP_CONFIG["FLOWDIRUP_PKL"], "rb") as f:
        flowdir_up = pickle.load(f)

    grid_df, grids = build_grid_attribute_df(TP_CONFIG, landuse, slope, flowdir_up)
    forcing = _load_forcing_csv(Path(TP_CONFIG["DAILY_DATA_CSV"]), int(TP_CONFIG["YEAR"]))
    forcing["fert"] = _build_fert_factor(forcing["date"], TP_CONFIG)

    rr = grid_df["row"].to_numpy(dtype=int)
    cc = grid_df["col"].to_numpy(dtype=int)
    lu = grid_df["landuse"].to_numpy(dtype=int)

    crop_mask = lu == 1
    imp_mask = lu == 8
    crop_w = np.full(int(np.sum(crop_mask)), 1.0 / max(int(np.sum(crop_mask)), 1), dtype=float)
    imp_w = np.full(int(np.sum(imp_mask)), 1.0 / max(int(np.sum(imp_mask)), 1), dtype=float)

    annual_crop_map = np.zeros_like(landuse, dtype=float)
    annual_imp_map = np.zeros_like(landuse, dtype=float)
    annual_crop_map[rr[crop_mask], cc[crop_mask]] = CROP_TOTAL * crop_w
    annual_imp_map[rr[imp_mask], cc[imp_mask]] = IMP_TOTAL * imp_w
    annual_total_map = annual_crop_map + annual_imp_map

    rain = forcing["rain"].to_numpy(dtype=float)
    runoff = forcing["runoff"].to_numpy(dtype=float)
    fert = forcing["fert"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(forcing["date"])

    rain_share = _normalize_positive(rain + 0.01)
    runoff_share = _normalize_positive(runoff + 0.01)
    fert_share = _normalize_positive(fert + 0.05)

    old_month_totals = {}
    for month in range(1, 13):
        arr = pd.read_csv(ROOT / "source" / f"09TP_2023{month:02d}.csv").to_numpy(dtype=float)
        old_month_totals[month] = float(np.nansum(arr))
    old_month_daily = np.array(
        [old_month_totals[int(d.month)] / d.days_in_month for d in dates],
        dtype=float,
    )
    old_month_share = _normalize_positive(old_month_daily)

    ang = 2.0 * np.pi * dates.dayofyear.to_numpy(dtype=float) / 365.0
    warm = _normalize_positive(0.55 + 0.45 * np.sin(ang - 0.7))

    crop_daily_share = _normalize_positive(
        0.20 * old_month_share + 0.32 * runoff_share + 0.16 * rain_share + 0.24 * fert_share + 0.08 * warm
    )
    imp_daily_share = _normalize_positive(
        0.20 * old_month_share + 0.42 * runoff_share + 0.25 * rain_share + 0.08 * warm + 0.05 * _normalize_positive(np.ones_like(rain))
    )

    prior_df = pd.DataFrame(
        {
            "date": dates,
            "crop_daily_share": crop_daily_share,
            "impervious_daily_share": imp_daily_share,
            "crop_daily_total_prior": crop_daily_share * CROP_TOTAL,
            "impervious_daily_total_prior": imp_daily_share * IMP_TOTAL,
            "daily_total_prior": crop_daily_share * CROP_TOTAL + imp_daily_share * IMP_TOTAL,
        }
    )
    prior_df.to_csv(DAILY_PRIOR_CSV, index=False, encoding="utf-8-sig")

    for month in range(1, 13):
        month_mask = dates.month == month
        month_crop_total = float((crop_daily_share[month_mask].sum()) * CROP_TOTAL)
        month_imp_total = float((imp_daily_share[month_mask].sum()) * IMP_TOTAL)
        month_map = np.zeros_like(landuse, dtype=float)
        month_map[rr[crop_mask], cc[crop_mask]] = month_crop_total * crop_w
        month_map[rr[imp_mask], cc[imp_mask]] = month_imp_total * imp_w
        pd.DataFrame(month_map).to_csv(OUT_DIR / f"09TP_2023{month:02d}.csv", index=False)

    summary = {
        "annual_total": YEAR_TOTAL,
        "crop_total": CROP_TOTAL,
        "impervious_total": IMP_TOTAL,
        "spatial_allocation_mode": "uniform_within_landuse",
        "forest_total": float(annual_total_map[(landuse == 2) | (landuse == 4)].sum()),
        "crop_cell_count": int(crop_mask.sum()),
        "impervious_cell_count": int(imp_mask.sum()),
        "annual_crop_map_sum": float(annual_crop_map.sum()),
        "annual_imp_map_sum": float(annual_imp_map.sum()),
        "annual_total_map_sum": float(annual_total_map.sum()),
        "max_crop_cell": float(annual_crop_map.max()),
        "max_imp_cell": float(annual_imp_map.max()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "annual_crop_map": annual_crop_map,
        "annual_imp_map": annual_imp_map,
        "annual_total_map": annual_total_map,
        "crop_daily_share": crop_daily_share,
        "imp_daily_share": imp_daily_share,
        "summary": summary,
        "dates": dates,
        "grid_df": grid_df,
    }


def main() -> None:
    result = build_corrected_sources()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved corrected source rasters to: {OUT_DIR}")
    print(f"Saved daily prior file to: {DAILY_PRIOR_CSV}")


if __name__ == "__main__":
    main()
