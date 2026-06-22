from __future__ import annotations

import json
import runpy
import sys
import traceback
from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parent
COMMON_DIR = ANALYSIS_ROOT / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from tp_model_access import build_full_model_prediction_df, build_full_model_spatial_data, build_process_ensemble
from tp_paper_config import FIG_DIR
from tp_paper_utils import read_prediction_file, split_train_val


SCRIPTS = [
    "01_baselines/tp_simple_physical_baseline.py",
    "01_baselines/tp_ml_baseline.py",
    "02_validation/tp_validation_diagnostics.py",
    "03_spatial_management/tp_spatial_hotspot_analysis.py",
    "03_spatial_management/tp_cumulative_contribution_analysis.py",
    "04_sensitivity_uncertainty/tp_local_sensitivity_improved.py",
    "04_sensitivity_uncertainty/tp_uncertainty_coverage_analysis.py",
    "04_sensitivity_uncertainty/tp_stability_scatter_enhanced.py",
]

TARGETS = [
    "figure_tp_baseline_model_comparison.png",
    "figure_tp_train_val_timeseries_residuals.png",
    "figure_tp_season_event_validation.png",
    "figure_tp_source_vs_export_hotspots.png",
    "figure_tp_cumulative_contribution_curve.png",
    "figure_tp_local_sensitivity_improved.png",
    "figure_tp_uncertainty_coverage_width.png",
    "figure_tp_stability_scatter_enhanced.png",
]


def main() -> None:
    df = read_prediction_file()
    train_df, val_df = split_train_val(df)
    print(f"Using TP predictions file: {df.attrs.get('source_path', 'unknown')}")
    print(f"Calibration period: {train_df['date'].iloc[0].date()} to {train_df['date'].iloc[-1].date()} ({len(train_df)} days)")
    print(f"Validation period: {val_df['date'].iloc[0].date()} to {val_df['date'].iloc[-1].date()} ({len(val_df)} days)")
    build_full_model_prediction_df(force=True)
    build_full_model_spatial_data(force=True)
    build_process_ensemble(force=True)

    all_ok = True
    for rel_path in SCRIPTS:
        script_path = ANALYSIS_ROOT / rel_path
        print(f"[RUN] {rel_path}")
        try:
            runpy.run_path(str(script_path), run_name="__main__")
            print(f"[OK ] {rel_path}")
        except Exception as exc:
            all_ok = False
            print(f"[ERR] {rel_path}: {exc}")
            traceback.print_exc()

    spatial_summary = ANALYSIS_ROOT / "outputs" / "intermediate" / "tp_spatial_hotspot_summary.json"
    if spatial_summary.exists():
        info = json.loads(spatial_summary.read_text(encoding="utf-8"))
        print(f"Annual TP source total: {info.get('annual_source_total'):.6f}")
        print(f"Annual cell contribution total: {info.get('annual_cell_contribution_total'):.6f}")
        print(f"Used contribution proxy: {info.get('used_proxy')}")

    unc_notes = ANALYSIS_ROOT / "outputs" / "intermediate" / "tp_uncertainty_notes.txt"
    if unc_notes.exists():
        print(f"Uncertainty ensemble source: {unc_notes.read_text(encoding='utf-8').splitlines()[1]}")
    stab_notes = ANALYSIS_ROOT / "outputs" / "intermediate" / "tp_stability_notes.txt"
    if stab_notes.exists():
        print(f"Stability ensemble source: {stab_notes.read_text(encoding='utf-8').splitlines()[0]}")

    missing = [name for name in TARGETS if not (FIG_DIR / name).exists()]
    for name in TARGETS:
        if (FIG_DIR / name).exists():
            print(f"Generated figure: {FIG_DIR / name}")
    print(f"All target figures generated: {len(missing) == 0 and all_ok}")
    if missing:
        print(f"Missing figures: {missing}")


if __name__ == "__main__":
    main()
