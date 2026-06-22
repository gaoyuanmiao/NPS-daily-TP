from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_paper_tp"
OUTPUT_ROOT = ANALYSIS_ROOT / "outputs"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
INTER_DIR = OUTPUT_ROOT / "intermediate"

for _path in (ANALYSIS_ROOT, OUTPUT_ROOT, TABLE_DIR, FIG_DIR, INTER_DIR):
    _path.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
LOCAL_PERTURB_FRAC = 0.10
GLOBAL_N_RUNS = 100
FIG_DPI = 300

FIGSIZE_WIDE = (13.5, 5.8)
FIGSIZE_GRID = (12.5, 10.0)
FIGSIZE_ROW3 = (15.0, 4.5)
FIGSIZE_SQUARE = (7.2, 6.5)

FONT_FAMILY = ["Times New Roman", "DejaVu Serif"]
FONT_SIZE = 11
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10

TP_PLOT_UNIT = "kg d$^{-1}$"
TP_SPATIAL_UNIT = "kg yr$^{-1}$ cell$^{-1}$"

FULL_MODEL_PREDICTION_CSV = INTER_DIR / "tp_full_model_predictions.csv"
FULL_MODEL_SPATIAL_NPZ = INTER_DIR / "tp_full_model_spatial_maps.npz"
FULL_MODEL_SPATIAL_SUMMARY_JSON = INTER_DIR / "tp_full_model_spatial_summary.json"
ENSEMBLE_PREDICTIONS_CSV = INTER_DIR / "tp_process_ensemble_predictions.csv"
