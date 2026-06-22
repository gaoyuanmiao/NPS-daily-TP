from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tp_ml_baselines import train_ml_baselines


if __name__ == "__main__":
    train_ml_baselines(
        ROOT / "results" / "predictions" / "tp_ml_best_predictions.csv",
        ROOT / "results" / "metrics" / "tp_ml_all_metrics.csv",
        ROOT / "results" / "metrics" / "tp_ml_best_metrics.json",
    )
