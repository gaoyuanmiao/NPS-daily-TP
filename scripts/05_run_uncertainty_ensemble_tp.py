from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tp_differentiable_model import train_differentiable_model
from src.tp_uncertainty_ensemble import build_ensemble


if __name__ == "__main__":
    pred_path = ROOT / "results" / "predictions" / "tp_differentiable_predictions.csv"
    metric_path = ROOT / "results" / "metrics" / "tp_differentiable_metrics.json"
    result = train_differentiable_model(pred_path, metric_path)
    build_ensemble(
        ROOT / "results" / "ensemble" / "tp_ensemble_predictions.csv",
        ROOT / "results" / "metrics" / "tp_ensemble_median_metrics.json",
        result.state_dict,
    )
