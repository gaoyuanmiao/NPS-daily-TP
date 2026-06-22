from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tp_stop_gradient_ablation import train_stop_gradient_ablation


if __name__ == "__main__":
    train_stop_gradient_ablation(
        ROOT / "results" / "predictions" / "tp_stop_gradient_predictions.csv",
        ROOT / "results" / "metrics" / "tp_stop_gradient_metrics.json",
    )
