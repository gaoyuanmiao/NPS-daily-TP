from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tp_differentiable_model import train_differentiable_model


if __name__ == "__main__":
    train_differentiable_model(
        ROOT / "results" / "predictions" / "tp_differentiable_predictions.csv",
        ROOT / "results" / "metrics" / "tp_differentiable_metrics.json",
    )
