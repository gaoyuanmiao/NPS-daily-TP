from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tp_nondiff_daily_model import train_nondiff_model


if __name__ == "__main__":
    train_nondiff_model(
        ROOT / "results" / "predictions" / "tp_nondiff_daily_predictions.csv",
        ROOT / "results" / "metrics" / "tp_nondiff_daily_metrics.json",
    )
