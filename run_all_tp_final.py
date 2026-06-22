from __future__ import annotations

import json
from pathlib import Path

from src.final_figures import make_final_figures
from src.tp_differentiable_model import train_differentiable_model
from src.tp_ml_baselines import train_ml_baselines
from src.tp_nondiff_daily_model import train_nondiff_model
from src.tp_sensitivity_analysis import run_sensitivity_analysis
from src.tp_stop_gradient_ablation import train_stop_gradient_ablation
from src.tp_uncertainty_ensemble import build_ensemble


ROOT = Path(__file__).resolve().parent


def _read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    pred_dir = ROOT / "results" / "predictions"
    metric_dir = ROOT / "results" / "metrics"

    diff_result = train_differentiable_model(
        pred_dir / "tp_differentiable_predictions.csv",
        metric_dir / "tp_differentiable_metrics.json",
    )
    train_ml_baselines(
        pred_dir / "tp_ml_best_predictions.csv",
        metric_dir / "tp_ml_all_metrics.csv",
        metric_dir / "tp_ml_best_metrics.json",
    )
    train_nondiff_model(
        pred_dir / "tp_nondiff_daily_predictions.csv",
        metric_dir / "tp_nondiff_daily_metrics.json",
    )
    train_stop_gradient_ablation(
        pred_dir / "tp_stop_gradient_predictions.csv",
        metric_dir / "tp_stop_gradient_metrics.json",
    )
    build_ensemble(
        ROOT / "results" / "ensemble" / "tp_ensemble_predictions.csv",
        metric_dir / "tp_ensemble_median_metrics.json",
        diff_result.state_dict,
    )
    run_sensitivity_analysis(ROOT / "results" / "sensitivity" / "tp_parameter_sensitivity.csv", diff_result.state_dict)
    make_final_figures()

    summary_paths = [
        ("Differentiable model", metric_dir / "tp_differentiable_metrics.json"),
        ("Best machine-learning model", metric_dir / "tp_ml_best_metrics.json"),
        ("Non-differentiable daily model", metric_dir / "tp_nondiff_daily_metrics.json"),
        ("Stop-gradient source-generation ablation", metric_dir / "tp_stop_gradient_metrics.json"),
        ("Ensemble median", metric_dir / "tp_ensemble_median_metrics.json"),
    ]
    print("TP final workflow completed.\n")
    for label, path in summary_paths:
        payload = _read_metrics(path)
        metrics = payload["metrics"]
        print(label)
        print(f"  Calibration NSE={metrics['calibration']['nse']:.3f}, R2={metrics['calibration']['r2']:.3f}, RMSE={metrics['calibration']['rmse']:.4f}, PBIAS={metrics['calibration']['pbias']:.2f}")
        print(f"  Validation  NSE={metrics['validation']['nse']:.3f}, R2={metrics['validation']['r2']:.3f}, RMSE={metrics['validation']['rmse']:.4f}, PBIAS={metrics['validation']['pbias']:.2f}")


if __name__ == "__main__":
    main()

