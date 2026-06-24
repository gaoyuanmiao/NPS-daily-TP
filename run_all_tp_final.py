from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.final_figures import make_final_figures
from src.tp_consistency_checks import run_prediction_consistency_checks
from src.tp_differentiable_model import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_SOURCE_GENERATION_PATH,
    train_differentiable_model,
)
from src.tp_ml_baselines import train_ml_baselines
from src.tp_nondiff_daily_model import train_nondiff_model
from src.tp_sensitivity_analysis import run_sensitivity_analysis
from src.tp_stop_gradient_ablation import train_stop_gradient_ablation
from src.tp_uncertainty_ensemble import build_ensemble


ROOT = Path(__file__).resolve().parent


def _read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_comparison_summary(metric_dir: Path, output_csv: Path) -> pd.DataFrame:
    model_specs = [
        ("Differentiable model", metric_dir / "tp_differentiable_metrics.json"),
        ("Best machine-learning model", metric_dir / "tp_ml_best_metrics.json"),
        ("Non-differentiable daily model", metric_dir / "tp_nondiff_daily_metrics.json"),
        ("Stop-gradient source-generation ablation", metric_dir / "tp_stop_gradient_metrics.json"),
    ]
    rows = []
    for model_name, path in model_specs:
        payload = _read_metrics(path)["metrics"]
        rows.append(
            {
                "model": model_name,
                "cal_nse": payload["calibration"]["nse"],
                "val_nse": payload["validation"]["nse"],
                "cal_r2": payload["calibration"]["r2"],
                "val_r2": payload["validation"]["r2"],
                "cal_rmse": payload["calibration"]["rmse"],
                "val_rmse": payload["validation"]["rmse"],
                "cal_pbias": payload["calibration"]["pbias"],
                "val_pbias": payload["validation"]["pbias"],
                "all_nse": payload["all"]["nse"],
                "all_r2": payload["all"]["r2"],
                "all_rmse": payload["all"]["rmse"],
                "all_pbias": payload["all"]["pbias"],
            }
        )
    summary = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    return summary


def main() -> None:
    pred_dir = ROOT / "results" / "predictions"
    metric_dir = ROOT / "results" / "metrics"
    ensemble_dir = ROOT / "results" / "ensemble"
    sensitivity_dir = ROOT / "results" / "sensitivity"

    diff_result = train_differentiable_model(
        pred_dir / "tp_differentiable_predictions.csv",
        metric_dir / "tp_differentiable_metrics.json",
        source_generation_csv=DEFAULT_SOURCE_GENERATION_PATH,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
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
        source_generation_csv=DEFAULT_SOURCE_GENERATION_PATH,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    )
    build_ensemble(
        ensemble_dir / "tp_ensemble_predictions.csv",
        metric_dir / "tp_ensemble_median_metrics.json",
        diff_result.state_dict,
        ensemble_size=120,
    )
    run_sensitivity_analysis(sensitivity_dir / "tp_parameter_sensitivity.csv", diff_result.state_dict)
    check_report = run_prediction_consistency_checks(ROOT)
    figure_paths = make_final_figures()
    summary = build_model_comparison_summary(metric_dir, metric_dir / "tp_final_model_comparison_summary.csv")

    expected_paths = [
        pred_dir / "tp_differentiable_predictions.csv",
        pred_dir / "tp_differentiable_source_generation.csv",
        pred_dir / "tp_ml_best_predictions.csv",
        pred_dir / "tp_nondiff_daily_predictions.csv",
        pred_dir / "tp_stop_gradient_predictions.csv",
        ensemble_dir / "tp_ensemble_predictions.csv",
        sensitivity_dir / "tp_parameter_sensitivity.csv",
        metric_dir / "tp_differentiable_metrics.json",
        metric_dir / "tp_ml_best_metrics.json",
        metric_dir / "tp_nondiff_daily_metrics.json",
        metric_dir / "tp_stop_gradient_metrics.json",
        metric_dir / "tp_ensemble_median_metrics.json",
        metric_dir / "tp_final_model_comparison_summary.csv",
        ROOT / "figures" / "figure_tp_timeseries_interval.png",
        ROOT / "figures" / "figure_tp_scatter_model_comparison.png",
        ROOT / "figures" / "figure_tp_parameter_sensitivity.png",
        ROOT / "figures" / "figure_tp_timeseries_final_model.png",
    ]
    missing = [str(path) for path in expected_paths if not path.exists()]
    warnings: list[str] = []
    stop_val = float(summary.loc[summary["model"] == "Stop-gradient source-generation ablation", "val_nse"].iloc[0])
    diff_val = float(summary.loc[summary["model"] == "Differentiable model", "val_nse"].iloc[0])
    if stop_val > diff_val:
        warnings.append(
            "Stop-gradient validation NSE is still higher than the full differentiable model. Check validation-period stability, source-prior strength, potential calibration overfit, ablation definition, or seed sensitivity."
        )

    print("TP final workflow completed.\n")
    for _, row in summary.iterrows():
        print(
            f"{row['model']}: "
            f"Calibration NSE={row['cal_nse']:.3f}, R2={row['cal_r2']:.3f}; "
            f"Validation NSE={row['val_nse']:.3f}, R2={row['val_r2']:.3f}"
        )
    print("\nFinal figures:")
    for path in figure_paths:
        print(f"  {path}")
    print("\nConsistency checks:")
    print(f"  Calibration/Validation counts consistent: {check_report['calibration_count']} / {check_report['validation_count']}")
    print(f"  Figure text contains TN: {check_report['figure_text_has_tn']}")
    print(f"  Missing files: {len(missing)}")
    if missing:
        for path in missing:
            print(f"    {path}")
    print(f"  Warning count: {len(warnings)}")
    for item in warnings:
        print(f"    WARNING: {item}")


if __name__ == "__main__":
    main()
