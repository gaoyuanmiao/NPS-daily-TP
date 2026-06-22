# TP Daily Model Final

This project is the finalized TP daily grid-scale model workflow. It keeps TP-specific data, parameters, predictions, figures, and diagnostics while aligning the experiment structure and figure style with the TN final project. The TN project is only a reference for workflow organization and plotting style; all TP results in this repository are recomputed from TP data.

## Scope

The finalized workflow covers:

1. Differentiable TP model based on corrected daily TP source priors.
2. Best machine-learning comparison model selected from multiple candidate regressors.
3. Non-differentiable daily TP model calibrated without automatic differentiation.
4. Stop-gradient source-generation ablation.
5. Uncertainty ensemble with 50% and 90% prediction intervals.
6. Raw physical TP output parameter sensitivity analysis.
7. Four final TP figures aligned with the TN final paper-style layout.

## Inputs

- `input/forcing/daily_data.csv`: daily rainfall and runoff for 2023.
- `input/forcing/obs.csv`: daily TP and TN observations for 2023.
- `source_corrected_90kg/tp_daily_source_prior_corrected.csv`: corrected daily TP source priors used by the finalized TP workflow.
- `source_corrected_90kg/09TP_*.csv`: monthly corrected TP source rasters retained for reproducibility.
- `input/landuse/`, `input/slope/`, `input/flow/`: spatial inputs retained from the TP project.

The finalized split follows the current TP main-model convention: 2023 daily records with a fixed 70% calibration period and 30% validation period. The exact split date is written into the metrics JSON for the differentiable model.

## Final Structure

- `src/`: final reusable TP workflow modules.
- `scripts/`: stepwise runnable scripts for each experiment.
- `results/predictions/`: daily predictions from each model.
- `results/metrics/`: calibration, validation, and full-period metrics.
- `results/ensemble/`: ensemble summary predictions and median metrics.
- `results/sensitivity/`: parameter sensitivity table.
- `results/spatial/`: reserved for final spatial outputs if needed.
- `figures/`: four final TP figures in PNG and PDF.
- `_archive_legacy/`: legacy code, intermediate outputs, and superseded files moved out of the main workflow.

## Experiments

### Differentiable TP model

- Module: `src/tp_differentiable_model.py`
- Script: `scripts/01_train_differentiable_tp.py`
- Outputs:
  - `results/predictions/tp_differentiable_predictions.csv`
  - `results/metrics/tp_differentiable_metrics.json`

### Best machine-learning model

- Module: `src/tp_ml_baselines.py`
- Script: `scripts/02_train_ml_baselines_tp.py`
- Candidate models include Random Forest, ExtraTrees, GradientBoosting, MLP, plus XGBoost or LightGBM when those libraries are available.
- Outputs:
  - `results/predictions/tp_ml_best_predictions.csv`
  - `results/metrics/tp_ml_all_metrics.csv`
  - `results/metrics/tp_ml_best_metrics.json`

### Non-differentiable daily TP model

- Module: `src/tp_nondiff_daily_model.py`
- Script: `scripts/03_train_nondiff_daily_tp.py`
- Output files:
  - `results/predictions/tp_nondiff_daily_predictions.csv`
  - `results/metrics/tp_nondiff_daily_metrics.json`

### Stop-gradient source-generation ablation

- Module: `src/tp_stop_gradient_ablation.py`
- Script: `scripts/04_train_stop_gradient_ablation_tp.py`
- Output files:
  - `results/predictions/tp_stop_gradient_predictions.csv`
  - `results/metrics/tp_stop_gradient_metrics.json`

## Uncertainty

- Module: `src/tp_uncertainty_ensemble.py`
- Script: `scripts/05_run_uncertainty_ensemble_tp.py`
- Output files:
  - `results/ensemble/tp_ensemble_predictions.csv`
  - `results/metrics/tp_ensemble_median_metrics.json`

## Sensitivity

- Module: `src/tp_sensitivity_analysis.py`
- Script: `scripts/06_run_sensitivity_tp.py`
- Output files:
  - `results/sensitivity/tp_parameter_sensitivity.csv`
  - `figures/figure_tp_parameter_sensitivity.png`
  - `figures/figure_tp_parameter_sensitivity.pdf`

## Final Figures

- Script: `scripts/07_make_final_figures_tp.py`
- Outputs:
  - `figures/figure_tp_timeseries_interval.png`
  - `figures/figure_tp_timeseries_interval.pdf`
  - `figures/figure_tp_scatter_model_comparison.png`
  - `figures/figure_tp_scatter_model_comparison.pdf`
  - `figures/figure_tp_parameter_sensitivity.png`
  - `figures/figure_tp_parameter_sensitivity.pdf`
  - `figures/figure_tp_timeseries_final_model.png`
  - `figures/figure_tp_timeseries_final_model.pdf`

## Reproduce Everything

Run the full finalized TP workflow with:

```bash
C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe run_all_tp_final.py
```

This runs, in order:

1. Differentiable TP model.
2. Machine-learning baselines.
3. Non-differentiable daily TP model.
4. Stop-gradient ablation.
5. Uncertainty ensemble.
6. Sensitivity analysis.
7. Final TP figure generation.
8. Metric summary printing.

## Versioning And Rollback

This repository already contains backup tags created during the cleanup process. To inspect or restore a saved version:

```bash
git tag
git checkout <tag-name>
git checkout -b restore-from-tag <tag-name>
git reset --hard <tag-name>
```

## Notes

- All plotted and reported loads use `TP load (kg d⁻¹)`.
- No TN predictions are reused here.
- TN only informed the layout and comparison logic of the final workflow.
