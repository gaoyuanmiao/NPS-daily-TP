# TP Daily Model Final

This repository is the cleaned TP daily grid-scale workflow. It keeps TP-only data, source priors, model code, figures, and diagnostics, while aligning the experiment logic with the retained TN final workflow. TN is used only as a workflow and plotting reference. All model fitting, metrics, ablations, uncertainty, sensitivity, and figures here are rebuilt from TP data.

## What Was Fixed

The current workflow removes the earlier experiment-logic errors:

1. Calibration is the only period used for model training.
2. Final validation is never used in loss functions, checkpoint selection, early stopping, or model selection.
3. The stop-gradient source-generation ablation is now a real detached-source ablation:
   - the full differentiable TP model is trained first
   - its daily source-generation outputs are exported
   - those source outputs are fixed and detached
   - only downstream release, transport, attenuation, and memory parameters are updated
4. PBIAS is unified as:

```text
PBIAS = 100 × sum(simulated - observed) / sum(observed)
```

Positive PBIAS means overestimation. Negative PBIAS means underestimation.

## Inputs

- `input/forcing/daily_data.csv`: 2023 daily rainfall and runoff
- `input/forcing/obs.csv`: 2023 daily TP and TN observations
- `source_corrected_90kg/tp_daily_source_prior_corrected.csv`: corrected daily TP source priors used by the final TP workflow
- `source_corrected_90kg/09TP_*.csv`: monthly corrected TP source rasters retained for reproducibility
- `input/landuse/`, `input/slope/`, `input/flow/`: TP spatial inputs retained from the original TP project

The split is fixed at 70% calibration and 30% validation within 2023 daily records. The differentiable-model metrics JSON records the exact split date.

## Main Outputs

### Predictions

- `results/predictions/tp_differentiable_predictions.csv`
- `results/predictions/tp_differentiable_source_generation.csv`
- `results/predictions/tp_ml_best_predictions.csv`
- `results/predictions/tp_nondiff_daily_predictions.csv`
- `results/predictions/tp_stop_gradient_predictions.csv`

### Metrics

- `results/metrics/tp_differentiable_metrics.json`
- `results/metrics/tp_ml_all_metrics.csv`
- `results/metrics/tp_ml_best_metrics.json`
- `results/metrics/tp_nondiff_daily_metrics.json`
- `results/metrics/tp_stop_gradient_metrics.json`
- `results/metrics/tp_ensemble_median_metrics.json`
- `results/metrics/tp_final_model_comparison_summary.csv`

### Ensemble And Sensitivity

- `results/ensemble/tp_ensemble_predictions.csv`
- `results/sensitivity/tp_parameter_sensitivity.csv`

### Figures

- `figures/figure_tp_timeseries_interval.png`
- `figures/figure_tp_timeseries_interval.pdf`
- `figures/figure_tp_scatter_model_comparison.png`
- `figures/figure_tp_scatter_model_comparison.pdf`
- `figures/figure_tp_parameter_sensitivity.png`
- `figures/figure_tp_parameter_sensitivity.pdf`
- `figures/figure_tp_timeseries_final_model.png`
- `figures/figure_tp_timeseries_final_model.pdf`

## Experiment Logic

### Differentiable TP model

- Module: `src/tp_differentiable_model.py`
- Script: `scripts/01_train_differentiable_tp.py`
- Uses corrected TP daily source priors
- Loss uses calibration observations only
- Final validation is computed only after training finishes
- Exports both:
  - `raw_physical_tp`
  - `simulated_tp`

### Stop-gradient source-generation ablation

- Module: `src/tp_stop_gradient_ablation.py`
- Script: `scripts/04_train_stop_gradient_ablation_tp.py`
- Formal meaning:
  - full-model daily source generation is exported first
  - exported source generation is reused as fixed detached input
  - source-generation parameters are not updated
  - downstream physical parameters remain trainable
- This is different from:
  - zeroing source coefficients
  - directly using the corrected source prior without detached full-model source outputs

### Non-differentiable daily model

- Module: `src/tp_nondiff_daily_model.py`
- Script: `scripts/03_train_nondiff_daily_tp.py`
- Uses the same TP daily physical structure in non-autodiff form
- Objective uses calibration only
- Validation is reported only after optimization

### Machine-learning comparison

- Module: `src/tp_ml_baselines.py`
- Script: `scripts/02_train_ml_baselines_tp.py`
- Models train only on calibration data
- Best model selection uses calibration-only internal time-series cross-validation
- Final validation is reported once and is not used for model choice

### Uncertainty ensemble

- Module: `src/tp_uncertainty_ensemble.py`
- Script: `scripts/05_run_uncertainty_ensemble_tp.py`
- Uses the calibration-only-trained differentiable base model
- Uses calibration residual bootstrap only
- Reports median, q05, q25, q75, and q95

### Sensitivity analysis

- Module: `src/tp_sensitivity_analysis.py`
- Script: `scripts/06_run_sensitivity_tp.py`
- Uses the corrected differentiable TP model
- Perturbs raw-physical-output parameters one by one
- Normalizes the largest sensitivity to 100%

## Consistency Checks

Before figures are created, the workflow checks:

1. `date` is identical across the four model prediction files
2. `observed_tp` is identical across the four model prediction files
3. `period` labels are identical across the four model prediction files
4. calibration and validation sample counts are consistent
5. the stop-gradient prediction is not identical to the full differentiable prediction
6. the stop-gradient metrics file declares `stop_gradient_source_generation`
7. figure text templates do not contain `TN`

The checker lives in `src/tp_consistency_checks.py`.

## One-Click Reproduction

Run the full finalized TP workflow with:

```bash
C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe run_all_tp_final.py
```

The full workflow runs in this order:

1. train differentiable TP model
2. train machine-learning baselines
3. train non-differentiable daily TP model
4. train stop-gradient source-generation ablation
5. run uncertainty ensemble
6. run sensitivity analysis
7. run consistency checks
8. make final figures
9. write model comparison summary

## Notes

- All plotted and reported TP loads use `TP load (kg d⁻¹)`.
- Validation metrics are independent evaluation outputs only.
- `_archive_legacy/` keeps older code and superseded files for reference, but the active final workflow is only the top-level `src/`, `scripts/`, `results/`, and `figures/` structure.
