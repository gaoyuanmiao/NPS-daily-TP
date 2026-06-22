TP Daily Model Package

This folder contains the cleaned TP-only daily model package aligned to the TN workflow.

Included:
- input/: all shared spatial inputs required by the TP model
- source/: legacy monthly TP source rasters kept only for backtracking
- source_corrected_90kg/: corrected monthly TP source rasters with annual total constrained to about 90 kg
- tp_daily_loader.py: loader configured to use the packaged relative source directory
- train_tp_direct_physical_model.py: primary physical-output calibration script
- train_torch_tp_split_calibration.py: legacy experiment script kept for comparison only
- shared runtime code: data_loader.py, model_components_numpy.py, model_components_torch.py
- raw TP data files: tp_parameter_raw.xlsx, tp_q_day.xlsx, tp_rain.xls, tp_water_quality.xlsx
- exported daily TP inputs: tp_daily_forcing_2023.csv, tp_daily_obs_2023.csv, tp_daily_meta_2023.json
- primary checkpoint and outputs:
  tp_direct_model_direct_physical_90kg_tp.pt
  tp_direct_summary_direct_physical_90kg_tp.json
  tp_direct_predictions_direct_physical_90kg_tp.csv
  tp_direct_spatial_direct_physical_90kg_tp_bestday.png/json
  tp_direct_annual_contribution_direct_physical_90kg_tp.png

Current preferred TP version:
- tag: direct_physical_90kg_tp
- full NSE/R2: about 0.6940 / 0.6962
- validation NSE/R2: about 0.8020 / 0.8059

Current package direction:
- preserve the actual outlet location `00940067`
- use topology stitching rather than outlet relocation
- keep physical output on the same scale as observations by using the corrected 90 kg source set
- treat the split-calibration observation head as a legacy branch, not the main result
