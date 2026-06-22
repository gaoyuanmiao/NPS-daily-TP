# TP Daily Model

This TP project has been reorganized to follow the TN daily-model workflow while keeping the true outlet position fixed at `00940067`.

## Primary workflow

Run the corrected physical model:

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\train_tp_direct_physical_model.py
```

This workflow:

- uses `source_corrected_90kg/` as the default TP source set
- keeps the nominal outlet and relies on topology stitching instead of outlet relocation
- produces physical-output metrics without the heavy observation-head correction used in legacy experiments

## Main files

- `tp_daily_loader.py`: shared TP daily-data loader
- `generate_corrected_tp_sources.py`: rebuilds the corrected 90 kg TP source rasters
- `train_tp_direct_physical_model.py`: preferred calibration and spatial-output script
- `source_corrected_90kg/`: corrected monthly source rasters and daily prior

## Legacy branch

`train_torch_tp_split_calibration.py` is kept only as a comparison branch. Its observation operator can hide physical-scale errors, so it should not be used as the main project result.
