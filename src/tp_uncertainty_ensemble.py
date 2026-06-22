from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data_loader import load_tp_dataset
from .metrics import metrics_by_period
from .tp_differentiable_model import DirectPhysicalTPModel


def build_ensemble(output_csv: Path, output_metrics_json: Path, state_dict: dict[str, torch.Tensor], ensemble_size: int = 120, seed: int = 2029) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    dataset = load_tp_dataset()
    df = dataset.frame.copy()
    base_model = DirectPhysicalTPModel(dataset=dataset, learn_source_weights=True)
    base_model.load_state_dict(state_dict, strict=False)

    with torch.no_grad():
        base_out = base_model()
    base_sim = base_out["sim"].detach().cpu().numpy()
    calib_resid = df.loc[df["period"] == "Calibration", "TP"].to_numpy(dtype=float) - base_sim[df["period"].to_numpy() == "Calibration"]

    members = []
    for _ in range(ensemble_size):
        member = DirectPhysicalTPModel(dataset=dataset, learn_source_weights=True)
        perturbed = {}
        for name, tensor in state_dict.items():
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            scale = 0.08 if arr.ndim > 0 else 0.06
            if arr.shape:
                noise = rng.normal(0.0, scale, size=arr.shape).astype(np.float32)
            else:
                noise = np.float32(rng.normal(0.0, scale))
            if arr.shape:
                perturbed[name] = torch.as_tensor(arr * (1.0 + noise), dtype=tensor.dtype)
            else:
                perturbed[name] = torch.as_tensor(arr + noise, dtype=tensor.dtype)
        member.load_state_dict(perturbed, strict=False)
        with torch.no_grad():
            sim = member()["sim"].detach().cpu().numpy()
        resid_sample = rng.choice(calib_resid, size=len(sim), replace=True)
        member_pred = np.clip(sim + 0.55 * resid_sample, 0.0, None)
        members.append(member_pred)

    ensemble = np.vstack(members)
    summary = pd.DataFrame(
        {
            "date": df["date"],
            "observed_tp": df["TP"],
            "median_tp": np.median(ensemble, axis=0),
            "q05_tp": np.quantile(ensemble, 0.05, axis=0),
            "q25_tp": np.quantile(ensemble, 0.25, axis=0),
            "q75_tp": np.quantile(ensemble, 0.75, axis=0),
            "q95_tp": np.quantile(ensemble, 0.95, axis=0),
            "period": df["period"],
        }
    )
    metrics = metrics_by_period(summary["observed_tp"].to_numpy(dtype=float), summary["median_tp"].to_numpy(dtype=float), summary["period"].to_numpy())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    output_metrics_json.write_text(
        json.dumps({"metrics": metrics, "ensemble_size": ensemble_size}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary, metrics
