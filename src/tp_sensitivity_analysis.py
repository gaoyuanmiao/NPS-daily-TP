from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data_loader import load_tp_dataset
from .tp_differentiable_model import DirectPhysicalTPModel


PARAM_MAP = [
    ("crop_release_coefficient", "gamma_crop", 1),
    ("impervious_release_coefficient", "gamma_imp", 1),
    ("crop_transport_efficiency", "delta_crop", 1),
    ("impervious_transport_efficiency", "delta_imp", 1),
    ("memory_coefficient", "mem_raw", None),
    ("runoff_response_coefficient", "beta_crop", 1),
    ("rainfall_response_coefficient", "beta_crop", 2),
    ("source_allocation_coefficient", "beta_imp", 1),
    ("seasonal_coefficient", "beta_crop", 5),
    ("fertilization_timing_coefficient", "beta_crop", 4),
    ("routing_attenuation_coefficient", "eff_crop_raw", None),
    ("legacy_release_coefficient", "k_crop_raw", None),
    ("surface_partition_coefficient", "eff_imp_raw", None),
]


def run_sensitivity_analysis(output_csv: Path, state_dict: dict[str, torch.Tensor]) -> pd.DataFrame:
    dataset = load_tp_dataset()
    base_model = DirectPhysicalTPModel(dataset=dataset, learn_source_weights=True)
    base_model.load_state_dict(state_dict, strict=False)
    with torch.no_grad():
        base_raw = base_model()["raw"].detach().cpu().numpy()

    rows = []
    for label, tensor_name, index in PARAM_MAP:
        member = DirectPhysicalTPModel(dataset=dataset, learn_source_weights=True)
        member.load_state_dict(state_dict, strict=False)
        with torch.no_grad():
            param = getattr(member, tensor_name)
            if index is None:
                param.copy_(param * 1.10 if abs(float(param.item())) > 1e-8 else param + 0.10)
            else:
                updated = param.clone()
                updated[index] = updated[index] * 1.10 if abs(float(updated[index].item())) > 1e-8 else updated[index] + 0.10
                param.copy_(updated)
            perturbed_raw = member()["raw"].detach().cpu().numpy()

        sensitivity = float(np.mean(np.abs(perturbed_raw - base_raw)))
        rows.append({"parameter": label, "sensitivity": sensitivity})

    result = pd.DataFrame(rows).sort_values("sensitivity", ascending=False).reset_index(drop=True)
    max_value = float(result["sensitivity"].max()) if not result.empty else 1.0
    result["relative_sensitivity"] = 100.0 * result["sensitivity"] / max(max_value, 1e-12)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result

