from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass
class MetricSet:
    nse: float
    r2: float
    rmse: float
    pbias: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _as_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    mask = np.isfinite(arr)
    return arr[mask]


def compute_metrics(observed: Iterable[float], simulated: Iterable[float]) -> MetricSet:
    obs = np.asarray(observed, dtype=float)
    sim = np.asarray(simulated, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    if obs.size == 0:
        return MetricSet(nse=float("nan"), r2=float("nan"), rmse=float("nan"), pbias=float("nan"))

    diff = sim - obs
    den = np.sum((obs - np.mean(obs)) ** 2)
    nse = float(1.0 - np.sum(diff ** 2) / den) if den > 1e-12 else float("nan")
    corr = np.corrcoef(obs, sim)[0, 1] if obs.size > 1 else np.nan
    r2 = float(corr * corr) if np.isfinite(corr) else float("nan")
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    pbias = float(100.0 * np.sum(obs - sim) / np.sum(obs)) if abs(np.sum(obs)) > 1e-12 else float("nan")
    return MetricSet(nse=nse, r2=r2, rmse=rmse, pbias=pbias)


def metrics_by_period(observed: np.ndarray, simulated: np.ndarray, period: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in ("Calibration", "Validation"):
        mask = period == label
        out[label.lower()] = compute_metrics(observed[mask], simulated[mask]).to_dict()
    out["all"] = compute_metrics(observed, simulated).to_dict()
    return out

