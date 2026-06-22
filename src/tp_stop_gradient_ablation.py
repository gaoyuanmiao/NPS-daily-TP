from __future__ import annotations

from pathlib import Path

from .tp_differentiable_model import TrainResult, train_differentiable_model


def train_stop_gradient_ablation(output_prediction_csv: Path, output_metrics_json: Path, epochs: int = 1600, seed: int = 2027) -> TrainResult:
    return train_differentiable_model(
        output_prediction_csv=output_prediction_csv,
        output_metrics_json=output_metrics_json,
        epochs=epochs,
        seed=seed,
        learn_source_weights=False,
    )
