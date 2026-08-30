"""Dispatch reproducible model training to the configured implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_model(
    data_dir: str | Path,
    model_path: str | Path,
    artifact_dir: str | Path,
    params: dict[str, Any],
) -> dict[str, float]:
    model_type = str(params.get("model_type", "mobilenet_v3_small_transfer"))
    if model_type != "mobilenet_v3_small_transfer":
        raise ValueError(f"unsupported configured model_type: {model_type}")
    from .transfer_learning import train_transfer_model

    return train_transfer_model(data_dir, model_path, artifact_dir, params)
