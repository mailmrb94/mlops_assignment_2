"""Serialization and prediction helpers shared by training and serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import extract_features

CLASS_NAMES = ["cat", "dog"]


def save_model(bundle: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)


def load_model(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    required = {"classifier", "class_names", "feature_size", "histogram_bins", "version"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"invalid model bundle; missing: {sorted(missing)}")
    return bundle


def predict_image(bundle: dict[str, Any], source: Any) -> dict[str, Any]:
    features = extract_features(
        source,
        feature_size=int(bundle["feature_size"]),
        histogram_bins=int(bundle["histogram_bins"]),
    ).reshape(1, -1)
    probabilities = bundle["classifier"].predict_proba(features)[0]
    classes = [str(name) for name in bundle["class_names"]]
    by_label = {name: float(probabilities[index]) for index, name in enumerate(classes)}
    label = max(by_label, key=by_label.get)
    return {
        "label": label,
        "confidence": by_label[label],
        "probabilities": by_label,
        "model_version": str(bundle["version"]),
    }

