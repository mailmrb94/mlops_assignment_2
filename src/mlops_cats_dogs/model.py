"""Serialization and prediction helpers shared by training and serving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import extract_features
from .preprocessing import preprocess_image

CLASS_NAMES = ["cat", "dog"]


def save_model(bundle: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)


def load_model(path: str | Path) -> dict[str, Any]:
    model_path = Path(path)
    bundle = joblib.load(model_path)
    backend = str(bundle.get("backend", "legacy_hog_random_forest"))
    required = {"classifier", "class_names", "version"}
    if backend == "legacy_hog_random_forest":
        required.update({"feature_size", "histogram_bins"})
    elif backend == "onnx_mobilenet_v3_small":
        required.update(
            {"input_size", "imagenet_mean", "imagenet_std", "onnx_model", "onnx_sha256"}
        )
    else:
        raise ValueError(f"unsupported model backend: {backend}")
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"invalid model bundle; missing: {sorted(missing)}")
    if backend == "onnx_mobilenet_v3_small":
        import hashlib

        import onnxruntime as ort

        onnx_path = model_path.with_name(str(bundle["onnx_model"]))
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX feature extractor not found: {onnx_path}")
        digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        if digest != str(bundle["onnx_sha256"]):
            raise ValueError("ONNX feature extractor checksum does not match model bundle")
        bundle["_onnx_session"] = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
    return bundle


def predict_image(bundle: dict[str, Any], source: Any) -> dict[str, Any]:
    backend = str(bundle.get("backend", "legacy_hog_random_forest"))
    if backend == "onnx_mobilenet_v3_small":
        input_size = int(bundle["input_size"])
        image = preprocess_image(source, size=(input_size, input_size))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray(bundle["imagenet_mean"], dtype=np.float32)
        std = np.asarray(bundle["imagenet_std"], dtype=np.float32)
        model_input = ((pixels - mean) / std).transpose(2, 0, 1)[None, ...]
        session = bundle.get("_onnx_session")
        if session is None:
            raise ValueError("ONNX model session is not initialized; use load_model()")
        input_name = session.get_inputs()[0].name
        features = np.asarray(session.run(None, {input_name: model_input})[0])
    else:
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
