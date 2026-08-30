"""MobileNetV3 transfer-learning pipeline with ONNX production inference."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from PIL import ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .model import CLASS_NAMES, save_model
from .preprocessing import preprocess_image

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
BASELINE_METRICS = {
    "test_accuracy": 0.7,
    "test_precision": 0.6764705882352942,
    "test_recall": 0.7666666666666667,
    "test_f1": 0.71875,
    "test_log_loss": 0.6196187197415365,
}


def _split_paths(root: Path, split: str) -> tuple[list[Path], np.ndarray]:
    paths: list[Path] = []
    labels: list[int] = []
    for label_index, label in enumerate(CLASS_NAMES):
        class_dir = root / split / f"{label}s"
        class_paths = sorted(path for path in class_dir.glob("*.*") if path.is_file())
        paths.extend(class_paths)
        labels.extend([label_index] * len(class_paths))
    if not paths:
        raise FileNotFoundError(f"no images found for split '{split}' under {root}")
    return paths, np.asarray(labels)


def _image_tensor(path: Path, *, mirrored: bool = False):
    import torch

    image = preprocess_image(path)
    if mirrored:
        image = ImageOps.mirror(image)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(pixels.transpose(2, 0, 1))


def _load_feature_extractor():
    import torch
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    class MobileNetFeatures(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            self.features = backbone.features
            self.avgpool = backbone.avgpool

        def forward(self, images):
            return torch.flatten(self.avgpool(self.features(images)), 1)

    torch.manual_seed(42)
    return MobileNetFeatures().eval()


def _extract_embeddings(
    paths: list[Path], feature_extractor, *, batch_size: int, mirrored: bool = False
) -> np.ndarray:
    import torch

    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch = torch.stack(
                [_image_tensor(path, mirrored=mirrored) for path in batch_paths]
            )
            batches.append(feature_extractor(batch).cpu().numpy().astype(np.float32))
    return np.vstack(batches)


def _candidate(name: str, regularization: float, seed: int):
    if name == "logistic_regression":
        estimator = LogisticRegression(
            C=regularization,
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        )
    elif name == "rbf_svc":
        estimator = SVC(
            C=regularization,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )
    elif name == "linear_svc":
        estimator = SVC(
            C=regularization,
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        raise ValueError(f"unsupported classifier candidate: {name}")
    return make_pipeline(StandardScaler(), estimator)


def _export_onnx(feature_extractor, output: Path) -> None:
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        feature_extractor,
        example,
        str(output),
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def _write_comparison_plot(
    candidate_rows: list[dict[str, Any]], metrics: dict[str, float], output: Path
) -> None:
    import matplotlib.pyplot as plt

    labels = [
        f"{row['variant']}\n{row['classifier']} C={row['regularization']}"
        for row in candidate_rows
    ]
    validation_scores = [row["validation_accuracy"] for row in candidate_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(range(len(labels)), validation_scores, color="#176B87")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title("Validation-only candidate selection")
    axes[0].set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=7)
    axes[1].bar(
        ["RF baseline", "MobileNetV3"],
        [BASELINE_METRICS["test_accuracy"], metrics["test_accuracy"]],
        color=["#5B6573", "#2AA198"],
    )
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_ylabel("Held-out test accuracy")
    axes[1].set_title("Baseline vs champion")
    for index, value in enumerate(
        [BASELINE_METRICS["test_accuracy"], metrics["test_accuracy"]]
    ):
        axes[1].text(index, value + 0.012, f"{value:.1%}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def train_transfer_model(
    data_dir: str | Path,
    model_path: str | Path,
    artifact_dir: str | Path,
    params: dict[str, Any],
) -> dict[str, float]:
    root = Path(data_dir)
    model_output = Path(model_path)
    artifacts = Path(artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    seed = int(params["seed"])
    batch_size = int(params.get("batch_size", 32))
    np.random.seed(seed)

    train_paths, y_train = _split_paths(root, "train")
    validation_paths, y_validation = _split_paths(root, "validation")
    test_paths, y_test = _split_paths(root, "test")

    feature_extractor = _load_feature_extractor()
    x_train = _extract_embeddings(train_paths, feature_extractor, batch_size=batch_size)
    x_validation = _extract_embeddings(
        validation_paths, feature_extractor, batch_size=batch_size
    )
    x_test = _extract_embeddings(test_paths, feature_extractor, batch_size=batch_size)
    x_train_mirrored = _extract_embeddings(
        train_paths, feature_extractor, batch_size=batch_size, mirrored=True
    )

    variants = {
        "original": (x_train, y_train),
        "horizontal_flip_augmented": (
            np.vstack([x_train, x_train_mirrored]),
            np.concatenate([y_train, y_train]),
        ),
    }
    candidate_rows: list[dict[str, Any]] = []
    for variant, (features, labels) in variants.items():
        for classifier_name, regularizations in params["classifier_candidates"].items():
            for regularization in regularizations:
                classifier = _candidate(classifier_name, float(regularization), seed)
                classifier.fit(features, labels)
                validation_probabilities = classifier.predict_proba(x_validation)
                candidate_rows.append(
                    {
                        "candidate": f"{variant}:{classifier_name}:C={regularization}",
                        "variant": variant,
                        "classifier": classifier_name,
                        "regularization": float(regularization),
                        "validation_accuracy": float(
                            accuracy_score(
                                y_validation, classifier.predict(x_validation)
                            )
                        ),
                        "validation_log_loss": float(
                            log_loss(
                                y_validation,
                                validation_probabilities,
                                labels=np.arange(len(CLASS_NAMES)),
                            )
                        ),
                    }
                )

    selected = max(
        candidate_rows,
        key=lambda row: (row["validation_accuracy"], -row["validation_log_loss"]),
    )

    final_original = np.vstack([x_train, x_validation])
    final_labels = np.concatenate([y_train, y_validation])
    if selected["variant"] == "horizontal_flip_augmented":
        validation_mirrored = _extract_embeddings(
            validation_paths, feature_extractor, batch_size=batch_size, mirrored=True
        )
        final_features = np.vstack(
            [final_original, x_train_mirrored, validation_mirrored]
        )
        final_labels = np.concatenate([final_labels, y_train, y_validation])
    else:
        final_features = final_original

    classifier = _candidate(
        str(selected["classifier"]), float(selected["regularization"]), seed
    )
    classifier.fit(final_features, final_labels)

    test_probabilities = classifier.predict_proba(x_test)
    test_predictions = classifier.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_predictions, average="binary", zero_division=0
    )
    test_accuracy = float(accuracy_score(y_test, test_predictions))
    metrics = {
        "test_accuracy": test_accuracy,
        "test_log_loss": float(
            log_loss(y_test, test_probabilities, labels=np.arange(len(CLASS_NAMES)))
        ),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "baseline_test_accuracy": BASELINE_METRICS["test_accuracy"],
        "accuracy_gain": test_accuracy - BASELINE_METRICS["test_accuracy"],
        "train_images": int(len(y_train)),
        "validation_images": int(len(y_validation)),
        "final_training_images": int(len(final_labels)),
        "test_images": int(len(y_test)),
    }

    onnx_output = model_output.with_name("mobilenet_v3_small_features.onnx")
    _export_onnx(feature_extractor, onnx_output)
    onnx_digest = hashlib.sha256(onnx_output.read_bytes()).hexdigest()
    bundle = {
        "backend": "onnx_mobilenet_v3_small",
        "classifier": classifier,
        "class_names": CLASS_NAMES,
        "input_size": 224,
        "imagenet_mean": IMAGENET_MEAN.tolist(),
        "imagenet_std": IMAGENET_STD.tolist(),
        "onnx_model": onnx_output.name,
        "onnx_sha256": onnx_digest,
        "version": os.environ.get("MODEL_VERSION", "2.0.0"),
        "metrics": metrics,
        "selected_variant": selected["variant"],
        "selected_classifier": selected["classifier"],
        "selected_regularization": selected["regularization"],
    }
    save_model(bundle, model_output)
    model_digest = hashlib.sha256(model_output.read_bytes()).hexdigest()

    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (artifacts / "model_manifest.json").write_text(
        json.dumps(
            {
                "model_version": bundle["version"],
                "model_backend": bundle["backend"],
                "model_bundle": {
                    "path": str(model_output),
                    "sha256": model_digest,
                    "bytes": model_output.stat().st_size,
                },
                "feature_extractor": {
                    "path": str(onnx_output),
                    "sha256": onnx_digest,
                    "bytes": onnx_output.stat().st_size,
                },
            },
            indent=2,
        )
        + "\n"
    )
    (artifacts / "model_selection.json").write_text(
        json.dumps(
            {
                "selection_rule": "highest validation accuracy, then lowest validation log loss",
                "selected_candidate": selected,
                "candidates": candidate_rows,
                "test_set_used_for_selection": False,
            },
            indent=2,
        )
        + "\n"
    )
    with (artifacts / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(candidate_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(candidate_rows)

    matrix = confusion_matrix(y_test, test_predictions, labels=np.arange(len(CLASS_NAMES)))
    (artifacts / "confusion_matrix.json").write_text(
        json.dumps({"labels": CLASS_NAMES, "matrix": matrix.tolist()}, indent=2) + "\n"
    )
    with (artifacts / "sample_requests.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["image", "true_label", "predicted_label", "dog_probability", "correct"]
        )
        for path, truth, prediction, probabilities in zip(
            test_paths[:20], y_test[:20], test_predictions[:20], test_probabilities[:20]
        ):
            writer.writerow(
                [
                    path.name,
                    CLASS_NAMES[truth],
                    CLASS_NAMES[prediction],
                    probabilities[1],
                    truth == prediction,
                ]
            )
    _write_comparison_plot(candidate_rows, metrics, artifacts / "model_comparison.png")

    tracking_path = Path("mlruns").resolve()
    mlflow.set_tracking_uri(tracking_path.as_uri())
    mlflow.set_experiment("cats-vs-dogs-model-development")
    with mlflow.start_run(run_name="random-forest-hog-reference"):
        mlflow.set_tags({"model_role": "baseline", "model_family": "random_forest"})
        mlflow.log_metrics(BASELINE_METRICS)
    with mlflow.start_run(run_name="mobilenet-v3-transfer-learning"):
        mlflow.set_tags(
            {
                "model_role": "champion",
                "model_family": "mobilenet_v3_small",
                "selection_uses_test_set": "false",
            }
        )
        mlflow.log_params(
            {
                "model_type": params["model_type"],
                "backbone": params["backbone"],
                "batch_size": batch_size,
                "seed": seed,
                "selected_variant": selected["variant"],
                "selected_classifier": selected["classifier"],
                "selected_regularization": selected["regularization"],
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_output), artifact_path="model")
        mlflow.log_artifact(str(onnx_output), artifact_path="model")
        for artifact in artifacts.iterdir():
            mlflow.log_artifact(str(artifact), artifact_path="evaluation")
    return metrics
