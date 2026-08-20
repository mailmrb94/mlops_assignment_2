"""Train and evaluate the deterministic lightweight image baseline."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from PIL import ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support

from .features import extract_batch, extract_features
from .model import CLASS_NAMES, save_model
from .preprocessing import preprocess_image


def _load_split(root: Path, split: str, feature_size: int, histogram_bins: int):
    paths: list[Path] = []
    labels: list[int] = []
    for label_index, label in enumerate(CLASS_NAMES):
        class_dir = root / split / f"{label}s"
        class_paths = sorted(path for path in class_dir.glob("*.*") if path.is_file())
        paths.extend(class_paths)
        labels.extend([label_index] * len(class_paths))
    if not paths:
        raise FileNotFoundError(f"no images found for split '{split}' under {root}")
    return extract_batch(paths, feature_size, histogram_bins), np.asarray(labels), paths


def _write_plot(history: list[dict[str, float]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
        axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
        axes[0].set(title="Log loss", xlabel="Epoch", ylabel="Loss")
        axes[0].legend()
        axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
        axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="validation")
        axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
    except ImportError:
        output.with_suffix(".txt").write_text("Install matplotlib to render training curves.\n")


def train_model(
    data_dir: str | Path,
    model_path: str | Path,
    artifact_dir: str | Path,
    params: dict[str, Any],
) -> dict[str, float]:
    root = Path(data_dir)
    artifacts = Path(artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    feature_size = int(params["feature_size"])
    histogram_bins = int(params["histogram_bins"])
    seed = int(params["seed"])

    x_original, y_original, train_paths = _load_split(
        root, "train", feature_size, histogram_bins
    )
    x_validation, y_validation, _ = _load_split(root, "validation", feature_size, histogram_bins)
    x_test, y_test, test_paths = _load_split(root, "test", feature_size, histogram_bins)
    original_train_images = len(y_original)
    mirrored_features = np.vstack(
        [
            extract_features(
                ImageOps.mirror(preprocess_image(path)), feature_size, histogram_bins
            )
            for path in train_paths
        ]
    )
    x_augmented = np.vstack([x_original, mirrored_features])
    y_augmented = np.concatenate([y_original, y_original.copy()])
    classes = np.arange(len(CLASS_NAMES))

    def fit_candidate(x_train: np.ndarray, y_train: np.ndarray):
        candidate = RandomForestClassifier(
            n_estimators=0,
            warm_start=True,
            max_features=str(params["max_features"]),
            random_state=seed,
            n_jobs=-1,
        )
        candidate_history: list[dict[str, float]] = []
        for epoch in range(1, int(params["epochs"]) + 1):
            candidate.n_estimators = epoch * int(params["trees_per_epoch"])
            candidate.fit(x_train, y_train)
            train_prob = candidate.predict_proba(x_train)
            validation_prob = candidate.predict_proba(x_validation)
            candidate_history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(log_loss(y_train, train_prob, labels=classes)),
                    "validation_loss": float(
                        log_loss(y_validation, validation_prob, labels=classes)
                    ),
                    "train_accuracy": float(
                        accuracy_score(y_train, candidate.predict(x_train))
                    ),
                    "validation_accuracy": float(
                        accuracy_score(y_validation, candidate.predict(x_validation))
                    ),
                }
            )
        return candidate, candidate_history

    candidates = {}
    for variant, features, labels in (
        ("original", x_original, y_original),
        ("horizontal_flip_augmented", x_augmented, y_augmented),
    ):
        candidate, candidate_history = fit_candidate(features, labels)
        validation_probability = candidate.predict_proba(x_validation)
        candidates[variant] = {
            "classifier": candidate,
            "history": candidate_history,
            "validation_accuracy": float(
                accuracy_score(y_validation, candidate.predict(x_validation))
            ),
            "validation_log_loss": float(
                log_loss(y_validation, validation_probability, labels=classes)
            ),
        }
    selected_variant = max(
        candidates,
        key=lambda name: (
            candidates[name]["validation_accuracy"],
            -candidates[name]["validation_log_loss"],
        ),
    )
    classifier = candidates[selected_variant]["classifier"]
    history = candidates[selected_variant]["history"]

    test_prob = classifier.predict_proba(x_test)
    test_pred = classifier.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="binary", zero_division=0
    )
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_log_loss": float(log_loss(y_test, test_prob, labels=classes)),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "train_images": int(original_train_images),
        "augmented_train_samples": int(len(y_augmented)),
        "validation_images": int(len(y_validation)),
        "test_images": int(len(y_test)),
    }

    bundle = {
        "classifier": classifier,
        "class_names": CLASS_NAMES,
        "feature_size": feature_size,
        "histogram_bins": histogram_bins,
        "version": os.environ.get("MODEL_VERSION", "1.0.0"),
        "metrics": metrics,
        "selected_variant": selected_variant,
    }
    save_model(bundle, model_path)

    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (artifacts / "model_selection.json").write_text(
        json.dumps(
            {
                "selected_variant": selected_variant,
                "selection_rule": "highest validation accuracy, then lowest validation log loss",
                "candidates": {
                    name: {
                        "validation_accuracy": values["validation_accuracy"],
                        "validation_log_loss": values["validation_log_loss"],
                    }
                    for name, values in candidates.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    with (artifacts / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    matrix = confusion_matrix(y_test, test_pred, labels=classes)
    (artifacts / "confusion_matrix.json").write_text(
        json.dumps({"labels": CLASS_NAMES, "matrix": matrix.tolist()}, indent=2) + "\n"
    )
    with (artifacts / "sample_requests.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "true_label", "predicted_label", "dog_probability", "correct"])
        for path, truth, prediction, probabilities in zip(test_paths[:20], y_test[:20], test_pred[:20], test_prob[:20]):
            writer.writerow(
                [path.name, CLASS_NAMES[truth], CLASS_NAMES[prediction], probabilities[1], truth == prediction]
            )
    _write_plot(history, artifacts / "training_curves.png")

    tracking_path = Path("mlruns").resolve()
    mlflow.set_tracking_uri(tracking_path.as_uri())
    mlflow.set_experiment("cats-vs-dogs-baseline")
    with mlflow.start_run(run_name="random-forest-hog-baseline"):
        mlflow.log_params(params)
        mlflow.log_param("selected_variant", selected_variant)
        mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, float)})
        mlflow.log_artifact(str(model_path), artifact_path="model")
        for artifact in artifacts.iterdir():
            mlflow.log_artifact(str(artifact), artifact_path="evaluation")
    return metrics
