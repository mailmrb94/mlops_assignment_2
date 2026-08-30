# Cats vs Dogs — Production-Style MLOps Assignment

[![MLOps CI/CD](https://github.com/mailmrb94/mlops_assignment_2/actions/workflows/mlops.yml/badge.svg)](https://github.com/mailmrb94/mlops_assignment_2/actions/workflows/mlops.yml)

This repository is a complete, reproducible Assignment 2 submission. It develops and
tracks a Cats-vs-Dogs classifier, serves it through FastAPI, packages it with Docker,
publishes immutable images through GitHub Actions, and exposes operational metrics.

## Result

| Model | Held-out accuracy | Precision | Recall | F1 | Log loss |
|---|---:|---:|---:|---:|---:|
| HOG + Random Forest baseline | 70.00% | 0.676 | 0.767 | 0.719 | 0.620 |
| **MobileNetV3-Small + Logistic Regression** | **98.33%** | **1.000** | **0.967** | **0.983** | **0.032** |

The champion improves test accuracy by **28.33 percentage points** and makes only one
error on the 60-image held-out test set. Candidate selection used validation data only;
the test split was opened once after the model and hyperparameters were locked.

## Architecture

```text
600 source images + SHA-256 provenance
          |
          v
DVC prepare: EXIF -> RGB -> center crop -> 224x224 -> 80/10/10 split
          |
          v
Frozen ImageNet MobileNetV3-Small -> 576-dimensional embeddings
          |
          v
12 validation candidates -> Logistic Regression (C=0.1) -> final train+validation fit
          |
          +--> MLflow metrics, parameters, plots, manifest, labeled request batch
          +--> ONNX feature extractor + checksum-verified joblib classifier
                          |
                          v
                FastAPI -> Docker/GHCR -> Compose CD
                          |
                          v
             health + prediction smoke tests + Prometheus metrics
```

The original Random Forest is retained as a reproducible baseline. The production
service uses ONNX Runtime rather than PyTorch, reducing the runtime dependency footprint
while preserving the pretrained feature extractor. `artifacts/model_manifest.json`
records SHA-256 hashes for both deployable model files.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt -r requirements-train.txt
pip install -e .
python scripts/download_data.py --output data/raw
dvc repro
pytest -q
uvicorn mlops_cats_dogs.api:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
curl http://localhost:8000/health
curl -X POST -F "file=@data/processed/test/cats/cat.0.jpg" http://localhost:8000/predict
curl http://localhost:8000/metrics
```

View both the baseline and champion experiments:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000> and select `cats-vs-dogs-model-development`.

## Reproducible model selection

- Fixed seed 42 and deterministic, class-balanced 480/60/60 split.
- Frozen MobileNetV3-Small ImageNet features prevent overfitting a deep network to only
  480 training images.
- Twelve candidates compare Logistic Regression, linear SVC, and RBF SVC across original
  and horizontally flipped representations.
- Selection rule: highest validation accuracy, then lowest validation log loss.
- Winner: original images + Logistic Regression, `C=0.1`, 98.33% validation accuracy.
- Winner is retrained on 540 train+validation images before the single final test.
- Six automated tests cover preprocessing, invalid inputs, API behavior, inference, and
  the ONNX/checksum contract.

## Docker and deployment

Start Docker Desktop first, then run:

```bash
docker build -t cats-dogs-mlops:local .
IMAGE_NAME=cats-dogs-mlops:local docker compose up -d
python scripts/smoke_test.py --base-url http://localhost:8000 \
  --image data/processed/test/cats/cat.0.jpg
docker compose down
```

The GitHub Actions workflow tests every change, builds the image, and publishes commit-SHA
and `latest` tags to GHCR. On `main`, deployment runs over SSH only when `DEPLOY_HOST`,
`DEPLOY_USER`, and `DEPLOY_SSH_KEY` are configured; otherwise the deploy job exits cleanly.
Set `DEPLOY_PATH` if the server checkout is not `/opt/cats-dogs-mlops`.

## Repository map

- `src/mlops_cats_dogs/transfer_learning.py`: embeddings, model selection, final fit,
  ONNX export, MLflow logging, and artifact creation
- `src/mlops_cats_dogs/model.py`: checksum-verified ONNX inference
- `src/mlops_cats_dogs/api.py`: validated API, safe logs, and Prometheus metrics
- `tests/`: six preprocessing, model, ONNX, and API tests
- `models/`: deployable ONNX extractor, classifier bundle, and preserved baseline
- `artifacts/`: metrics, comparison plot, candidate table, confusion matrices, manifest
- `dvc.yaml` / `dvc.lock`: dependency-aware data and training lineage
- `.github/workflows/mlops.yml`: CI, GHCR publishing, conditional CD, smoke test
- `output/docx/`: fully editable Word report
- `output/pdf/`: final PDF report
- `docs/MLOps_Assignment_2_Demo.mp4`: concise demonstration video

## Versioning policy

Source, deployable model binaries, and reports are Git-tracked so a fresh clone can build
the exact container without access to a private DVC cache. DVC tracks the dataset pipeline,
dependencies, parameters, evaluation outputs, and signed model manifest. This preserves
pipeline lineage while keeping deployment self-contained.

Predictions are educational and are not intended for safety-critical use. The 600-image
teaching subset is small; production promotion would require a larger external test set,
slice-level fairness checks, calibration monitoring, and drift-triggered retraining.
