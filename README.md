# Cats vs Dogs - End-to-End MLOps Assignment

This repository is a complete, reproducible submission for Assignment 2. It trains a
binary Cats-vs-Dogs image classifier, logs experiments to MLflow, serves predictions
through FastAPI, packages the service with Docker, and defines CI/CD deployment to a
Docker Compose target.

## Architecture

```text
Cats/Dogs images -> DVC prepare stage -> 224x224 RGB images -> feature extraction
                 -> DVC train stage -> MLflow + model.joblib -> FastAPI -> Docker/GHCR
                 -> GitHub Actions -> Docker Compose deployment -> smoke tests/metrics
```

The baseline intentionally uses a lightweight, incrementally grown random forest. Every
image is first normalized to 224x224 RGB, then represented by HOG-style edge features,
a compact RGB thumbnail, and color histograms. Deterministic horizontal flips double the
training samples for augmentation while keeping the workflow practical on a laptop.
The pipeline compares original-only and augmented candidates on validation data and
serializes the better candidate; the test set remains untouched until final evaluation.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install .
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

MLflow runs are stored under `mlruns/`. View them with:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Docker and deployment

```bash
docker build -t cats-dogs-mlops:local .
IMAGE_NAME=cats-dogs-mlops:local docker compose up -d
python scripts/smoke_test.py --base-url http://localhost:8000 \
  --image data/processed/test/cats/cat.0.jpg
docker compose down
```

The GitHub Actions workflow runs tests, builds the image, pushes it to GHCR, and (on
`main`) deploys it over SSH to a Docker Compose host. Configure repository secrets
`DEPLOY_HOST`, `DEPLOY_USER`, and `DEPLOY_SSH_KEY`; set `DEPLOY_PATH` as a repository
variable if the checkout on the host is not `/opt/cats-dogs-mlops`.

## Repository map

- `src/mlops_cats_dogs/`: preprocessing, feature extraction, training, inference, API
- `tests/`: preprocessing, inference, and API tests
- `scripts/`: dataset download, preparation, training wrapper, smoke test, demo
- `models/model.joblib`: trained model bundle
- `artifacts/`: metrics, plots, confusion matrix, sample requests
- `.github/workflows/mlops.yml`: CI, registry publishing, CD, post-deploy smoke test
- `docker-compose.yml`: deployment manifest
- `dvc.yaml` / `dvc.lock`: reproducible data and model pipeline
- `docs/`: final report and demonstration assets
- `output/docx/MLOps_Assignment_2_Report_Editable.docx`: editable Word report

## Reproducibility notes

- Seeds are fixed to 42 and image sampling/splits are deterministic.
- The downloaded teaching subset is derived from the public Cats and Dogs dataset and
  its URL and SHA-256 are recorded by the downloader.
- The trained model is included in the submission. Raw and processed datasets are not
  copied into the ZIP; DVC metadata and the downloader reproduce them.
- Predictions are educational and are not intended for safety-critical use.
