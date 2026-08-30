# Four-Minute Video Presentation Guide

Use this as a speaking script while recording the repository, terminal, MLflow, FastAPI,
and GitHub Actions. Keep the browser zoom near 100% and enlarge the terminal font.

## Before recording

Start Docker Desktop. Open three terminal tabs in the project and activate the environment
in each one:

```bash
cd "/Users/mahesh/Documents/ChatGPT/MLops 2"
source .venv/bin/activate
```

Start MLflow in terminal 1:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Start the container in terminal 2:

```bash
IMAGE_NAME=cats-dogs-mlops:local docker compose up -d --build
```

If the image is already built, omit `--build`. Open these pages before recording:

- <http://127.0.0.1:5000> — MLflow
- <http://localhost:8000/docs> — FastAPI Swagger UI
- <http://localhost:8000/metrics> — Prometheus metrics
- <https://github.com/mailmrb94/mlops_assignment_2/actions> — GitHub Actions

## Recording sequence and narration

### 0:00–0:25 — Goal and headline result

Show the README result table.

Say: “This is an end-to-end MLOps pipeline for Cats versus Dogs. I kept the original HOG
and Random Forest as a measured baseline, then added a frozen MobileNetV3-Small transfer
model. Held-out accuracy improved from 70.00% to 98.33%, and F1 improved from 0.719 to
0.983.”

### 0:25–0:55 — Repository and architecture

Scroll through the architecture and repository map.

Say: “Git versions the code and deployable artifacts. DVC versions the data pipeline,
parameters, metrics, and model manifest. MLflow tracks experiments. ONNX Runtime serves
the model through FastAPI. Docker, GitHub Actions, GHCR, and Compose cover delivery.”

### 0:55–1:25 — Reproducibility and test integrity

Run:

```bash
dvc status
cat params.yaml
```

Say: “The dataset has 600 balanced images with deterministic 480/60/60 splits and seed
42. Twelve candidates were compared using validation data only. The test set was not used
for model selection; it was evaluated once after the winner was locked and retrained on
train plus validation.”

### 1:25–2:00 — MLflow evidence

In MLflow, select `cats-vs-dogs-model-development`, open the latest champion run, and show
the metrics, parameters, and artifacts.

Say: “MLflow records the backbone, classifier, regularization, seed, selection result, final
metrics, confusion matrix, comparison plot, request batch, joblib classifier, ONNX graph,
and SHA-256 manifest. The final confusion matrix is 30 correct cats, 29 correct dogs, and
one missed dog.”

### 2:00–2:20 — Automated quality gate

Run:

```bash
pytest -q
```

Say: “Six tests validate preprocessing, invalid images, inference, API behavior, and the
ONNX checksum and probability contract. A failure blocks the container build.”

### 2:20–3:00 — Live service

Show `/docs`, expand `POST /predict`, upload `tests/fixtures/sample_cat.jpg`, and execute it.
Then show `/metrics`.

Say: “The API accepts validated JPEG, PNG, or WebP files, returns the class probabilities
and model version 2.0.0, and rejects unsupported or oversized inputs. It exports request
counts and latency for Prometheus while avoiding image contents and filenames in logs.”

Optionally run the automated smoke test:

```bash
python scripts/smoke_test.py \
  --base-url http://localhost:8000 \
  --image tests/fixtures/sample_cat.jpg
```

### 3:00–3:35 — CI/CD proof

Show the latest successful GitHub Actions run and expand the test and Docker build steps.

Say: “Every push installs pinned dependencies, runs the tests, builds the runtime image,
and publishes immutable commit-SHA and latest tags to GHCR. The deployment job uses
repository secrets for SSH and intentionally skips when no production host is configured.
A configured deployment must pass health and prediction smoke tests.”

### 3:35–4:00 — Monitoring, limitations, and close

Show the report or README limitations section.

Say: “The system includes structured privacy-conscious logs, Prometheus metrics, delayed-
label request samples, and a retraining path. The 98.33% result is strong for this fixed
teaching subset, but production use still requires a larger external test set, slice-level
fairness checks, calibration monitoring, and drift detection. The source, editable Word
report, PDF, trained artifacts, and demonstration video are included.”

## After recording

Stop the service cleanly:

```bash
docker compose down
```

Keep the final recording below five minutes, use 1080p if possible, and ensure the terminal
output and metric values are readable without pausing the video.
