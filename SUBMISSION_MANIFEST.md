# Submission Manifest

## Verified results

- Dataset sample: 600 images (300 cats, 300 dogs)
- Split: 480 train, 60 validation, 60 untouched test
- Candidate search: 12 combinations across two image variants and three classifier families
- Selection rule: highest validation accuracy, then lowest validation log loss
- Champion: frozen MobileNetV3-Small embeddings + Logistic Regression (`C=0.1`)
- Champion validation accuracy: 0.9833
- Held-out test accuracy: 0.9833 (baseline 0.7000; gain +0.2833)
- Held-out precision / recall / F1: 1.0000 / 0.9667 / 0.9831
- Held-out confusion matrix: `[[30, 0], [1, 29]]`
- Automated tests: 6 passed
- DVC status: data and pipelines reproducible from `dvc.yaml` and `dvc.lock`
- MLflow: baseline and champion runs, parameters, metrics, models, plots, and request batch
- Serving: FastAPI 2.0.0 with checksum-verified ONNX Runtime inference
- Delivery: Docker, Compose, GHCR CI/CD, post-deployment health/prediction smoke test
- Reports: editable Word document plus visually verified PDF
- Demo: sub-five-minute H.264 walkthrough

## Primary artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `models/model.joblib` | 18,945 | `a1f93aa106ebc398d130ffafc09f34416275d32931f5b3cab63f3bae21b28df0` |
| `models/mobilenet_v3_small_features.onnx` | 3,717,020 | `355230de403faec174a9e0f19b4201645b11151961c9a5bb8c0808b9f37336e1` |
| `output/docx/MLOps_Assignment_2_Report_Editable.docx` | 393,091 | `c216e373537c053f1fac3f423f3e3e10af63f23888d8240a707316364f03de02` |
| `output/pdf/MLOps_Assignment_2_Report.pdf` | 145,030 | `957ff8ab9412409851ed88286264dc52439ff4b5fcaf7c8944799846bcf24459` |
| `docs/MLOps_Assignment_2_Demo.mp4` | 1,285,045 | `956ed6daad029ea640fd759042a969a5b3ac60c93132d029e382f9c712720b32` |

The model hashes also appear in `artifacts/model_manifest.json` and are verified at
service startup.

## Reproduction note

Install production, development, and training dependencies before `dvc repro`. The
production container needs only `requirements.txt`; PyTorch is used for training and ONNX
export but is deliberately excluded from serving. Deployment is conditional on repository
secrets so forks can run CI safely without possessing an external server.
