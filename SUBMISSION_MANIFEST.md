# Submission Manifest

## Verified results

- Dataset sample: 600 images (300 cats, 300 dogs)
- Split: 480 train, 60 validation, 60 test
- Training augmentation: deterministic horizontal flips (960 candidate samples)
- Model selection: original-only candidate selected by validation accuracy (0.733 vs 0.667)
- Held-out test accuracy: 0.700
- Held-out test F1: 0.719
- Automated tests: 5 passed
- Live smoke test: health HTTP 200 and prediction HTTP 200
- DVC status: data and pipelines up to date
- Report: 8 A4 pages, visually rendered and inspected
- Demo video: 3 minutes 6 seconds, H.264 1280x720

## Primary artifacts and SHA-256

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `models/model.joblib` | 528266 | `2fb9005ae11b02201d3b952c94fc5f6d31f8e42f1ce895b45c6cbd873bc8a5e7` |
| `output/pdf/MLOps_Assignment_2_Report.pdf` | 71578 | `cb1ff3247e1328ec6806208da7d4e3593b69f46d583829fd040f2ae5488be6d6` |
| `docs/MLOps_Assignment_2_Demo.mp4` | 1254994 | `4ec261451a0b27a4d0c02b0504cd6eebc9bb193b46783d7febd5402db882af50` |

## Environment note

The service, tests, smoke test, DVC pipeline, MLflow logging, YAML manifests, report,
and video were executed locally. Docker Desktop 29.6.1 was available, but the external
Docker Hub base-image metadata request timed out in this environment. The Dockerfile
and Compose manifest are included, and the GitHub Actions workflow performs the image
build and GHCR publish when run in the repository.

