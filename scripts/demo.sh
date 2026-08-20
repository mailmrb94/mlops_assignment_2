#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q
docker build -t cats-dogs-mlops:demo .
IMAGE_NAME=cats-dogs-mlops:demo docker compose up -d
trap 'docker compose down' EXIT
python scripts/smoke_test.py --base-url http://localhost:8000 --image "${1:-data/processed/test/cats/cat.0.jpg}"
curl --fail --silent http://localhost:8000/metrics | grep inference_requests_total

