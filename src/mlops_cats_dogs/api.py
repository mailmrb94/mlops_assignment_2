"""FastAPI inference service with safe logging and Prometheus metrics."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .model import load_model, predict_image
from .preprocessing import InvalidImageError, MAX_UPLOAD_BYTES

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("cats_dogs_api")
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
REQUESTS = Counter("inference_requests_total", "Inference requests", ["endpoint", "status"])
LATENCY = Histogram("inference_latency_seconds", "Inference latency", ["endpoint"])

@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = load_model(MODEL_PATH)
    LOGGER.info(
        "model_loaded path=%s version=%s", MODEL_PATH, application.state.model["version"]
    )
    yield


app = FastAPI(title="Cats vs Dogs Classifier", version="1.0.0", lifespan=lifespan)
app.state.model = None


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        LOGGER.info(
            "request id=%s method=%s path=%s status=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status,
            elapsed * 1000,
        )


@app.get("/health")
def health() -> dict[str, str]:
    REQUESTS.labels("health", "ok").inc()
    return {"status": "healthy", "model_version": str(app.state.model["version"])}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    start = time.perf_counter()
    try:
        if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            REQUESTS.labels("predict", "invalid_type").inc()
            raise HTTPException(status_code=415, detail="upload JPEG, PNG, or WebP image")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            REQUESTS.labels("predict", "too_large").inc()
            raise HTTPException(status_code=413, detail="image exceeds 10 MB")
        result = predict_image(app.state.model, payload)
        REQUESTS.labels("predict", "ok").inc()
        LOGGER.info(
            "prediction label=%s confidence=%.4f content_type=%s bytes=%d",
            result["label"],
            result["confidence"],
            file.content_type,
            len(payload),
        )
        return result
    except InvalidImageError as exc:
        REQUESTS.labels("predict", "invalid_image").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        LATENCY.labels("predict").observe(time.perf_counter() - start)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
