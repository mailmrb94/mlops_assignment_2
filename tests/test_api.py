from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from mlops_cats_dogs.api import app


class StubClassifier:
    def predict_proba(self, features):
        return np.array([[0.8, 0.2]])


def model_bundle():
    return {
        "classifier": StubClassifier(),
        "class_names": ["cat", "dog"],
        "feature_size": 32,
        "histogram_bins": 16,
        "version": "test",
    }


def test_health_and_prediction() -> None:
    app.state.model = model_bundle()
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.model = model_bundle()
        health = client.get("/health")
        assert health.status_code == 200
        image = Image.new("RGB", (64, 64), color=(200, 100, 20))
        payload = BytesIO()
        image.save(payload, format="JPEG")
        response = client.post(
            "/predict", files={"file": ("sample.jpg", payload.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 200
        assert response.json()["label"] == "cat"


def test_prediction_rejects_non_image_content_type() -> None:
    app.state.model = model_bundle()
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.model = model_bundle()
        response = client.post(
            "/predict", files={"file": ("sample.txt", b"text", "text/plain")}
        )
        assert response.status_code == 415

