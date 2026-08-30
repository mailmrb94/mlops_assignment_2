import numpy as np
from PIL import Image

from mlops_cats_dogs.features import extract_features
from mlops_cats_dogs.model import predict_image


class StubClassifier:
    def predict_proba(self, features):
        assert features.shape[0] == 1
        return np.array([[0.25, 0.75]])


class StubInput:
    name = "images"


class StubOnnxSession:
    def get_inputs(self):
        return [StubInput()]

    def run(self, output_names, inputs):
        assert output_names is None
        assert inputs["images"].shape == (1, 3, 224, 224)
        return [np.ones((1, 576), dtype=np.float32)]


def test_feature_vector_and_prediction_contract() -> None:
    image = Image.new("RGB", (80, 120), color=(20, 60, 200))
    features = extract_features(image)
    assert features.ndim == 1
    assert features.size > 1000
    bundle = {
        "classifier": StubClassifier(),
        "class_names": ["cat", "dog"],
        "feature_size": 32,
        "histogram_bins": 16,
        "version": "test",
    }
    result = predict_image(bundle, image)
    assert result["label"] == "dog"
    assert result["confidence"] == 0.75
    assert sum(result["probabilities"].values()) == 1.0


def test_onnx_transfer_prediction_contract() -> None:
    image = Image.new("RGB", (80, 120), color=(20, 60, 200))
    bundle = {
        "backend": "onnx_mobilenet_v3_small",
        "classifier": StubClassifier(),
        "class_names": ["cat", "dog"],
        "input_size": 224,
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "version": "test-transfer",
        "_onnx_session": StubOnnxSession(),
    }
    result = predict_image(bundle, image)
    assert result["label"] == "dog"
    assert result["confidence"] == 0.75
    assert result["model_version"] == "test-transfer"
