from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from mlops_cats_dogs.preprocessing import InvalidImageError, image_to_array, preprocess_image


def test_preprocess_converts_grayscale_and_resizes() -> None:
    source = Image.new("L", (400, 200), color=128)
    processed = preprocess_image(source)
    array = image_to_array(processed)
    assert processed.mode == "RGB"
    assert processed.size == (224, 224)
    assert array.shape == (224, 224, 3)
    assert array.dtype == np.float32
    assert 0.49 < float(array.mean()) < 0.51


def test_preprocess_rejects_invalid_bytes() -> None:
    with pytest.raises(InvalidImageError):
        preprocess_image(BytesIO(b"not an image"))

