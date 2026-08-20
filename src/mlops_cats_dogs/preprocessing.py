"""Image validation and deterministic preprocessing utilities."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class InvalidImageError(ValueError):
    """Raised when an input cannot be safely decoded as an image."""


def preprocess_image(
    source: str | Path | bytes | BinaryIO | Image.Image,
    size: tuple[int, int] = IMAGE_SIZE,
) -> Image.Image:
    """Decode, orient, convert to RGB, and center-crop an image to ``size``.

    The returned image is detached from any underlying file handle.
    """

    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        elif isinstance(source, bytes):
            if len(source) > MAX_UPLOAD_BYTES:
                raise InvalidImageError("image exceeds the 10 MB size limit")
            image = Image.open(BytesIO(source))
        else:
            image = Image.open(source)
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    except InvalidImageError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("file is not a valid supported image") from exc


def image_to_array(source: str | Path | bytes | BinaryIO | Image.Image) -> np.ndarray:
    """Return a float32 RGB array in [0, 1] with shape (224, 224, 3)."""

    image = preprocess_image(source)
    return np.asarray(image, dtype=np.float32) / 255.0

