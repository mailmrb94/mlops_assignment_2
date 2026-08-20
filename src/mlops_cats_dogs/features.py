"""Compact color and edge feature extraction from standardized RGB images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .preprocessing import preprocess_image


def _hog_features(gray: np.ndarray, cells: int = 8, bins: int = 9) -> np.ndarray:
    """Compute a small HOG-style descriptor without an additional dependency."""

    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = (np.degrees(np.arctan2(gradient_y, gradient_x)) + 180.0) % 180.0
    cell_height = gray.shape[0] // cells
    cell_width = gray.shape[1] // cells
    histograms = np.zeros((cells, cells, bins), dtype=np.float32)
    for row in range(cells):
        for column in range(cells):
            y_slice = slice(row * cell_height, (row + 1) * cell_height)
            x_slice = slice(column * cell_width, (column + 1) * cell_width)
            cell_angles = orientation[y_slice, x_slice]
            cell_magnitudes = magnitude[y_slice, x_slice]
            bin_indices = np.floor(cell_angles / (180.0 / bins)).astype(int) % bins
            histograms[row, column] = np.bincount(
                bin_indices.ravel(), weights=cell_magnitudes.ravel(), minlength=bins
            )
    blocks = []
    for row in range(cells - 1):
        for column in range(cells - 1):
            block = histograms[row : row + 2, column : column + 2].ravel()
            blocks.append(block / np.sqrt(np.dot(block, block) + 1e-6))
    return np.concatenate(blocks).astype(np.float32)


def extract_features(
    source: str | Path | bytes | Image.Image,
    feature_size: int = 32,
    histogram_bins: int = 16,
) -> np.ndarray:
    """Create HOG-style edge, compact RGB, and color histogram features."""

    image = preprocess_image(source)
    edge_image = image.resize((feature_size, feature_size), Image.Resampling.BILINEAR)
    edge_pixels = np.asarray(edge_image, dtype=np.float32) / 255.0
    gray = (
        0.299 * edge_pixels[:, :, 0]
        + 0.587 * edge_pixels[:, :, 1]
        + 0.114 * edge_pixels[:, :, 2]
    )
    edges = _hog_features(gray)
    color_thumbnail = image.resize((16, 16), Image.Resampling.BILINEAR)
    color_features = np.asarray(color_thumbnail, dtype=np.float32).reshape(-1) / 255.0
    histogram_parts = []
    full_pixels = np.asarray(image)
    for channel in range(3):
        histogram, _ = np.histogram(
            full_pixels[:, :, channel], bins=histogram_bins, range=(0, 256), density=True
        )
        histogram_parts.append(histogram.astype(np.float32))
    return np.concatenate([edges, color_features, *histogram_parts]).astype(np.float32)


def extract_batch(
    paths: list[Path], feature_size: int = 32, histogram_bins: int = 16
) -> np.ndarray:
    """Extract features for a list of image paths."""

    return np.vstack(
        [extract_features(path, feature_size, histogram_bins) for path in paths]
    )
