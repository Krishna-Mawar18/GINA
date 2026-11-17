"""Tests for the drawing-to-MNIST preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from inference.preprocess import PreprocessingConfig, preprocess_drawing


def test_preprocess_produces_normalized_tensor(digit_canvas: Image.Image) -> None:
    result = preprocess_drawing(digit_canvas)

    assert tuple(result.tensor.shape) == (1, 28, 28)
    assert result.tensor.dtype == torch.float32
    assert not result.is_empty
    assert result.ink_pixels > 0
    assert result.bounding_box is not None
    for stage in ("grayscale", "inverted", "thresholded", "cropped", "resized", "centered"):
        assert stage in result.stages


def test_empty_canvas_is_handled(empty_canvas: Image.Image) -> None:
    result = preprocess_drawing(empty_canvas)

    assert result.is_empty
    assert result.bounding_box is None
    assert torch.allclose(result.tensor, torch.tensor((0.0 - 0.1307) / 0.3081), atol=1e-5)
    assert any("empty" in warning.lower() for warning in result.warnings)


def test_light_background_is_inverted(digit_canvas: Image.Image) -> None:
    inverted_canvas = Image.new("RGBA", (128, 128), (255, 255, 255, 255))
    pixels = np.asarray(digit_canvas)
    dark_strokes = np.all(pixels[:, :, :3] > 0, axis=2)
    output = np.asarray(inverted_canvas).copy()
    output[dark_strokes] = (0, 0, 0, 255)
    result = preprocess_drawing(Image.fromarray(output))

    assert result.inverted
    assert not result.is_empty
    assert result.ink_pixels > 0


def test_bounding_box_crops_empty_space() -> None:
    canvas = Image.new("RGBA", (128, 128), (0, 0, 0, 255))
    ImageDraw.Draw(canvas).rectangle((20, 30, 45, 80), fill=(255, 255, 255, 255))

    result = preprocess_drawing(canvas)

    assert result.bounding_box is not None
    x0, y0, x1, y1 = result.bounding_box
    assert 15 <= x0 <= 20
    assert 25 <= y0 <= 30
    assert 45 <= x1 <= 50
    assert 80 <= y1 <= 85


def test_aspect_ratio_is_preserved() -> None:
    canvas = Image.new("RGBA", (128, 128), (0, 0, 0, 255))
    ImageDraw.Draw(canvas).rectangle((40, 20, 60, 100), fill=(255, 255, 255, 255))

    result = preprocess_drawing(canvas, PreprocessingConfig(digit_size=20))
    crop = result.stages["cropped"]
    resized = result.stages["resized"]

    crop_ratio = crop.shape[1] / crop.shape[0]
    resized_ratio = resized.shape[1] / resized.shape[0]
    assert abs(crop_ratio - resized_ratio) < 0.1
    assert max(resized.shape) == 20


def test_tiny_drawing_does_not_crash() -> None:
    canvas = Image.new("RGBA", (128, 128), (0, 0, 0, 255))
    ImageDraw.Draw(canvas).point((64, 64), fill=(255, 255, 255, 255))

    result = preprocess_drawing(canvas)

    assert not result.is_empty
    assert tuple(result.tensor.shape) == (1, 28, 28)
