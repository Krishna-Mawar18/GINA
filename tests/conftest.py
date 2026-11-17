"""Shared test fixtures."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def digit_canvas() -> Image.Image:
    """A simple white '7' on a black 128×128 canvas."""
    canvas = Image.new("RGBA", (128, 128), (0, 0, 0, 255))
    draw = ImageDraw.Draw(canvas)
    draw.line((35, 28, 95, 28), fill=(255, 255, 255, 255), width=10)
    draw.line((95, 28, 58, 105), fill=(255, 255, 255, 255), width=10)
    return canvas


@pytest.fixture
def empty_canvas() -> Image.Image:
    """An empty black 128×128 canvas."""
    return Image.new("RGBA", (128, 128), (0, 0, 0, 255))
