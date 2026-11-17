"""Convert a 128×128 drawing into a normalized 28×28 MNIST-style tensor.

The canvas is never blindly downscaled: the digit is thresholded, cropped,
aspect-preserved resized and recentred using its centre of mass before the
training normalization is applied. Every intermediate image is retained so
that the pipeline can be inspected visually.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image

from data.dataset import MNIST_MEAN, MNIST_STD


@dataclass(frozen=True)
class PreprocessingConfig:
    """Tunable parameters for the drawing-to-model conversion."""

    canvas_size: int = 128
    model_size: int = 28
    digit_size: int = 20
    mean: float = MNIST_MEAN
    std: float = MNIST_STD
    border_fraction: float = 0.08
    min_component_area: int = 2

    def __post_init__(self) -> None:
        if self.canvas_size < self.model_size:
            raise ValueError("canvas_size must be at least model_size")
        if not 0 < self.digit_size <= self.model_size:
            raise ValueError("digit_size must be in (0, model_size]")
        if not 0.0 < self.border_fraction < 0.5:
            raise ValueError("border_fraction must be in (0, 0.5)")


@dataclass
class PreprocessingResult:
    """Model-ready tensor plus diagnostics and visualization stages."""

    tensor: torch.Tensor
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    is_empty: bool = False
    bounding_box: tuple[int, int, int, int] | None = None
    ink_pixels: int = 0
    inverted: bool = False
    warnings: list[str] = field(default_factory=list)

    def final_image(self, config: PreprocessingConfig | None = None) -> np.ndarray:
        """Return the final model input as a float array in ``[0, 1]``."""
        config = config or PreprocessingConfig()
        array = self.tensor.squeeze(0).detach().cpu().numpy()
        return np.clip(array * config.std + config.mean, 0.0, 1.0)


def _to_grayscale(
    image: np.ndarray | Image.Image,
    canvas_size: int,
) -> np.ndarray:
    """Convert RGBA/RGB/grayscale input to a square grayscale uint8 array."""
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGBA"))
    else:
        array = np.asarray(image)

    if array.ndim == 3 and array.shape[2] == 4:
        rgba = array.astype(np.float32) / 255.0
        # Composite over a black background: transparent regions become black.
        rgb = rgba[..., :3] * rgba[..., 3:4]
        gray = cv2.cvtColor((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif array.ndim == 2:
        gray = array.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported drawing shape: {array.shape}")

    if gray.shape != (canvas_size, canvas_size):
        interpolation = cv2.INTER_AREA if gray.shape[0] > canvas_size else cv2.INTER_CUBIC
        gray = cv2.resize(gray, (canvas_size, canvas_size), interpolation=interpolation)
    return np.ascontiguousarray(gray)


def _invert_if_light_background(gray: np.ndarray, fraction: float) -> tuple[np.ndarray, bool]:
    """Invert when the canvas border indicates a dark digit on light ground."""
    strip = max(1, int(gray.shape[0] * fraction))
    border = np.concatenate(
        (
            gray[:strip].ravel(),
            gray[-strip:].ravel(),
            gray[:, :strip].ravel(),
            gray[:, -strip:].ravel(),
        )
    )
    if float(np.median(border)) > 127:
        return cv2.bitwise_not(gray), True
    return gray, False


def _threshold(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold the image, with a fixed threshold for flat images."""
    threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if threshold == 0 or not np.any(mask):
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return mask


def _remove_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop tiny disconnected components while preserving multi-stroke digits."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == component] = 255
    return cleaned if np.any(cleaned) else mask


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return ``(x0, y0, x1, y1)`` with x1/y1 exclusive."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    height, width = mask.shape
    return (
        max(0, int(xs.min()) - 1),
        max(0, int(ys.min()) - 1),
        min(width, int(xs.max()) + 2),
        min(height, int(ys.max()) + 2),
    )


def preprocess_drawing(
    image: np.ndarray | Image.Image,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Run the complete preprocessing pipeline on one canvas image.

    An empty drawing produces an all-black model input with ``is_empty=True``
    rather than an exception, so callers decide how to report it.
    """
    config = config or PreprocessingConfig()
    gray = _to_grayscale(image, config.canvas_size)
    inverted_gray, was_inverted = _invert_if_light_background(gray, config.border_fraction)
    mask = _remove_specks(_threshold(inverted_gray), config.min_component_area)

    stages: dict[str, np.ndarray] = {
        "grayscale": gray,
        "inverted": inverted_gray,
        "thresholded": mask,
    }
    ink_pixels = int(np.count_nonzero(mask))
    bbox = _bounding_box(mask)
    warnings: list[str] = []
    if was_inverted:
        warnings.append("Canvas colors were inverted to match MNIST.")
    if ink_pixels < 12:
        warnings.append("Very little ink was detected; the drawing may be too small.")

    if bbox is None:
        blank = np.zeros((config.model_size, config.model_size), dtype=np.uint8)
        stages["centered"] = blank
        tensor = torch.from_numpy(
            (blank.astype(np.float32) / 255.0 - config.mean) / config.std
        ).unsqueeze(0)
        return PreprocessingResult(
            tensor=tensor,
            stages=stages,
            is_empty=True,
            ink_pixels=0,
            inverted=was_inverted,
            warnings=warnings + ["The canvas is empty."],
        )

    x0, y0, x1, y1 = bbox
    digit_gray = inverted_gray[y0:y1, x0:x1].copy()
    digit_mask = mask[y0:y1, x0:x1] > 0
    digit_gray[~digit_mask] = 0
    stages["cropped"] = digit_gray

    crop_h, crop_w = digit_gray.shape
    scale = config.digit_size / max(crop_h, crop_w)
    new_w = max(2, int(round(crop_w * scale)))
    new_h = max(2, int(round(crop_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(digit_gray, (new_w, new_h), interpolation=interpolation)
    stages["resized"] = resized

    size = config.model_size
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    pasted = np.zeros((size, size), dtype=np.float32)
    pasted[top : top + new_h, left : left + new_w] = resized.astype(np.float32)

    total = float(pasted.sum())
    if total > 0:
        ys, xs = np.mgrid[0:size, 0:size]
        center_x = float((xs * pasted).sum() / total)
        center_y = float((ys * pasted).sum() / total)
        dx = int(round((size - 1) / 2 - center_x))
        dy = int(round((size - 1) / 2 - center_y))
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        centered = cv2.warpAffine(
            pasted,
            transform,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
    else:
        centered = pasted

    centered_uint8 = np.clip(centered, 0, 255).astype(np.uint8)
    stages["centered"] = centered_uint8
    normalized = (centered.astype(np.float32) / 255.0 - config.mean) / config.std
    tensor = torch.from_numpy(np.ascontiguousarray(normalized)).unsqueeze(0)

    if min(new_w, new_h) <= 2:
        warnings.append("The drawing was extremely small and was enlarged.")

    return PreprocessingResult(
        tensor=tensor,
        stages=stages,
        is_empty=False,
        bounding_box=bbox,
        ink_pixels=ink_pixels,
        inverted=was_inverted,
        warnings=warnings,
    )
