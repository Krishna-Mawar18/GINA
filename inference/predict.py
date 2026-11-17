"""Model loading and prediction service for interactive inference."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from data.dataset import CLASS_NAMES
from inference.preprocess import PreprocessingConfig, PreprocessingResult, preprocess_drawing
from models.model_factory import create_model
from training.trainer import load_checkpoint
from utils.device import clear_cuda_cache, resolve_device


class EmptyCanvasError(ValueError):
    """Raised when a prediction is requested for an empty drawing."""


@dataclass(frozen=True)
class PredictionResult:
    """Prediction, probabilities, timing and preprocessing diagnostics."""

    digit: int
    confidence: float
    probabilities: tuple[float, ...]
    top_predictions: tuple[tuple[int, float], ...]
    latency_ms: float
    model_latency_ms: float
    device: str
    model_name: str
    checkpoint_epoch: int | None
    preprocessing: PreprocessingResult


class PredictionService:
    """Loads a checkpoint once and runs preprocess → model → softmax."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        preprocess_config: PreprocessingConfig | None = None,
        class_names: Sequence[str] = CLASS_NAMES,
        model_name: str = "unknown",
        checkpoint_epoch: int | None = None,
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.preprocess_config = preprocess_config or PreprocessingConfig()
        self.class_names = tuple(class_names)
        self.model_name = model_name
        self.checkpoint_epoch = checkpoint_epoch

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        preprocess_config: PreprocessingConfig | None = None,
    ) -> PredictionService:
        """Build a service from a locally saved GINA checkpoint."""
        target = resolve_device(device or "auto")
        checkpoint = load_checkpoint(checkpoint_path, device=target)
        model_kwargs = checkpoint.get("model_kwargs") or {"dropout": 0.25}
        if not isinstance(model_kwargs, dict):
            raise ValueError("Checkpoint model_kwargs must be a dictionary.")

        class_names = tuple(checkpoint.get("class_names") or CLASS_NAMES)
        model = create_model(
            checkpoint["model_name"],
            num_classes=len(class_names),
            **model_kwargs,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(target).eval()
        return cls(
            model,
            target,
            preprocess_config=preprocess_config,
            class_names=class_names,
            model_name=str(checkpoint["model_name"]),
            checkpoint_epoch=int(checkpoint["epoch"]) if checkpoint.get("epoch") else None,
        )

    def warmup(self) -> None:
        """Run a dummy forward pass to initialize CUDA kernels before timing."""
        dummy = torch.zeros(1, 1, 28, 28, device=self.device)
        with torch.no_grad():
            self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def predict(self, image: np.ndarray | Image.Image) -> PredictionResult:
        """Predict the digit in a 128×128 drawing.

        Raises:
            EmptyCanvasError: If no ink is detected in the drawing.
            RuntimeError: If CUDA runs out of memory during inference.
        """
        started = time.perf_counter()
        preprocessed = preprocess_drawing(image, self.preprocess_config)
        if preprocessed.is_empty:
            raise EmptyCanvasError("The canvas is empty. Draw a digit first.")

        batch = preprocessed.tensor.unsqueeze(0).to(self.device, non_blocking=True)
        try:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                logits = self.model(batch)
                end_event.record()
                torch.cuda.synchronize()
                model_latency_ms = float(start_event.elapsed_time(end_event))
            else:
                model_started = time.perf_counter()
                logits = self.model(batch)
                model_latency_ms = (time.perf_counter() - model_started) * 1000.0
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                clear_cuda_cache(self.device)
                raise RuntimeError(
                    "CUDA out of memory during inference. The cached model remains "
                    "loaded; try reducing other GPU workloads and retry."
                ) from exc
            raise

        probabilities = torch.softmax(logits[0], dim=0).detach().cpu().numpy().astype(float)
        digit = int(np.argmax(probabilities))
        top = tuple(
            sorted(
                ((index, float(value)) for index, value in enumerate(probabilities)),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        )
        return PredictionResult(
            digit=digit,
            confidence=float(probabilities[digit]),
            probabilities=tuple(float(value) for value in probabilities),
            top_predictions=top,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_latency_ms=model_latency_ms,
            device=str(self.device),
            model_name=self.model_name,
            checkpoint_epoch=self.checkpoint_epoch,
            preprocessing=preprocessed,
        )
