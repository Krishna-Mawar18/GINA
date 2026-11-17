"""Prediction service and user-sample accumulation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from data.accumulation import SampleStore
from inference.predict import EmptyCanvasError, PredictionService
from models.cnn import SimpleCNN
from training.config import TrainingConfig
from training.trainer import save_checkpoint


@pytest.fixture
def service(tmp_path: Path) -> PredictionService:
    model = SimpleCNN()
    checkpoint = save_checkpoint(
        tmp_path / "test.pt",
        model=model,
        config=TrainingConfig(epochs=1),
        epoch=1,
        history=[],
    )
    prediction_service = PredictionService.from_checkpoint(checkpoint, device="cpu")
    prediction_service.warmup()
    return prediction_service


def test_prediction_output(service: PredictionService, digit_canvas: Image.Image) -> None:
    result = service.predict(digit_canvas)

    assert 0 <= result.digit <= 9
    assert abs(sum(result.probabilities) - 1.0) < 1e-5
    assert len(result.top_predictions) == 3
    assert result.top_predictions[0][0] == result.digit
    assert result.top_predictions[0][1] >= result.top_predictions[1][1]
    assert result.latency_ms >= 0
    assert result.device == "cpu"
    assert tuple(result.preprocessing.tensor.shape) == (1, 28, 28)


def test_empty_canvas_raises(service: PredictionService, empty_canvas: Image.Image) -> None:
    with pytest.raises(EmptyCanvasError):
        service.predict(empty_canvas)


def test_sample_store_requires_explicit_label(tmp_path: Path, digit_canvas: Image.Image) -> None:
    store = SampleStore(tmp_path)

    with pytest.raises(ValueError, match="user_label"):
        store.add_sample(digit_canvas, 12)  # type: ignore[arg-type]

    sample = store.add_sample(
        digit_canvas,
        7,
        predicted_digit=3,
        model_version="test.pt@1",
        model_name="simple_cnn",
    )

    assert sample.user_label == 7
    assert sample.corrected
    assert len(store.list_samples()) == 1
    assert store.class_counts()[7] == 1
    assert store.export_csv(tmp_path / "samples.csv").is_file()
    assert (store.export_for_training(tmp_path / "export") / "7").is_dir()
