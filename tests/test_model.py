"""Model architecture, factory and checkpoint tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import Adam

from models.cnn import SimpleCNN
from models.model_factory import create_model, list_models
from training.config import TrainingConfig
from training.trainer import load_checkpoint, save_checkpoint


def test_model_initialization_and_forward_shape() -> None:
    model = SimpleCNN()
    logits = model(torch.randn(4, 1, 28, 28))

    assert tuple(logits.shape) == (4, 10)
    assert not torch.isnan(logits).any()
    assert model.count_parameters() > 0


def test_model_has_no_softmax_layer() -> None:
    model = SimpleCNN()
    assert not any(isinstance(layer, nn.Softmax) for layer in model.modules())


def test_model_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        SimpleCNN(dropout=1.0)


def test_factory_lists_and_builds_models() -> None:
    assert "simple_cnn" in list_models()
    assert isinstance(create_model("simple_cnn"), SimpleCNN)
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("not_a_model")


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = SimpleCNN()
    optimizer = Adam(model.parameters(), lr=0.001)
    config = TrainingConfig(epochs=3)
    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=2,
        history=[{"epoch": 1, "val_accuracy": 0.5}],
    )

    restored = SimpleCNN()
    checkpoint = load_checkpoint(path, device=torch.device("cpu"), model=restored)

    assert checkpoint["epoch"] == 2
    assert checkpoint["config"]["batch_size"] == 64
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


@pytest.mark.cuda
def test_cuda_forward_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment.")
    model = SimpleCNN().to("cuda")
    logits = model(torch.randn(2, 1, 28, 28, device="cuda"))

    assert logits.device.type == "cuda"
    assert tuple(logits.shape) == (2, 10)
