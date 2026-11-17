"""Smoke tests for the training loop and evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from models.cnn import SimpleCNN
from training.config import TrainingConfig
from training.evaluate import evaluate_model
from training.trainer import Trainer, write_history_csv
from utils.metrics import compute_metrics


def test_training_loop_saves_checkpoints_and_history(tmp_path: Path) -> None:
    torch.manual_seed(0)
    dataset = TensorDataset(
        torch.randn(16, 1, 28, 28),
        torch.randint(0, 10, (16,)),
    )
    loader = DataLoader(dataset, batch_size=4)
    config = TrainingConfig(
        epochs=1,
        batch_size=4,
        show_progress=False,
        checkpoint_dir=tmp_path,
    )
    model = SimpleCNN()
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=Adam(model.parameters(), lr=config.learning_rate),
        device=torch.device("cpu"),
        config=config,
        run_dir=tmp_path / "run",
    )

    history = trainer.fit(loader, loader)

    assert len(history) == 1
    assert {"epoch", "train_loss", "val_accuracy", "learning_rate"} <= history[0].keys()
    assert (tmp_path / "run" / "best.pt").is_file()
    assert (tmp_path / "run" / "last.pt").is_file()
    assert write_history_csv(history, tmp_path / "history.csv").is_file()


def test_evaluate_model_returns_complete_metrics() -> None:
    dataset = TensorDataset(
        torch.randn(8, 1, 28, 28),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
    )
    loader = DataLoader(dataset, batch_size=4)

    result, collector = evaluate_model(
        SimpleCNN().eval(),
        loader,
        torch.device("cpu"),
        nn.CrossEntropyLoss(),
        show_progress=False,
    )

    assert result.num_samples == 8
    assert 0.0 <= result.accuracy <= 1.0
    assert len(result.confusion_matrix) == 10
    assert set(result.per_class) == {str(digit) for digit in range(10)}
    assert len(collector.misclassified) + int(round(result.accuracy * 8)) == 8


def test_compute_metrics_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_metrics([0, 1], [0])
