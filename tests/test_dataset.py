"""Tests for transforms, dataset splitting and DataLoader construction."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.dataset import (
    CLASS_NAMES,
    INPUT_SHAPE,
    build_transforms,
    class_distribution,
    create_dataloaders,
    load_datasets,
    supported_datasets,
)
from data.download import missing_mnist_files


class _FakeLabeledDataset(Dataset[tuple[torch.Tensor, int]]):
    targets = [0, 1, 1, 9]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(1, 2, 2), self.targets[index]


def test_transforms_produce_normalized_model_input() -> None:
    transform = build_transforms(train=False)
    image = Image.fromarray(torch.zeros(28, 28, dtype=torch.uint8).numpy())

    tensor = transform(image)

    assert tuple(tensor.shape) == INPUT_SHAPE
    assert torch.isclose(
        tensor.mean(),
        torch.tensor(-0.1307 / 0.3081),
        atol=1e-4,
    )
    assert tensor.min() >= -1.0


def test_class_distribution_counts_all_digits() -> None:
    counts = class_distribution(_FakeLabeledDataset())

    assert counts == {
        digit: (2 if digit == 1 else 1 if digit in (0, 9) else 0) for digit in range(10)
    }
    assert len(CLASS_NAMES) == 10


def test_registry_contains_mnist() -> None:
    assert "mnist" in supported_datasets()


@pytest.mark.integration
def test_mnist_download_and_load() -> None:
    if missing_mnist_files(Path("data/raw")):
        pytest.skip("MNIST has not been downloaded; run 'make download-data'.")

    first = load_datasets(validation_split=0.05, seed=7, augmentation=False)
    second = load_datasets(validation_split=0.05, seed=7, augmentation=False)

    assert all(
        first.counts["train"][digit] + first.counts["val"][digit] > first.counts["test"][digit]
        for digit in range(10)
    )
    assert first.counts == second.counts
    assert sum(first.counts["train"].values()) + sum(first.counts["val"].values()) == 60_000
    assert sum(first.counts["test"].values()) == 10_000
    assert len(first.train) + len(first.val) == 60_000

    loaders = create_dataloaders(first, batch_size=8, num_workers=0)
    images, labels = next(iter(loaders.train))
    assert tuple(images.shape[1:]) == INPUT_SHAPE
    assert labels.shape == (8,)
