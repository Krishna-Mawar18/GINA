"""Dataset construction, transforms and train/validation splitting.

The dataset registry deliberately separates dataset choice from the rest of the
application. MNIST is implemented now; EMNIST can be added as another builder
without changing the training loop or inference code.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from data.download import DEFAULT_DATA_DIR

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CLASS_NAMES: tuple[str, ...] = tuple(str(digit) for digit in range(10))
INPUT_SHAPE: tuple[int, int, int] = (1, 28, 28)

DatasetBuilder = Callable[..., Dataset[tuple[torch.Tensor, int]]]


@dataclass(frozen=True)
class DatasetBundle:
    """Train/validation/test datasets plus metadata used across GINA."""

    train: Dataset[tuple[torch.Tensor, int]]
    val: Dataset[tuple[torch.Tensor, int]]
    test: Dataset[tuple[torch.Tensor, int]]
    class_names: tuple[str, ...]
    input_shape: tuple[int, int, int]
    counts: dict[str, dict[int, int]]
    dataset_name: str


@dataclass(frozen=True)
class DataLoaders:
    """DataLoaders created from a :class:`DatasetBundle`."""

    train: DataLoader[tuple[torch.Tensor, int]]
    val: DataLoader[tuple[torch.Tensor, int]]
    test: DataLoader[tuple[torch.Tensor, int]]


def build_transforms(*, train: bool, augmentation: bool = True) -> transforms.Compose:
    """Build the torchvision transform pipeline.

    Augmentation is applied only to the training split and before ``ToTensor``
    so that it operates on PIL images.
    """
    operations: list[object] = []
    if train and augmentation:
        operations.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                fill=0,
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    return transforms.Compose(operations)


def _build_mnist(
    root: str | Path,
    train: bool,
    transform: transforms.Compose,
    download: bool,
) -> Dataset[tuple[torch.Tensor, int]]:
    return MNIST(
        root=str(root),
        train=train,
        transform=transform,
        download=download,
    )


# Adding EMNIST later: register a builder with the same signature here.
DATASET_REGISTRY: dict[str, DatasetBuilder] = {"mnist": _build_mnist}


def supported_datasets() -> tuple[str, ...]:
    """Return names accepted by :func:`load_datasets`."""
    return tuple(DATASET_REGISTRY)


def load_datasets(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    validation_split: float = 0.1,
    seed: int = 42,
    augmentation: bool = True,
    download: bool = True,
    dataset_name: str = "mnist",
) -> DatasetBundle:
    """Load a dataset and split its training portion into train and validation.

    Args:
        data_dir: Parent directory containing ``MNIST/raw``.
        validation_split: Fraction of the official training set used for validation.
        seed: Seed for the reproducible split.
        augmentation: Enable random affine augmentation on the training split.
        download: Download data when it is missing.
        dataset_name: Key in :data:`DATASET_REGISTRY`.

    Returns:
        A :class:`DatasetBundle` with train, validation and test datasets.
    """
    name = dataset_name.lower().strip()
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Registered datasets: "
            f"{', '.join(supported_datasets())}."
        )
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")

    builder = DATASET_REGISTRY[name]
    train_transform = build_transforms(train=True, augmentation=augmentation)
    eval_transform = build_transforms(train=False)

    train_full = builder(data_dir, True, train_transform, download)
    train_eval = builder(data_dir, True, eval_transform, False)
    test = builder(data_dir, False, eval_transform, download)

    size = len(train_full)  # type: ignore[arg-type]
    val_size = max(1, int(size * validation_split))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(size, generator=generator).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]

    bundle = DatasetBundle(
        train=Subset(train_full, train_indices),
        val=Subset(train_eval, val_indices),
        test=test,
        class_names=CLASS_NAMES,
        input_shape=INPUT_SHAPE,
        counts={
            "train": class_distribution(Subset(train_full, train_indices)),
            "val": class_distribution(Subset(train_eval, val_indices)),
            "test": class_distribution(test),
        },
        dataset_name=name,
    )
    return bundle


def create_dataloaders(
    bundle: DatasetBundle,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    shuffle_train: bool = True,
) -> DataLoaders:
    """Create DataLoaders with reproducible shuffling.

    ``num_workers > 0`` requires a ``if __name__ == "__main__"`` guard in the
    launching script on Windows; the training CLI provides one.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return DataLoaders(
        train=DataLoader(
            bundle.train,
            shuffle=shuffle_train,
            generator=generator if shuffle_train else None,
            **common,
        ),
        val=DataLoader(bundle.val, shuffle=False, **common),
        test=DataLoader(bundle.test, shuffle=False, **common),
    )


def class_distribution(dataset: Dataset[tuple[torch.Tensor, int]]) -> dict[int, int]:
    """Count samples per class without iterating over image tensors."""
    if isinstance(dataset, Subset):
        labels = dataset.dataset.targets  # type: ignore[attr-defined]
        indices = dataset.indices
        counts = Counter(int(labels[index]) for index in indices)
    else:
        counts = Counter(int(label) for label in dataset.targets)  # type: ignore[attr-defined]

    return {digit: counts.get(digit, 0) for digit in range(10)}


def dataset_size(dataset: Dataset[Any]) -> int:
    """Return the length of a dataset or subset."""
    return len(dataset)
