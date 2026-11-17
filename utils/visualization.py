"""Matplotlib helpers shared by the training CLI and the Streamlit app."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import Dataset

from data.dataset import CLASS_NAMES, MNIST_MEAN, MNIST_STD


def denormalize_image(
    image: torch.Tensor,
    mean: float = MNIST_MEAN,
    std: float = MNIST_STD,
) -> np.ndarray:
    """Undo MNIST normalization and return a H×W array in ``[0, 1]``."""
    array = image.detach().cpu().squeeze().numpy().astype(np.float32, copy=True)
    return np.clip(array * std + mean, 0.0, 1.0)


def plot_sample_grid(
    dataset: Dataset[tuple[torch.Tensor, int]],
    *,
    indices: Sequence[int] | None = None,
    num_samples: int = 16,
    ncols: int = 4,
    seed: int = 42,
    title: str = "Dataset samples",
) -> Figure:
    """Plot a grid of normalized dataset samples with their labels."""
    available = len(dataset)  # type: ignore[arg-type]
    count = min(num_samples, available)
    if count < 1:
        raise ValueError("The dataset is empty.")

    if indices is None:
        rng = np.random.default_rng(seed)
        selected = rng.choice(available, size=count, replace=False).tolist()
    else:
        selected = [int(index) for index in indices[:count]]

    nrows = ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 1.9 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    for axis in flat_axes:
        axis.axis("off")

    for axis, index in zip(flat_axes, selected, strict=False):
        image, label = dataset[index]
        axis.imshow(denormalize_image(image), cmap="gray", vmin=0.0, vmax=1.0)
        label_int = int(label)
        axis.set_title(CLASS_NAMES[label_int] if 0 <= label_int < 10 else str(label_int))

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_class_distribution(
    counts: Mapping[int, int] | Sequence[int],
    *,
    title: str = "Class distribution",
) -> Figure:
    """Plot sample counts for digits 0–9."""
    values = (
        [counts.get(digit, 0) for digit in range(10)]  # type: ignore[union-attr]
        if isinstance(counts, Mapping)
        else [int(counts[digit]) for digit in range(10)]
    )
    fig, axis = plt.subplots(figsize=(8, 4))
    bars = axis.bar(CLASS_NAMES, values, color="#4f7cac")
    axis.set_xlabel("Digit")
    axis.set_ylabel("Samples")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    return fig


def save_figure(figure: Figure, path: str | Path, *, dpi: int = 150) -> Path:
    """Save a matplotlib figure, creating parent directories as needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    return output


def figure_to_png(figure: Figure) -> bytes:
    """Serialize a matplotlib figure to PNG bytes for Streamlit."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    return buffer.getvalue()


def plot_training_history(
    history: Sequence[Mapping[str, float | int]],
    *,
    title: str = "Training history",
) -> Figure:
    """Plot train/validation loss and accuracy over epochs."""
    if not history:
        raise ValueError("Training history is empty.")

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axis = axes[0]
    axis.plot(epochs, [float(row["train_loss"]) for row in history], label="Train")
    axis.plot(epochs, [float(row["val_loss"]) for row in history], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Loss")
    axis.legend()
    axis.grid(alpha=0.25)

    axis = axes[1]
    axis.plot(epochs, [float(row["train_accuracy"]) for row in history], label="Train")
    axis.plot(epochs, [float(row["val_accuracy"]) for row in history], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy")
    axis.legend()
    axis.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    confusion: Sequence[Sequence[int]] | np.ndarray,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> Figure:
    """Plot a labelled 10×10 confusion matrix."""
    matrix = np.asarray(confusion, dtype=float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted digit")
    axis.set_ylabel("True digit")
    axis.set_title(title)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            label = f"{value:.2f}" if normalize else str(int(value))
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > threshold else "black",
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_per_class_metrics(
    per_class: Mapping[str, Mapping[str, float]],
    *,
    title: str = "Per-class performance",
) -> Figure:
    """Grouped bar chart of per-class precision, recall and F1."""
    names = list(per_class)
    x_positions = np.arange(len(names))
    width = 0.26
    fig, axis = plt.subplots(figsize=(10, 4.5))
    for offset, (metric, color) in zip(
        (-width, 0.0, width),
        (("precision", "#4f7cac"), ("recall", "#63a375"), ("f1", "#d19a4b")),
        strict=True,
    ):
        values = [float(per_class[name][metric]) for name in names]
        axis.bar(x_positions + offset, values, width, label=metric, color=color)

    axis.set_xticks(x_positions, names)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Digit")
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    return fig


def plot_preprocessing_stages(
    stages: Mapping[str, np.ndarray],
    *,
    original: np.ndarray | None = None,
    title: str = "Preprocessing pipeline",
) -> Figure:
    """Visualize the intermediate images produced by the preprocessing pipeline."""
    available = dict(stages)
    if original is not None:
        available = {"original": np.asarray(original), **available}

    order = (
        "original",
        "grayscale",
        "inverted",
        "thresholded",
        "cropped",
        "resized",
        "centered",
    )
    names = [name for name in order if name in available]
    names += [name for name in available if name not in names]
    if not names:
        raise ValueError("No preprocessing stages were provided.")

    columns = min(4, len(names))
    rows = ceil(len(names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.5 * columns, 2.5 * rows), squeeze=False)
    flat_axes = axes.ravel()
    for axis in flat_axes:
        axis.axis("off")
    for axis, name in zip(flat_axes, names, strict=False):
        axis.imshow(available[name], cmap="gray", vmin=0, vmax=255)
        axis.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_misclassified(
    dataset: Dataset[tuple[torch.Tensor, int]],
    examples: Sequence[tuple[int, int, int]],
    *,
    ncols: int = 4,
    title: str = "Misclassified examples (index, true → predicted)",
) -> Figure:
    """Plot misclassified samples as ``(index, true, predicted)`` tuples."""
    if not examples:
        raise ValueError("There are no misclassified examples to plot.")

    nrows = ceil(len(examples) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 2.1 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    for axis in flat_axes:
        axis.axis("off")
    for axis, (index, true_digit, predicted_digit) in zip(flat_axes, examples, strict=False):
        image, _ = dataset[int(index)]
        axis.imshow(denormalize_image(image), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"#{index}: {true_digit} → {predicted_digit}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
