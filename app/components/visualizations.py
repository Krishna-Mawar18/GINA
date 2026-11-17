"""Streamlit wrappers around GINA's plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from inference.predict import PredictionResult
from utils.metrics import EvaluationResult
from utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_preprocessing_stages,
    plot_training_history,
)


def show_preprocessing(result: PredictionResult) -> None:
    """Display the preprocessing stages for the latest prediction."""
    stages = dict(result.preprocessing.stages)
    figure = plot_preprocessing_stages(stages)
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def show_training_history(history: list[dict[str, object]], title: str) -> None:
    """Display saved training curves."""
    figure = plot_training_history(history, title=title)  # type: ignore[arg-type]
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def show_evaluation(result: EvaluationResult) -> None:
    """Display confusion matrix and per-class metrics."""
    left, right = st.columns(2)
    with left:
        figure = plot_confusion_matrix(result.confusion_matrix)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with right:
        figure = plot_per_class_metrics(result.per_class)
        st.pyplot(figure, width="stretch")
        plt.close(figure)


def show_dataset_samples(images: list[np.ndarray], labels: list[int]) -> None:
    """Display a row of MNIST samples with their labels."""
    columns = st.columns(len(images))
    for column, image, label in zip(columns, images, labels, strict=True):
        column.image(image, width=90, caption=str(label))


def list_result_files(directory: str | Path) -> list[Path]:
    """List saved training result files, newest first."""
    path = Path(directory)
    if not path.is_dir():
        return []
    return sorted(path.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
