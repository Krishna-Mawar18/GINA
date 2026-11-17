"""Dataset Explorer page."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.components.visualizations import show_dataset_samples
from data.accumulation import SampleStore
from data.dataset import load_datasets
from utils.visualization import denormalize_image, plot_class_distribution


@st.cache_data(show_spinner="Loading MNIST…")
def dataset_summary(
    data_dir: str, seed: int
) -> tuple[dict[str, dict[int, int]], list[np.ndarray], list[int], dict[str, int]]:
    """Load MNIST once and return counts plus a random sample grid."""
    bundle = load_datasets(data_dir=data_dir, seed=seed, augmentation=False)
    indices = np.random.default_rng(seed).choice(len(bundle.train), size=8, replace=False)
    images = []
    labels = []
    for index in indices:
        image, label = bundle.train[int(index)]
        images.append(denormalize_image(image))
        labels.append(int(label))
    sizes = {
        "train": len(bundle.train),
        "validation": len(bundle.val),
        "test": len(bundle.test),
    }
    return bundle.counts, images, labels, sizes


st.title("Dataset Explorer")
st.caption("Official MNIST splits and the separately stored, user-approved samples.")

try:
    counts, images, labels, sizes = dataset_summary("data/raw", 42)
except Exception as exc:  # noqa: BLE001 - dataset may not be downloaded yet
    st.error(f"Could not load MNIST: {exc}")
    st.stop()

st.subheader("Split sizes")
st.write(sizes)

st.subheader("Training class distribution")
figure = plot_class_distribution(counts["train"])
st.pyplot(figure, width="stretch")
plt.close(figure)

st.subheader("Sample images")
show_dataset_samples(images, labels)

st.subheader("Custom dataset")
custom_counts = SampleStore().class_counts()
st.bar_chart({str(digit): count for digit, count in custom_counts.items()})
st.caption("Custom samples are stored separately from MNIST and are never used automatically.")
