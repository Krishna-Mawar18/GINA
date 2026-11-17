"""Model Information page."""

from __future__ import annotations

import streamlit as st
import torch

from app.components.prediction import (
    discover_checkpoints,
    render_device_summary,
    render_model_summary,
)
from training.trainer import load_checkpoint

st.title("Model Information")
st.caption("Inspect the checkpoint currently available to the inference service.")

checkpoints = discover_checkpoints()
if not checkpoints:
    st.warning("No checkpoint found. Train a model with `make train` first.")
    st.stop()

selected = st.selectbox(
    "Checkpoint",
    checkpoints,
    format_func=lambda path: path.replace("\\", "/"),
)
try:
    checkpoint = load_checkpoint(selected, device=torch.device("cpu"))
except Exception as exc:  # noqa: BLE001 - report broken checkpoints in the UI
    st.error(f"Could not read checkpoint: {exc}")
    st.stop()

render_model_summary(checkpoint)

st.subheader("Baseline architecture")
st.code(
    """Input 1×28×28 grayscale
Conv2d(1→32, 3×3) → ReLU → MaxPool2d(2×2)   # 32×13×13
Conv2d(32→64, 3×3) → ReLU → MaxPool2d(2×2)  # 64×5×5
Flatten → Linear(1600→128) → ReLU → Dropout(0.25)
Linear(128→10) → 10 logits (softmax applied only at inference)""",
    language="text",
)

st.subheader("Training configuration")
if isinstance(checkpoint.get("config"), dict):
    st.json(checkpoint["config"])

st.subheader("Compute environment")
render_device_summary()
