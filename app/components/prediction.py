"""Prediction display, cached model loading and user feedback controls."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from data.accumulation import SampleStore
from inference.predict import PredictionResult, PredictionService
from models.model_factory import list_models
from utils.device import describe_device, inspect_device, resolve_device


def discover_checkpoints(root: str | Path = "models/checkpoints") -> list[str]:
    """List available checkpoints, with stable aliases first."""
    directory = Path(root)
    if not directory.is_dir():
        return []
    found = {str(path) for path in directory.rglob("*.pt")}
    preferred = [str(directory / name) for name in ("best.pt", "last.pt")]
    ordered = [path for path in preferred if path in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


@st.cache_resource(show_spinner="Loading GINA model…")
def load_service(checkpoint_path: str, device: str) -> PredictionService:
    """Load and cache one prediction service per checkpoint/device pair."""
    service = PredictionService.from_checkpoint(checkpoint_path, device=device)
    service.warmup()
    return service


def render_prediction(result: PredictionResult) -> None:
    """Show the predicted digit, confidence and timing."""
    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted digit", result.digit)
        st.metric("Confidence", f"{result.confidence * 100:.1f}%")
    with right:
        st.subheader("Top three")
        for rank, (digit, probability) in enumerate(result.top_predictions, start=1):
            st.write(f"**{rank}. {digit}** — {probability * 100:.2f}%")
        st.caption(
            f"Inference: {result.latency_ms:.2f} ms total "
            f"({result.model_latency_ms:.2f} ms model) · device: {result.device}"
        )
        if result.preprocessing.warnings:
            for warning in result.preprocessing.warnings:
                st.caption(f"⚠ {warning}")


def render_probabilities(result: PredictionResult) -> None:
    """Show probabilities for all ten classes as a chart and table."""
    frame = pd.DataFrame(
        {
            "Digit": list(range(10)),
            "Probability": result.probabilities,
        }
    ).set_index("Digit")
    st.bar_chart(frame, height=220, width="stretch")
    st.dataframe(
        frame.style.format({"Probability": "{:.4f}"}),
        width="stretch",
    )


def render_feedback(result: PredictionResult, checkpoint_path: str) -> None:
    """Collect an explicit user label before storing a sample."""
    st.subheader("Improve GINA")
    st.caption(
        "Confirm or correct the digit below. Predictions are never stored as labels "
        "without your explicit confirmation."
    )
    label = st.selectbox(
        "Correct digit",
        options=list(range(10)),
        index=result.digit,
        key="feedback_label",
    )
    if st.button("Save approved sample", width="stretch"):
        store = SampleStore()
        sample = store.add_sample(
            result.preprocessing.stages.get("grayscale"),
            label,
            predicted_digit=result.digit,
            model_version=checkpoint_path,
            model_name=result.model_name,
        )
        st.success(f"Saved digit {sample.user_label} to the custom dataset.")


def render_device_summary() -> None:
    """Show the selected compute device in the sidebar."""
    with st.sidebar:
        st.divider()
        st.subheader("Compute")
        st.code(describe_device(inspect_device(resolve_device("auto"))))


def render_model_summary(checkpoint: dict[str, object]) -> None:
    """Show checkpoint metadata and available architectures."""
    config = checkpoint.get("config", {})
    st.write(
        {
            "Model": checkpoint.get("model_name"),
            "Checkpoint epoch": checkpoint.get("epoch"),
            "Classes": len(checkpoint.get("class_names", [])),
            "Input shape": checkpoint.get("input_shape"),
            "Seed": config.get("seed") if isinstance(config, dict) else None,
        }
    )
    st.caption(f"Registered architectures: {', '.join(list_models())}")


def append_history(result: PredictionResult) -> None:
    """Append a prediction to the in-session history."""
    history = st.session_state.setdefault("prediction_history", [])
    history.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "predicted": result.digit,
            "confidence": round(result.confidence, 4),
            "device": result.device,
        }
    )
