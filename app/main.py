"""GINA Streamlit entry point: ``streamlit run app/main.py``."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.canvas import render_canvas, render_settings  # noqa: E402
from app.components.prediction import (  # noqa: E402
    append_history,
    discover_checkpoints,
    load_service,
    render_device_summary,
    render_feedback,
    render_prediction,
    render_probabilities,
)
from app.components.visualizations import show_preprocessing  # noqa: E402
from inference.predict import EmptyCanvasError  # noqa: E402

PAGES_DIR = Path(__file__).parent / "pages"


def recognition_page() -> None:
    """Home page: draw a digit and inspect the prediction."""
    st.title("GINA")
    st.markdown("**Draw a number. Let intelligence recognize it.**")
    st.caption("128×128 canvas → smart crop → 28×28 model input → digit 0–9")

    stroke_width, tool, realtime, device = render_settings()
    render_device_summary()

    checkpoints = discover_checkpoints()
    if not checkpoints:
        st.warning(
            "No model checkpoint found. Run `make download-data` then `make train` "
            "before using recognition. You can still explore the other pages."
        )
        return

    with st.sidebar:
        checkpoint_path = st.selectbox(
            "Checkpoint",
            checkpoints,
            format_func=lambda path: Path(path).name,
        )

    if st.session_state.get("active_checkpoint") != checkpoint_path:
        st.session_state["active_checkpoint"] = checkpoint_path
        st.session_state["last_prediction"] = None
        st.session_state["last_canvas_hash"] = None

    try:
        service = load_service(checkpoint_path, device)
    except Exception as exc:  # noqa: BLE001 - UI must report any checkpoint load failure
        st.error(f"Could not load the checkpoint: {exc}")
        return

    canvas_column, result_column = st.columns([1, 2], gap="large")
    with canvas_column:
        drawing = render_canvas(
            stroke_width=stroke_width,
            tool=tool,
        )
    with result_column:
        if drawing is None:
            st.info("Draw a digit on the canvas, then press **Predict**.")
            return

        canvas_hash = hashlib.sha256(np.asarray(drawing).tobytes()).hexdigest()
        changed = canvas_hash != st.session_state.get("last_canvas_hash")
        if st.button("Predict", type="primary", width="stretch") or (realtime and changed):
            try:
                result = service.predict(drawing)
                st.session_state["last_prediction"] = result
                st.session_state["last_canvas_hash"] = canvas_hash
                append_history(result)
            except EmptyCanvasError as exc:
                st.warning(str(exc))
            except RuntimeError as exc:
                st.error(str(exc))

        result = st.session_state.get("last_prediction")
        if result is None:
            st.info("Ready when you are — press **Predict** to classify the drawing.")
            return

        render_prediction(result)
        st.divider()
        render_probabilities(result)
        with st.expander("Preprocessing steps"):
            show_preprocessing(result)
        with st.expander("Approve this sample for future training"):
            render_feedback(result, checkpoint_path)


def configure_app() -> None:
    """Apply global styling and navigation."""
    css_path = Path(__file__).parent / "styles" / "main.css"
    if css_path.is_file():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True
        )

    pages = [
        st.Page(recognition_page, title="Recognition", icon="✏️"),
        st.Page(PAGES_DIR / "model_info.py", title="Model Information", icon="🧠"),
        st.Page(PAGES_DIR / "training_experiments.py", title="Training & Experiments", icon="📈"),
        st.Page(PAGES_DIR / "dataset.py", title="Dataset Explorer", icon="🗂️"),
        st.Page(PAGES_DIR / "history.py", title="Prediction History", icon="🕘"),
        st.Page(PAGES_DIR / "about.py", title="About GINA", icon="ℹ️"),
    ]
    navigation = st.navigation(pages)
    navigation.run()


st.set_page_config(
    page_title="GINA — Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)
configure_app()
