"""Drawing canvas component and its sidebar controls."""

from __future__ import annotations

from typing import Literal

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

Tool = Literal["Brush", "Eraser"]


def render_settings() -> tuple[int, Tool, bool, str]:
    """Render canvas/inference controls and return their values."""
    with st.sidebar:
        st.header("Drawing controls")
        stroke_width = st.slider("Brush thickness", 1, 30, 12)
        tool: Tool = st.radio("Tool", ("Brush", "Eraser"), horizontal=True)
        realtime = st.checkbox("Real-time prediction", value=False)
        st.divider()
        st.header("Inference")
        device = st.selectbox("Compute device", ("auto", "cpu", "cuda"), index=0)
        st.caption("CUDA is used automatically when available with 'auto'.")
    return stroke_width, tool, realtime, device


def render_canvas(*, stroke_width: int, tool: Tool) -> Image.Image | None:
    """Render the 128×128 drawing canvas and return its image data."""
    control_col, _ = st.columns([1, 3])
    with control_col:
        if st.button("Clear canvas", width="stretch"):
            st.session_state["canvas_generation"] = (
                int(st.session_state.get("canvas_generation", 0)) + 1
            )
            st.session_state["last_prediction"] = None
            st.rerun()

    generation = int(st.session_state.get("canvas_generation", 0))
    stroke_color = "#FFFFFF" if tool == "Brush" else "#000000"
    canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#000000",
        height=128,
        width=128,
        drawing_mode="freedraw",
        key=f"gina_canvas_{generation}",
        update_streamlit=True,
        return_image_data=True,
    )
    image_data = canvas.image_data
    return Image.fromarray(image_data) if image_data is not None else None
