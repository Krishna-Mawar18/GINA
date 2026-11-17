"""About GINA page."""

from __future__ import annotations

import streamlit as st

st.title("About GINA")
st.markdown("### *Draw a number. Let intelligence recognize it.*")

st.write(
    "GINA is a handwritten digit recognition system built with PyTorch and CUDA. "
    "It combines a custom CNN trained on MNIST with an interactive drawing canvas, "
    "a crop-and-centre preprocessing pipeline, measurable evaluation and a "
    "controlled pipeline for collecting user-approved handwriting samples."
)

st.subheader("How recognition works")
st.markdown(
    """1. Draw a digit on the 128×128 canvas.
2. GINA thresholds and inks the drawing, crops the digit and preserves its aspect ratio.
3. The digit is centred by mass in a 28×28 image and normalized with MNIST statistics.
4. The CNN produces ten logits; softmax converts them to probabilities.
5. The predicted digit, confidence, top-three classes and timing are displayed."""
)

st.subheader("Accumulative learning")
st.write(
    "You may confirm or correct a prediction to save a sample. Samples are stored "
    "with a timestamp, the model version and the original prediction. They remain "
    "separate from MNIST and are exported only for reviewed, controlled retraining. "
    "GINA never updates its production model automatically."
)

st.subheader("Current limitations")
st.markdown(
    """- Trained on MNIST, which differs from many real handwriting styles.
- A high MNIST test score does not guarantee the same accuracy on canvas drawings.
- EMNIST support and controlled retraining are planned, not yet enabled.
- GPU memory on a 4 GB card limits very large training configurations."""
)
