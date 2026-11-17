"""Prediction History page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data.accumulation import SampleStore

st.title("Prediction History")
st.caption("Predictions from this browser session and approved samples saved on disk.")

st.subheader("Session predictions")
history = st.session_state.get("prediction_history", [])
if history:
    st.dataframe(pd.DataFrame(history), width="stretch")
    if st.button("Clear session history"):
        st.session_state["prediction_history"] = []
        st.rerun()
else:
    st.info("No predictions in this session yet.")

st.subheader("Approved custom samples")
store = SampleStore()
samples = store.to_records()
if not samples:
    st.info("No samples saved yet. Approve a prediction from the Recognition page.")
    st.stop()

frame = pd.DataFrame(samples)
st.dataframe(frame, width="stretch")
st.bar_chart(frame["user_label"].value_counts().sort_index())

st.download_button(
    "Download sample metadata (CSV)",
    data=frame.to_csv(index=False).encode("utf-8"),
    file_name="gina_custom_samples.csv",
    mime="text/csv",
)

if st.button("Export 28×28 images for future training"):
    destination = store.export_for_training("data/custom_samples/export")
    st.success(f"Exported reviewed samples to {destination}")
