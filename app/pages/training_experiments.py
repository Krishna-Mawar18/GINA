"""Training & Experiments page."""

from __future__ import annotations

import json

import streamlit as st

from app.components.visualizations import list_result_files, show_training_history

st.title("Training & Experiments")
st.caption("Inspect metrics from completed runs. Numbers shown here come from saved result files.")

results = list_result_files("experiments/results")
if not results:
    st.info(
        "No training results yet. Run `make download-data` and `make train`, "
        "or `python -m training.train --epochs 12`."
    )
    st.stop()

selected = st.selectbox("Run", results, format_func=lambda path: path.stem)
try:
    report = json.loads(selected.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"Could not read result file: {exc}")
    st.stop()

history = report.get("history", [])
if not history:
    st.warning("This result file contains no per-epoch history.")
    st.stop()

best = report.get("best_val_accuracy")
final = report.get("final_val_accuracy")
columns = st.columns(3)
columns[0].metric("Best validation accuracy", f"{float(best):.2%}" if best else "—")
columns[1].metric("Final validation accuracy", f"{float(final):.2%}" if final else "—")
columns[2].metric("Epochs recorded", len(history))

show_training_history(history, title=f"GINA training — {report.get('run_id', selected.stem)}")

with st.expander("Run configuration"):
    st.json(report.get("config", {}))
with st.expander("Device information"):
    st.json(report.get("device", {}))

st.caption("Launch TensorBoard in a separate terminal:")
st.code("make tensorboard", language="bash")
