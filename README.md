# GINA — Generative Interactive Neural Arithmetic

> **Draw a number. Let intelligence recognize it.**

GINA is a full-stack handwritten digit recognition application built with Python, PyTorch and CUDA. It combines a custom-trained CNN with an interactive 128×128 drawing canvas, an MNIST-aware crop-and-centre preprocessing pipeline, reproducible training/evaluation, TensorBoard experiment tracking and a controlled system for accumulating user-approved handwriting samples.

This is deliberately a machine learning application rather than a single-file MNIST demo: data handling, model construction, training, inference, UI and tests are separated into independently testable modules.

## Objectives

- Recognize handwritten digits 0–9 with ten-class softmax probabilities.
- Train a convolutional network from scratch; no pretrained models.
- Use NVIDIA CUDA automatically when available, with a reliable CPU fallback.
- Convert freehand 128×128 drawings into MNIST-compatible 28×28 inputs without naive whole-canvas downscaling.
- Measure real performance with standard classification metrics and publish per-run artifacts.
- Collect user-confirmed samples separately for future controlled dataset improvements.

## Features

**Implemented** 

- Custom two-block CNN (`1→32→64` conv channels, `1600→128→10` classifier, dropout 0.25).
- MNIST download, normalization (`mean=0.1307`, `std=0.3081`), train/validation split and optional affine augmentation.
- CUDA/CPU device resolution with GPU name, compute capability and memory reporting.
- Reproducible seeding, checkpoint save/load (best + last + run-specific copies), TensorBoard logging and per-epoch CSV/JSON history.
- Accuracy, macro precision/recall/F1, classification report, confusion matrix, per-class plot and optional misclassified-sample grid.
- Modular preprocessing: grayscale/alpha compositing, polarity detection, Otsu thresholding, bounding-box crop, aspect-ratio-preserving resize, centre-of-mass placement, MNIST normalization and stage visualization.
- Streamlit app with Recognition, Model Information, Training & Experiments, Dataset Explorer, Prediction History and About pages; adjustable brush, eraser mode, clear, real-time prediction, probabilities, top-three, timings and preprocessing views.
- Sample accumulation requiring explicit user labels, with timestamps, model version, counts, CSV export and 28×28 training-folder export.
- Pytest suite covering model, preprocessing, inference, dataset, device, training loop and checkpoints.

**Planned / not yet implemented**

- EMNIST training (the dataset registry is ready, but no EMNIST builder is registered).
- Controlled retraining on accumulated samples (export exists; training remains MNIST-only).
- Automatic continual learning (explicitly out of scope).

## Technology stack

| Purpose | Technology |
|---|---|
| Language | Python 3.13 |
| Deep learning | PyTorch 2.11 (`+cu130` wheels), torchvision 0.26 |
| GPU | NVIDIA CUDA 13 runtime via RTX 3050 Laptop 4 GB (verified at setup) |
| Dataset | MNIST (torchvision) |
| Interface | Streamlit 1.67.0 + streamlit-drawable-canvas 0.13.0 |
| Imaging | OpenCV 4.14.0.94, Pillow 12.3.0 |
| Numerical / ML | NumPy 2.5.5, scikit-learn 1.9.1, pandas 2.3.3 |
| Plots | Matplotlib 3.11.2 |
| Experiment tracking | TensorBoard 2.21.0 |
| Progress | tqdm |
| Testing / quality | pytest 9.2.0, ruff 0.17.5, mypy 2.4.1 |
| Automation | pip, venv, Makefile |

Versions in the table are the ones installed and verified in the project `.venv`. No TensorFlow/Keras and no pretrained vision models are used.

## Architecture

```text
128×128 canvas
  → grayscale / alpha composite on black
  → polarity check + Otsu threshold
  → bounding-box crop (empty space removed)
  → aspect-preserving resize (largest side 20 px)
  → centre-of-mass placement in 28×28
  → (x - 0.1307) / 0.3081
  → SimpleCNN → 10 logits
  → softmax → digit, confidence, top-3
```

The baseline network (see `models/cnn.py`) is:

```text
Input 1×28×28
Conv2d(1→32, 3×3) → ReLU → MaxPool2d(2×2)   → 32×13×13
Conv2d(32→64, 3×3) → ReLU → MaxPool2d(2×2)  → 64×5×5
Flatten → Linear(1600→128) → ReLU → Dropout(0.25) → Linear(128→10)
```

Softmax is intentionally absent from `forward()` because training uses `CrossEntropyLoss`; it is applied only in `inference/predict.py`. Models are created through `models/model_factory.py`, and datasets through the registry in `data/dataset.py`, so variants can be benchmarked without rewriting training or UI code.

## Project structure

```text
GINA/
├── app/
│   ├── main.py                 # Streamlit entry point + recognition page
│   ├── pages/                  # Model info, training, dataset, history, about
│   ├── components/             # Canvas, prediction, visualization widgets
│   └── styles/main.css
├── models/
│   ├── cnn.py                  # SimpleCNN
│   ├── model_factory.py        # Model registry
│   └── checkpoints/            # best.pt, last.pt and per-run checkpoints
├── training/
│   ├── config.py               # CLI/config dataclass
│   ├── trainer.py              # Training loop + checkpoint I/O
│   ├── train.py                # python -m training.train
│   └── evaluate.py             # python -m training.evaluate
├── inference/
│   ├── preprocess.py           # 128×128 → 28×28 pipeline
│   └── predict.py              # PredictionService
├── data/
│   ├── dataset.py              # Dataset registry, transforms, splits, loaders
│   ├── download.py             # MNIST download + integrity check
│   └── accumulation.py         # User-approved sample store
├── experiments/
│   ├── results/                # JSON, CSV and plots per run
│   └── logs/                   # TensorBoard event files
├── utils/
│   ├── device.py               # CUDA/CPU resolution and memory info
│   ├── metrics.py              # sklearn-based evaluation
│   ├── seed.py                 # Reproducibility
│   └── visualization.py        # Reusable matplotlib figures
├── tests/                      # Pytest modules and shared fixtures
├── notebooks/exploration.ipynb
├── requirements.txt
├── Makefile
├── pytest.ini
├── ruff.toml
└── README.md
```

Generated data and artifacts (`data/raw`, checkpoints, experiment results/logs, custom images) are ignored by `.gitignore`.

## Installation (Windows / PowerShell)

### 1. Create and activate the virtual environment

The environment used during development was created as follows (do this from the project root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, allow it for the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Pinned CUDA wheels are declared in `requirements.txt`: `torch==2.11.0+cu130` and `torchvision==0.26.0+cu130`. The audit found a CPU-only global PyTorch installation; the project venv intentionally replaces it without touching global packages.

### 2. Verify CUDA

```powershell
# Make target (or activate the venv first)
make check-cuda
```

Equivalent direct command:

```powershell
.\.venv\Scripts\python.exe -c "from utils.device import describe_device, inspect_device; print(describe_device(inspect_device()))"
```

Also verify with NVIDIA's own tool:

```powershell
nvidia-smi
```

Expected output includes `cuda available`, the GPU model (e.g. RTX 3050 Laptop), CUDA runtime and available memory. CPU fallback is automatic if CUDA is unavailable.

### 3. Prepare the dataset

```powershell
make download-data
# or
.\.venv\Scripts\python.exe -m data.download
```

MNIST is downloaded to `data/raw/MNIST/raw` and its four IDX files are verified. The training/validation split uses a seeded permutation; the official test set stays untouched.

## Training

```powershell
make train
# or
.\.venv\Scripts\python.exe -m training.train
```

Useful overrides:

```powershell
.\.venv\Scripts\python.exe -m training.train --epochs 15 --batch-size 64 --lr 0.001 --device cuda
.\.venv\Scripts\python.exe -m training.train --no-augmentation --scheduler cosine --no-deterministic
.\.venv\Scripts\python.exe -m training.train --resume models\checkpoints\<run>\last.pt
```

For a 4 GB GPU, keep `batch_size` around 64 and `num_workers` modest (0–2). If CUDA runs out of memory, GINA reports a clear error and suggests reducing the batch size.

Each run writes:

- `models/checkpoints/<run_id>/best.pt`, `last.pt` (also copied to stable `models/checkpoints/best.pt` and `last.pt`).
- `experiments/results/<run_id>.json` and `.csv` with configuration, device details and all epoch metrics.
- `experiments/results/<run_id>_history.png`.
- TensorBoard events under `experiments/logs/<run_id>`.

## Evaluation

```powershell
make evaluate
# or
.\.venv\Scripts\python.exe -m training.evaluate
```

Options include `--checkpoint`, `--split test|val`, `--batch-size`, `--device`, `--save-misclassified 12` and `--download`. Evaluation prints the classification report and writes a JSON report plus confusion-matrix and per-class PNGs into `experiments/results/`.

### Measured baseline (this machine)

Training was run for 12 epochs with the default seed, batch size 64, augmentation and CUDA on an RTX 3050 Laptop GPU. The run ID is `baseline_20261117_105545`; full history, configuration and plots are in `experiments/results/`.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy (best epoch) | **99.28%** | — |
| Accuracy | 99.08% (final epoch) | **99.38%** |
| Macro-F1 | — | **99.38%** |
| Cross-entropy loss | 0.0318 (final epoch) | 0.0191 |

These are real MNIST results from `models/checkpoints/best.pt` (epoch 11). MNIST accuracy does not directly measure accuracy on freehand canvas drawings; the interactive page is for qualitative validation of that domain gap.

## Running the application

```powershell
make app
# or
.\.venv\Scripts\streamlit.exe run app/main.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Recognition is the primary page: draw with the mouse or touch, choose brush/eraser, press **Predict**, and inspect the digit, confidence, top-three classes, probability chart, timings and preprocessing stages. Enable **Real-time prediction** to classify automatically whenever the canvas finishes an update. Approve or correct a prediction to store a sample; the History page lists session predictions and approved samples and can export them.

UI screenshot placeholders (add images under `docs/screenshots/` later):

- `Recognition` — 128×128 canvas, metrics, probability chart.
- `Preprocessing stages` — original, grayscale, thresholded, cropped, centered.
- `Training & Experiments` — saved loss/accuracy curves.

## Experiment tracking

```powershell
make tensorboard
# or
.\.venv\Scripts\tensorboard.exe logdir experiments/logs --port 6006
```

TensorBoard logs scalars for train/validation loss, train/validation accuracy and learning rate per epoch, plus the run configuration.

## Accumulative learning

`data/accumulation.py` implements `SampleStore`:

- `add_sample(...)` requires an explicit integer `user_label`; model predictions are stored only as separate metadata (`predicted_digit`, `corrected`).
- Each record includes a UUID, UTC timestamp, model version/name and paths to the original drawing and its processed 28×28 PNG.
- `class_counts()`, `to_records()`, `export_csv()` and `export_for_training()` support inspection and controlled exports.
- `export_for_training()` writes digit-named folders plus a summary note. Exports must be reviewed before any future retraining; nothing modifies the production model automatically, and MNIST data remains separate.

## Testing and code quality

```powershell
make test     # or: .\.venv\Scripts\python.exe -m pytest -q
make lint     # or: .\.venv\Scripts\python.exe -m ruff check .
make format   # or: .\.venv\Scripts\python.exe -m ruff format .
```

Tests marked `@pytest.mark.integration` need MNIST downloaded first; tests marked `@pytest.mark.cuda` skip automatically when CUDA is unavailable. The suite covers model shape/forward pass, CPU and optional CUDA inference, preprocessing edge cases, empty canvases, prediction output, dataset loading/splitting, checkpoint round-trips and a one-epoch training smoke test.

## Known limitations

- Model accuracy on MNIST does not guarantee equal accuracy on drawings from the canvas (different stroke style, thickness and distribution).
- EMNIST and retraining on custom samples are not enabled in this release.
- Deterministic mode can slow training; disable it with `--no-deterministic` when exact reproducibility is not needed.
- A 4 GB GPU limits very large batches or wider architectures.
- The canvas uses a Streamlit custom component; touch behaviour depends on the browser/device.

## Future improvements

1. Register an EMNIST `digits` builder and benchmark it against MNIST.
2. Add model variants to `MODEL_REGISTRY` and compare them with identical seeds.
3. Analyze misclassified canvas drawings and tune stroke normalization.
4. Build a reviewed, versioned retraining pipeline for accumulated samples.
5. Benchmark CPU vs GPU latency and publish measured numbers.
6. Expand automated tests with real user-drawing fixtures and browser-level UI checks.
