# GINA development commands (Windows / PowerShell).
PYTHON ?= .venv\Scripts\python.exe
STREAMLIT ?= .venv\Scripts\streamlit.exe
TENSORBOARD ?= .venv\Scripts\tensorboard.exe

.PHONY: install check-cuda download-data train evaluate app test lint format tensorboard clean

install:
	$(PYTHON) -m pip install -r requirements.txt

check-cuda:
	$(PYTHON) -c "from utils.device import describe_device, inspect_device; print(describe_device(inspect_device()))"

download-data:
	$(PYTHON) -m data.download

train:
	$(PYTHON) -m training.train

evaluate:
	$(PYTHON) -m training.evaluate

app:
	$(STREAMLIT) run app/main.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

tensorboard:
	$(TENSORBOARD) logdir experiments/logs --port 6006

clean:
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
