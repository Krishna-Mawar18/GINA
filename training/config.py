"""Configuration and command-line arguments for training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data.dataset import INPUT_SHAPE

DEFAULT_DATA_DIR = Path("data") / "raw"
DEFAULT_CHECKPOINT_DIR = Path("models") / "checkpoints"
DEFAULT_RESULTS_DIR = Path("experiments") / "results"
DEFAULT_LOG_DIR = Path("experiments") / "logs"


@dataclass
class TrainingConfig:
    """All tunable training parameters.

    Values here are also saved in every checkpoint and metrics file so that a
    run can be reproduced or inspected later.
    """

    seed: int = 42
    model_name: str = "simple_cnn"
    dataset_name: str = "mnist"
    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    validation_split: float = 0.1
    num_workers: int = 0
    pin_memory: bool = True
    device: str = "auto"
    data_dir: Path = DEFAULT_DATA_DIR
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR
    log_dir: Path = DEFAULT_LOG_DIR
    run_name: str = "baseline"
    augmentation: bool = True
    download: bool = True
    deterministic: bool = True
    dropout: float = 0.25
    scheduler: str = "none"
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.5
    early_stopping_patience: int = 0
    grad_clip: float = 0.0
    resume: Path | None = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        """Validate numeric ranges early, before any expensive work."""
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 0.0 < self.validation_split < 1.0:
            raise ValueError("validation_split must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.scheduler not in {"none", "cosine", "step"}:
            raise ValueError("scheduler must be 'none', 'cosine' or 'step'")

        for field in ("data_dir", "checkpoint_dir", "results_dir", "log_dir"):
            setattr(self, field, Path(getattr(self, field)))
        if self.resume is not None:
            self.resume = Path(self.resume)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the configuration."""
        data = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in data.items()
        }

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Expected model input shape."""
        return INPUT_SHAPE

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Build the CLI parser used by ``python -m training.train``."""
        defaults = TrainingConfig()
        parser = argparse.ArgumentParser(
            description="Train a GINA CNN on MNIST.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--seed", type=int, default=defaults.seed)
        parser.add_argument("--model", dest="model_name", default=defaults.model_name)
        parser.add_argument("--dataset", dest="dataset_name", default=defaults.dataset_name)
        parser.add_argument("--epochs", type=int, default=defaults.epochs)
        parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
        parser.add_argument(
            "--lr", dest="learning_rate", type=float, default=defaults.learning_rate
        )
        parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
        parser.add_argument("--validation-split", type=float, default=defaults.validation_split)
        parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
        parser.add_argument(
            "--pin-memory", action=argparse.BooleanOptionalAction, default=defaults.pin_memory
        )
        parser.add_argument("--device", default=defaults.device, help="auto, cuda, cuda:N or cpu")
        parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
        parser.add_argument("--checkpoint-dir", type=Path, default=defaults.checkpoint_dir)
        parser.add_argument("--results-dir", type=Path, default=defaults.results_dir)
        parser.add_argument("--log-dir", type=Path, default=defaults.log_dir)
        parser.add_argument("--run-name", default=defaults.run_name)
        parser.add_argument(
            "--augmentation", action=argparse.BooleanOptionalAction, default=defaults.augmentation
        )
        parser.add_argument(
            "--download", action=argparse.BooleanOptionalAction, default=defaults.download
        )
        parser.add_argument(
            "--deterministic", action=argparse.BooleanOptionalAction, default=defaults.deterministic
        )
        parser.add_argument("--dropout", type=float, default=defaults.dropout)
        parser.add_argument(
            "--scheduler", choices=("none", "cosine", "step"), default=defaults.scheduler
        )
        parser.add_argument("--scheduler-step-size", type=int, default=defaults.scheduler_step_size)
        parser.add_argument("--scheduler-gamma", type=float, default=defaults.scheduler_gamma)
        parser.add_argument(
            "--early-stopping-patience", type=int, default=defaults.early_stopping_patience
        )
        parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
        parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
        parser.add_argument(
            "--show-progress", action=argparse.BooleanOptionalAction, default=defaults.show_progress
        )
        return parser

    @classmethod
    def from_args(cls, args: Sequence[str] | None = None) -> TrainingConfig:
        """Parse CLI arguments into a validated :class:`TrainingConfig`."""
        namespace = vars(cls.build_parser().parse_args(args))
        return cls(**namespace)
