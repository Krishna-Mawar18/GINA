"""Training loop, checkpoint I/O and TensorBoard logging."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import CLASS_NAMES, INPUT_SHAPE
from training.config import TrainingConfig
from utils.device import clear_cuda_cache

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config: TrainingConfig,
    epoch: int,
    history: list[dict[str, float | int]],
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save model, optimizer and run metadata to a single checkpoint file.

    ``weights_only=False`` is required when loading because the payload also
    contains configuration dictionaries; only load checkpoints you trust.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_VERSION,
        "model_name": config.model_name,
        "model_kwargs": {"dropout": config.dropout},
        "num_classes": len(CLASS_NAMES),
        "class_names": list(CLASS_NAMES),
        "input_shape": list(INPUT_SHAPE),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "history": history,
        "metrics": metrics,
        "config": config.to_dict(),
    }
    torch.save(payload, output)
    return output


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> dict[str, Any]:
    """Load a GINA checkpoint and optionally restore module states."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version: {checkpoint.get('format_version')}"
        )

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


class Trainer:
    """Own the training/validation loop for one model and configuration."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        config: TrainingConfig,
        *,
        scheduler: LRScheduler | None = None,
        writer: SummaryWriter | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.writer = writer
        self.run_dir = run_dir or Path(config.checkpoint_dir) / config.run_name

    def _batch_to_device(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        non_blocking = self.device.type == "cuda" and self.config.pin_memory
        return (
            images.to(self.device, non_blocking=non_blocking),
            labels.to(self.device, non_blocking=non_blocking),
        )

    def train_one_epoch(
        self, loader: DataLoader[tuple[torch.Tensor, int]], epoch: int
    ) -> dict[str, float]:
        """Run one training epoch and return mean loss and accuracy."""
        self.model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch}/{self.config.epochs} [train]",
            disable=not self.config.show_progress,
        )
        try:
            for images, labels in progress:
                images, labels = self._batch_to_device(images, labels)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                batch_size = labels.size(0)
                loss_sum += float(loss.item()) * batch_size
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                seen += batch_size
                progress.set_postfix(
                    loss=f"{loss_sum / seen:.4f}",
                    accuracy=f"{correct / seen:.3f}",
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                clear_cuda_cache(self.device)
                raise RuntimeError(
                    "CUDA out of memory during training. Reduce --batch-size, "
                    "disable --pin-memory or use a smaller model."
                ) from exc
            raise

        if seen == 0:
            raise RuntimeError("Training DataLoader produced no batches.")
        return {"loss": loss_sum / seen, "accuracy": correct / seen}

    @torch.no_grad()
    def validate(self, loader: DataLoader[tuple[torch.Tensor, int]]) -> dict[str, float]:
        """Evaluate the current model on the validation split."""
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        seen = 0

        try:
            for images, labels in loader:
                images, labels = self._batch_to_device(images, labels)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                batch_size = labels.size(0)
                loss_sum += float(loss.item()) * batch_size
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                seen += batch_size
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                clear_cuda_cache(self.device)
                raise RuntimeError(
                    "CUDA out of memory during validation. Reduce --batch-size."
                ) from exc
            raise

        if seen == 0:
            raise RuntimeError("Validation DataLoader produced no batches.")
        return {"loss": loss_sum / seen, "accuracy": correct / seen}

    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, int]],
        val_loader: DataLoader[tuple[torch.Tensor, int]],
        *,
        start_epoch: int = 1,
        initial_history: list[dict[str, float | int]] | None = None,
    ) -> list[dict[str, float | int]]:
        """Train for the configured number of epochs, saving best/last checkpoints."""
        history: list[dict[str, float | int]] = list(initial_history or [])
        best_accuracy = max(
            (float(row["val_accuracy"]) for row in history),
            default=float("-inf"),
        )
        stale_epochs = 0

        for epoch in range(start_epoch, self.config.epochs + 1):
            train_stats = self.train_one_epoch(train_loader, epoch)
            val_stats = self.validate(val_loader)
            if self.scheduler is not None:
                self.scheduler.step()

            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(row)

            if self.writer is not None:
                self.writer.add_scalars(
                    "Loss",
                    {"train": train_stats["loss"], "val": val_stats["loss"]},
                    epoch,
                )
                self.writer.add_scalars(
                    "Accuracy",
                    {"train": train_stats["accuracy"], "val": val_stats["accuracy"]},
                    epoch,
                )
                self.writer.add_scalar("LearningRate", row["learning_rate"], epoch)
                self.writer.flush()

            save_checkpoint(
                self.run_dir / "last.pt",
                model=self.model,
                config=self.config,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                history=history,
            )

            if val_stats["accuracy"] > best_accuracy:
                best_accuracy = val_stats["accuracy"]
                stale_epochs = 0
                save_checkpoint(
                    self.run_dir / "best.pt",
                    model=self.model,
                    config=self.config,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    history=history,
                    metrics={"best_val_accuracy": best_accuracy},
                )
            else:
                stale_epochs += 1

            print(
                f"Epoch {epoch:02d}/{self.config.epochs} | "
                f"train loss {train_stats['loss']:.4f} acc {train_stats['accuracy']:.4f} | "
                f"val loss {val_stats['loss']:.4f} acc {val_stats['accuracy']:.4f}"
            )

            if (
                self.config.early_stopping_patience > 0
                and stale_epochs >= self.config.early_stopping_patience
            ):
                print(f"Early stopping after {stale_epochs} epochs without validation improvement.")
                break

        return history


def write_history_csv(history: list[dict[str, float | int]], path: str | Path) -> Path:
    """Write per-epoch metrics to CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return output
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return output
