"""Command-line evaluation entry point: ``python -m training.evaluate``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.dataset import CLASS_NAMES, load_datasets
from models.model_factory import create_model
from training.config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    TrainingConfig,
)
from training.trainer import load_checkpoint
from utils.device import clear_cuda_cache, describe_device, inspect_device, resolve_device
from utils.metrics import EvaluationResult, PredictionCollector
from utils.seed import set_seed
from utils.visualization import (
    plot_confusion_matrix,
    plot_misclassified,
    plot_per_class_metrics,
    save_figure,
)


def config_from_checkpoint(checkpoint: dict[str, object]) -> TrainingConfig:
    """Rebuild a configuration from checkpoint metadata."""
    stored = checkpoint.get("config", {})
    if not isinstance(stored, dict):
        raise ValueError("Checkpoint does not contain a configuration dictionary.")
    known = {field.name for field in fields(TrainingConfig)}
    return TrainingConfig(**{key: value for key, value in stored.items() if key in known})


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    criterion: nn.Module,
    *,
    show_progress: bool = True,
) -> tuple[EvaluationResult, PredictionCollector]:
    """Run inference over a loader and compute all evaluation metrics."""
    model.eval()
    collector = PredictionCollector(CLASS_NAMES)
    progress = tqdm(loader, desc="Evaluating", disable=not show_progress)
    try:
        for images, labels in progress:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            loss = criterion(logits, labels)
            collector.update(logits, labels, loss=float(loss.item()))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            clear_cuda_cache(device)
            raise RuntimeError(
                "CUDA out of memory during evaluation. Reduce --batch-size."
            ) from exc
        raise
    return collector.result(), collector


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GINA checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Overrides checkpoint batch size"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Overrides checkpoint worker count"
    )
    parser.add_argument("--device", default=None, help="auto, cuda, cuda:N or cpu")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--download", action="store_true", help="Download MNIST if missing")
    parser.add_argument(
        "--save-misclassified", type=int, default=0, help="How many error examples to plot"
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Evaluate one checkpoint and write reports and plots."""
    checkpoint = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    config = config_from_checkpoint(checkpoint)
    set_seed(config.seed, deterministic=config.deterministic)

    device = resolve_device(args.device or config.device)
    device_info = inspect_device(device)
    print(describe_device(device_info))
    print(f"Checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']})")

    bundle = load_datasets(
        data_dir=args.data_dir,
        validation_split=config.validation_split,
        seed=config.seed,
        augmentation=False,
        download=args.download,
        dataset_name=config.dataset_name,
    )
    dataset: Dataset[tuple[torch.Tensor, int]] = bundle.test if args.split == "test" else bundle.val
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.batch_size,
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
    )

    model_kwargs = checkpoint.get("model_kwargs") or {"dropout": config.dropout}
    if not isinstance(model_kwargs, dict):
        raise ValueError("Checkpoint model_kwargs must be a dictionary.")
    model = create_model(
        checkpoint["model_name"],
        num_classes=len(bundle.class_names),
        **model_kwargs,
    )
    # Re-apply the checkpoint weights on the selected device.
    checkpoint = load_checkpoint(args.checkpoint, device=device, model=model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    result, collector = evaluate_model(
        model, loader, device, criterion, show_progress=not args.no_progress
    )
    print("\n" + result.classification_report)
    print(
        f"Accuracy: {result.accuracy:.4f} | macro-F1: {result.f1_macro:.4f} | "
        f"loss: {result.loss:.4f} | samples: {result.num_samples}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"evaluation_{args.split}_{timestamp}.json"
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "split": args.split,
        "device": device_info.to_dict(),
        "metrics": result.to_dict(),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    cm_path = save_figure(
        plot_confusion_matrix(result.confusion_matrix),
        args.output_dir / f"confusion_matrix_{args.split}_{timestamp}.png",
    )
    per_class_path = save_figure(
        plot_per_class_metrics(result.per_class),
        args.output_dir / f"per_class_{args.split}_{timestamp}.png",
    )

    if args.save_misclassified > 0:
        indices = collector.misclassified[: args.save_misclassified]
        examples = [
            (int(index), collector.y_true[int(index)], collector.y_pred[int(index)])
            for index in indices
        ]
        if examples:
            save_figure(
                plot_misclassified(dataset, examples),
                args.output_dir / f"misclassified_{timestamp}.png",
            )

    print(f"Report: {report_path}")
    print(f"Plots: {cm_path.name}, {per_class_path.name}")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and launch evaluation."""
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
