"""Command-line training entry point: ``python -m training.train``."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from torch.utils.tensorboard import SummaryWriter

from data.dataset import create_dataloaders, load_datasets
from models.model_factory import create_model
from training.config import TrainingConfig
from training.trainer import Trainer, load_checkpoint, write_history_csv
from utils.device import describe_device, inspect_device, resolve_device
from utils.seed import set_seed
from utils.visualization import plot_training_history, save_figure


def build_scheduler(optimizer: Adam, config: TrainingConfig) -> LRScheduler | None:
    """Create the configured learning-rate scheduler."""
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.scheduler == "step":
        return StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    return None


def run(config: TrainingConfig) -> dict[str, object]:
    """Execute one complete training run and write its artifacts."""
    set_seed(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    device_info = inspect_device(device)
    print(describe_device(device_info))

    bundle = load_datasets(
        data_dir=config.data_dir,
        validation_split=config.validation_split,
        seed=config.seed,
        augmentation=config.augmentation,
        download=config.download,
        dataset_name=config.dataset_name,
    )
    print(
        f"Dataset: {bundle.dataset_name} | "
        f"train {len(bundle.train)}, val {len(bundle.val)}, test {len(bundle.test)}"
    )

    loaders = create_dataloaders(
        bundle,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        seed=config.seed,
    )

    model = create_model(
        config.model_name,
        num_classes=len(bundle.class_names),
        dropout=config.dropout,
    ).to(device)
    if hasattr(model, "count_parameters"):
        print(f"Model: {config.model_name} ({model.count_parameters():,} parameters)")  # type: ignore[attr-defined]

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, config)

    run_id = f"{config.run_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(config.checkpoint_dir) / run_id
    writer = SummaryWriter(log_dir=str(Path(config.log_dir) / run_id))
    writer.add_text("Configuration", json.dumps(config.to_dict(), indent=2))

    start_epoch = 1
    history: list[dict[str, float | int]] = []
    if config.resume is not None:
        checkpoint = load_checkpoint(
            config.resume,
            device=device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
        print(f"Resumed from {config.resume} at epoch {start_epoch}.")
        if start_epoch > config.epochs:
            print("Checkpoint epoch already exceeds --epochs; nothing to train.")
            writer.close()
            return {"run_id": run_id, "status": "already_complete"}

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=config,
        scheduler=scheduler,
        writer=writer,
        run_dir=run_dir,
    )
    try:
        history = trainer.fit(
            loaders.train,
            loaders.val,
            start_epoch=start_epoch,
            initial_history=history,
        )
    finally:
        writer.close()

    results_dir = Path(config.results_dir)
    results_path = results_dir / f"{run_id}.json"
    results: dict[str, object] = {
        "run_id": run_id,
        "config": config.to_dict(),
        "device": device_info.to_dict(),
        "history": history,
        "best_val_accuracy": max(float(row["val_accuracy"]) for row in history),
        "final_val_accuracy": float(history[-1]["val_accuracy"]),
        "checkpoint_dir": str(run_dir),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_history_csv(history, results_dir / f"{run_id}.csv")
    save_figure(
        plot_training_history(history, title=f"GINA training — {run_id}"),
        results_dir / f"{run_id}_history.png",
    )

    # Stable aliases let the app find the latest trained model without a config edit.
    for filename in ("best.pt", "last.pt"):
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, Path(config.checkpoint_dir) / filename)

    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Results: {results_path}")
    print(f"Checkpoints: {run_dir} (also copied to {config.checkpoint_dir})")
    return results


def main() -> None:
    """Parse CLI arguments and launch training."""
    run(TrainingConfig.from_args())


if __name__ == "__main__":
    main()
