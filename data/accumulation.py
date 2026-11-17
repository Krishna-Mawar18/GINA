"""Controlled accumulation of user-approved handwriting samples.

Predictions are never silently converted into labels: every stored sample
requires an explicit digit supplied by the user together with the model
prediction it may be correcting.
"""

from __future__ import annotations

import csv
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from inference.preprocess import PreprocessingConfig, preprocess_drawing

DEFAULT_SAMPLE_DIR = Path("data") / "custom_samples"


@dataclass(frozen=True)
class AccumulatedSample:
    """Metadata for one user-approved drawing."""

    sample_id: str
    timestamp: str
    user_label: int
    predicted_digit: int | None
    corrected: bool
    model_version: str | None
    model_name: str | None
    image_path: str
    processed_path: str
    source: str


class SampleStore:
    """JSONL metadata store plus original and 28×28 PNG sample files."""

    def __init__(self, root: str | Path = DEFAULT_SAMPLE_DIR) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.processed_dir = self.root / "processed"
        self.log_path = self.root / "samples.jsonl"
        for directory in (self.images_dir, self.processed_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def add_sample(
        self,
        image: Image.Image | np.ndarray,
        user_label: int,
        *,
        predicted_digit: int | None = None,
        model_version: str | None = None,
        model_name: str | None = None,
        preprocess_config: PreprocessingConfig | None = None,
        source: str = "canvas",
    ) -> AccumulatedSample:
        """Store a drawing with its explicitly user-confirmed label."""
        if user_label is None or not isinstance(user_label, (int, np.integer)):
            raise ValueError("An explicit integer user_label is required.")
        label = int(user_label)
        if not 0 <= label <= 9:
            raise ValueError("user_label must be a digit from 0 to 9.")

        prepared = preprocess_drawing(image, preprocess_config)
        if prepared.is_empty:
            raise ValueError("Cannot save an empty drawing.")

        sample_id = uuid.uuid4().hex
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        original_path = self.images_dir / f"{sample_id}.png"
        processed_path = self.processed_dir / f"{sample_id}.png"

        if isinstance(image, Image.Image):
            original = image.convert("RGBA")
        else:
            array = np.asarray(image)
            original = Image.fromarray(array.astype(np.uint8))
        original.save(original_path)
        Image.fromarray((prepared.final_image(preprocess_config) * 255).astype(np.uint8)).save(
            processed_path
        )

        sample = AccumulatedSample(
            sample_id=sample_id,
            timestamp=timestamp,
            user_label=label,
            predicted_digit=predicted_digit,
            corrected=predicted_digit is not None and int(predicted_digit) != label,
            model_version=model_version,
            model_name=model_name,
            image_path=str(original_path.relative_to(self.root)),
            processed_path=str(processed_path.relative_to(self.root)),
            source=source,
        )
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(sample)) + "\n")
        return sample

    def list_samples(self) -> list[AccumulatedSample]:
        """Read all stored sample records in insertion order."""
        if not self.log_path.is_file():
            return []
        samples: list[AccumulatedSample] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append(AccumulatedSample(**json.loads(line)))
        return samples

    def class_counts(self) -> dict[int, int]:
        """Count approved samples per digit."""
        counts = {digit: 0 for digit in range(10)}
        for sample in self.list_samples():
            counts[sample.user_label] += 1
        return counts

    def export_csv(self, path: str | Path) -> Path:
        """Export sample metadata to CSV for inspection."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        samples = self.list_samples()
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(AccumulatedSample.__dataclass_fields__),
            )
            writer.writeheader()
            writer.writerows(asdict(sample) for sample in samples)
        return output

    def export_for_training(self, target_dir: str | Path) -> Path:
        """Copy processed 28×28 samples into digit-named folders.

        The export is deliberately separate from MNIST and must be inspected
        and explicitly selected for a future controlled retraining run.
        """
        destination = Path(target_dir)
        counts = self.class_counts()
        for digit in range(10):
            (destination / str(digit)).mkdir(parents=True, exist_ok=True)

        for sample in self.list_samples():
            source = self.root / sample.processed_path
            if source.is_file():
                shutil.copy2(
                    source, destination / str(sample.user_label) / f"{sample.sample_id}.png"
                )

        (destination / "summary.json").write_text(
            json.dumps(
                {
                    "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "counts": counts,
                    "note": "User-approved samples only. Review before retraining.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return destination

    def to_records(self) -> list[dict[str, Any]]:
        """Return samples as plain dictionaries for UI tables."""
        return [asdict(sample) for sample in self.list_samples()]
