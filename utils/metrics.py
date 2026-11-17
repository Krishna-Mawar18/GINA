"""Evaluation metrics computed with scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from data.dataset import CLASS_NAMES


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate and per-class metrics for one evaluation run."""

    accuracy: float
    loss: float | None
    precision_macro: float
    recall_macro: float
    f1_macro: float
    num_samples: int
    confusion_matrix: list[list[int]]
    classification_report: str
    per_class: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    loss: float | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> EvaluationResult:
    """Compute accuracy, macro and per-class precision/recall/F1 and a CM."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("Cannot compute metrics on an empty set of predictions.")

    names = list(class_names)
    digits = list(range(len(names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=digits,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=digits,
        average="macro",
        zero_division=0,
    )

    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        loss=loss,
        precision_macro=float(macro_precision),
        recall_macro=float(macro_recall),
        f1_macro=float(macro_f1),
        num_samples=len(y_true),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=digits).tolist(),
        classification_report=classification_report(
            y_true,
            y_pred,
            labels=digits,
            target_names=names,
            digits=4,
            zero_division=0,
        ),
        per_class={
            names[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": float(support[index]),
            }
            for index in range(len(names))
        },
    )


class PredictionCollector:
    """Accumulate predictions across batches without storing image tensors."""

    def __init__(self, class_names: Sequence[str] = CLASS_NAMES) -> None:
        self.class_names = list(class_names)
        self.y_true: list[int] = []
        self.y_pred: list[int] = []
        self._loss_sum = 0.0

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        loss: float | None = None,
    ) -> None:
        """Add one batch of logits and ground-truth labels."""
        predictions = logits.detach().cpu().argmax(dim=1).tolist()
        self.y_pred.extend(int(value) for value in predictions)
        self.y_true.extend(int(value) for value in labels.detach().cpu().tolist())
        if loss is not None:
            self._loss_sum += float(loss) * len(predictions)

    def result(self) -> EvaluationResult:
        """Compute the final :class:`EvaluationResult`."""
        mean_loss = self._loss_sum / len(self.y_true) if self.y_true else None
        return compute_metrics(
            self.y_true,
            self.y_pred,
            loss=mean_loss,
            class_names=self.class_names,
        )

    @property
    def misclassified(self) -> np.ndarray:
        """Indices of misclassified samples in encounter order."""
        return np.flatnonzero(np.asarray(self.y_true) != np.asarray(self.y_pred))
