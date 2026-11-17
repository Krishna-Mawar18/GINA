"""Custom CNN architectures trained from scratch on MNIST."""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Baseline two-block CNN for 1×28×28 grayscale digit images.

    Tensor shapes (batch dimension omitted):

    ``1×28×28 -> 32×26×26 -> 32×13×13 -> 64×11×11 -> 64×5×5
    -> 128 -> 10``

    No softmax is applied here: the model returns ten raw logits so that
    ``CrossEntropyLoss`` can be used during training and softmax can be
    applied explicitly during inference.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, num_classes)`` logits."""
        return self.classifier(self.features(images))

    def count_parameters(self, *, trainable_only: bool = True) -> int:
        """Count parameters, optionally excluding frozen ones."""
        parameters = self.parameters()
        if trainable_only:
            parameters = (p for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)
