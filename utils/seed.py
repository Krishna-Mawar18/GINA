"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all random number generators used by GINA.

    Args:
        seed: Master seed for Python, NumPy and PyTorch (CPU and CUDA).
        deterministic: Request deterministic CUDA/cuDNN kernels. With
            ``warn_only=True`` a kernel without a deterministic implementation
            warns instead of failing the run.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def make_generator(seed: int = 42) -> torch.Generator:
    """Return a seeded CPU generator for dataset splits and DataLoaders."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - required DataLoader signature
    """Seed a DataLoader worker derived from the main process seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
