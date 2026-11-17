"""Tests for device resolution and reporting."""

from __future__ import annotations

import pytest
import torch

from utils.device import inspect_device, move_to_device, resolve_device


def test_cpu_can_always_be_selected() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported device preference"):
        resolve_device("tpu")


def test_device_info_is_serializable() -> None:
    info = inspect_device(resolve_device("cpu"))

    assert info.device.type == "cpu"
    assert "device" in info.to_dict()


def test_move_to_device_handles_nested_batches() -> None:
    batch = {"images": torch.zeros(2, 2), "labels": [torch.tensor(1), torch.tensor(2)]}
    moved = move_to_device(batch, torch.device("cpu"))

    assert moved["images"].shape == (2, 2)
    assert len(moved["labels"]) == 2
