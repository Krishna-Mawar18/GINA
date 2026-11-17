"""Device discovery and CUDA memory reporting.

Both the training CLI and the Streamlit application use these helpers so that
CUDA is used when it is really available and CPU fallback is explicit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch

CUDA_PREFERENCES = frozenset({"auto", "cuda", "gpu"})
_MIB = 1024**2


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a device preference into an available :class:`torch.device`.

    Args:
        preference: ``"auto"``, ``"cuda"``/``"gpu"``, ``"cuda:N"`` or ``"cpu"``.

    Returns:
        A CUDA device when requested and available, otherwise CPU.

    Raises:
        ValueError: If the preference string is not recognised.
    """
    pref = preference.strip().lower()

    if pref == "cpu":
        return torch.device("cpu")

    if pref in CUDA_PREFERENCES or pref.startswith("cuda:"):
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA was requested but is not available in this PyTorch build "
                "or on this machine. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")

        if pref.startswith("cuda:"):
            try:
                index = int(pref.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device index in {preference!r}.") from exc
            if not 0 <= index < torch.cuda.device_count():
                warnings.warn(
                    f"CUDA device {index} does not exist (count="
                    f"{torch.cuda.device_count()}); using the default CUDA device.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return torch.device("cuda")
            return torch.device("cuda", index)

        return torch.device("cuda")

    raise ValueError(
        f"Unsupported device preference {preference!r}. Use 'auto', 'cuda', 'cuda:N' or 'cpu'."
    )


@dataclass(frozen=True)
class DeviceInfo:
    """Snapshot of the compute device and, when relevant, GPU memory."""

    device: torch.device
    cuda_available: bool
    gpu_name: str | None = None
    compute_capability: str | None = None
    cuda_runtime_version: str | None = None
    cudnn_version: str | None = None
    memory_total_mib: float | None = None
    memory_free_mib: float | None = None
    memory_allocated_mib: float | None = None
    memory_reserved_mib: float | None = None

    def to_dict(self) -> dict[str, str | bool | float | None]:
        """Return a plain dictionary suitable for logs and JSON reports."""
        return {
            "device": str(self.device),
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "compute_capability": self.compute_capability,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cudnn_version": self.cudnn_version,
            "memory_total_mib": self.memory_total_mib,
            "memory_free_mib": self.memory_free_mib,
            "memory_allocated_mib": self.memory_allocated_mib,
            "memory_reserved_mib": self.memory_reserved_mib,
        }


def inspect_device(device: torch.device | None = None) -> DeviceInfo:
    """Collect device name, driver/runtime and GPU memory information."""
    selected = device if device is not None else resolve_device("auto")
    cuda_available = torch.cuda.is_available()

    info = DeviceInfo(
        device=selected,
        cuda_available=cuda_available,
        cuda_runtime_version=torch.version.cuda,
        cudnn_version=str(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else None,
    )

    if selected.type != "cuda" or not cuda_available:
        return info

    index = selected.index if selected.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)

    return DeviceInfo(
        device=selected,
        cuda_available=True,
        gpu_name=props.name,
        compute_capability=f"{props.major}.{props.minor}",
        cuda_runtime_version=torch.version.cuda,
        cudnn_version=str(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else None,
        memory_total_mib=round(total_bytes / _MIB, 1),
        memory_free_mib=round(free_bytes / _MIB, 1),
        memory_allocated_mib=round(torch.cuda.memory_allocated(index) / _MIB, 1),
        memory_reserved_mib=round(torch.cuda.memory_reserved(index) / _MIB, 1),
    )


def describe_device(info: DeviceInfo | None = None) -> str:
    """Build a short human-readable device summary for CLI output and the UI."""
    info = info or inspect_device()
    lines = [f"Device: {info.device}"]
    if info.gpu_name:
        lines.append(f"GPU: {info.gpu_name} (compute capability {info.compute_capability})")
    if info.cuda_runtime_version:
        lines.append(f"PyTorch CUDA runtime: {info.cuda_runtime_version}")
    if info.cudnn_version:
        lines.append(f"cuDNN: {info.cudnn_version}")
    if info.memory_total_mib is not None:
        lines.append(
            f"GPU memory: {info.memory_free_mib:.0f} MiB free / "
            f"{info.memory_total_mib:.0f} MiB total"
        )
    return "\n".join(lines)


def move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors in a nested batch (tensor/dict/list/tuple) to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def clear_cuda_cache(device: torch.device | None = None) -> None:
    """Release unused CUDA caching-allocator memory when CUDA is usable."""
    if device is not None and device.type != "cuda":
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
