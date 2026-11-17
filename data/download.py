"""MNIST download and integrity checks.

torchvision handles the actual download from its maintained mirror. This module
adds verification so that a partially downloaded dataset fails early.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import MNIST

DEFAULT_DATA_DIR = Path("data") / "raw"
RAW_SUBDIR = Path("MNIST") / "raw"

EXPECTED_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def raw_directory(root: str | Path) -> Path:
    """Return the expected torchvision raw-data directory for ``root``."""
    return Path(root).expanduser() / RAW_SUBDIR


def missing_mnist_files(root: str | Path) -> list[str]:
    """List MNIST IDX files that are missing from ``root``."""
    raw_dir = raw_directory(root)
    return [name for name in EXPECTED_FILES if not (raw_dir / name).is_file()]


def download_mnist(root: str | Path = DEFAULT_DATA_DIR) -> Path:
    """Download MNIST if necessary and verify the extracted files.

    Args:
        root: Parent directory; torchvision writes to ``root/MNIST/raw``.

    Returns:
        Path to the verified raw directory.
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    for train in (True, False):
        MNIST(root=str(root), train=train, download=True)

    missing = missing_mnist_files(root)
    if missing:
        raise RuntimeError(f"MNIST download incomplete; missing files: {missing}")

    return raw_directory(root)


def main() -> None:
    """Command-line entry point: ``python -m data.download``."""
    parser = argparse.ArgumentParser(description="Download and verify the MNIST dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dataset root directory (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    raw_dir = download_mnist(args.data_dir)
    sizes = {path.name: path.stat().st_size for path in sorted(raw_dir.glob("*idx*-ubyte"))}
    print(f"MNIST ready at: {raw_dir.resolve()}")
    for name, size in sizes.items():
        print(f"  {name}: {size / 1_048_576:.2f} MiB")


if __name__ == "__main__":
    main()
