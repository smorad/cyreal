"""CIFAR-10 dataset that stays within NumPy/JAX dependencies."""
from __future__ import annotations

import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..dataset_protocol import DatasetProtocol

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _download(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def _ensure_extracted(archive: Path, extract_root: Path) -> Path:
    target = extract_root / "cifar-10-batches-py"
    if target.exists():
        return target
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=extract_root)
    return target


@dataclass
class CIFAR10Dataset(DatasetProtocol):
    """Download-free CIFAR-10 access compatible with `dataset_to_jax`."""

    split: Literal["train", "test"] = "train"
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        base_dir = (
            Path(self.cache_dir)
            if self.cache_dir is not None
            else Path.home() / ".cache" / "cereal_cifar10"
        )
        archive_path = base_dir / "cifar-10-python.tar.gz"
        _download(CIFAR10_URL, archive_path)
        extract_dir = _ensure_extracted(archive_path, base_dir)

        if self.split == "train":
            batch_names = [f"data_batch_{i}" for i in range(1, 6)]
        elif self.split == "test":
            batch_names = ["test_batch"]
        else:
            raise ValueError("split must be 'train' or 'test'.")

        images = []
        labels = []
        for name in batch_names:
            batch_path = extract_dir / name
            if not batch_path.exists():
                raise FileNotFoundError(f"Missing CIFAR-10 batch '{name}'.")
            with open(batch_path, "rb") as f:
                batch = pickle.load(f, encoding="latin1")
            data = batch["data"].reshape(-1, 3, 32, 32)
            # Move channel dimension to the end for consistency with MNIST image layout.
            images.append(np.transpose(data, (0, 2, 3, 1)).astype(np.uint8))
            labels.append(np.asarray(batch["labels"], dtype=np.int32))

        self._images = np.concatenate(images, axis=0)
        self._labels = np.concatenate(labels, axis=0)

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int):
        return {
            "image": self._images[index],
            "label": self._labels[index],
        }
