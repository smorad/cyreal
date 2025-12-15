"""MNIST dataset implemented without Torch dependencies."""
from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .dataset_protocol import DatasetProtocol

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _ensure_file(path: Path, url: str) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        magic, num, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST image file magic number: {magic}.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        magic, num = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label file magic number: {magic}.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num)


@dataclass
class MNISTDataset(DatasetProtocol):
    """Lightweight MNIST dataset that leaves preprocessing to transforms."""

    split: Literal["train", "test"] = "train"
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        base_dir = (
            Path(self.cache_dir)
            if self.cache_dir is not None
            else Path.home() / ".cache" / "jax_mnist"
        )
        self.cache_dir = base_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        images_file = self.cache_dir / f"{self.split}_images.gz"
        labels_file = self.cache_dir / f"{self.split}_labels.gz"

        _ensure_file(images_file, MNIST_URLS[f"{self.split}_images"])
        _ensure_file(labels_file, MNIST_URLS[f"{self.split}_labels"])

        images = _read_idx_images(images_file)[..., None].astype(np.uint8)
        labels = _read_idx_labels(labels_file).astype(np.int32)

        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int):
        return {
            "image": self._images[index],
            "label": self._labels[index],
        }
