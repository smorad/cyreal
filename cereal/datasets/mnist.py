"""MNIST dataset utilities without Torch dependencies."""
from __future__ import annotations

import gzip
import shutil
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..dataset_protocol import DatasetProtocol
from ..sources import DiskSampleSource

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


@dataclass
class MNISTDiskSource:
    """Stream MNIST samples directly from on-disk IDX files via io_callback."""

    split: Literal["train", "test"] = "train"
    cache_dir: str | Path | None = None
    ordering: Literal["sequential", "shuffle"] = "shuffle"
    prefetch_size: int = 64

    def __post_init__(self) -> None:
        base_dir = (
            Path(self.cache_dir)
            if self.cache_dir is not None
            else Path.home() / ".cache" / "jax_mnist"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        images_gz = base_dir / f"{self.split}_images.gz"
        labels_gz = base_dir / f"{self.split}_labels.gz"
        _ensure_file(images_gz, MNIST_URLS[f"{self.split}_images"])
        _ensure_file(labels_gz, MNIST_URLS[f"{self.split}_labels"])

        images_path = self._ensure_uncompressed(images_gz)
        labels_path = self._ensure_uncompressed(labels_gz)

        num_images, rows, cols = self._read_image_header(images_path)
        num_labels = self._read_label_header(labels_path)
        if num_images != num_labels:
            raise ValueError("MNIST image/label counts do not match.")

        self._images_memmap = np.memmap(
            images_path,
            dtype=np.uint8,
            mode="r",
            offset=16,
            shape=(num_images, rows, cols),
        )
        self._labels_memmap = np.memmap(
            labels_path,
            dtype=np.uint8,
            mode="r",
            offset=8,
            shape=(num_labels,),
        )

        self._num_samples = int(num_images)
        sample_spec = {
            "image": jax.ShapeDtypeStruct(shape=(rows, cols, 1), dtype=jnp.uint8),
            "label": jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
        }

        self._disk_source = DiskSampleSource(
            length=self._num_samples,
            sample_fn=self._read_sample,
            sample_spec=sample_spec,
            ordering=self.ordering,
            prefetch_size=self.prefetch_size,
        )
        self.steps_per_epoch = self._disk_source.steps_per_epoch

    def element_spec(self):
        return self._disk_source.element_spec()

    def init_state(self, key=None):
        return self._disk_source.init_state(key)

    def next(self, state):
        return self._disk_source.next(state)

    def _read_sample(self, index: int | np.ndarray) -> dict[str, np.ndarray]:
        idx = int(np.asarray(index))
        image = np.asarray(self._images_memmap[idx], dtype=np.uint8)[..., None]
        label = np.asarray(self._labels_memmap[idx], dtype=np.int32)
        return {"image": image, "label": label}

    @staticmethod
    def _ensure_uncompressed(path: Path) -> Path:
        target = path.with_suffix("")
        if target.exists():
            return target
        with gzip.open(path, "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return target

    @staticmethod
    def _read_image_header(path: Path) -> tuple[int, int, int]:
        with open(path, "rb") as f:
            header = f.read(16)
            magic, num, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST image file magic number: {magic}.")
        return int(num), int(rows), int(cols)

    @staticmethod
    def _read_label_header(path: Path) -> int:
        with open(path, "rb") as f:
            header = f.read(8)
            magic, num = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label file magic number: {magic}.")
        return int(num)