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


def _to_host_jax_array(array: np.ndarray) -> jax.Array:
    cpu_devices = jax.devices("cpu")
    if cpu_devices:
        with jax.default_device(cpu_devices[0]):
            return jnp.asarray(array)
    return jnp.asarray(array)


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


def _ensure_uncompressed_idx(path: Path) -> Path:
    target = path.with_suffix("")
    if target.exists():
        return target
    with gzip.open(path, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def _read_image_header(path: Path) -> tuple[int, int, int]:
    with open(path, "rb") as f:
        header = f.read(16)
        magic, num, rows, cols = struct.unpack(">IIII", header)
    if magic != 2051:
        raise ValueError(f"Unexpected MNIST image file magic number: {magic}.")
    return int(num), int(rows), int(cols)


def _read_label_header(path: Path) -> int:
    with open(path, "rb") as f:
        header = f.read(8)
        magic, num = struct.unpack(">II", header)
    if magic != 2049:
        raise ValueError(f"Unexpected MNIST label file magic number: {magic}.")
    return int(num)


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

        self._images = _to_host_jax_array(images)
        self._labels = _to_host_jax_array(labels)

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int):
        return {
            "image": self._images[index],
            "label": self._labels[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        """Expose the full dataset as a PyTree of JAX arrays."""

        return {
            "image": self._images,
            "label": self._labels,
        }

    @classmethod
    def make_disk_source(
        cls,
        *,
        split: Literal["train", "test"] = "train",
        cache_dir: str | Path | None = None,
        ordering: Literal["sequential", "shuffle"] = "shuffle",
        prefetch_size: int = 64,
    ) -> DiskSampleSource:
        base_dir = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "jax_mnist"
        base_dir.mkdir(parents=True, exist_ok=True)

        images_gz = base_dir / f"{split}_images.gz"
        labels_gz = base_dir / f"{split}_labels.gz"
        _ensure_file(images_gz, MNIST_URLS[f"{split}_images"])
        _ensure_file(labels_gz, MNIST_URLS[f"{split}_labels"])

        images_path = _ensure_uncompressed_idx(images_gz)
        labels_path = _ensure_uncompressed_idx(labels_gz)

        num_images, rows, cols = _read_image_header(images_path)
        num_labels = _read_label_header(labels_path)
        if num_images != num_labels:
            raise ValueError("MNIST image/label counts do not match.")

        images_memmap = np.memmap(
            images_path,
            dtype=np.uint8,
            mode="r",
            offset=16,
            shape=(num_images, rows, cols),
        )
        labels_memmap = np.memmap(
            labels_path,
            dtype=np.uint8,
            mode="r",
            offset=8,
            shape=(num_labels,),
        )

        def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
            idx = int(np.asarray(index))
            image = np.asarray(images_memmap[idx], dtype=np.uint8)[..., None]
            label = np.asarray(labels_memmap[idx], dtype=np.int32)
            return {"image": image, "label": label}

        sample_spec = {
            "image": jax.ShapeDtypeStruct(shape=(rows, cols, 1), dtype=jnp.uint8),
            "label": jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
        }

        return DiskSampleSource(
            length=int(num_images),
            sample_fn=_read_sample,
            sample_spec=sample_spec,
            ordering=ordering,
            prefetch_size=prefetch_size,
        )


@dataclass
class MNISTDiskSource:
    """Stream MNIST samples directly from on-disk IDX files via io_callback."""

    split: Literal["train", "test"] = "train"
    cache_dir: str | Path | None = None
    ordering: Literal["sequential", "shuffle"] = "shuffle"
    prefetch_size: int = 64

    def __post_init__(self) -> None:
        self._disk_source = MNISTDataset.make_disk_source(
            split=self.split,
            cache_dir=self.cache_dir,
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
