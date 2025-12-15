"""MNIST dataset utilities without Torch dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..dataset_protocol import DatasetProtocol
from ..sources import DiskSource
from .utils import (
    ensure_file as _ensure_file,
    resolve_cache_dir,
    to_host_jax_array as _to_host_jax_array,
)
from .mnist_utils import (
    ensure_uncompressed_idx as _ensure_uncompressed_idx,
    read_image_header as _read_image_header,
    read_idx_images as _read_idx_images,
    read_idx_labels as _read_idx_labels,
    read_label_header as _read_label_header,
)

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


@dataclass
class MNISTDataset(DatasetProtocol):
    """Lightweight MNIST dataset that leaves preprocessing to transforms."""

    split: Literal["train", "test"] = "train"
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        base_dir = resolve_cache_dir(self.cache_dir, default_name="jax_mnist")
        self.cache_dir = base_dir
        images_file = base_dir / f"{self.split}_images.gz"
        labels_file = base_dir / f"{self.split}_labels.gz"

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
    ) -> DiskSource:
        base_dir = resolve_cache_dir(cache_dir, default_name="jax_mnist")

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

        return DiskSource(
            length=int(num_images),
            sample_fn=_read_sample,
            sample_spec=sample_spec,
            ordering=ordering,
            prefetch_size=prefetch_size,
        )