"""EMNIST dataset utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..dataset_protocol import DatasetProtocol
from ..sources import DiskSampleSource
from ._mnist_like import (
    ensure_file as _ensure_file,
    ensure_uncompressed_idx as _ensure_uncompressed_idx,
    read_image_header as _read_image_header,
    read_idx_images as _read_idx_images,
    read_idx_labels as _read_idx_labels,
    read_label_header as _read_label_header,
    to_host_jax_array as _to_host_jax_array,
)

EMNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
EMNIST_URLS = {
    "balanced": {
        "train_images": f"{EMNIST_BASE}/emnist-balanced-train-images-idx3-ubyte.gz",
        "train_labels": f"{EMNIST_BASE}/emnist-balanced-train-labels-idx1-ubyte.gz",
        "test_images": f"{EMNIST_BASE}/emnist-balanced-test-images-idx3-ubyte.gz",
        "test_labels": f"{EMNIST_BASE}/emnist-balanced-test-labels-idx1-ubyte.gz",
    },
    "byclass": {
        "train_images": f"{EMNIST_BASE}/emnist-byclass-train-images-idx3-ubyte.gz",
        "train_labels": f"{EMNIST_BASE}/emnist-byclass-train-labels-idx1-ubyte.gz",
        "test_images": f"{EMNIST_BASE}/emnist-byclass-test-images-idx3-ubyte.gz",
        "test_labels": f"{EMNIST_BASE}/emnist-byclass-test-labels-idx1-ubyte.gz",
    },
    "bymerge": {
        "train_images": f"{EMNIST_BASE}/emnist-bymerge-train-images-idx3-ubyte.gz",
        "train_labels": f"{EMNIST_BASE}/emnist-bymerge-train-labels-idx1-ubyte.gz",
        "test_images": f"{EMNIST_BASE}/emnist-bymerge-test-images-idx3-ubyte.gz",
        "test_labels": f"{EMNIST_BASE}/emnist-bymerge-test-labels-idx1-ubyte.gz",
    },
    "digits": {
        "train_images": f"{EMNIST_BASE}/emnist-digits-train-images-idx3-ubyte.gz",
        "train_labels": f"{EMNIST_BASE}/emnist-digits-train-labels-idx1-ubyte.gz",
        "test_images": f"{EMNIST_BASE}/emnist-digits-test-images-idx3-ubyte.gz",
        "test_labels": f"{EMNIST_BASE}/emnist-digits-test-labels-idx1-ubyte.gz",
    },
    "letters": {
        "train_images": f"{EMNIST_BASE}/emnist-letters-train-images-idx3-ubyte.gz",
        "train_labels": f"{EMNIST_BASE}/emnist-letters-train-labels-idx1-ubyte.gz",
        "test_images": f"{EMNIST_BASE}/emnist-letters-test-images-idx3-ubyte.gz",
        "test_labels": f"{EMNIST_BASE}/emnist-letters-test-labels-idx1-ubyte.gz",
    },
}


def _resolve_cache_dir(cache_dir: str | Path | None, subset: str) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return Path.home() / ".cache" / "emnist" / subset


@dataclass
class EMNISTDataset(DatasetProtocol):
    """Extended MNIST dataset family that covers multiple subsets."""

    subset: Literal["balanced", "byclass", "bymerge", "digits", "letters"] = "balanced"
    split: Literal["train", "test"] = "train"
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.subset not in EMNIST_URLS:
            raise ValueError(f"Unknown EMNIST subset '{self.subset}'.")
        base_dir = _resolve_cache_dir(self.cache_dir, self.subset)
        base_dir.mkdir(parents=True, exist_ok=True)
        images_file = base_dir / f"{self.split}_images.gz"
        labels_file = base_dir / f"{self.split}_labels.gz"

        urls = EMNIST_URLS[self.subset]
        _ensure_file(images_file, urls[f"{self.split}_images"])
        _ensure_file(labels_file, urls[f"{self.split}_labels"])

        images = _read_idx_images(images_file)[..., None].astype(np.uint8)
        labels = _read_idx_labels(labels_file).astype(np.int32)

        self._images = _to_host_jax_array(images)
        self._labels = _to_host_jax_array(labels)
        self.cache_dir = base_dir

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int):
        return {
            "image": self._images[index],
            "label": self._labels[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        return {
            "image": self._images,
            "label": self._labels,
        }

    @classmethod
    def make_disk_source(
        cls,
        *,
        subset: Literal["balanced", "byclass", "bymerge", "digits", "letters"] = "balanced",
        split: Literal["train", "test"] = "train",
        cache_dir: str | Path | None = None,
        ordering: Literal["sequential", "shuffle"] = "shuffle",
        prefetch_size: int = 64,
    ) -> DiskSampleSource:
        if subset not in EMNIST_URLS:
            raise ValueError(f"Unknown EMNIST subset '{subset}'.")
        base_dir = _resolve_cache_dir(cache_dir, subset)
        base_dir.mkdir(parents=True, exist_ok=True)

        images_gz = base_dir / f"{split}_images.gz"
        labels_gz = base_dir / f"{split}_labels.gz"
        urls = EMNIST_URLS[subset]
        _ensure_file(images_gz, urls[f"{split}_images"])
        _ensure_file(labels_gz, urls[f"{split}_labels"])

        images_path = _ensure_uncompressed_idx(images_gz)
        labels_path = _ensure_uncompressed_idx(labels_gz)

        num_images, rows, cols = _read_image_header(images_path)
        num_labels = _read_label_header(labels_path)
        if num_images != num_labels:
            raise ValueError("EMNIST image/label counts do not match.")

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