"""CIFAR-10 dataset that stays within NumPy/JAX dependencies."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..dataset_protocol import DatasetProtocol
from ..sources import DiskSource
from .utils import (
    download_archive,
    ensure_tar_extracted,
    resolve_cache_dir,
    to_host_jax_array as _to_host_jax_array,
)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _batch_names(split: Literal["train", "test"]) -> list[str]:
    if split == "train":
        return [f"data_batch_{i}" for i in range(1, 6)]
    if split == "test":
        return ["test_batch"]
    raise ValueError("split must be 'train' or 'test'.")


def _load_split_numpy(split: Literal["train", "test"], extract_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for name in _batch_names(split):
        batch_path = extract_dir / name
        if not batch_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10 batch '{name}'.")
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="latin1")
        data = batch["data"].reshape(-1, 3, 32, 32)
        images.append(np.transpose(data, (0, 2, 3, 1)).astype(np.uint8))
        labels.append(np.asarray(batch["labels"], dtype=np.int32))

    images_np = np.concatenate(images, axis=0)
    labels_np = np.concatenate(labels, axis=0)
    return images_np, labels_np

def _ensure_split_numpy_cache(
    split: Literal["train", "test"],
    base_dir: Path,
    extract_dir: Path,
) -> tuple[Path, Path]:
    cache_root = base_dir / "disk_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    images_path = cache_root / f"{split}_images.npy"
    labels_path = cache_root / f"{split}_labels.npy"
    if not images_path.exists() or not labels_path.exists():
        images_np, labels_np = _load_split_numpy(split, extract_dir)
        np.save(images_path, images_np)
        np.save(labels_path, labels_np)
    return images_path, labels_path


@dataclass
class CIFAR10Dataset(DatasetProtocol):
    """Download-free CIFAR-10 access ready for `ArraySampleSource`."""

    split: Literal["train", "test"] = "train"
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        base_dir = resolve_cache_dir(self.cache_dir, default_name="cyreal_cifar10")
        archive_path = base_dir / "cifar-10-python.tar.gz"
        download_archive(CIFAR10_URL, archive_path)
        extract_dir = ensure_tar_extracted(archive_path, base_dir, target_dir="cifar-10-batches-py")

        images_np, labels_np = _load_split_numpy(self.split, extract_dir)
        self._images = _to_host_jax_array(images_np)
        self._labels = _to_host_jax_array(labels_np)

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
        base_dir = resolve_cache_dir(cache_dir, default_name="cyreal_cifar10")
        archive_path = base_dir / "cifar-10-python.tar.gz"
        download_archive(CIFAR10_URL, archive_path)
        extract_dir = ensure_tar_extracted(archive_path, base_dir, target_dir="cifar-10-batches-py")

        images_path, labels_path = _ensure_split_numpy_cache(split, base_dir, extract_dir)
        images_memmap = np.load(images_path, mmap_mode="r")
        labels_memmap = np.load(labels_path, mmap_mode="r")

        if images_memmap.shape[0] != labels_memmap.shape[0]:
            raise ValueError("CIFAR-10 image and label counts do not match.")

        def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
            idx = int(np.asarray(index))
            image = np.asarray(images_memmap[idx], dtype=np.uint8)
            label = np.asarray(labels_memmap[idx], dtype=np.int32)
            return {"image": image, "label": label}

        sample_spec = {
            "image": jax.ShapeDtypeStruct(shape=tuple(images_memmap.shape[1:]), dtype=jnp.uint8),
            "label": jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
        }

        return DiskSource(
            length=int(labels_memmap.shape[0]),
            sample_fn=_read_sample,
            sample_spec=sample_spec,
            ordering=ordering,
            prefetch_size=prefetch_size,
        )