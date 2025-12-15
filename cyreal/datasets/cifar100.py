"""CIFAR-100 dataset helpers."""
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

CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def _load_split_numpy(split: Literal["train", "test"], extract_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    file_name = "train" if split == "train" else "test"
    batch_path = extract_dir / file_name
    if not batch_path.exists():
        raise FileNotFoundError(f"Missing CIFAR-100 split file '{file_name}'.")
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    data = batch["data"].reshape(-1, 3, 32, 32)
    images = np.transpose(data, (0, 2, 3, 1)).astype(np.uint8)
    fine = np.asarray(batch["fine_labels"], dtype=np.int32)
    coarse = np.asarray(batch["coarse_labels"], dtype=np.int32)
    return images, fine, coarse

def _ensure_split_numpy_cache(
    split: Literal["train", "test"],
    base_dir: Path,
    extract_dir: Path,
) -> tuple[Path, Path, Path]:
    cache_root = base_dir / "disk_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    images_path = cache_root / f"{split}_images.npy"
    fine_labels_path = cache_root / f"{split}_fine_labels.npy"
    coarse_labels_path = cache_root / f"{split}_coarse_labels.npy"
    if not images_path.exists() or not fine_labels_path.exists() or not coarse_labels_path.exists():
        images_np, fine_np, coarse_np = _load_split_numpy(split, extract_dir)
        np.save(images_path, images_np)
        np.save(fine_labels_path, fine_np)
        np.save(coarse_labels_path, coarse_np)
    return images_path, fine_labels_path, coarse_labels_path


@dataclass
class CIFAR100Dataset(DatasetProtocol):
    """CIFAR-100 dataset that keeps preprocessing within NumPy/JAX."""

    split: Literal["train", "test"] = "train"
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        base_dir = resolve_cache_dir(self.cache_dir, default_name="cifar100")
        archive_path = base_dir / "cifar-100-python.tar.gz"
        download_archive(CIFAR100_URL, archive_path)
        extract_dir = ensure_tar_extracted(archive_path, base_dir, target_dir="cifar-100-python")

        images_np, fine_np, coarse_np = _load_split_numpy(self.split, extract_dir)
        self._images = _to_host_jax_array(images_np)
        self._fine_labels = _to_host_jax_array(fine_np)
        self._coarse_labels = _to_host_jax_array(coarse_np)

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int):
        return {
            "image": self._images[index],
            "label": self._fine_labels[index],
            "coarse_label": self._coarse_labels[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        return {
            "image": self._images,
            "label": self._fine_labels,
            "coarse_label": self._coarse_labels,
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
        base_dir = resolve_cache_dir(cache_dir, default_name="cifar100")
        archive_path = base_dir / "cifar-100-python.tar.gz"
        download_archive(CIFAR100_URL, archive_path)
        extract_dir = ensure_tar_extracted(archive_path, base_dir, target_dir="cifar-100-python")

        images_path, fine_labels_path, coarse_labels_path = _ensure_split_numpy_cache(
            split, base_dir, extract_dir
        )
        images_memmap = np.load(images_path, mmap_mode="r")
        fine_labels_memmap = np.load(fine_labels_path, mmap_mode="r")
        coarse_labels_memmap = np.load(coarse_labels_path, mmap_mode="r")

        lengths = {
            images_memmap.shape[0],
            fine_labels_memmap.shape[0],
            coarse_labels_memmap.shape[0],
        }
        if len(lengths) != 1:
            raise ValueError("CIFAR-100 arrays do not share the same length.")

        def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
            idx = int(np.asarray(index))
            image = np.asarray(images_memmap[idx], dtype=np.uint8)
            label = np.asarray(fine_labels_memmap[idx], dtype=np.int32)
            coarse_label = np.asarray(coarse_labels_memmap[idx], dtype=np.int32)
            return {"image": image, "label": label, "coarse_label": coarse_label}

        sample_spec = {
            "image": jax.ShapeDtypeStruct(shape=tuple(images_memmap.shape[1:]), dtype=jnp.uint8),
            "label": jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
            "coarse_label": jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
        }

        return DiskSource(
            length=int(images_memmap.shape[0]),
            sample_fn=_read_sample,
            sample_spec=sample_spec,
            ordering=ordering,
            prefetch_size=prefetch_size,
        )