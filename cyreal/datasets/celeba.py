"""CelebA dataset loader with minimal dependencies."""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .dataset_protocol import DatasetProtocol
from ..sources import DiskSource
from .utils import ensure_file as _ensure_file
from .utils import resolve_cache_dir, to_host_jax_array as _to_host_jax_array

CELEBA_URLS = {
    "images_zip": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip",
    "partition": "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pY0NSMzRuSXJEVkk",
    "attributes": "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pblRyaVFSWGxPY0U",
}

CELEBA_FILES = {
    "images_zip": "img_align_celeba.zip",
    "partition": "list_eval_partition.txt",
    "attributes": "list_attr_celeba.txt",
}

_SPLIT_TO_PARTITION = {"train": 0, "valid": 1, "test": 2}


def _ensure_text_file(path: Path, url: str, description: str) -> Path:
    try:
        return _ensure_file(path, url)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not download CelebA {description} from {url}."
        ) from exc


def _ensure_image_archive(path: Path, url: str) -> Path:
    if path.exists() and zipfile.is_zipfile(path):
        return path

    if path.exists():
        path.unlink()

    try:
        _ensure_file(path, url)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not download CelebA image archive from {url}."
        ) from exc

    if not zipfile.is_zipfile(path):
        path.unlink(missing_ok=True)
        raise FileNotFoundError(
            "Downloaded CelebA archive is not a valid zip file. "
            "Please provide a local `data_dir` containing `img_align_celeba/`."
        )

    return path


def _ensure_image_dir(root: Path, image_archive_url: str) -> Path:
    image_dir = root / "img_align_celeba"
    nested_dir = image_dir / "img_align_celeba"
    if image_dir.exists() and nested_dir.exists():
        return nested_dir
    if image_dir.exists():
        return image_dir

    archive_path = root / CELEBA_FILES["images_zip"]
    _ensure_image_archive(archive_path, image_archive_url)

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(path=root)

    image_dir = root / "img_align_celeba"
    nested_dir = image_dir / "img_align_celeba"
    if image_dir.exists() and nested_dir.exists():
        return nested_dir
    if image_dir.exists():
        return image_dir

    raise FileNotFoundError("Could not find `img_align_celeba/` after extraction.")


def _read_split_filenames(partition_path: Path, split: Literal["train", "valid", "test"]) -> list[str]:
    split_id = _SPLIT_TO_PARTITION[split]
    filenames: list[str] = []

    with partition_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            name, partition = parts
            if partition.isdigit() and int(partition) == split_id:
                filenames.append(name)

    if not filenames:
        raise ValueError(f"No CelebA samples found for split '{split}'.")

    return filenames


def _read_attributes(attributes_path: Path, filenames: list[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    wanted = set(filenames)
    attr_by_name: dict[str, np.ndarray] = {}

    with attributes_path.open("r", encoding="utf-8") as f:
        count_line = f.readline()
        if not count_line:
            raise ValueError("CelebA attributes file is empty.")

        names_line = f.readline().strip()
        if not names_line:
            raise ValueError("CelebA attributes header line is missing.")

        attribute_names = tuple(names_line.split())

        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_name = parts[0]
            if image_name not in wanted:
                continue
            values = np.asarray(parts[1:], dtype=np.int8)
            attr_by_name[image_name] = ((values + 1) // 2).astype(np.int8)

    missing = [name for name in filenames if name not in attr_by_name]
    if missing:
        raise ValueError(f"Missing CelebA attributes for image {missing[0]}.")

    attributes = np.stack([attr_by_name[name] for name in filenames], axis=0)
    return attributes, attribute_names


def _load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("CelebA image loading requires Pillow (`pip install pillow`).") from exc

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


@dataclass
class CelebADataset(DatasetProtocol):
    """CelebA face attributes dataset."""

    split: Literal["train", "valid", "test"] = "train"
    cache_dir: str | Path | None = None
    data_dir: str | Path | None = None
    include_attributes: bool = True
    image_archive_url: str = CELEBA_URLS["images_zip"]
    partition_url: str = CELEBA_URLS["partition"]
    attributes_url: str = CELEBA_URLS["attributes"]

    def __post_init__(self) -> None:
        root = Path(self.data_dir) if self.data_dir is not None else resolve_cache_dir(
            self.cache_dir, default_name="cyreal_celeba"
        )
        root.mkdir(parents=True, exist_ok=True)

        image_dir = _ensure_image_dir(root, self.image_archive_url)
        partition_path = _ensure_text_file(root / CELEBA_FILES["partition"], self.partition_url, "partition file")

        filenames = _read_split_filenames(partition_path, self.split)
        image_paths = [image_dir / name for name in filenames]
        missing = [p.name for p in image_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing CelebA image file, e.g. {missing[0]}.")

        images_np = np.stack([_load_image(path) for path in image_paths], axis=0)
        self._images = _to_host_jax_array(images_np)

        self._attribute_names: tuple[str, ...] = tuple()
        self._attributes: jax.Array | None = None
        if self.include_attributes:
            attributes_path = _ensure_text_file(
                root / CELEBA_FILES["attributes"],
                self.attributes_url,
                "attributes file",
            )
            attributes_np, names = _read_attributes(attributes_path, filenames)
            self._attributes = _to_host_jax_array(attributes_np.astype(np.int32))
            self._attribute_names = names

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int):
        sample = {"image": self._images[index]}
        if self._attributes is not None:
            sample["attributes"] = self._attributes[index]
        return sample

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return self._attribute_names

    def as_array_dict(self) -> dict[str, jax.Array]:
        data: dict[str, jax.Array] = {"image": self._images}
        if self._attributes is not None:
            data["attributes"] = self._attributes
        return data

    @classmethod
    def make_disk_source(
        cls,
        *,
        split: Literal["train", "valid", "test"] = "train",
        cache_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        include_attributes: bool = True,
        image_archive_url: str = CELEBA_URLS["images_zip"],
        partition_url: str = CELEBA_URLS["partition"],
        attributes_url: str = CELEBA_URLS["attributes"],
        ordering: Literal["sequential", "shuffle"] = "shuffle",
        prefetch_size: int = 64,
    ) -> DiskSource:
        root = Path(data_dir) if data_dir is not None else resolve_cache_dir(
            cache_dir, default_name="cyreal_celeba"
        )
        root.mkdir(parents=True, exist_ok=True)

        image_dir = _ensure_image_dir(root, image_archive_url)
        partition_path = _ensure_text_file(root / CELEBA_FILES["partition"], partition_url, "partition file")

        filenames = _read_split_filenames(partition_path, split)
        image_paths = [image_dir / name for name in filenames]
        missing = [p.name for p in image_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing CelebA image file, e.g. {missing[0]}.")

        first_image = _load_image(image_paths[0])
        image_shape = tuple(first_image.shape)

        attributes_np: np.ndarray | None = None
        if include_attributes:
            attributes_path = _ensure_text_file(
                root / CELEBA_FILES["attributes"],
                attributes_url,
                "attributes file",
            )
            attributes_np, _ = _read_attributes(attributes_path, filenames)

        def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
            idx = int(np.asarray(index))
            sample = {"image": _load_image(image_paths[idx])}
            if attributes_np is not None:
                sample["attributes"] = np.asarray(attributes_np[idx], dtype=np.int32)
            return sample

        sample_spec: dict[str, jax.ShapeDtypeStruct] = {
            "image": jax.ShapeDtypeStruct(shape=image_shape, dtype=jnp.uint8)
        }
        if attributes_np is not None:
            sample_spec["attributes"] = jax.ShapeDtypeStruct(
                shape=(int(attributes_np.shape[1]),),
                dtype=jnp.int32,
            )

        return DiskSource(
            length=int(len(image_paths)),
            sample_fn=_read_sample,
            sample_spec=sample_spec,
            ordering=ordering,
            prefetch_size=prefetch_size,
        )
