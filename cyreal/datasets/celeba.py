"""CelebA dataset loader without Torch dependencies."""
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

CELEBA_FILES = {
    "images_zip": "img_align_celeba.zip",
    "partition": "list_eval_partition.txt",
    "attributes": "list_attr_celeba.txt",
}

_SPLIT_TO_PARTITION = {"train": 0, "valid": 1, "test": 2}


def _ensure_celeba_layout(
    *,
    base_dir: Path,
    data_dir: str | Path | None,
    image_archive_url: str | None,
    partition_url: str | None,
    attributes_url: str | None,
) -> tuple[Path, Path, Path | None]:
    root = Path(data_dir) if data_dir is not None else base_dir
    root.mkdir(parents=True, exist_ok=True)

    partition_path = root / CELEBA_FILES["partition"]
    if not partition_path.exists():
        if partition_url is None:
            raise FileNotFoundError(
                "CelebA partition file not found. Provide `data_dir` with "
                "`list_eval_partition.txt` or pass `partition_url`."
            )
        _ensure_file(partition_path, partition_url)

    attributes_path = root / CELEBA_FILES["attributes"]
    if not attributes_path.exists() and attributes_url is not None:
        _ensure_file(attributes_path, attributes_url)

    image_dir = root / "img_align_celeba"
    nested_image_dir = image_dir / "img_align_celeba"
    if image_dir.exists() and nested_image_dir.exists():
        image_dir = nested_image_dir

    if not image_dir.exists():
        archive_path = root / CELEBA_FILES["images_zip"]
        if not archive_path.exists():
            if image_archive_url is None:
                raise FileNotFoundError(
                    "CelebA image directory not found. Provide `data_dir` with "
                    "`img_align_celeba/` or pass `image_archive_url`."
                )
            _ensure_file(archive_path, image_archive_url)

        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(path=root)

        image_dir = root / "img_align_celeba"
        nested_image_dir = image_dir / "img_align_celeba"
        if image_dir.exists() and nested_image_dir.exists():
            image_dir = nested_image_dir

    if not image_dir.exists():
        raise FileNotFoundError("Could not locate `img_align_celeba` after archive extraction.")

    return image_dir, partition_path, attributes_path if attributes_path.exists() else None


def _read_split_filenames(partition_path: Path, split: Literal["train", "valid", "test"]) -> list[str]:
    split_index = _SPLIT_TO_PARTITION[split]
    filenames: list[str] = []
    with partition_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            name, partition = parts
            if partition.isdigit() and int(partition) == split_index:
                filenames.append(name)
    if not filenames:
        raise ValueError(f"No CelebA samples found for split '{split}'.")
    return filenames


def _read_attributes(attributes_path: Path, filenames: list[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    wanted = set(filenames)
    name_to_attributes: dict[str, np.ndarray] = {}

    with attributes_path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            raise ValueError("CelebA attributes file is empty.")
        names_line = f.readline().strip()
        if not names_line:
            raise ValueError("CelebA attributes header line is missing.")
        attribute_names = tuple(names_line.split())

        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_name, raw_values = parts[0], parts[1:]
            if image_name not in wanted:
                continue
            values = np.asarray(raw_values, dtype=np.int8)
            name_to_attributes[image_name] = ((values + 1) // 2).astype(np.int8)

    missing = [name for name in filenames if name not in name_to_attributes]
    if missing:
        raise ValueError(
            "Missing CelebA attributes for some images, e.g. "
            f"{missing[0]}"
        )

    attributes = np.stack([name_to_attributes[name] for name in filenames], axis=0)
    return attributes, attribute_names


def _load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "CelebA image loading requires Pillow. Install it with `pip install pillow`."
        ) from exc

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


@dataclass
class CelebADataset(DatasetProtocol):
    """CelebA face attributes dataset.

    By default this loader expects a local CelebA layout with:
    - `img_align_celeba/`
    - `list_eval_partition.txt`
    - optional `list_attr_celeba.txt`

    You can also pass URLs for files that are not present locally.
    """

    split: Literal["train", "valid", "test"] = "train"
    cache_dir: str | Path | None = None
    data_dir: str | Path | None = None
    include_attributes: bool = True
    image_archive_url: str | None = None
    partition_url: str | None = None
    attributes_url: str | None = None

    def __post_init__(self) -> None:
        base_dir = resolve_cache_dir(self.cache_dir, default_name="cyreal_celeba")
        image_dir, partition_path, attributes_path = _ensure_celeba_layout(
            base_dir=base_dir,
            data_dir=self.data_dir,
            image_archive_url=self.image_archive_url,
            partition_url=self.partition_url,
            attributes_url=self.attributes_url,
        )

        filenames = _read_split_filenames(partition_path, self.split)
        image_paths = [image_dir / name for name in filenames]
        missing = [p.name for p in image_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing CelebA image file, e.g. {missing[0]}")

        images_np = np.stack([_load_image(path) for path in image_paths], axis=0)
        self._images = _to_host_jax_array(images_np)

        self._attribute_names: tuple[str, ...] = tuple()
        self._attributes: jax.Array | None = None
        if self.include_attributes:
            if attributes_path is None:
                raise FileNotFoundError(
                    "`include_attributes=True` but `list_attr_celeba.txt` was not found."
                )
            attributes_np, attribute_names = _read_attributes(attributes_path, filenames)
            self._attributes = _to_host_jax_array(attributes_np.astype(np.int32))
            self._attribute_names = attribute_names

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int):
        sample = {
            "image": self._images[index],
        }
        if self._attributes is not None:
            sample["attributes"] = self._attributes[index]
        return sample

    @property
    def attribute_names(self) -> tuple[str, ...]:
        """Tuple of attribute names aligned to the `attributes` vector."""

        return self._attribute_names

    def as_array_dict(self) -> dict[str, jax.Array]:
        """Expose the full dataset as a PyTree of JAX arrays."""

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
        image_archive_url: str | None = None,
        partition_url: str | None = None,
        attributes_url: str | None = None,
        ordering: Literal["sequential", "shuffle"] = "shuffle",
        prefetch_size: int = 64,
    ) -> DiskSource:
        """Return the dataset in a disk streaming format."""

        base_dir = resolve_cache_dir(cache_dir, default_name="cyreal_celeba")
        image_dir, partition_path, attributes_path = _ensure_celeba_layout(
            base_dir=base_dir,
            data_dir=data_dir,
            image_archive_url=image_archive_url,
            partition_url=partition_url,
            attributes_url=attributes_url,
        )

        filenames = _read_split_filenames(partition_path, split)
        image_paths = [image_dir / name for name in filenames]
        missing = [p.name for p in image_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing CelebA image file, e.g. {missing[0]}")

        first_image = _load_image(image_paths[0])
        image_shape = tuple(first_image.shape)

        attributes_np: np.ndarray | None = None
        if include_attributes:
            if attributes_path is None:
                raise FileNotFoundError(
                    "`include_attributes=True` but `list_attr_celeba.txt` was not found."
                )
            attributes_np, _ = _read_attributes(attributes_path, filenames)

        def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
            idx = int(np.asarray(index))
            sample = {
                "image": _load_image(image_paths[idx]),
            }
            if attributes_np is not None:
                sample["attributes"] = np.asarray(attributes_np[idx], dtype=np.int32)
            return sample

        sample_spec: dict[str, jax.ShapeDtypeStruct] = {
            "image": jax.ShapeDtypeStruct(shape=image_shape, dtype=jnp.uint8),
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
