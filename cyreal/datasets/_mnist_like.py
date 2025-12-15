"""Shared helpers for IDX-formatted vision datasets."""
from __future__ import annotations

import gzip
import shutil
import struct
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def to_host_jax_array(array: np.ndarray) -> jax.Array:
    cpu_devices = jax.devices("cpu")
    if cpu_devices:
        with jax.default_device(cpu_devices[0]):
            return jnp.asarray(array)
    return jnp.asarray(array)


def ensure_file(path: Path, url: str) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        magic, num, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Unexpected image file magic number: {magic}.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        magic, num = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Unexpected label file magic number: {magic}.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num)


def ensure_uncompressed_idx(path: Path) -> Path:
    target = path.with_suffix("")
    if target.exists():
        return target
    with gzip.open(path, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def read_image_header(path: Path) -> tuple[int, int, int]:
    with open(path, "rb") as f:
        header = f.read(16)
        magic, num, rows, cols = struct.unpack(">IIII", header)
    if magic != 2051:
        raise ValueError(f"Unexpected image file magic number: {magic}.")
    return int(num), int(rows), int(cols)


def read_label_header(path: Path) -> int:
    with open(path, "rb") as f:
        header = f.read(8)
        magic, num = struct.unpack(">II", header)
    if magic != 2049:
        raise ValueError(f"Unexpected label file magic number: {magic}.")
    return int(num)
