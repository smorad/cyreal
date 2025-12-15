"""Filesystem helpers shared across dataset modules."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def resolve_cache_dir(cache_dir: str | Path | None, *, default_name: str) -> Path:
    """Return a writable cache directory, creating it if needed."""
    base = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / default_name
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_file(path: Path, url: str) -> Path:
    """Download ``url`` into ``path`` if it doesn't already exist."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def to_host_jax_array(array: np.ndarray) -> jax.Array:
    """Copy a NumPy array onto the default CPU device for JAX consumption."""
    cpu_devices = jax.devices("cpu")
    if cpu_devices:
        with jax.default_device(cpu_devices[0]):
            return jnp.asarray(array)
    return jnp.asarray(array)
