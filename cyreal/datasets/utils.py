"""Filesystem helpers shared across dataset modules."""
from __future__ import annotations

import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


def download_with_progress(url: str, path: Path) -> Path:
    """Download ``url`` to ``path`` while printing percent and window speed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_percent = [-1]
    last_bytes = [None]
    last_time = [None]

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percent = int((100 * downloaded) / total_size)

        now = time.monotonic()
        if last_bytes[0] is None or last_time[0] is None:
            speed_text = "?? MB/s"
        else:
            delta_bytes = max(downloaded - int(last_bytes[0]), 0)
            delta_time = max(now - float(last_time[0]), 1e-6)
            bytes_per_sec = int(delta_bytes / delta_time)
            if bytes_per_sec >= 1024 * 1024:
                speed_text = f"{bytes_per_sec // (1024 * 1024)} MB/s"
            else:
                speed_text = f"{bytes_per_sec // 1024} KB/s"

        last_bytes[0] = downloaded
        last_time[0] = now

        if percent != last_percent[0]:
            last_percent[0] = percent
            print(f"\rDownloading {path.name}: {percent}% ({speed_text})", end="", flush=True)

    urllib.request.urlretrieve(url, path, reporthook=_reporthook)
    if last_percent[0] >= 0:
        print(flush=True)
    return path


def resolve_cache_dir(cache_dir: str | Path | None, *, default_name: str) -> Path:
    """Return a writable cache directory, creating it if needed."""
    base = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / default_name
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_file(path: Path, url: str) -> Path:
    """Download ``url`` into ``path`` if it doesn't already exist."""
    if path.exists():
        return path
    return download_with_progress(url, path)


def ensure_csv(
    cache_dir: Path,
    filename: str,
    url: str,
    data_path: Optional[str | Path] = None,
) -> Path:
    """Resolve a CSV file via cache download or user-provided path."""
    if data_path is not None:
        return Path(data_path)
    target = cache_dir / filename
    if not target.exists():
        download_with_progress(url, target)
    return target


def download_archive(url: str, path: Path) -> Path:
    """Download an archive to ``path`` if needed."""
    return ensure_file(path, url)


def ensure_tar_extracted(archive: Path, extract_root: Path, target_dir: str) -> Path:
    """Extract a tar.gz archive into ``extract_root`` and return the target dir."""
    target_path = extract_root / target_dir
    if target_path.exists():
        return target_path
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=extract_root)
    return target_path


def to_host_jax_array(array: np.ndarray) -> jax.Array:
    """Copy a NumPy array onto the default CPU device for JAX consumption."""
    cpu_devices = jax.devices("cpu")
    if cpu_devices:
        with jax.default_device(cpu_devices[0]):
            return jnp.asarray(array)
    return jnp.asarray(array)
