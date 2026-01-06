"""Shared helpers for time-series datasets."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..sources import DiskSource
from .utils import ensure_csv, resolve_cache_dir


def load_value_column(path, *, skip_header: int, value_column: int) -> np.ndarray:
    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=skip_header,
        usecols=[value_column],
        dtype=np.float32,
    )
    if data.ndim == 1:
        values = data
    else:
        values = data[:, 0]
    if np.isnan(values).any():
        raise ValueError(f"Series at {path} contains NaNs.")
    return values


def select_split(
    values: np.ndarray,
    *,
    split: Literal["train", "test"],
    train_fraction: float,
    context_length: int,
) -> np.ndarray:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    train_len = max(int(len(values) * train_fraction), 1)
    train_len = min(train_len, len(values))
    if split == "train":
        return values[:train_len]
    overlap = max(context_length, 1)
    start = max(train_len - overlap, 0)
    return values[start:]


def sliding_window_series(
    series: np.ndarray,
    *,
    overlapping: bool,
    context_length: int,
    prediction_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare sliding windows from a time series.

    Args:
        series: The time series to prepare windows from.
        overlapping: Whether the context and target windows should overlap. Useful for training neural CDE models.
        context_length: The length of the context window.
        prediction_length: The length of the prediction window.
    """
    if context_length <= 0 or prediction_length <= 0:
        raise ValueError("context_length and prediction_length must be positive.")
    total = len(series) - (context_length + prediction_length) + 1
    if total <= 0:
        raise ValueError(
            "Series too short for requested window configuration (context + prediction)."
        )
    contexts = []
    targets = []
    if overlapping:
        for i in range(total):
            ctx = series[i : i + context_length]
            tgt = series[i : i + context_length + prediction_length]
            contexts.append(ctx)
            targets.append(tgt)
    else:
        for i in range(total):
            ctx = series[i : i + context_length]
            tgt = series[i + context_length : i + context_length + prediction_length]
            contexts.append(ctx)
            targets.append(tgt)

    return np.stack(contexts, axis=0), np.stack(targets, axis=0)


def prepare_time_windows(
    *,
    dataset_name: str,
    filename: str,
    url: str,
    value_column: int,
    skip_header: int,
    split: Literal["train", "test"],
    overlapping: bool,
    context_length: int,
    prediction_length: int,
    train_fraction: float,
    cache_dir: str | None,
    data_path: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    base_dir = resolve_cache_dir(cache_dir, default_name=f"cyreal_{dataset_name}")
    csv_path = ensure_csv(base_dir, filename, url, data_path)
    values = load_value_column(
        csv_path, skip_header=skip_header, value_column=value_column
    )
    split_values = select_split(
        values,
        split=split,
        train_fraction=train_fraction,
        context_length=context_length,
    )
    contexts, targets = sliding_window_series(
        split_values,
        overlapping=overlapping,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    return contexts.astype(np.float32), targets.astype(np.float32)


def make_sequence_disk_source(
    *,
    contexts: np.ndarray,
    targets: np.ndarray,
    ordering: Literal["sequential", "shuffle"],
    prefetch_size: int,
) -> DiskSource:
    """Create a DiskSource for a time series dataset.

    Internally resolves the context and target lengths from the input arrays.

    Args:
        contexts: The context windows.
        targets: The target windows.
        ordering: The ordering of the samples.
        prefetch_size: The number of samples to prefetch.

    Returns:
        A DiskSource for the time series dataset.
    """
    contexts_np = np.array(contexts, copy=True)
    targets_np = np.array(targets, copy=True)

    context_length = int(contexts_np.shape[1])
    prediction_length = int(targets_np.shape[1])

    def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
        idx = int(np.asarray(index))
        return {
            "context": np.asarray(contexts_np[idx], dtype=np.float32),
            "target": np.asarray(targets_np[idx], dtype=np.float32),
        }

    sample_spec = {
        "context": jax.ShapeDtypeStruct(shape=(context_length,), dtype=jnp.float32),
        "target": jax.ShapeDtypeStruct(shape=(prediction_length,), dtype=jnp.float32),
    }

    return DiskSource(
        length=int(contexts_np.shape[0]),
        sample_fn=_read_sample,
        sample_spec=sample_spec,
        ordering=ordering,
        prefetch_size=prefetch_size,
    )


__all__ = [
    "load_value_column",
    "select_split",
    "sliding_window_series",
    "prepare_time_windows",
    "make_sequence_disk_source",
]
