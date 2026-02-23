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
    array: np.ndarray,
    split: Literal["train", "val", "test"],
    train_fraction: float,
    context_length: int,
    val_fraction: float = 0.0,
) -> np.ndarray:
    """Slice time-aligned arrays into train/val/test, with history overlap for non-train splits.

    For (T,) or (T, D): slices along axis 0
    For (B, T, D): slices along axis 1 (time)
    """
    # Detect time axis: if 3+ dims assume (batch, time, ...)
    time_axis = 1 if array.ndim >= 3 else 0
    n = int(array.shape[time_axis])

    if n <= 0:
        raise ValueError("Array must be non-empty along time axis.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")
    if context_length < 0:
        raise ValueError("context_length must be >= 0.")

    train_end = min(max(int(n * train_fraction), 1), n)

    if val_fraction > 0.0:
        val_end = min(max(int(n * (train_fraction + val_fraction)), train_end + 1), n)
    else:
        val_end = train_end

    if split == "train":
        start, end = 0, train_end
    elif split == "val":
        if val_fraction == 0.0:
            raise ValueError("val_fraction must be > 0 when split='val'.")
        start, end = train_end, val_end
    else:
        start, end = val_end, n

    if split != "train":
        overlap = context_length
        start = max(start - overlap, 0)

    if time_axis == 0:
        return array[start:end]
    return array[:, start:end]


def sliding_window_many(
    array: np.ndarray,
    window_size: int,
    stride: int = 1,
    offset: int = 0,
) -> np.ndarray:
    """Create aligned sliding windows along the time axis.

    For (T,) or (T, D): windows along axis 0
    For (B, T, D): windows along axis 1
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if offset < 0:
        raise ValueError("offset must be >= 0.")

    # Detect time axis: if 3+ dims assume (batch, time, ...)
    time_axis = 1 if array.ndim >= 3 else 0
    t = int(array.shape[time_axis])

    if t <= 0:
        raise ValueError("Array must be non-empty along time axis.")

    total = t - (offset + window_size) + 1
    if total <= 0:
        raise ValueError("Series too short for requested window_size/offset.")

    all_windows = np.lib.stride_tricks.sliding_window_view(
        array,
        window_shape=window_size,
        axis=time_axis,
    )

    # pick starts: offset + [0, stride, 2*stride, ...]
    if time_axis == 0:
        return all_windows[offset : offset + total : stride]
    return all_windows[:, offset : offset + total : stride]


def load_time_series_from_csv(
    *,
    cache_dir: str | None,
    dataset_name: str,
    filename: str,
    url: str,
    data_path: str | None,
    skip_header: int,
    value_column: int,
) -> np.ndarray:
    base_dir = resolve_cache_dir(cache_dir, default_name=f"cyreal_{dataset_name}")
    csv_path = ensure_csv(base_dir, filename, url, data_path)
    values = load_value_column(csv_path, skip_header=skip_header, value_column=value_column)
    return values


def prepare_time_series_windows(
    series: np.ndarray,
    split: Literal["train", "val", "test"],
    context_length: int,
    prediction_length: int,
    train_fraction: float,
    val_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare time windows for a time series array.

    This returns aligned windows built from the (possibly overlap-extended) split slice
    produced by :func:`select_split`.

    Diagram: context/target windowing (forecasting)

        series:  [0 1 2 3 4 5 6 7 8 9 ...]

        context_length = 4
        prediction_length = 3

        for start s = 0:
          context: [0 1 2 3]
          target:          [4 5 6]

        for start s = 1:
          context:   [1 2 3 4]
          target:             [5 6 7]

    Args:
        array: The time series array.
        split: The split to prepare windows for.
        context_length: The length of the context window.
        prediction_length: The length of the prediction window.
        train_fraction: The fraction of the data to use for training.
        val_fraction: The fraction of the data to use for validation.

    Returns:
        A tuple of context and target arrays.
    """
    split_values = select_split(
        series,
        split=split,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        context_length=context_length,
    )
    contexts = sliding_window_many(
        split_values,
        window_size=context_length,
    )
    targets = sliding_window_many(
        split_values,
        window_size=prediction_length,
        offset=context_length,
    )
    # Align counts: `targets` needs `context_length + prediction_length` points, so it
    # is always the limiting factor. `contexts` may contain extra suffix windows that
    # cannot be paired with a future target.
    if contexts.ndim == 2:
        contexts = contexts[: targets.shape[0]]
    else:
        contexts = contexts[:, : targets.shape[1]]
    return contexts, targets


def prepare_seq_to_seq_windows(
    input_sequence: np.ndarray,
    target_sequence: np.ndarray,
    split: Literal["train", "val", "test"],
    input_window_len: int,
    target_window_len: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.0,
    sliding_window_stride: int = 1,
    target_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare aligned context/target windows from two time-aligned sequences.

    This is the multi-array version of :func:`prepare_time_series_windows`:
    it first slices *both* sequences with :func:`select_split` (including overlap for
    non-train splits), then creates forecasting-style windows where each target window
    starts immediately after its corresponding context window.

    Diagram 1: Standard seq-to-seq style

        series:  [0 1 2 3 4 5 6 7 8 9 ...]

        input_window_len = 4
        target_window_len = 4
        target_offset = 0

        for start s = 0:
          context: [0 1 2 3]
          target:  [0 1 2 3]

        for start s = 1:
          context:   [1 2 3 4]
          target:    [1 2 3 4]

    Diagram 2: Dynamics-conditioned forecasting style

        series:  [0 1 2 3 4 5 6 7 8 9 ...]

        input_window_len = 4
        target_window_len = 6
        target_offset = 0

        for start s = 0:
          context: [0 1 2 3]
          target:  [0 1 2 3 4 5]

        for start s = 1:
          context:   [1 2 3 4]
          target:    [1 2 3 4 5 6]

    Diagram 3: Pure forecasting style (uncommon)

        series:  [0 1 2 3 4 5 6 7 8 9 ...]

        input_window_len = 4
        target_window_len = 4
        target_offset = input_window_len

        for start s = 0:
          context: [0 1 2 3]
          target:          [4 5 6 7]

        for start s = 1:
          context:   [1 2 3 4]
          target:            [5 6 7 8]

    Args:
        input_sequence: The time-aligned input sequence (e.g. features).
        target_sequence: The time-aligned target sequence (e.g. labels).
        split: The split to prepare windows for.
        input_window_len: The length of the input window.
        target_window_len: The length of the target window.
        train_fraction: The fraction of the data to use for training.
        val_fraction: The fraction of the data to use for validation.

    Returns:
        A tuple of context and target arrays.
    """
    if input_sequence.ndim == 0 or target_sequence.ndim == 0:
        raise ValueError("All sequences must have at least 1 dimension.")
    if input_sequence.ndim == 1 and target_sequence.ndim == 1:
        if int(input_sequence.shape[0]) != int(target_sequence.shape[0]):
            raise ValueError("All sequences must have the same length.")
    else:
        if int(input_sequence.shape[1]) != int(target_sequence.shape[1]):
            raise ValueError("All sequences must have the same length along axis 1 (time).")
        if int(input_sequence.shape[0]) != int(target_sequence.shape[0]):
            raise ValueError("All sequences must have the same batch size along axis 0.")
    split_inputs = select_split(
        input_sequence,
        split=split,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        context_length=input_window_len,
    )
    split_targets = select_split(
        target_sequence,
        split=split,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        context_length=input_window_len,
    )

    contexts = sliding_window_many(
        split_inputs,
        window_size=input_window_len,
        stride=sliding_window_stride,
    )
    targets = sliding_window_many(
        split_targets,
        window_size=target_window_len,
        stride=sliding_window_stride,
        offset=target_offset,
    )

    # Align the number of windows along axis 1.
    contexts = contexts[:, : targets.shape[1]]

    # For batched (B, T, D) inputs, reshape to (B*N, W, D) for training
    if input_sequence.ndim >= 3:
        # Currently: (B, N, D, W) → move window to axis 1 → (B, W, N, D) → wait no
        # Currently: (B, N, D, W)
        # Need: (B*N, W, D)
        # Step 1: moveaxis -1 to 2: (B, N, W, D)
        # Step 2: reshape to (B*N, W, D)
        contexts = np.moveaxis(contexts, -1, 2)
        targets = np.moveaxis(targets, -1, 2)

    return contexts, targets


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

    if contexts_np.ndim < 2:
        raise ValueError("contexts must have shape (N, context_len, ...).")
    if targets_np.ndim < 2:
        raise ValueError("targets must have shape (N, target_len, ...).")
    context_sample_shape = tuple(int(x) for x in contexts_np.shape[1:])
    target_sample_shape = tuple(int(x) for x in targets_np.shape[1:])

    def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
        idx = int(np.asarray(index))
        return {
            "context": np.asarray(contexts_np[idx], dtype=np.float32),
            "target": np.asarray(targets_np[idx], dtype=np.float32),
        }

    sample_spec = {
        "context": jax.ShapeDtypeStruct(shape=context_sample_shape, dtype=jnp.float32),
        "target": jax.ShapeDtypeStruct(shape=target_sample_shape, dtype=jnp.float32),
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
    "sliding_window_many",
    "load_time_series_from_csv",
    "prepare_time_series_windows",
    "prepare_seq_to_seq_windows",
    "make_sequence_disk_source",
]
