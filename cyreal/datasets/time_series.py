"""Time-series datasets built on simple CSV downloads."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from ..dataset_protocol import DatasetProtocol
from ..sources import DiskSampleSource
from .fs_utils import resolve_cache_dir, to_host_jax_array as _to_host_jax_array

DAILY_MIN_TEMPS_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
)
SUNSPOTS_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv"
)
def _ensure_csv(
    cache_dir: Path,
    filename: str,
    url: str,
    data_path: str | Path | None,
) -> Path:
    if data_path is not None:
        return Path(data_path)
    target = cache_dir / filename
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    return target


def _load_value_column(path: Path, *, skip_header: int, value_column: int) -> np.ndarray:
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


def _select_split(
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


def _window_series(
    series: np.ndarray,
    *,
    context_length: int,
    prediction_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if context_length <= 0 or prediction_length <= 0:
        raise ValueError("context_length and prediction_length must be positive.")
    total = len(series) - (context_length + prediction_length) + 1
    if total <= 0:
        raise ValueError(
            "Series too short for requested window configuration (context + prediction)."
        )
    contexts = []
    targets = []
    for i in range(total):
        ctx = series[i : i + context_length]
        tgt = series[i + context_length : i + context_length + prediction_length]
        contexts.append(ctx)
        targets.append(tgt)
    return np.stack(contexts, axis=0), np.stack(targets, axis=0)


def _prepare_windows(
    *,
    dataset_name: str,
    filename: str,
    url: str,
    value_column: int,
    skip_header: int,
    split: Literal["train", "test"],
    context_length: int,
    prediction_length: int,
    train_fraction: float,
    cache_dir: str | Path | None,
    data_path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    base_dir = resolve_cache_dir(cache_dir, default_name=f"cyreal_{dataset_name}")
    csv_path = _ensure_csv(base_dir, filename, url, data_path)
    values = _load_value_column(csv_path, skip_header=skip_header, value_column=value_column)
    split_values = _select_split(
        values,
        split=split,
        train_fraction=train_fraction,
        context_length=context_length,
    )
    contexts, targets = _window_series(
        split_values,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    return contexts.astype(np.float32), targets.astype(np.float32)


def _make_sequence_disk_source(
    *,
    contexts: np.ndarray,
    targets: np.ndarray,
    context_length: int,
    prediction_length: int,
    ordering: Literal["sequential", "shuffle"],
    prefetch_size: int,
) -> DiskSampleSource:
    contexts_np = np.array(contexts, copy=True)
    targets_np = np.array(targets, copy=True)

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

    return DiskSampleSource(
        length=int(contexts_np.shape[0]),
        sample_fn=_read_sample,
        sample_spec=sample_spec,
        ordering=ordering,
        prefetch_size=prefetch_size,
    )
@dataclass
class DailyMinTemperaturesDataset(DatasetProtocol):
    """Sliding-window dataset built from Bureau of Meteorology temperatures."""

    split: Literal["train", "test"] = "train"
    context_length: int = 30
    prediction_length: int = 1
    train_fraction: float = 0.8
    cache_dir: str | Path | None = None
    data_path: str | Path | None = None

    def __post_init__(self) -> None:
        contexts, targets = _prepare_windows(
            dataset_name="daily_min_temperatures",
            filename="daily-min-temperatures.csv",
            url=DAILY_MIN_TEMPS_URL,
            skip_header=1,
            value_column=1,
            split=self.split,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            train_fraction=self.train_fraction,
            cache_dir=self.cache_dir,
            data_path=self.data_path,
        )
        self._contexts = _to_host_jax_array(contexts)
        self._targets = _to_host_jax_array(targets)

    def __len__(self) -> int:
        return int(self._contexts.shape[0])

    def __getitem__(self, index: int):
        return {
            "context": self._contexts[index],
            "target": self._targets[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        return {"context": self._contexts, "target": self._targets}

    @classmethod
    def make_disk_source(
        cls,
        *,
        split: Literal["train", "test"] = "train",
        context_length: int = 30,
        prediction_length: int = 1,
        train_fraction: float = 0.8,
        cache_dir: str | Path | None = None,
        data_path: str | Path | None = None,
        ordering: Literal["sequential", "shuffle"] = "shuffle",
        prefetch_size: int = 64,
    ) -> DiskSampleSource:
        contexts, targets = _prepare_windows(
            dataset_name="daily_min_temperatures",
            filename="daily-min-temperatures.csv",
            url=DAILY_MIN_TEMPS_URL,
            skip_header=1,
            value_column=1,
            split=split,
            context_length=context_length,
            prediction_length=prediction_length,
            train_fraction=train_fraction,
            cache_dir=cache_dir,
            data_path=data_path,
        )
        return _make_sequence_disk_source(
            contexts=contexts,
            targets=targets,
            context_length=context_length,
            prediction_length=prediction_length,
            ordering=ordering,
            prefetch_size=prefetch_size,
        )


@dataclass
class SunspotsDataset(DatasetProtocol):
    """Monthly mean sunspot counts from SILSO."""

    split: Literal["train", "test"] = "train"
    context_length: int = 24
    prediction_length: int = 1
    train_fraction: float = 0.8
    cache_dir: str | Path | None = None
    data_path: str | Path | None = None

    def __post_init__(self) -> None:
        contexts, targets = _prepare_windows(
            dataset_name="sunspots",
            filename="monthly-sunspots.csv",
            url=SUNSPOTS_URL,
            skip_header=1,
            value_column=1,
            split=self.split,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            train_fraction=self.train_fraction,
            cache_dir=self.cache_dir,
            data_path=self.data_path,
        )
        self._contexts = _to_host_jax_array(contexts)
        self._targets = _to_host_jax_array(targets)

    def __len__(self) -> int:
        return int(self._contexts.shape[0])

    def __getitem__(self, index: int):
        return {
            "context": self._contexts[index],
            "target": self._targets[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        return {"context": self._contexts, "target": self._targets}

    @classmethod
    def make_disk_source(
        cls,
        *,
        split: Literal["train", "test"] = "train",
        context_length: int = 24,
        prediction_length: int = 1,
        train_fraction: float = 0.8,
        cache_dir: str | Path | None = None,
        data_path: str | Path | None = None,
        ordering: Literal["sequential", "shuffle"] = "shuffle",
        prefetch_size: int = 64,
    ) -> DiskSampleSource:
        contexts, targets = _prepare_windows(
            dataset_name="sunspots",
            filename="monthly-sunspots.csv",
            url=SUNSPOTS_URL,
            skip_header=1,
            value_column=1,
            split=split,
            context_length=context_length,
            prediction_length=prediction_length,
            train_fraction=train_fraction,
            cache_dir=cache_dir,
            data_path=data_path,
        )
        return _make_sequence_disk_source(
            contexts=contexts,
            targets=targets,
            context_length=context_length,
            prediction_length=prediction_length,
            ordering=ordering,
            prefetch_size=prefetch_size,
        )
