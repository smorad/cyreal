"""Daily minimum temperature dataset built from CSV windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax

from .dataset_protocol import DatasetProtocol
from ..sources import DiskSource
from .time_utils import load_time_series_from_csv, prepare_time_series_windows, make_sequence_disk_source
from .utils import to_host_jax_array as _to_host_jax_array

DAILY_MIN_TEMPS_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"


@dataclass
class DailyMinTemperaturesDataset(DatasetProtocol):
    """A time series regression dataset of daily minimum temperatures from the Bureau of Meteorology."""

    split: Literal["train", "val", "test"] = "train"
    overlapping: bool = False
    context_length: int = 30
    prediction_length: int = 1
    train_fraction: float = 0.8
    val_fraction: float = 0.0
    cache_dir: str | None = None
    data_path: str | None = None

    def __post_init__(self) -> None:
        values = load_time_series_from_csv(
            cache_dir=self.cache_dir,
            dataset_name="daily_min_temperatures",
            filename="daily-min-temperatures.csv",
            url=DAILY_MIN_TEMPS_URL,
            data_path=self.data_path,
            skip_header=1,
            value_column=1,
        )
        contexts, targets = prepare_time_series_windows(
            series=values,
            split=self.split,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            train_fraction=self.train_fraction,
            val_fraction=self.val_fraction,
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
        """Expose the full dataset as a PyTree of JAX arrays."""
        return {"context": self._contexts, "target": self._targets}

    @classmethod
    def make_disk_source(
        cls,
        *,
        split: Literal["train", "val", "test"] = "train",
        context_length: int = 30,
        prediction_length: int = 1,
        train_fraction: float = 0.8,
        val_fraction: float = 0.0,
        cache_dir: str | None = None,
        data_path: str | None = None,
        ordering: Literal["sequential", "shuffle"] = "shuffle",
        prefetch_size: int = 64,
    ) -> DiskSource:
        """Return the dataset in a disk streaming format."""

        values = load_time_series_from_csv(
            cache_dir=cache_dir,
            dataset_name="daily_min_temperatures",
            filename="daily-min-temperatures.csv",
            url=DAILY_MIN_TEMPS_URL,
            data_path=data_path,
            skip_header=1,
            value_column=1,
        )
        contexts, targets = prepare_time_series_windows(
            series=values,
            split=split,
            context_length=context_length,
            prediction_length=prediction_length,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        return make_sequence_disk_source(
            contexts=contexts,
            targets=targets,
            ordering=ordering,
            prefetch_size=prefetch_size,
        )
