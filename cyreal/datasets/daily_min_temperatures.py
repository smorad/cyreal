"""Daily minimum temperature dataset built from CSV windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax

from ..dataset_protocol import DatasetProtocol
from ..sources import DiskSource
from .time_utils import make_sequence_disk_source, prepare_time_windows
from .utils import to_host_jax_array as _to_host_jax_array

DAILY_MIN_TEMPS_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
)


@dataclass
class DailyMinTemperaturesDataset(DatasetProtocol):
    """Sliding-window dataset built from Bureau of Meteorology temperatures."""

    split: Literal["train", "test"] = "train"
    context_length: int = 30
    prediction_length: int = 1
    train_fraction: float = 0.8
    cache_dir: str | None = None
    data_path: str | None = None

    def __post_init__(self) -> None:
        contexts, targets = prepare_time_windows(
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
        cache_dir: str | None = None,
        data_path: str | None = None,
        ordering: Literal["sequential", "shuffle"] = "shuffle",
        prefetch_size: int = 64,
    ) -> DiskSource:
        contexts, targets = prepare_time_windows(
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
        return make_sequence_disk_source(
            contexts=contexts,
            targets=targets,
            context_length=context_length,
            prediction_length=prediction_length,
            ordering=ordering,
            prefetch_size=prefetch_size,
        )
