"""Simple dataset protocol used by the jittable dataloader."""
from __future__ import annotations

from typing import Any, Protocol


class DatasetProtocol(Protocol):
    """Minimal interface for indexable, length-known datasets."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Any:
        ...
