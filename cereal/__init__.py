"""Jittable dataset utilities for JAX."""
from __future__ import annotations

from .dataset_protocol import DatasetProtocol
from .datasets import CIFAR10Dataset, MNISTDataset, MNISTDiskSource
from .loader import (
    DataLoader,
    LoaderState,
    dataset_to_jax,
)
from .sources import ArraySampleSource, DiskSampleSource, GymnaxSource, Source
from .transforms import (
    BatchTransform,
    DevicePutTransform,
    FlattenImageTransform,
    HostCallbackTransform,
    MapTransform,
    NormalizeImageTransform,
)

__all__ = [
    "DatasetProtocol",
    "DataLoader",
    "LoaderState",
    "dataset_to_jax",
    "CIFAR10Dataset",
    "MNISTDataset",
    "MNISTDiskSource",
    "ArraySampleSource",
    "DiskSampleSource",
    "GymnaxSource",
    "Source",
    "BatchTransform",
    "DevicePutTransform",
    "FlattenImageTransform",
    "HostCallbackTransform",
    "MapTransform",
    "NormalizeImageTransform",
]
