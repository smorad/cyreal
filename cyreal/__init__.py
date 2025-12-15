"""Jittable dataset utilities for JAX."""
from __future__ import annotations

from .dataset_protocol import DatasetProtocol
from .datasets import CIFAR10Dataset, CIFAR10DiskSource, MNISTDataset, MNISTDiskSource
from .loader import (
    DataLoader,
    LoaderState,
)
from .sources import ArraySampleSource, DiskSampleSource, GymnaxSource, Source
from .transforms import (
    BatchTransform,
    DevicePutTransform,
    FlattenTransform,
    HostCallbackTransform,
    MapTransform,
    NormalizeImageTransform,
)

__all__ = [
    "DatasetProtocol",
    "DataLoader",
    "LoaderState",
    "CIFAR10Dataset",
    "CIFAR10DiskSource",
    "MNISTDataset",
    "MNISTDiskSource",
    "ArraySampleSource",
    "DiskSampleSource",
    "GymnaxSource",
    "Source",
    "BatchTransform",
    "DevicePutTransform",
    "FlattenTransform",
    "HostCallbackTransform",
    "MapTransform",
    "NormalizeImageTransform",
]
