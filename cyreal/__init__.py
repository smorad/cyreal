"""Jittable dataset utilities for JAX."""
from __future__ import annotations

from .dataset_protocol import DatasetProtocol
from .datasets import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    EMNISTDataset,
    FashionMNISTDataset,
    KMNISTDataset,
    MNISTDataset,
)
from .loader import (
    DataLoader,
    LoaderState,
)
from .sources import ArraySource, DiskSource, GymnaxSource, Source
from .transforms import (
    BatchTransform,
    DevicePutTransform,
    FlattenTransform,
    HostCallbackTransform,
    MapTransform,
    NormalizeImageTransform,
    TimeSeriesBatchTransform,
)

__all__ = [
    "DatasetProtocol",
    "DataLoader",
    "LoaderState",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "EMNISTDataset",
    "FashionMNISTDataset",
    "KMNISTDataset",
    "Kuzushiji49Dataset",
    "MNISTDataset",
    "ArraySource",
    "DiskSource",
    "GymnaxSource",
    "VectorGymnaxSource",
    "Source",
    "BatchTransform",
    "DevicePutTransform",
    "FlattenTransform",
    "HostCallbackTransform",
    "MapTransform",
    "NormalizeImageTransform",
    "TimeSeriesBatchTransform",
]
