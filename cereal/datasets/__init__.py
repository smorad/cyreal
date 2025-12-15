"""Dataset helpers bundled with cereal."""
from __future__ import annotations

from .cifar10 import CIFAR10Dataset
from .mnist import MNISTDataset, MNISTDiskSource, MNIST_URLS, _ensure_file

__all__ = [
    "CIFAR10Dataset",
    "MNISTDataset",
    "MNISTDiskSource",
    "MNIST_URLS",
    "_ensure_file",
]
