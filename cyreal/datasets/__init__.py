"""Dataset helpers bundled with cyreal."""
from __future__ import annotations

from .celeba import CelebADataset
from .cifar10 import CIFAR10Dataset
from .cifar100 import CIFAR100Dataset
from .daily_min_temperatures import DailyMinTemperaturesDataset
from .emnist import EMNISTDataset, EMNIST_URLS
from .fashion_mnist import FashionMNISTDataset, FASHION_MNIST_URLS
from .kmnist import KMNISTDataset, KMNIST_URLS
from .mnist import MNISTDataset, MNIST_URLS
from .sunspots import SunspotsDataset

__all__ = [
    "CelebADataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "DailyMinTemperaturesDataset",
    "EMNISTDataset",
    "EMNIST_URLS",
    "FashionMNISTDataset",
    "FASHION_MNIST_URLS",
    "KMNISTDataset",
    "KMNIST_URLS",
    "MNISTDataset",
    "MNIST_URLS",
    "SunspotsDataset",
]
