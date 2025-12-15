"""Dataset helpers bundled with cyreal."""
from __future__ import annotations

from .cifar10 import CIFAR10Dataset
from .cifar100 import CIFAR100Dataset
from .emnist import EMNISTDataset, EMNIST_URLS
from .fashion_mnist import FashionMNISTDataset, FASHION_MNIST_URLS
from .kmnist import KMNISTDataset, KMNIST_URLS
from .kuzushiji49 import Kuzushiji49Dataset, K49_URLS
from .mnist import MNISTDataset, MNIST_URLS

__all__ = [
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "EMNISTDataset",
    "EMNIST_URLS",
    "FashionMNISTDataset",
    "FASHION_MNIST_URLS",
    "KMNISTDataset",
    "KMNIST_URLS",
    "Kuzushiji49Dataset",
    "K49_URLS",
    "MNISTDataset",
    "MNIST_URLS",
]
