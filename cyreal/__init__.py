"""Jittable dataset utilities for JAX.

We use a producer and transform model. Sources stream data from datasets and Transforms transform the data stream. The DataLoader class composes sources and transforms into jittable data pipelines.


- `cyreal.datasets` contains datasets such as MNIST, CIFAR-10, etc and associated utilities.
- `cyreal.loader` contains the `DataLoader` class for building jittable data pipelines.
- `cyreal.sources` contains data sources such as `ArraySource` and `GymnaxSource`.
- `cyreal.transforms` contains data transforms such as `BatchTransform` and `DevicePutTransform`.
- `cyreal.rl` contains some RL-specific utilities for the DataLoader.
"""

# from .datasets import (
#     CIFAR10Dataset,
#     CIFAR100Dataset,
#     EMNISTDataset,
#     FashionMNISTDataset,
#     KMNISTDataset,
#     MNISTDataset,
# )
# from .loader import (
#     DataLoader,
# )
# from .transforms import (
#     BatchTransform,
#     DevicePutTransform,
#     FlattenTransform,
#     HostCallbackTransform,
#     MapTransform,
#     NormalizeImageTransform,
#     BufferTransform,
#     TimeSeriesBatchTransform,
# )
