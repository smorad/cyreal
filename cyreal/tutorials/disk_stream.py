"""Streaming from disk using `DiskSource`. This is much slower than an in-memory dataset
but it allows you to work with datasets that do not fit into RAM. The key is to call
`make_disk_source` on your dataset class to get a disk-backed source.

```python
import jax

from cyreal.transforms import BatchTransform
from cyreal.loader import DataLoader
from cyreal.datasets import MNISTDataset

pipeline = [
    # Prefetch 1024 examples for each disk read
    MNISTDataset.make_disk_source(split="train", ordering="shuffle", prefetch_size=1024),
    BatchTransform(batch_size=128),
]

loader = DataLoader(pipeline=pipeline)
state = loader.init_state(jax.random.PRNGKey(0))

for batch, mask in loader.iterate(state):
    ...  # stream without holding the dataset in RAM
```
"""
