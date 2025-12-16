"""Example using `BufferTransform` for reservoir replay or FIFO buffering.

```python
import jax
import jax.numpy as jnp

from cyreal.sources import ArraySource
from cyreal.transforms import BufferTransform, BatchTransform, DevicePutTransform
from cyreal.loader import DataLoader
from cyreal.datasets import MNISTDataset

train_data = MNISTDataset(split="train").as_array_dict()
pipeline = [
    ArraySource(train_data, ordering="shuffle"),
    # We have a lot of options for the BufferTransform
    # You can use it for either reservoir sampling or FIFO buffering
    # Prefill determines how many samples to wait before yielding batches
    BufferTransform(capacity=128, prefill=16, sample_size=16, mode="shuffled", write_mode="reservoir"),
    # BufferTransform yields 16 samples, and we can perform additional subsampling with
    # BatchTransform if necessary
    BatchTransform(batch_size=8),
    DevicePutTransform(),
]
loader = DataLoader(pipeline)
loader_state = loader.init_state(jax.random.key(0))
sample, mask, loader_state = jax.jit(loader.next)(loader_state)
```
"""
