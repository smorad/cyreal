"""For those premature optimizers, you can `jit` your entire training epoch.

```python
import jax
import jax.numpy as jnp

from cyreal.sources import ArraySource
from cyreal.transforms import BatchTransform, DevicePutTransform
from cyreal.loader import DataLoader
from cyreal.datasets import MNISTDataset

train_data = MNISTDataset(split="test").as_array_dict()
pipeline = [
    ArraySource(train_data, ordering="shuffle"),
    BatchTransform(batch_size=128),
    DevicePutTransform(),
]
loader = DataLoader(pipeline)
loader_state = loader.init_state(jax.random.key(0))

model_state = {"params": jnp.array(0)}

@jax.jit
def train_epoch(model_state, loader_state):
    def model_update(model_state, batch, mask):
        # Update the network using your train fn
        new_model_state = {"params": model_state['params'] + 1}
        return new_model_state, None

    # scan_epoch is a helper method, but the loader itself is fully JIT-compatible
    # in case you want to roll your own training loop.
    loader_state, model_state, _ = loader.scan_epoch(loader_state, model_state, model_update)
    return model_state, loader_state

model_state, loader_state = train_epoch(model_state, loader_state)
```
"""
