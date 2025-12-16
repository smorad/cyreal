"""`HostCallbackTransform` allows you to log metrics and call other impure IO within jit.

```python
import jax
import jax.numpy as jnp
import numpy as np

from cyreal.sources import ArraySource
from cyreal.transforms import BatchTransform, HostCallbackTransform
from cyreal.loader import DataLoader
from cyreal.datasets import MNISTDataset

def model(images):
    return jnp.mean(images.astype(jnp.float32), axis=(1, 2, 3))

def cross_entropy(logits, labels):
    labels = labels.astype(jnp.float32)
    return (logits - labels) ** 2

def log_loss(batch, mask):
    logits = model(batch["image"])
    loss = jnp.mean(cross_entropy(logits, batch["label"]) * mask[:, None])
    print("loss:", float(np.asarray(loss)))
    return batch

loader = DataLoader(
    pipeline=[
        ArraySource(MNISTDataset(split="train").as_array_dict(), ordering="shuffle"),
        BatchTransform(batch_size=128),
        HostCallbackTransform(fn=log_loss),
    ],
)
# Still jittable
state = loader.init_state(jax.random.key(0))
sample, mask, state = jax.jit(loader.next)(state)
```
"""
