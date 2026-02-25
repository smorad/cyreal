
"""Write `torch`-style dataloaders without `torch`. Here is an in-memory MNIST example.

```python
import jax
import jax.numpy as jnp

from cyreal.sources import ArraySource
from cyreal.transforms import BatchTransform

from cyreal.loader import DataLoader
from cyreal.datasets import MNISTDataset

# Start with a dataset
train_data = MNISTDataset(split="train").as_array_dict()
pipeline = [
  # Construct an in-memory array from the dataset
  ArraySource(train_data, ordering="shuffle"),
  # Batch the data
  BatchTransform(batch_size=128),
]
# Construct the dataloader
loader = DataLoader(pipeline=pipeline)
# Initialize the loader state
state = loader.init_state(jax.random.key(0))
# Iterate over the data for one epoch
for _ in range(loader.steps_per_epoch):
  batch, state, mask = jax.jit(loader.next)(state)
  ...  # train your network!
```
"""