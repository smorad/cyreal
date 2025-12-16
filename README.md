# Cyreal - Another JAX DataLoader 

> `grain` for the corporations, `cyreal` for the people

Pure `jax` utilities for iterating over finite and infinite datasets without ever touching `torch` or `tensorflow`. Dataloaders are fast and support `jax.jit`, `jax.grad`, `jax.lax.scan`, and other function transformations.

## Installation

`pip install cyreal`

The only dependency is `jax`.


## Quick Start
Write fast `torch`-style dataloaders without `torch`

```python
import jax
import jax.numpy as jnp

from cyreal.transforms import BatchTransform, DevicePutTransform
from cyreal.loader import DataLoader
from cyreal.rl import set_loader_policy_state, set_source_policy_state
from cyreal.sources import ArraySource
from cyreal.datasets import MNISTDataset

train_data = MNISTDataset(split="train").as_array_dict()
pipeline = [
  # Load dataset into memory-backed array
  ArraySource(train_data, ordering="shuffle"),
  # Batch it
  BatchTransform(batch_size=128),
  # Move the batch to the GPU
  DevicePutTransform(),
]
loader = DataLoader(pipeline)
state = loader.init_state(jax.random.key(0))

for epoch in range(2):
    for batch, mask in loader.iterate(state):
        ...  # train your network!
```

Go ahead and `jit` the loader, it is stateless

```python
for epoch in range(2):
    for _ in range(loader.steps_per_epoch):
        batch, state, mask = jax.jit(loader.next)(state)
        ... # Train your network
```

For maximum throughput, `jit` an entire training epoch, fusing together loads of operations

```python
model_state = {"params": jnp.array(0)}

def update(model_state, batch, mask):
    model_state = {"params": model_state['params'] + 1}
    return model_state, None

for epoch in range(2):
    state, model_state, _ = loader.scan_epoch(state, model_state, update)
```

## Examples and Documentation

See our [documentation](https://smorad.github.io/cyreal/cyreal.html) for more examples.
- Do you enjoy premature optimization? [Why not `jit` the entire training epoch?](https://smorad.github.io/cyreal/cyreal/tutorials/scan_and_jit.html)
- For the dirty and impure, we support [logging metrics from within a `jit`ted loader.](https://smorad.github.io/cyreal/cyreal/tutorials/host_callback.html)
- Got yourself a huge dataset? [Stream from a disk-backed source.](https://smorad.github.io/cyreal/cyreal/tutorials/disk_stream.html)
- Afraid of finite datasets? We provide [`gymnax`-backed data sources for online reinforcement learning.](https://smorad.github.io/cyreal/cyreal/tutorials/rl_quickstart.html)
- Are you a starving researcher/temporarily embarrassed hyperscaler? We support continual learning via [reservoir sampling and replay buffers.](https://smorad.github.io/cyreal/cyreal/tutorials/buffer_quickstart.html)

We also provide full end to end training examples
- [MNIST](examples/mnist_equinox.py)
- [Time Series](examples/time_series_rnn.py)
- [Reinforcement Learning](examples/cartpole_pg.py)