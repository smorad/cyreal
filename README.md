# Cyreal - Another JAX DataLoader 

> `grain` for the corporations, `cyreal` for the people

Pure `jax` utilities for iterating over finite and infinite datasets without ever touching `torch` or `tensorflow`. Dataloaders are fast and support `jax.jit`, `jax.grad`, `jax.lax.scan`, and other function transformations.

## Installation

`pip install cyreal`

The only dependency is `jax`.


## Quick Start
Write fast dataloaders without `torch` or `tensorflow`

```python
import jax
import jax.numpy as jnp

from cyreal.transforms import BatchTransform
from cyreal.loader import DataLoader
from cyreal.sources import ArraySource
from cyreal.datasets import MNISTDataset

train_data = MNISTDataset(split="train").as_array_dict()
pipeline = [
  # Load dataset into memory-backed array
  ArraySource(train_data, ordering="shuffle"),
  # Batch it
  BatchTransform(batch_size=128),
]
loader = DataLoader(pipeline)
state = loader.init_state(jax.random.key(0))

for epoch in range(2):
    for _ in range(loader.steps_per_epoch):
        # JIT the loader on the CPU or GPU
        batch, state, mask = jax.jit(loader.next, backend='cpu')(state)
        ... # Train your network
```

Use `scan_epoch` to `jit` and avoid boilerplate

```python
model_state = {"params": jnp.array(0)}

def update(model_state, batch, mask):
    model_state = {"params": model_state['params'] + 1}
    return model_state, None

for epoch in range(2):
    # Runs everything on the default jax device
    state, model_state, _ = loader.scan_epoch(state, model_state, update)
```

## Examples and Documentation

See our [documentation](https://smorad.github.io/cyreal/cyreal.html) for more examples.
- Do you enjoy premature optimization? [Why not `jit` the entire training epoch?](https://smorad.github.io/cyreal/cyreal/tutorials/scan_and_jit.html)
- For the dirty and impure, we support [logging metrics from within a `jit`ted loader.](https://smorad.github.io/cyreal/cyreal/tutorials/host_callback.html)
- Got yourself a huge dataset? [Stream from a disk-backed source.](https://smorad.github.io/cyreal/cyreal/tutorials/disk_stream.html)
- Afraid of finite datasets? We provide [`gymnax`-backed data sources for online reinforcement learning.](https://smorad.github.io/cyreal/cyreal/tutorials/rl_quickstart.html)
- Starving researcher/temporarily embarrassed hyperscaler? We support continual learning via [reservoir sampling and replay buffers.](https://smorad.github.io/cyreal/cyreal/tutorials/buffer_quickstart.html)

We also provide full end to end training examples
- [MNIST](examples/mnist_equinox.py)
- [Time Series](examples/time_series_rnn.py)
- [Reinforcement Learning](examples/cartpole_pg.py)

## Speed Test
You can compare the speed to the `grain` dataloader using [this script](cyreal/examples/speed_test.py). This is how long it takes to iterate though one epoch of MNIST

### MacBook M4 Pro

|Library|Device|Method|Time (s)|
|---|---|---|---|
|`grain`|CPU| Iterator| 1.05
|`cyreal`|CPU| `jit(loader.next)`| 0.04
|`cyreal`|CPU| `scan_epoch`| 0.08


### RTX 5090

|Library|Device|Method|Time (s)|
|---|---|---|---|
|`grain`|GPU| Iterator| 4.04
|`cyreal`|GPU| `jit(loader.next)`| 0.30
|`cyreal`|GPU| `scan_epoch`| 1.44