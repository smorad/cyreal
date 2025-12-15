# Cyreal - Another JAX DataLoader 

> `grain` for the corporations, `cyreal` for the people

Pure `jax` utilities for iterating over finite datasets without ever touching `torch` or `tensorflow`. Dataloaders support `jax.jit`, `jax.grad`, `jax.lax.scan`, and other function transformations.

## Installation

The only dependency is `jax`. On GPU machines, install the
appropriate JAX build for your CUDA version.

## Quick start with MNIST
Write `torch`-style dataloaders without `torch`

```python
import jax
import jax.numpy as jnp

from cyreal import (
  ArraySampleSource,
  BatchTransform,
  DataLoader,
  DevicePutTransform,
  MNISTDataset,
)

train_data = MNISTDataset(split="train").as_array_dict()
pipeline = [
  ArraySampleSource(train_data, ordering="shuffle"),
  BatchTransform(batch_size=128),
  DevicePutTransform(),
]
loader = DataLoader(pipeline=pipeline)
state = loader.init_state(jax.random.PRNGKey(0))

for batch, mask in loader.iterate(state):
  ...  # train your network!
```

## Scan and Avoid Boilerplate 

`DataLoader.scan_epoch` will run a full pass through the dataset into a single
`jax.lax.scan` to minimize dispatch overhead. This will `jit` the `body_fn`.

```python
def body_fn(model_state, batch, mask):
  model_state = update_model(model_state, batch, mask)
  return model_state, None

loader_state, model_state, _ = loader.scan_epoch(loader_state, model_state, body_fn)
```

## JIT Capabilities
Do you enjoy premature optimization? Why not `jit` the entire train epoch?


```python
import jax
import jax.numpy as jnp

from cyreal import (
  ArraySampleSource,
  BatchTransform,
  DataLoader,
  DevicePutTransform,
  MNISTDataset,
)

train_data = MNISTDataset(split="train").as_array_dict()
pipeline = [
  ArraySampleSource(train_data, ordering="shuffle"),
  BatchTransform(batch_size=128),
  DevicePutTransform(),
]
loader = DataLoader(pipeline)
loader_state = loader.init_state(jax.random.PRNGKey(0))
model_state = model_init()

@jax.jit
def train_epoch(model_state, loader_state):
  def body_fn(model_state, batch, mask):
    # Update the network using your train fn
    new_model_state = model_update(model_state, batch, mask)
    return new_model_state, None

  loader_state, model_state, _ = loader.scan_epoch(loader_state, model_state, body_fn)
  return model_state, loader_state

model_state, loader_state = train_epoch(model_state, loader_state)
```


## Streaming from Disk
Is your dataset enormous? Swap in a disk-backed source.

```python
import jax

from cyreal import (
  BatchTransform,
  DataLoader,
  DevicePutTransform,
  MNISTDataset,
)

pipeline = [
  MNISTDataset.make_disk_source(split="train", ordering="shuffle", prefetch_size=1024),
  BatchTransform(batch_size=128),
  DevicePutTransform(),
]

loader = DataLoader(pipeline=pipeline)
state = loader.init_state(jax.random.PRNGKey(0))

for batch, mask in loader.iterate(state):
  ...  # stream without holding the dataset in RAM
```

## For the Dirty and Impure
Want to `jit` but also log some metrics? Use `HostCallbackTransform` which utilizes `jax.experimental.io_callback` under the hood.

```python
import jax.numpy as jnp
import numpy as np

from cyreal import (
  ArraySampleSource,
  BatchTransform,
  DataLoader,
  HostCallbackTransform,
  MNISTDataset,
)

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
    ArraySampleSource(MNISTDataset(split="train").as_array_dict(), ordering="shuffle"),
    BatchTransform(batch_size=128),
    HostCallbackTransform(fn=log_loss),
  ],
)
```