# Jittable DataLoader for JAX

Pure-JAX utilities for iterating over finite datasets without ever touching
PyTorch. The dataset is materialized into host (CPU) memory once, then all
subsequent batching is performed with JAX index math so it can participate in
`jax.jit`, `jax.grad`, or `jax.lax.scan`.

## Installation

Dependencies are `jax`, `jaxlib`, and `numpy`. On GPU machines, install the
appropriate JAX build for your CUDA version.

## Quick start with MNIST

```python
import jax
import jax.numpy as jnp

from cereal import (
  ArraySampleSource,
  BatchTransform,
  DataLoader,
  DevicePutTransform,
  MNISTDataset,
  dataset_to_jax,
)

dataset = MNISTDataset(split="train")
train_arrays = dataset_to_jax(dataset, storage_device="cpu")

pipeline = [
  ArraySampleSource(train_arrays, ordering="shuffle"),
  BatchTransform(batch_size=128, pad_last_batch=True),
  DevicePutTransform(),
]
loader = DataLoader(pipeline=pipeline)
loader_state = loader.init_state(jax.random.PRNGKey(0))

def step(train_state, loader_state):
    batch, loader_state, mask = loader.next_batch(loader_state)
    images, labels = batch
    logits = train_state.apply_fn(train_state.params, images)
    loss = jnp.mean(cross_entropy(logits, labels) * mask[:, None])
    grads = jax.grad(lambda params: jnp.mean(loss))(train_state.params)
    new_state = train_state.apply_gradients(grads=grads)
    return new_state, loader_state

train_state, loader_state = jax.jit(step)(train_state, loader_state)

# To move batches onto a GPU, wrap your source with `DevicePutTransform` (see below).
```

## Features

- Grain-style **sources** describe how to iterate over the underlying dataset.
  The default `ArraySampleSource` handles sequential or shuffled epochs, while
  padding/drop-last semantics now live in `BatchTransform` so batch shapes stay static.
- Composable **transforms** wrap sources to build a pipeline: `BatchTransform`
  accumulates items into batches, `DevicePutTransform` moves them onto accelerators,
  and `MapTransform` lets you inject arbitrary post-processing.
- `dataset_to_jax(dataset, storage_device="cpu")` materializes samples inside a
  `with jax.default_device(jax.devices("cpu")[0])` context so the full dataset
  resides on host memory even when GPUs are present.
- `DataLoader` simply wires a source + transforms combo and exposes
  `next_batch` and `scan_epoch` for use inside `jax.jit` or `jax.lax.scan`.
  Pass `pipeline=[ArraySampleSource(...), BatchTransform(...), ...]`
  (or a pre-composed `Source`) for full control over the stages.
- Insert `HostCallbackTransform` wherever you need host-side logging or metrics.
  It relies on `jax.experimental.io_callback`, so the pipeline stays compatible
  with `jax.jit` even though Python code is running in the middle of the graph.
- `MNISTDataset` and `CIFAR10Dataset` now return raw uint8 images/labels, and
  preprocessing steps (normalization, flattening, device moves, etc.) can be
  expressed explicitly with `NormalizeImageTransform`, `FlattenImageTransform`,
  and `MapTransform`.
- `DiskSampleSource` lets you wrap any callable that retrieves individual samples
  (e.g., from disk, databases, or RPC) and still benefit from the batching/padding
  logic shared with in-memory pipelines.
- `MNISTDiskSource` streams samples directly from IDX files via `io_callback`,
  letting you iterate over MNIST without ever materializing the full dataset in
  memory.
- `GymnaxSource` can turn any Gymnax environment + policy into a transition
  stream that plugs into the same batching and transform primitives.

## Handling large datasets

- By default `dataset_to_jax(..., storage_device="cpu")` stores samples on the CPU. Set
  `storage_device=None` if you explicitly want to keep them on the default JAX
  device instead.
- Use `DevicePutTransform` (examples below) to move batches onto a target device
  at the edge of your pipeline.
- Use `batch_transform=lambda batch: data_augmentation(batch)` if you need to
  run custom host work before handing the batch to jitted code.

## API reference

- `dataset_to_jax(dataset, storage_device="cpu") -> PyTree`
- `DataLoader.next_batch(state) -> (batch, new_state, mask)`
- `DataLoader.scan_epoch(state, carry, body_fn)`
- `MNISTDataset(split="train", cache_dir=None)`
- `CIFAR10Dataset(split="train", cache_dir=None)`
- `NormalizeImageTransform(inner, image_key="image", dtype=jnp.float32)`
- `FlattenImageTransform(inner, image_key="image")`
- `HostCallbackTransform(inner, fn, element_spec_override=None)`
- `DiskSampleSource(length, sample_fn, sample_spec, prefetch_size=64, ...)`
- `MNISTDiskSource(split="train", prefetch_size=64, cache_dir=None, ...)`
- `GymnaxSource(env, env_params, policy_fn, policy_params=None, steps_per_epoch=1024)`

The boolean mask returned alongside each batch stays `True` for real samples and
`False` for any padding that was injected to maintain static shapes when
`drop_last=False`.

## Building custom pipelines

You can recreate the convenience loader manually if you need finer control:

```python
import jax

from cereal import ArraySampleSource, BatchTransform, DataLoader, DevicePutTransform

source = ArraySampleSource(data, ordering="sequential")
pipeline = BatchTransform(batch_size=128)(source)
pipeline = DevicePutTransform()(pipeline)

state = pipeline.init_state(jax.random.PRNGKey(0))
batch, mask, state = pipeline.next(state)

# Or wrap the stages directly in `DataLoader` for convenience:
loader = DataLoader(
  pipeline=[
    ArraySampleSource(data, ordering="sequential"),
    BatchTransform(batch_size=128),
    DevicePutTransform(),
  ]
)
loader_state = loader.init_state(jax.random.PRNGKey(0))
batch, loader_state, mask = loader.next_batch(loader_state)
```

## JIT example

Because loaders are pure pytrees, you can jit entire training steps without
additional synchronization:

```python
def train_step(train_state, loader_state, rng):
  batch, loader_state, mask = loader.next_batch(loader_state)
  logits = model_apply(train_state.params, batch, rng)
  loss = jnp.mean(loss_fn(logits, batch["label"]) * mask[:, None])
  grads = jax.grad(lambda params: loss)(train_state.params)
  train_state = train_state.apply_gradients(grads=grads)
  return train_state, loader_state

compiled_step = jax.jit(train_step)
train_state, loader_state = compiled_step(train_state, loader_state, jax.random.PRNGKey(0))
```

`DevicePutTransform` defaults to the first device returned by `jax.devices()`, so
the compiled step works even when multiple accelerators are present.

## `scan_epoch` example

`DataLoader.scan_epoch` fuses a full pass through the dataset into a single
`jax.lax.scan` to minimize dispatch overhead:

```python
def body_fn(carry, _):
  opt_state, loader_state = carry
  batch, loader_state, mask = loader.next_batch(loader_state)
  opt_state = optimizer_update(opt_state, batch, mask)
  return (opt_state, loader_state), None

(opt_state, loader_state), _ = loader.scan_epoch((opt_state, loader_state), None, body_fn)
```

## Host callbacks inside pipelines

When you need host-side side effects (like printing metrics), insert a
`HostCallbackTransform` wherever you want Python to run. The transform receives
NumPy views of the batch and mask via `io_callback`, so the pipeline stays
compatible with `jax.jit`:

```python
import numpy as np
from cereal import ArraySampleSource, BatchTransform, DataLoader, HostCallbackTransform, MapTransform

def attach_loss(batch, mask):
  logits = model(batch["image"])
  loss = jnp.mean(cross_entropy(logits, batch["label"]) * mask[:, None])
  return {**batch, "loss": loss}

def log_loss(batch, mask):
  del mask
  print("loss:", float(np.asarray(batch["loss"])))

loader = DataLoader(
  pipeline=[
    ArraySampleSource(train_data, ordering="shuffle"),
    BatchTransform(batch_size=128),
    MapTransform(fn=attach_loss),
    HostCallbackTransform(fn=log_loss),
  ],
)
```

Host transforms may optionally return a modified batch (as NumPy arrays) before
it leaves the loader.

Adding new behaviour (e.g., prioritized replay) becomes a matter of implementing
another `Source` or wrapping the existing one with a custom transform.
