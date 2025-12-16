"""Compares the speed of grain to cyreal on MNIST."""

import time

import jax
import jax.numpy as jnp

from cyreal.sources import ArraySource
from cyreal.transforms import BufferTransform, BatchTransform, DevicePutTransform
from cyreal.loader import DataLoader
from cyreal.datasets import MNISTDataset


train_data = MNISTDataset(split="test").as_array_dict()

def test_cpu_iter(batch_size: int=32):
    pipeline = [
        ArraySource(train_data, ordering="shuffle"),
        BatchTransform(batch_size=batch_size),
    ]
    loader = DataLoader(pipeline)
    loader_state = loader.init_state(jax.random.key(0))
    start = time.time()
    for batch, mask in loader.iterate(loader_state):
        mask.block_until_ready()
    end = time.time()
    return end - start

def test_cpu_gpu_iter(batch_size: int=32):
    pipeline = [
        ArraySource(train_data, ordering="shuffle"),
        BatchTransform(batch_size=batch_size),
        DevicePutTransform(),
    ]
    loader = DataLoader(pipeline)
    loader_state = loader.init_state(jax.random.key(0))
    start = time.time()
    for batch, mask in loader.iterate(loader_state):
        mask.block_until_ready()
    end = time.time()
    return end - start

def test_cpu_gpu_jit(batch_size: int=32):
    pipeline = [
        ArraySource(train_data, ordering="shuffle"),
        BatchTransform(batch_size=batch_size),
        DevicePutTransform(),
    ]
    loader = DataLoader(pipeline)
    loader_state = loader.init_state(jax.random.key(0))
    # compile 
    load_fn = jax.jit(loader.next)
    for i in range(5):
        _ = load_fn(loader_state)

    start = time.time()
    for _ in range(loader.steps_per_epoch):
        batch, loader_state, mask = load_fn(loader_state)
        mask.block_until_ready()
    end = time.time()
    return end - start

def test_cpu_jit(batch_size: int=32):
    pipeline = [
        ArraySource(train_data, ordering="shuffle"),
        BatchTransform(batch_size=batch_size),
    ]
    loader = DataLoader(pipeline)
    loader_state = loader.init_state(jax.random.key(0))
    # compile 
    load_fn = jax.jit(loader.next)
    for i in range(5):
        _ = load_fn(loader_state)

    start = time.time()
    for _ in range(loader.steps_per_epoch):
        batch, loader_state, mask = load_fn(loader_state)
        mask.block_until_ready()
    end = time.time()
    return end - start

def test_gpu_jit(batch_size: int=32):
    pipeline = [
        ArraySource(train_data, ordering="shuffle"),
        # Move entire dataset to GPU
        DevicePutTransform(),
        BatchTransform(batch_size=batch_size),
    ]
    loader = DataLoader(pipeline)
    loader_state = loader.init_state(jax.random.key(0))
    load_fn = jax.jit(loader.next)
    # compile and warm up
    for i in range(5):
        _ = load_fn(loader_state)

    start = time.time()
    for _ in range(loader.steps_per_epoch):
        batch, loader_state, mask = load_fn(loader_state)
        mask.block_until_ready()
    end = time.time()
    return end - start

def test_cpu_scan(batch_size: int=32):
    pipeline = [
        ArraySource(train_data, ordering="shuffle"),
        BatchTransform(batch_size=batch_size),
    ]
    loader = DataLoader(pipeline)
    loader_state = loader.init_state(jax.random.key(0))

    def scan_step(model_state, batch, mask):
        return jnp.zeros((), dtype=bool) + mask[0], None

    # compile and warmup
    init_state = jnp.zeros((), dtype=jnp.bool_)
    loader_state, model_state, _ = loader.scan_epoch(loader_state, init_state, scan_step)
    start = time.time()
    loader_state, model_state, _ = loader.scan_epoch(loader_state, init_state, scan_step)
    end = time.time()
    return end - start

def test_cpu_gpu_scan(batch_size: int=32):
    pipeline = [
        ArraySource(train_data, ordering="shuffle"),
        BatchTransform(batch_size=batch_size),
        DevicePutTransform(),
    ]
    loader = DataLoader(pipeline)
    loader_state = loader.init_state(jax.random.key(0))

    def scan_step(model_state, batch, mask):
        return mask[0], None

    # compile and warmup
    init_state = jnp.zeros((), dtype=jnp.bool_)
    loader_state, model_state, _ = loader.scan_epoch(loader_state, init_state, scan_step)
    start = time.time()
    loader_state, model_state, _ = loader.scan_epoch(loader_state, init_state, scan_step)
    end = time.time()
    return end - start

def test_gpu_scan(batch_size: int=32):
    pipeline = [
        ArraySource(train_data, ordering="shuffle"),
        DevicePutTransform(),
        BatchTransform(batch_size=batch_size),
    ]
    loader = DataLoader(pipeline)
    loader_state = loader.init_state(jax.random.key(0))

    def scan_step(model_state, batch, mask):
        return mask[0], None

    # compile and warmup
    init_state = jnp.zeros((), dtype=jnp.bool_)
    loader_state, model_state, _ = loader.scan_epoch(loader_state, init_state, scan_step)
    start = time.time()
    loader_state, model_state, _ = loader.scan_epoch(loader_state, init_state, scan_step)
    end = time.time()
    return end - start

def test_grain_cpu(batch_size: int=32):
    try:
        import tensorflow_datasets as tfds
        import tensorflow as tf
        import grain.python as pygrain
    except ImportError:
        print("Grain/tfds not installed, skipping Grain speed test.")
        return float('nan')

    tf.config.set_visible_devices([], device_type='GPU')

    ds = tfds.data_source("mnist", split="test")
    dl_grain = pygrain.DataLoader(
        data_source=ds,
        sampler=pygrain.IndexSampler(
            num_records=len(ds),
            shard_options=pygrain.NoSharding(),
            shuffle=True,
            num_epochs=1,
            seed=0,
        ),
        operations=[pygrain.Batch(batch_size=32)],
        worker_count=0,
    )

    start = time.time()
    for element in dl_grain:
        tmp = jnp.array(element['image'], device=jax.devices("cpu")[0]).block_until_ready()
    end = time.time()
    return end - start

def test_grain_gpu(batch_size: int=32):
    try:
        import tensorflow_datasets as tfds
        import tensorflow as tf
        import grain.python as pygrain
    except ImportError:
        print("Grain/tfds not installed, skipping Grain speed test.")
        return float('nan')

    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), device_type='GPU')
    data_source = tfds.data_source("mnist", split="test")

    # To shuffle the data, use a sampler:
    ds = tfds.data_source("mnist", split="test")
    dl_grain = pygrain.DataLoader(
        data_source=ds,
        sampler=pygrain.IndexSampler(
            num_records=len(ds),
            shard_options=pygrain.NoSharding(),
            shuffle=True,
            num_epochs=1,
            seed=0,
        ),
        operations=[pygrain.Batch(batch_size=32)],
        worker_count=0,
    )
    start = time.time()
    for element in dl_grain:
        tmp = jnp.array(element['image'], device=jax.devices("gpu")[0]).block_until_ready()
    end = time.time()
    return end - start

if __name__ == "__main__":
    has_gpu = jax.devices()[0].platform == "gpu"
    tests = {
        "Grain CPU Dataset Iterator": test_grain_cpu,
        "Grain GPU Dataset Iterator": test_grain_gpu if has_gpu else lambda: float('nan'),

        "CPU Dataset GPU JIT Batch": test_cpu_gpu_jit if has_gpu else lambda: float('nan'),
        "CPU Dataset JIT Batch": test_cpu_jit,
        "GPU Dataset JIT Batch": test_gpu_jit if has_gpu else lambda: float('nan'),

        "CPU Dataset Scan": test_cpu_scan,
        "CPU Dataset GPU Scan": test_cpu_gpu_scan if has_gpu else lambda: float('nan'),
        "GPU Dataset Scan": test_gpu_scan if has_gpu else lambda: float('nan'),

        # # Warning: Very slow!
        # "CPU Dataset Iterator": test_cpu_iter,
        # "CPU Dataset GPU Batch Iterator": test_cpu_gpu_iter if has_gpu else lambda: float('nan'),

    }
    for test in tests:
        duration = tests[test]()
        print(f"{test:<30}: {duration:>4.4f} seconds")
