"""Executable versions of the README snippets."""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cyreal.loader import DataLoader

from cyreal.sources import (
    DiskSource,
    ArraySource,
    GymnaxSource,
)

from cyreal.transforms import (
    BatchTransform,
    HostCallbackTransform,
)

from cyreal.datasets import MNISTDataset

from cyreal.rl import set_loader_policy_state, set_source_policy_state

import gymnax

def test_readme():
    import jax
    import jax.numpy as jnp

    from cyreal.transforms import BatchTransform
    from cyreal.loader import DataLoader
    from cyreal.rl import set_loader_policy_state, set_source_policy_state
    from cyreal.sources import ArraySource
    from cyreal.datasets import MNISTDataset

    train_data = MNISTDataset(split="test").as_array_dict()
    pipeline = [
    # Load dataset into memory-backed array
    ArraySource(train_data, ordering="shuffle"),
    # Batch it
    BatchTransform(batch_size=128),
    # Move the batch to the GPU
    ]
    loader = DataLoader(pipeline)
    state = loader.init_state(jax.random.key(0))

    for epoch in range(2):
        for batch, mask in loader.iterate(state):
            ...  # train your network!

    for epoch in range(2):
        for _ in range(loader.steps_per_epoch):
            batch, state, mask = jax.jit(loader.next)(state)
            ... # Train your network

    model_state = {"params": jnp.array(0)}

    def update(model_state, batch, mask):
        model_state = {"params": model_state['params'] + 1}
        return model_state, None

    for epoch in range(2):
        state, model_state, _ = loader.scan_epoch(state, model_state, update)