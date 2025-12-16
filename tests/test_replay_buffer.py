"""Tests for the replay buffer transform."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import jax
import jax.numpy as jnp

from cyreal.sources import ArraySource
from cyreal.transforms import BatchTransform, BufferTransform, TimeSeriesBatchTransform


def _sequential_source(length: int = 32) -> ArraySource:
    return ArraySource({"value": jnp.arange(length)}, ordering="sequential")


def _sequence_source(num_sequences: int = 16, time_steps: int = 3, features: int = 2) -> ArraySource:
    context = jnp.arange(num_sequences * time_steps * features, dtype=jnp.float32)
    context = context.reshape(num_sequences, time_steps, features)
    targets = jnp.arange(num_sequences)
    return ArraySource({"context": context, "target": targets}, ordering="sequential")


def _simulate_reservoir(expected_values: Sequence[int], capacity: int, key: jax.Array) -> np.ndarray:
    buffer = np.zeros(capacity, dtype=np.int32)
    count = 0
    write_index = 0
    seen = 0
    rng = jax.random.fold_in(key, 1)
    for value in expected_values:
        rng, write_key, _ = jax.random.split(rng, 3)
        seen += 1
        if count < capacity:
            buffer[write_index] = value
            write_index = (write_index + 1) % capacity
            count += 1
        else:
            rand_idx = int(jax.random.randint(write_key, (), minval=0, maxval=max(seen, 1)))
            if rand_idx < capacity:
                buffer[rand_idx] = value
    return buffer


def test_buffer_sequential_mode_prefill_passthrough():
    buffer = BufferTransform(capacity=4, prefill=3, sample_size=1, mode="sequential")(
        _sequential_source(10)
    )
    state = buffer.init_state(jax.random.PRNGKey(0))

    values: list[int] = []
    masks: list[bool] = []
    for _ in range(6):
        sample, mask, state = buffer.next(state)
        values.append(int(np.asarray(sample["value"])))
        masks.append(bool(np.asarray(mask)))

    assert values[:3] == [0, 1, 2]
    assert all(masks[:3])
    # Once the buffer is warm, sequential mode iterates through cached entries.
    assert values[3:6] == [0, 1, 2]
    assert all(masks)


def test_buffer_shuffled_mode_emits_chunks():
    buffer = BufferTransform(capacity=16, prefill=8, sample_size=4, mode="shuffled")(
        _sequential_source(32)
    )
    state = buffer.init_state(jax.random.PRNGKey(42))

    for _ in range(8):
        _, _, state = buffer.next(state)

    batch, mask, state = buffer.next(state)

    values = np.asarray(batch["value"])
    mask_np = np.asarray(mask)
    assert values.shape == (4,)
    assert mask_np.shape == (4,)
    assert mask_np.dtype == np.bool_
    assert mask_np.all()
    assert values.min() >= 0
    assert values.max() < 32


def test_buffer_with_time_series_batch_transform():
    time_steps = 3
    features = 2
    source = _sequence_source(num_sequences=16, time_steps=time_steps, features=features)
    buffer = BufferTransform(capacity=12, prefill=12, sample_size=3, mode="sequential")(source)
    pipeline = TimeSeriesBatchTransform(sequence_key="context", mode="packed")(buffer)
    state = pipeline.init_state(jax.random.PRNGKey(7))

    for _ in range(12):
        _, _, state = pipeline.next(state)

    batch, mask, state = pipeline.next(state)

    context = np.asarray(batch["context"])
    mask_np = np.asarray(mask)
    sequence_start = np.asarray(batch["sequence_start"])
    target = np.asarray(batch["target"])

    assert context.shape == (3 * time_steps, features)
    assert mask_np.shape == (3 * time_steps,)
    assert sequence_start.shape == (3 * time_steps,)
    assert target.shape == (3,)
    assert mask_np.all()
    assert sequence_start.tolist() == [True, False, False, True, False, False, True, False, False]
    assert target.min() >= 0
    assert target.max() < 16

    for seq_idx, seq_id in enumerate(target.tolist()):
        start = seq_id * time_steps * features
        expected = np.arange(start, start + time_steps * features, dtype=np.float32)
        expected = expected.reshape(time_steps, features)
        block = context[seq_idx * time_steps : (seq_idx + 1) * time_steps]
        np.testing.assert_array_equal(block, expected)


def test_reservoir_write_matches_reference():
    capacity = 4
    total = 20
    key = jax.random.PRNGKey(123)
    buffer = BufferTransform(
        capacity=capacity,
        prefill=capacity,
        sample_size=1,
        mode="sequential",
        write_mode="reservoir",
    )(_sequential_source(total))
    state = buffer.init_state(key)

    for _ in range(total):
        _, _, state = buffer.next(state)

    actual = np.asarray(state.buffer["value"])
    expected = _simulate_reservoir(range(total), capacity, key)
    np.testing.assert_array_equal(actual, expected)


def test_buffer_with_sample_size_then_batch_transform():
    k = 6
    source = _sequential_source(length=128)
    buffer = BufferTransform(
        capacity=32,
        prefill=12,
        sample_size=k,
        mode="shuffled",
        write_mode="fifo",
    )(source)
    pipeline = BatchTransform(batch_size=k // 2)(buffer)
    state = pipeline.init_state(jax.random.PRNGKey(0))

    # Warm up to ensure buffer emits cached samples
    for _ in range(6):
        _, _, state = pipeline.next(state)

    batch, mask, state = pipeline.next(state)

    assert batch["value"].shape == (k // 2, k)
    assert mask.shape == (k // 2,)
    mask_np = np.asarray(mask)
    assert mask_np.dtype == np.bool_
    assert mask_np.all()
