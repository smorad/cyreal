"""Unit tests for the JAX DataLoader stack."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cyreal import (
    ArraySource,
    BatchTransform,
    DataLoader,
    DevicePutTransform,
    DiskSource,
    FlattenTransform,
    GymnaxSource,
    HostCallbackTransform,
    MapTransform,
    NormalizeImageTransform,
    Source,
    TimeSeriesBatchTransform,
)
from cyreal.rl import set_loader_policy_state, set_source_policy_state


def test_next_padding_and_epoch_reset():
    data = {
        "inputs": jnp.arange(5, dtype=jnp.float32).reshape(5, 1),
    }
    source = ArraySource(data=data, ordering="sequential")
    pipeline = BatchTransform(
        batch_size=3,
        pad_last_batch=True,
        drop_last=False,
    )(source)
    loader = DataLoader(pipeline=pipeline)
    state = loader.init_state(jax.random.PRNGKey(0))

    batch, state, mask = loader.next(state)
    np.testing.assert_array_equal(np.asarray(batch["inputs"]).ravel(), np.array([0, 1, 2]))
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True, True]))

    batch, state, mask = loader.next(state)
    np.testing.assert_array_equal(np.asarray(batch["inputs"]).ravel(), np.array([3, 4, 0]))
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True, False]))
    inner_state = state.inner_state.inner_state  # BatchTransformState -> ArraySourceState
    assert int(inner_state.epoch) == 1


def test_batch_transform_and_device_put_applied():
    data = {"inputs": jnp.arange(4, dtype=jnp.float32).reshape(4, 1)}
    target_device = jax.devices()[0]

    source = ArraySource(data=data, ordering="sequential")
    pipeline = BatchTransform(
        batch_size=2,
        drop_last=True,
    )(source)

    def transform(batch, mask):
        del mask
        return {"inputs": batch["inputs"] + 10.0}

    pipeline = MapTransform(fn=transform)(pipeline)
    pipeline = DevicePutTransform(device=target_device)(pipeline)
    loader = DataLoader(pipeline=pipeline)
    state = loader.init_state(jax.random.PRNGKey(0))

    batch, state, mask = loader.next(state)
    np.testing.assert_array_equal(np.asarray(batch["inputs"]).ravel(), np.array([10.0, 11.0]))
    assert batch["inputs"].device == target_device
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

def test_manual_pipeline_composition_without_operator_overloads():
    data = {"inputs": jnp.arange(4, dtype=jnp.float32).reshape(4, 1)}

    def add_one(batch, mask):
        del mask
        return {"inputs": batch["inputs"] + 1.0}

    source = ArraySource(data=data, ordering="sequential")
    pipeline = BatchTransform(batch_size=2)(source)
    pipeline = MapTransform(fn=add_one)(pipeline)

    state = pipeline.init_state(jax.random.PRNGKey(0))
    batch, mask, state = pipeline.next(state)
    np.testing.assert_array_equal(np.asarray(batch["inputs"]).ravel(), np.array([1.0, 2.0]))
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

    batch, mask, _ = pipeline.next(state)
    np.testing.assert_array_equal(np.asarray(batch["inputs"]).ravel(), np.array([3.0, 4.0]))
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))


def test_dataloader_accepts_pipeline_sequence():
    data = {"inputs": jnp.arange(4, dtype=jnp.float32).reshape(4, 1)}

    def add_two(batch, mask):
        del mask
        return {"inputs": batch["inputs"] + 2.0}

    stages = [
        ArraySource(data=data, ordering="sequential"),
        BatchTransform(batch_size=2),
        MapTransform(fn=add_two),
    ]

    loader = DataLoader(pipeline=stages)
    state = loader.init_state(jax.random.PRNGKey(0))
    batch, state, mask = loader.next(state)

    np.testing.assert_array_equal(np.asarray(batch["inputs"]).ravel(), np.array([2.0, 3.0]))
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))
    assert loader.steps_per_epoch == 2


def test_dataloader_next_is_jittable():
    data = {"inputs": jnp.arange(6, dtype=jnp.float32).reshape(6, 1)}
    source = ArraySource(data=data, ordering="sequential")
    pipeline = BatchTransform(
        batch_size=2,
        pad_last_batch=True,
        drop_last=False,
    )(source)
    loader = DataLoader(pipeline=pipeline)
    state = loader.init_state(jax.random.PRNGKey(0))

    ref_batch, ref_state, ref_mask = loader.next(state)
    ref_batch2, _, ref_mask2 = loader.next(ref_state)

    @jax.jit
    def jit_step(loader_state):
        return loader.next(loader_state)

    batch, jit_state, mask = jit_step(state)
    batch2, _, mask2 = loader.next(jit_state)

    np.testing.assert_array_equal(np.asarray(batch["inputs"]), np.asarray(ref_batch["inputs"]))
    np.testing.assert_array_equal(np.asarray(mask), np.asarray(ref_mask))
    np.testing.assert_array_equal(np.asarray(batch2["inputs"]), np.asarray(ref_batch2["inputs"]))
    np.testing.assert_array_equal(np.asarray(mask2), np.asarray(ref_mask2))


def test_loader_iterate_returns_python_iterator():
    data = {"inputs": jnp.arange(4, dtype=jnp.float32).reshape(4, 1)}
    source = ArraySource(data=data, ordering="sequential")
    pipeline = BatchTransform(batch_size=2, drop_last=False)(source)
    loader = DataLoader(pipeline=pipeline)

    initial_state = loader.init_state(jax.random.PRNGKey(0))
    iterator = loader.iterate(initial_state)

    batches = []
    for batch, mask in iterator:
        np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))
        batches.append(np.asarray(batch["inputs"]).ravel())

    assert len(batches) == loader.steps_per_epoch
    with pytest.raises(StopIteration):
        next(iterator)

    manual_state = loader.init_state(jax.random.PRNGKey(0))
    manual_batches = []
    for _ in range(loader.steps_per_epoch):
        batch, manual_state, _ = loader.next(manual_state)
        manual_batches.append(np.asarray(batch["inputs"]).ravel())

    np.testing.assert_array_equal(np.stack(batches), np.stack(manual_batches))

    next_batch, _, _ = loader.next(iterator.state)
    np.testing.assert_array_equal(np.asarray(next_batch["inputs"]).ravel(), np.array([0.0, 1.0]))

    multi_state_key = jax.random.PRNGKey(123)
    multi_state = loader.init_state(multi_state_key)
    total_steps = loader.steps_per_epoch * 2
    multi_iterator = loader.iterate(multi_state, steps=total_steps)

    multi_batches = []
    for batch, mask in multi_iterator:
        np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))
        multi_batches.append(np.asarray(batch["inputs"]).ravel())

    assert len(multi_batches) == total_steps
    with pytest.raises(StopIteration):
        next(multi_iterator)

    manual_state = loader.init_state(multi_state_key)
    manual_batches = []
    for _ in range(total_steps):
        batch, manual_state, _ = loader.next(manual_state)
        manual_batches.append(np.asarray(batch["inputs"]).ravel())

    np.testing.assert_array_equal(np.stack(multi_batches), np.stack(manual_batches))

def test_scan_epoch_supports_multiple_epochs():
    data = {"inputs": jnp.arange(6, dtype=jnp.float32).reshape(6, 1)}
    source = ArraySource(data=data, ordering="sequential")
    pipeline = BatchTransform(batch_size=3, pad_last_batch=True, drop_last=False)(source)
    loader = DataLoader(pipeline=pipeline)

    state = loader.init_state(jax.random.PRNGKey(0))
    carry = jnp.array(0.0, dtype=jnp.float32)

    def body_fn(total, batch, mask):
        values = jnp.squeeze(batch["inputs"], axis=-1)
        summed = jnp.sum(values * mask.astype(jnp.float32))
        return total + summed, summed

    per_epoch_sum = float(np.sum(np.arange(6, dtype=np.float32)))
    for epoch in range(3):
        state, carry, outputs = loader.scan_epoch(state, carry, body_fn)
        assert outputs.shape == (loader.steps_per_epoch,)
        assert pytest.approx(float(carry)) == pytest.approx((epoch + 1) * per_epoch_sum)

def test_non_jittable_transform_runs_under_jit():
    data = {"inputs": jnp.arange(4, dtype=jnp.float32).reshape(4, 1)}
    calls: list[float] = []

    def log_batch(batch, mask):
        del mask
        calls.append(float(np.sum(batch["inputs"])))

    source = ArraySource(data=data, ordering="sequential")
    pipeline = BatchTransform(
        batch_size=2,
        drop_last=True,
    )(source)
    pipeline = HostCallbackTransform(fn=log_batch)(pipeline)
    loader = DataLoader(pipeline=pipeline)
    state = loader.init_state(jax.random.PRNGKey(0))

    @jax.jit
    def run(state):
        return loader.next(state)

    batch, state, mask = run(state)
    assert len(calls) == 1
    np.testing.assert_array_equal(np.asarray(batch["inputs"]).ravel(), np.array([0.0, 1.0]))
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))


def test_non_jittable_transform_can_modify_batch():
    data = {"inputs": jnp.arange(4, dtype=jnp.float32).reshape(4, 1)}

    def add_offset(batch, mask):
        del mask
        updated = dict(batch)
        updated["inputs"] = batch["inputs"] + 5.0
        return updated

    source = ArraySource(data=data, ordering="sequential")
    pipeline = BatchTransform(
        batch_size=2,
        drop_last=True,
    )(source)
    pipeline = HostCallbackTransform(fn=add_offset)(pipeline)
    loader = DataLoader(pipeline=pipeline)

    state = loader.init_state(jax.random.PRNGKey(0))
    batch, _, _ = loader.next(state)
    np.testing.assert_array_equal(
        np.asarray(batch["inputs"]).ravel(),
        np.array([5.0, 6.0]),
    )


def test_host_callback_transform_interleaves_with_map_transforms():
    data = {"inputs": jnp.arange(4, dtype=jnp.float32).reshape(4, 1)}
    logs: list[float] = []

    def add_one(batch, mask):
        del mask
        return {"inputs": batch["inputs"] + 1.0}

    def host_callback(batch, mask):
        del mask
        logs.append(float(np.sum(batch["inputs"])))
        updated = dict(batch)
        updated["inputs"] = batch["inputs"] + 50.0
        return updated

    def double(batch, mask):
        del mask
        return {"inputs": batch["inputs"] * 2.0}

    source = ArraySource(data=data, ordering="sequential")
    pipeline = BatchTransform(
        batch_size=2,
        drop_last=True,
    )(source)
    pipeline = MapTransform(fn=add_one)(pipeline)
    pipeline = HostCallbackTransform(fn=host_callback)(pipeline)
    pipeline = MapTransform(fn=double)(pipeline)
    loader = DataLoader(pipeline=pipeline)

    state = loader.init_state(jax.random.PRNGKey(0))
    batch, _, _ = loader.next(state)

    assert logs == [3.0]
    np.testing.assert_array_equal(
        np.asarray(batch["inputs"]).ravel(),
        np.array([102.0, 104.0]),
    )


def test_disk_sample_source_streams_via_callback():
    samples = [
        {"value": np.array(i, dtype=np.int32)}
        for i in range(5)
    ]

    def sample_fn(idx: int):
        return {"value": np.array(samples[idx]["value"], dtype=np.int32)}

    spec = {"value": jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32)}
    source = DiskSource(
        length=len(samples),
        sample_fn=sample_fn,
        sample_spec=spec,
        ordering="sequential",
        prefetch_size=2,
    )
    batched = BatchTransform(
        batch_size=2,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, state = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch["value"]), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

    batch2, mask2, state = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch2["value"]), np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask2), np.array([True, True]))

    batch3, mask3, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch3["value"]), np.array([4, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask3), np.array([True, False]))


def test_disk_sample_source_infers_spec_when_missing():
    samples = [
        {"value": np.array([i, i + 1], dtype=np.float32)}
        for i in (0, 2, 4)
    ]

    def sample_fn(idx: int):
        return {"value": np.array(samples[idx]["value"], dtype=np.float32)}

    source = DiskSource(
        length=len(samples),
        sample_fn=sample_fn,
        ordering="sequential",
        prefetch_size=2,
    )
    spec = source.element_spec()
    assert spec["value"].shape == (2,)
    assert spec["value"].dtype == np.float32

    batched = BatchTransform(
        batch_size=2,
        element_spec_override=spec,
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, state = batched.next(state)
    np.testing.assert_array_equal(
        np.asarray(batch["value"]),
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

    batch2, mask2, _ = batched.next(state)
    np.testing.assert_array_equal(
        np.asarray(batch2["value"][:1]),
        np.array([[4.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(np.asarray(mask2), np.array([True, False]))


def test_time_series_batch_transform_batched_mode_adds_feature_axis():
    num_samples = 4
    context_length = 3
    contexts = jnp.arange(num_samples * context_length, dtype=jnp.float32)
    contexts = contexts.reshape(num_samples, context_length)
    data = {"context": contexts}

    source = ArraySource(data=data, ordering="sequential")
    pipeline = BatchTransform(batch_size=2)(source)
    pipeline = TimeSeriesBatchTransform(mode="batched")(pipeline)

    state = pipeline.init_state(jax.random.PRNGKey(0))
    batch, mask, _ = pipeline.next(state)

    assert batch["context"].shape == (2, context_length, 1)
    np.testing.assert_array_equal(
        np.asarray(batch["context"][0, :, 0]),
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))


def test_time_series_batch_transform_packed_mode_marks_sequence_boundaries():
    num_samples = 3
    context_length = 2
    contexts = jnp.arange(num_samples * context_length, dtype=jnp.float32)
    contexts = contexts.reshape(num_samples, context_length)
    data = {"context": contexts}

    source = ArraySource(data=data, ordering="sequential")
    pipeline = BatchTransform(batch_size=2)(source)
    pipeline = TimeSeriesBatchTransform(mode="packed")(pipeline)

    state = pipeline.init_state(jax.random.PRNGKey(0))
    batch, mask, state = pipeline.next(state)

    np.testing.assert_array_equal(
        np.asarray(batch["context"]).ravel(),
        np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(batch["sequence_start"]),
        np.array([True, False, True, False]),
    )
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True, True, True]))

    batch2, mask2, _ = pipeline.next(state)
    np.testing.assert_array_equal(
        np.asarray(batch2["context"]).ravel(),
        np.array([4.0, 5.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(batch2["sequence_start"]),
        np.array([True, False, False, False]),
    )
    np.testing.assert_array_equal(np.asarray(mask2), np.array([True, True, False, False]))


def test_mnist_image_transforms_normalize_and_flatten():
    data = {
        "image": jnp.array(
            [
                [[[0.0], [255.0]]],
                [[[10.0], [20.0]]],
            ],
            dtype=jnp.uint8,
        ),
        "label": jnp.array([0, 1], dtype=jnp.int32),
    }

    source: Source = ArraySource(
        data=data,
        ordering="sequential",
    )
    source = NormalizeImageTransform(dtype=jnp.float32)(source)
    source = FlattenTransform(data_key="image", start_index=1)(source)
    batched = BatchTransform(
        batch_size=1,
        drop_last=True,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batches = []
    masks = []
    for _ in range(2):
        batch, mask, state = batched.next(state)
        batches.append(np.asarray(batch["image"]))
        masks.append(np.asarray(mask))

    stacked = np.concatenate(batches, axis=0)
    np.testing.assert_array_equal(np.concatenate(masks, axis=0), np.array([True, True]))
    assert batch["image"].dtype == jnp.float32
    np.testing.assert_allclose(
        stacked,
        np.array(
            [
                [[0.0, 1.0]],
                [[10.0 / 255.0, 20.0 / 255.0]],
            ],
            dtype=np.float32,
        ),
    )


class _ToyGymnaxEnv:
    def reset(self, key, params):
        del key, params
        obs = jnp.array([0.0], dtype=jnp.float32)
        state = jnp.array(0.0, dtype=jnp.float32)
        return obs, state

    def step(self, key, state, action, params):
        del key
        next_state = state + action
        obs = jnp.array([next_state])
        reward = -jnp.abs(next_state)
        done = next_state >= params["threshold"]
        info = {}
        return obs, next_state, reward, done, info


def _toy_policy_step(obs, policy_state, new_episode, key):
    del obs, new_episode, key
    action = jnp.asarray(policy_state["action"], dtype=jnp.float32)
    return action, policy_state


def test_gymnax_source_rollout_and_epoch_reset():
    env = _ToyGymnaxEnv()
    source = GymnaxSource(
        env=env,
        env_params={"threshold": 2.0},
        policy_step_fn=_toy_policy_step,
        policy_state_template={"action": jnp.array(1.0, dtype=jnp.float32)},
        steps_per_epoch=3,
    )

    state = source.init_state(jax.random.PRNGKey(0))
    state = set_source_policy_state(state, {"action": jnp.array(1.0, dtype=jnp.float32)})
    transition, mask, state = source.next(state)
    assert bool(mask)
    assert set(transition.keys()) == {"state", "action", "reward", "next_state", "done", "info"}
    assert transition["info"] == {}

    # Run through an epoch and ensure epoch counter advances.
    for _ in range(2):
        _, _, state = source.next(state)
    assert int(state.epoch) == 1

    # Ensure policy state continues to propagate through epochs.
    assert jnp.asarray(state.policy_state["action"]).shape == ()


def test_loader_set_policy_state_delegates_through_transforms():
    env = _ToyGymnaxEnv()
    source = GymnaxSource(
        env=env,
        env_params={"threshold": 2.0},
        policy_step_fn=_toy_policy_step,
        policy_state_template={"action": jnp.array(1.0, dtype=jnp.float32)},
        steps_per_epoch=4,
    )
    pipeline = [
        source,
        BatchTransform(batch_size=2, drop_last=False),
        MapTransform(lambda batch, mask: batch),
    ]
    loader = DataLoader(pipeline=pipeline)
    loader_state = loader.init_state(jax.random.PRNGKey(1))
    policy_state = {"action": jnp.array(2.0, dtype=jnp.float32)}
    loader_state = set_loader_policy_state(loader_state, policy_state)

    batch, loader_state, mask = loader.next(loader_state)
    assert batch["action"].shape[0] == 2
    assert bool(mask[0])