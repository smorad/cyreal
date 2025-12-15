"""Unit tests for the JAX DataLoader stack."""
from __future__ import annotations

import gzip
import pickle
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cyreal import (
    ArraySampleSource,
    BatchTransform,
    CIFAR10Dataset,
    CIFAR10DiskSource,
    DataLoader,
    DevicePutTransform,
    DiskSampleSource,
    FlattenTransform,
    GymnaxSource,
    HostCallbackTransform,
    MapTransform,
    MNISTDataset,
    MNISTDiskSource,
    NormalizeImageTransform,
    Source,
)


def test_next_padding_and_epoch_reset():
    data = {
        "inputs": jnp.arange(5, dtype=jnp.float32).reshape(5, 1),
    }
    source = ArraySampleSource(data=data, ordering="sequential")
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

    source = ArraySampleSource(data=data, ordering="sequential")
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

    source = ArraySampleSource(data=data, ordering="sequential")
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
        ArraySampleSource(data=data, ordering="sequential"),
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
    source = ArraySampleSource(data=data, ordering="sequential")
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
    source = ArraySampleSource(data=data, ordering="sequential")
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
    source = ArraySampleSource(data=data, ordering="sequential")
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

    source = ArraySampleSource(data=data, ordering="sequential")
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

    source = ArraySampleSource(data=data, ordering="sequential")
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

    source = ArraySampleSource(data=data, ordering="sequential")
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


def _write_idx_images(path, images):
    with gzip.open(path, "wb") as f:
        num, rows, cols = images.shape
        f.write(struct.pack(">IIII", 2051, num, rows, cols))
        f.write(images.tobytes())


def _write_idx_labels(path, labels):
    with gzip.open(path, "wb") as f:
        num = labels.shape[0]
        f.write(struct.pack(">II", 2049, num))
        f.write(labels.tobytes())


def _seed_fake_cifar10(tmp_path, split="train", samples_per_batch=1):
    archive_path = tmp_path / "cifar-10-python.tar.gz"
    archive_path.write_bytes(b"")
    batches_dir = tmp_path / "cifar-10-batches-py"
    batches_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        names = [f"data_batch_{i}" for i in range(1, 6)]
    elif split == "test":
        names = ["test_batch"]
    else:
        raise ValueError("split must be 'train' or 'test'.")

    current_label = 0
    for name in names:
        data = []
        labels = []
        for _ in range(samples_per_batch):
            pixel_value = current_label % 256
            image = np.full((3, 32, 32), pixel_value, dtype=np.uint8)
            data.append(image.reshape(-1))
            labels.append(current_label)
            current_label += 1
        batch = {"data": np.stack(data, axis=0), "labels": labels}
        with open(batches_dir / name, "wb") as f:
            pickle.dump(batch, f, protocol=2)


def test_mnist_dataset_reads_cached_idx(tmp_path):
    num, rows, cols = 3, 2, 2
    images = np.arange(num * rows * cols, dtype=np.uint8).reshape(num, rows, cols)
    labels = np.arange(num, dtype=np.uint8)

    images_path = tmp_path / "train_images.gz"
    labels_path = tmp_path / "train_labels.gz"
    _write_idx_images(images_path, images)
    _write_idx_labels(labels_path, labels)

    dataset = MNISTDataset(split="train", cache_dir=tmp_path)
    assert len(dataset) == num
    example = dataset[0]
    img = example["image"]
    label = example["label"]
    assert img.shape == (rows, cols, 1)
    assert img.dtype == np.uint8
    assert label == 0
    np.testing.assert_array_equal(img[..., 0], images[0])


def test_disk_sample_source_streams_via_callback():
    samples = [
        {"value": np.array(i, dtype=np.int32)}
        for i in range(5)
    ]

    def sample_fn(idx: int):
        return {"value": np.array(samples[idx]["value"], dtype=np.int32)}

    spec = {"value": jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32)}
    source = DiskSampleSource(
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

    source = DiskSampleSource(
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


@pytest.mark.parametrize("factory", ["class", "helper"])
def test_mnist_disk_source_streams_from_disk(tmp_path, factory):
    num, rows, cols = 3, 2, 2
    images = np.arange(num * rows * cols, dtype=np.uint8).reshape(num, rows, cols)
    labels = np.arange(num, dtype=np.uint8)

    images_path = tmp_path / "train_images.gz"
    labels_path = tmp_path / "train_labels.gz"
    _write_idx_images(images_path, images)
    _write_idx_labels(labels_path, labels)

    if factory == "helper":
        source = MNISTDataset.make_disk_source(
            split="train",
            cache_dir=tmp_path,
            ordering="sequential",
            prefetch_size=2,
        )
    else:
        source = MNISTDiskSource(
            split="train",
            cache_dir=tmp_path,
            ordering="sequential",
            prefetch_size=2,
        )
    batched = BatchTransform(
        batch_size=2,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, state = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch["image"][..., 0]), images[:2])
    np.testing.assert_array_equal(np.asarray(batch["label"]), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

    batch2, mask2, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch2["image"][0, ..., 0]), images[2])
    np.testing.assert_array_equal(np.asarray(batch2["label"][:1]), np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask2), np.array([True, False]))


@pytest.mark.parametrize("factory", ["class", "helper"])
def test_cifar10_disk_source_streams_from_disk(tmp_path, factory):
    _seed_fake_cifar10(tmp_path, split="test", samples_per_batch=3)

    if factory == "helper":
        source = CIFAR10Dataset.make_disk_source(
            split="test",
            cache_dir=tmp_path,
            ordering="sequential",
            prefetch_size=2,
        )
    else:
        source = CIFAR10DiskSource(
            split="test",
            cache_dir=tmp_path,
            ordering="sequential",
            prefetch_size=2,
        )
    batched = BatchTransform(
        batch_size=2,
        element_spec_override=source.element_spec(),
    )(source)

    state = batched.init_state(jax.random.PRNGKey(0))
    batch, mask, state = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch["label"]), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(batch["image"][0, 0, 0]),
        np.array([0, 0, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(batch["image"][1, 0, 0]),
        np.array([1, 1, 1], dtype=np.uint8),
    )
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True]))

    batch2, mask2, _ = batched.next(state)
    np.testing.assert_array_equal(np.asarray(batch2["label"][:1]), np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mask2), np.array([True, False]))
    np.testing.assert_array_equal(
        np.asarray(batch2["image"][1]),
        np.zeros((32, 32, 3), dtype=np.uint8),
    )


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

    source: Source = ArraySampleSource(
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


def _toy_policy(obs, params, key):
    del obs, key
    return jnp.asarray(params["action"], dtype=jnp.float32)


def test_gymnax_source_rollout_and_epoch_reset():
    env = _ToyGymnaxEnv()
    source = GymnaxSource(
        env=env,
        env_params={"threshold": 2.0},
        policy_fn=_toy_policy,
        policy_params={"action": 1.0},
        steps_per_epoch=3,
    )

    state = source.init_state(jax.random.PRNGKey(0))
    transition, mask, state = source.next(state)
    assert bool(mask)
    assert set(transition.keys()) == {"state", "action", "reward", "next_state", "done"}

    # Run through an epoch and ensure epoch counter advances.
    for _ in range(2):
        _, _, state = source.next(state)
    assert int(state.epoch) == 1