"""Executable versions of the README snippets."""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cyreal import (
    ArraySampleSource,
    BatchTransform,
    DataLoader,
    DevicePutTransform,
    GymnaxSource,
    HostCallbackTransform,
    MNISTDataset,
)
from cyreal.rl import set_loader_policy_state, set_source_policy_state

import gymnax


def _write_idx_images(path: Path, images: np.ndarray) -> None:
    with gzip.open(path, "wb") as f:
        num, rows, cols = images.shape
        f.write(struct.pack(">IIII", 2051, num, rows, cols))
        f.write(images.tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as f:
        num = labels.shape[0]
        f.write(struct.pack(">II", 2049, num))
        f.write(labels.tobytes())


@pytest.fixture(autouse=True)
def fake_mnist_cache(tmp_path, monkeypatch):
    """Populate the default MNIST cache path with tiny IDX files."""

    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: home)

    cache_dir = home / ".cache" / "jax_mnist"
    cache_dir.mkdir(parents=True, exist_ok=True)

    images = np.arange(16, dtype=np.uint8).reshape(4, 2, 2)
    labels = np.arange(4, dtype=np.uint8)

    for split in ("train", "test"):
        _write_idx_images(cache_dir / f"{split}_images.gz", images)
        _write_idx_labels(cache_dir / f"{split}_labels.gz", labels)

    return cache_dir


def test_readme_quickstart_example_runs():
    train_data = MNISTDataset(split="train").as_array_dict()
    pipeline = [
        ArraySampleSource(train_data, ordering="shuffle"),
        BatchTransform(batch_size=128),
        DevicePutTransform(),
    ]
    loader = DataLoader(pipeline=pipeline)
    state = loader.init_state(jax.random.PRNGKey(0))

    iterator = loader.iterate(state)
    batch, mask = next(iterator)

    assert batch["image"].shape == (128, 2, 2, 1)
    assert mask.shape == (128,)


def test_readme_scan_example_runs():
    train_data = MNISTDataset(split="train").as_array_dict()
    pipeline = [
        ArraySampleSource(train_data, ordering="shuffle"),
        BatchTransform(batch_size=128),
        DevicePutTransform(),
    ]
    loader = DataLoader(pipeline=pipeline)
    loader_state = loader.init_state(jax.random.PRNGKey(1))

    def update_model(model_state, batch, mask):
        del batch
        return model_state + jnp.sum(mask.astype(jnp.int32))

    def body_fn(model_state, batch, mask):
        new_model_state = update_model(model_state, batch, mask)
        return new_model_state, None

    loader_state, model_state, outputs = loader.scan_epoch(
        loader_state,
        jnp.array(0, dtype=jnp.int32),
        body_fn,
    )

    assert model_state > 0
    assert outputs is None
    assert isinstance(loader_state, type(loader.init_state(jax.random.PRNGKey(2))))


def test_readme_manual_jit_example_runs():
    train_data = MNISTDataset(split="train").as_array_dict()
    pipeline = [
        ArraySampleSource(train_data, ordering="shuffle"),
        BatchTransform(batch_size=128),
        DevicePutTransform(),
    ]
    loader = DataLoader(pipeline)
    loader_state = loader.init_state(jax.random.PRNGKey(2))

    def model_init():
        return jnp.array(0.0, dtype=jnp.float32)

    def model_update(model_state, batch, mask):
        del batch
        return model_state + jnp.sum(mask.astype(jnp.float32))

    model_state = model_init()

    @jax.jit
    def train_epoch(model_state, loader_state):
        def body_fn(model_state, batch, mask):
            new_model_state = model_update(model_state, batch, mask)
            return new_model_state, None

        loader_state, model_state, _ = loader.scan_epoch(loader_state, model_state, body_fn)
        return model_state, loader_state

    model_state, loader_state = train_epoch(model_state, loader_state)
    assert float(model_state) > 0
    assert isinstance(loader_state, type(loader.init_state(jax.random.PRNGKey(3))))


def test_readme_streaming_example_runs():
    pipeline = [
        MNISTDataset.make_disk_source(split="train", ordering="shuffle", prefetch_size=1024),
        BatchTransform(batch_size=128),
        DevicePutTransform(),
    ]
    loader = DataLoader(pipeline)
    state = loader.init_state(jax.random.PRNGKey(3))

    batch, state, mask = loader.next(state)
    assert batch["image"].shape == (128, 2, 2, 1)
    assert mask.shape == (128,)


def test_readme_host_callback_example_runs(capsys):
    train_data = MNISTDataset(split="train").as_array_dict()

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
            ArraySampleSource(train_data, ordering="shuffle"),
            BatchTransform(batch_size=128),
            HostCallbackTransform(fn=log_loss),
        ],
    )

    state = loader.init_state(jax.random.PRNGKey(4))
    batch, state, mask = loader.next(state)

    assert mask.shape == (128,)
    assert "loss:" in capsys.readouterr().out
    assert batch.keys() == {"image", "label"}


def test_readme_rl_example_runs():
    env = gymnax.environments.classic_control.cartpole.CartPole()
    env_params = env.default_params

    def policy_step(obs, policy_state, new_episode, key):
        del new_episode
        logits = obs @ policy_state["params"]
        action = jax.random.categorical(key, logits=logits)
        return action, policy_state

    policy_state = {
        "params": jnp.zeros((4, 2)),
        "recurrent_state": jnp.zeros((3,)),
    }

    source = GymnaxSource(
        env=env,
        env_params=env_params,
        policy_step_fn=policy_step,
        policy_state_template=policy_state,
        steps_per_epoch=16,
    )
    pipeline = [
        source,
        BatchTransform(batch_size=16, drop_last=True),
    ]
    loader = DataLoader(pipeline)
    state = loader.init_state(jax.random.PRNGKey(0))
    state = set_loader_policy_state(state, policy_state)

    batch, state, mask = loader.next(state)
    assert batch["state"].shape[0] == 16
    assert mask.shape == (16,)

    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    batched_state = jax.vmap(source.init_state)(keys)
    batched_state = jax.vmap(lambda s: set_source_policy_state(s, policy_state))(batched_state)
    transition, batched_mask, _ = jax.vmap(source.next)(batched_state)
    assert transition["action"].shape[0] == 4
    assert batched_mask.shape[0] == 4