"""Train a simple MNIST classifier with Equinox and cyreal.

Usage:
  python examples/mnist_equinox.py --epochs 3 --batch-size 256 --learning-rate 3e-4
The script runs on CPU or GPU depending on your JAX install.
"""
from __future__ import annotations

import argparse
from time import time
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm

from cyreal import ArraySampleSource, BatchTransform, DataLoader, DevicePutTransform, MNISTDataset, FlattenTransform

Batch = dict[str, jax.Array]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5, help="Number of passes over the training set.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for model and data loader.")
    return parser.parse_args()


def build_loader(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_data = MNISTDataset(split="train").as_array_dict()
    test_data = MNISTDataset(split="test").as_array_dict()
    pipeline = [
        ArraySampleSource(train_data, ordering="shuffle"),
        BatchTransform(batch_size=batch_size),
        FlattenTransform(data_key="image"),
        DevicePutTransform(),
    ]
    loader = DataLoader(pipeline=pipeline)
    test_pipeline = [
        ArraySampleSource(test_data, ordering="sequential"),
        BatchTransform(batch_size=test_data["image"].shape[0]),
        FlattenTransform(data_key="image"),
        DevicePutTransform(),
    ]
    test_loader = DataLoader(pipeline=test_pipeline)
    return loader, test_loader


def make_model(rng: jax.Array) -> eqx.Module:
    return eqx.nn.MLP(
        in_size=28 * 28,
        out_size=10,
        width_size=256,
        depth=2,
        key=rng,
    )


def make_epoch_fn(loader, optimizer, model_static):
    def loss_fn(params, batch, mask):
        model = eqx.combine(params, model_static)
        logits = eqx.filter_vmap(model)(batch["image"])
        labels = jax.nn.one_hot(batch["label"], 10)
        per_example = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
        valid = mask.astype(jnp.float32)
        return jnp.sum(per_example * valid) / jnp.maximum(valid.sum(), 1.0)

    def update_step(carry, batch, mask):
        params, opt_state = carry
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, batch, mask)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = eqx.apply_updates(params, updates)
        return (params, opt_state), loss

    @eqx.filter_jit
    def epoch_fn(params, opt_state, loader_state):
        loader_state, (params, opt_state), losses = loader.scan_epoch(
            loader_state,
            (params, opt_state),
            update_step,
        )
        return params, opt_state, loader_state, losses

    return epoch_fn


def evaluate(model_params, model_static, test_loader: DataLoader, test_loader_state) -> float:
    data, test_loader_state, _ = test_loader.next(test_loader_state)
    model = eqx.combine(model_params, model_static)
    logits = eqx.filter_jit(eqx.filter_vmap(model))(data["image"])
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == data["label"])
    return float(accuracy)


def main() -> None:
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)
    model_key, loader_key = jax.random.split(rng)

    loader, test_loader = build_loader(args.batch_size)
    loader_state = loader.init_state(loader_key)
    test_loader_state = test_loader.init_state(loader_key)

    model = make_model(model_key)
    model_params, model_static = eqx.partition(model, eqx.is_array)
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(model_params)
    epoch_fn = make_epoch_fn(loader, optimizer, model_static)

    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        start_t = time()
        model_params, opt_state, loader_state, losses = epoch_fn(
            model_params,
            opt_state,
            loader_state,
        )
        mean_loss = float(jnp.mean(losses))
        accuracy = evaluate(model_params, model_static, test_loader, test_loader_state)
        end_t = time()
        tqdm.tqdm.write(f"Epoch {epoch}: loss={mean_loss:.4f}, test_accuracy={accuracy:.3f}, time={end_t - start_t:.2f}s")


if __name__ == "__main__":
    main()