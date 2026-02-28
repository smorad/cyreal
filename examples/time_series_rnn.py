"""Train a simple RNN on cyreal's built-in time-series datasets.

Usage:
  python examples/time_series_rnn.py --epochs 20 --batch-size 64 --dataset daily-min
"""

from __future__ import annotations

import argparse
from time import time
from typing import Literal, Tuple, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm

from cyreal.transforms import (
    BatchTransform,
    TimeSeriesBatchTransform,
)
from cyreal.loader import DataLoader
from cyreal.datasets import DailyMinTemperaturesDataset, SunspotsDataset

SequenceBatch = dict[str, jax.Array]
DatasetCls = Type[DailyMinTemperaturesDataset | SunspotsDataset]

DATASETS: dict[str, DatasetCls] = {
    "daily-min": DailyMinTemperaturesDataset,
    "sunspots": SunspotsDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        default="daily-min",
        help="Sequence dataset to train on.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=30,
        help="History window length for each sample.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=1,
        help="Number of steps to predict ahead.",
    )
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size of the simple RNN.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.85,
        help="Fraction of the series used for the training split.",
    )
    parser.add_argument(
        "--prefetch-size",
        type=int,
        default=256,
        help="Samples to prefetch inside DiskSampleSource.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional directory for caching raw CSV files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for both model and loader state.",
    )
    return parser.parse_args()


def _build_pipeline(
    dataset_cls: DatasetCls,
    *,
    split: Literal["train", "test"],
    ordering: Literal["sequential", "shuffle"],
    args: argparse.Namespace,
) -> DataLoader:
    source = dataset_cls.make_disk_source(
        split=split,
        ordering=ordering,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        train_fraction=args.train_fraction,
        cache_dir=args.cache_dir,
        prefetch_size=args.prefetch_size,
    )
    pipeline = [
        source,
        BatchTransform(batch_size=args.batch_size, pad_last_batch=True),
        TimeSeriesBatchTransform(sequence_key="context", mode="batched"),
    ]
    return DataLoader(pipeline=pipeline)


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    dataset_cls = DATASETS[args.dataset]
    train_loader = _build_pipeline(dataset_cls, split="train", ordering="shuffle", args=args)
    test_loader = _build_pipeline(dataset_cls, split="test", ordering="sequential", args=args)
    return train_loader, test_loader


class SimpleRNN(eqx.Module):
    Wx: jax.Array
    Wh: jax.Array
    b: jax.Array
    readout: eqx.nn.Linear

    def __init__(self, input_size: int, hidden_size: int, output_size: int, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        limit = 1.0 / jnp.sqrt(max(input_size, 1))
        self.Wx = jax.random.uniform(k1, (input_size, hidden_size), minval=-limit, maxval=limit)
        self.Wh = jax.random.uniform(k2, (hidden_size, hidden_size), minval=-limit, maxval=limit)
        self.b = jnp.zeros((hidden_size,), dtype=jnp.float32)
        self.readout = eqx.nn.Linear(hidden_size, output_size, key=k3)

    def __call__(self, sequence: jax.Array) -> jax.Array:
        def step(h, x):
            h_new = jnp.tanh(jnp.dot(x, self.Wx) + jnp.dot(h, self.Wh) + self.b)
            return h_new, None

        hidden0 = jnp.zeros(self.b.shape, dtype=sequence.dtype)
        final_hidden, _ = jax.lax.scan(step, hidden0, sequence)
        return self.readout(final_hidden)


def make_epoch_fn(loader: DataLoader, optimizer: optax.GradientTransformation, model_static):
    def loss_fn(params, batch: SequenceBatch, mask: jax.Array) -> jax.Array:
        model = eqx.combine(params, model_static)
        preds = eqx.filter_vmap(model)(batch["context"])
        targets = batch["target"]
        per_example = jnp.mean((preds - targets) ** 2, axis=-1)
        weights = mask.astype(jnp.float32)
        return jnp.sum(per_example * weights) / jnp.maximum(jnp.sum(weights), 1.0)

    def update_step(carry, batch, mask):
        params, opt_state = carry
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, batch, mask)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = eqx.apply_updates(params, updates)
        return (params, opt_state), loss

    @jax.jit
    def epoch_fn(params, opt_state, loader_state):
        loader_state, (params, opt_state), losses = loader.scan_epoch(
            loader_state,
            (params, opt_state),
            update_step,
        )
        return params, opt_state, loader_state, losses

    return epoch_fn


def evaluate(model_params, model_static, loader: DataLoader, loader_state):
    model = eqx.combine(model_params, model_static)
    iterator = loader.iterate(loader_state)
    total_loss = 0.0
    total_weight = 0.0
    for batch, mask in iterator:
        preds = eqx.filter_vmap(model)(batch["context"])
        per_example = jnp.mean((preds - batch["target"]) ** 2, axis=-1)
        weights = mask.astype(jnp.float32)
        total_loss += float(jnp.sum(per_example * weights))
        total_weight += float(jnp.sum(weights))
    mean_loss = total_loss / max(total_weight, 1.0)
    return mean_loss, iterator.state


def main() -> None:
    args = parse_args()
    train_loader, test_loader = build_loaders(args)

    rng = jax.random.PRNGKey(args.seed)
    model_key, train_key, test_key = jax.random.split(rng, 3)
    train_state = train_loader.init_state(train_key)
    test_state = test_loader.init_state(test_key)

    context_spec = train_loader._source.element_spec()["context"]  # noqa: SLF001 - example script
    feature_size = int(context_spec.shape[-1])
    model = SimpleRNN(feature_size, args.hidden_size, args.prediction_length, key=model_key)
    model_params, model_static = eqx.partition(model, eqx.is_array)
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(model_params)
    epoch_fn = make_epoch_fn(train_loader, optimizer, model_static)

    for epoch in tqdm.trange(1, args.epochs + 1):
        start_t = time()
        model_params, opt_state, train_state, losses = epoch_fn(
            model_params,
            opt_state,
            train_state,
        )
        train_loss = float(jnp.mean(losses))
        eval_loss, test_state = evaluate(model_params, model_static, test_loader, test_state)
        end_t = time()
        tqdm.tqdm.write(
            f"Epoch {epoch}: train_mse={train_loss:.5f}, test_mse={eval_loss:.5f}, time={end_t - start_t:.2f}s"
        )


if __name__ == "__main__":
    main()
