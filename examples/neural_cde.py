"""Train a simple Neural CDE on cyreal's built-in time-series datasets.

Usage:
  python examples/neural_cde.py --epochs 20 --batch-size 64 --dataset daily-min
"""

from __future__ import annotations

import argparse
from time import time
from collections.abc import Callable
from typing import Literal, Tuple, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm
import diffrax
from cyreal.transforms import (
    BatchTransform,
    DevicePutTransform,
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
        DevicePutTransform(),
    ]
    return DataLoader(pipeline=pipeline)


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    dataset_cls = DATASETS[args.dataset]
    train_loader = _build_pipeline(dataset_cls, split="train", ordering="shuffle", args=args)
    test_loader = _build_pipeline(dataset_cls, split="test", ordering="sequential", args=args)
    return train_loader, test_loader


class NeuralCDE(eqx.Module):
    """
    Neural Controlled Differential Equation model.

    Usage
    - Provide `ts` and either a `diffrax` control path or cubic interpolation coeffs.
    - The model solves the induced ODE and applies a readout on the hidden state.
    """

    # Modules
    initial_cond_mlp: eqx.nn.MLP
    vf_mlp: eqx.nn.MLP
    readout_layer: eqx.nn.Linear
    cde_state_dim: int
    input_path_dim: int

    # Static configuration
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)

    # Solver configuration
    solver: diffrax.AbstractAdaptiveSolver = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        cde_state_dim: int,
        output_path_dim: int,
        init_hidden_dim: int,
        initial_cond_mlp_depth: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        *,
        key: jax.Array,
        readout_activation: Callable[[jax.Array], jax.Array] = lambda x: x,
        solver: diffrax.AbstractAdaptiveSolver = diffrax.Tsit5(),
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
            rtol=1e-2, atol=1e-3, dtmin=1e-6
        ),
        evolving_out: bool = False,
    ) -> None:
        if init_hidden_dim != cde_state_dim:
            raise ValueError(
                "This example expects init_hidden_dim == cde_state_dim so that the "
                "initial condition and vector field operate on the same hidden state size."
            )
        k1, k2, k3 = jax.random.split(key, 3)

        # Modules
        self.initial_cond_mlp = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=cde_state_dim,
            width_size=vf_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=jax.nn.softplus,
            key=k1,
        )
        self.vf_mlp = eqx.nn.MLP(
            in_size=cde_state_dim,
            out_size=cde_state_dim
            * input_path_dim,  # Shaped as such to reshape into (cde_state_dim, input_path_dim) matrix for dx/dt multiplication
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=jax.nn.softplus,
            key=k2,
        )
        self.readout_layer = eqx.nn.Linear(
            in_features=cde_state_dim,
            out_features=output_path_dim,
            use_bias=True,
            key=k3,
        )
        self.readout_activation = readout_activation
        self.cde_state_dim = cde_state_dim
        self.input_path_dim = input_path_dim

        # Static configuration
        self.evolving_out = evolving_out

        # Solver configuration
        self.solver = solver
        self.stepsize_controller = stepsize_controller

    def _apply_readout(self, hidden_states: jax.Array) -> jax.Array:
        """Apply readout to hidden states."""

        def apply_single(y: jax.Array) -> jax.Array:
            activation = self.readout_activation(self.readout_layer(y))
            return activation

        return jax.vmap(apply_single)(hidden_states)

    def __call__(
        self,
        control: jax.Array,
    ) -> jax.Array:
        """
        Forward pass.

        Given control path, build interpolation and solve the CDE.
        """
        # `control` is the observed path X(t) with shape (T, input_path_dim).
        # Use an index-based time grid; the dataset does not provide explicit timestamps.
        ts = jnp.arange(control.shape[0], dtype=jnp.float32)  # (T,)
        coeffs = diffrax.backward_hermite_coefficients(ts=ts, ys=control)
        interpolated_control = diffrax.CubicInterpolation(ts, coeffs)
        x0 = interpolated_control.evaluate(ts[0])  # (input_path_dim,)
        y0 = self.initial_cond_mlp(x0)  # (cde_state_dim,)

        def vf(t: diffrax._custom_types.RealScalarLike, y: jax.Array, args: object) -> jax.Array:
            del t, args
            return self.vf_mlp(y).reshape(self.cde_state_dim, self.input_path_dim)

        term = diffrax.ControlTerm(vf, interpolated_control).to_ode()
        saveat = diffrax.SaveAt(ts=ts)

        solution = diffrax.diffeqsolve(
            terms=term,
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=self.stepsize_controller,
            saveat=saveat,
        )

        assert solution.ys is not None

        if self.evolving_out:
            return self._apply_readout(solution.ys)

        return self.readout_activation(self.readout_layer(solution.ys[-1]))


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

    @eqx.filter_jit
    def predict(context: jax.Array) -> jax.Array:
        return eqx.filter_vmap(model)(context)

    iterator = loader.iterate(loader_state)
    total_loss = 0.0
    total_weight = 0.0
    for batch, mask in iterator:
        preds = predict(batch["context"])
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
    model = NeuralCDE(
        feature_size,
        args.hidden_size,
        args.prediction_length,
        init_hidden_dim=args.hidden_size,
        initial_cond_mlp_depth=2,
        vf_hidden_dim=args.hidden_size,
        vf_mlp_depth=2,
        key=model_key,
    )
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
