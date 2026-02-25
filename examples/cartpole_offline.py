"""Offline RL tutorial using cyreal."""
from __future__ import annotations

import argparse
import os
from typing import Any, Tuple

import equinox as eqx
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from gymnax.wrappers import purerl

from cyreal.transforms import BatchTransform
from cyreal.loader import DataLoader
from cyreal.sources import GymnaxSource, ArraySource
from cyreal.rl import set_loader_policy_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50, help="Optimization steps.")
    parser.add_argument("--rollout-length", type=int, default=512, help="Steps per policy rollout.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Policy MLP width.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Reward discount factor.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for policy and environment.")
    parser.add_argument("--dataset-path", type=str, default="cartpole_dataset.npz", help="Path to save/load dataset.")
    parser.add_argument("--collect-only", action="store_true", help="Only collect data, do not train offline.")
    parser.add_argument("--train-only", action="store_true", help="Only train offline, do not collect data.")
    return parser.parse_args()


class PolicyNetwork(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, key: jax.Array):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=output_dim,
            width_size=hidden_dim,
            depth=2,
            key=key,
        )

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self.mlp(obs)


def build_env() -> Tuple[Any, Any]:
    base_env = gymnax.environments.classic_control.cartpole.CartPole()
    env = purerl.LogWrapper(base_env)
    return env, base_env.default_params


def make_policy_state(params: Any) -> dict[str, Any]:
    return {"params": params}


def collect_dataset(args: argparse.Namespace) -> None:
    print(f"Collecting dataset to {args.dataset_path}...")
    env, env_params = build_env()

    rng = jax.random.PRNGKey(args.seed)
    policy_key, loader_key = jax.random.split(rng)

    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)

    # We use a random policy for data collection
    policy = PolicyNetwork(obs_dim, args.hidden_size, action_dim, key=policy_key)
    policy_params, policy_static = eqx.partition(policy, eqx.is_array)
    policy_state_template = make_policy_state(policy_params)

    def act(obs, policy_state, new_episode, key):
        # Random actions for exploration
        action = jax.random.randint(key, shape=(), minval=0, maxval=action_dim)
        return action, policy_state

    source = GymnaxSource(
        env=env,
        env_params=env_params,
        policy_step_fn=act,
        policy_state_template=policy_state_template,
        steps_per_epoch=args.rollout_length,
    )
    pipeline = [
        source,
        BatchTransform(batch_size=args.rollout_length, drop_last=True),
    ]
    loader = DataLoader(pipeline=pipeline)
    loader_state = loader.init_state(loader_key)
    loader_state = set_loader_policy_state(loader_state, policy_state_template)

    all_batches = []
    for _ in tqdm.trange(args.epochs, desc="Collecting"):
        batch, loader_state, _ = jax.jit(loader.next)(loader_state)
        # Bring data to CPU
        batch_cpu = jax.device_get(batch)
        all_batches.append(batch_cpu)

    # Concatenate all epochs
    full_dataset = jax.tree_util.tree_map(lambda *arrays: np.concatenate(arrays, axis=0), *all_batches)
    
    # Drop info to avoid saving object arrays
    if "info" in full_dataset:
        del full_dataset["info"]
    
    # Save to disk
    np.savez_compressed(args.dataset_path, **full_dataset)
    print(f"Saved offline dataset to {args.dataset_path} with {full_dataset['state'].shape[0]} transitions.")


def train_offline(args: argparse.Namespace) -> None:
    print(f"Training offline from {args.dataset_path}...")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}. Run without --train-only first.")

    # Load dataset and create loader
    loaded = np.load(args.dataset_path)
    dataset_dict = {key: jnp.array(loaded[key]) for key in loaded.files}
    source = ArraySource(
        data=dataset_dict,
        ordering="shuffle",
    )
    batch_size = 256
    pipeline = [
        source,
        BatchTransform(batch_size=batch_size, drop_last=True),
    ]
    loader = DataLoader(pipeline=pipeline)
    
    rng = jax.random.PRNGKey(args.seed + 1)
    policy_key, loader_key = jax.random.split(rng)
    loader_state = loader.init_state(loader_key)

    # Setup Policy
    base_env = gymnax.environments.classic_control.cartpole.CartPole()
    env_params = base_env.default_params
    obs_dim = int(base_env.observation_space(env_params).shape[0])
    action_dim = int(base_env.action_space(env_params).n)

    policy = PolicyNetwork(obs_dim, args.hidden_size, action_dim, key=policy_key)
    policy_params, policy_static = eqx.partition(policy, eqx.is_array)

    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(policy_params)

    # Simple Behavioral Cloning loss
    @jax.jit
    def update(params, opt_state, batch):
        def loss_fn(params, batch):
            model = eqx.combine(params, policy_static)
            logits = jax.vmap(model)(batch["state"])
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            indices = batch["action"].astype(jnp.int32)[..., None]
            action_log_probs = jnp.take_along_axis(log_probs, indices, axis=-1)[..., 0]
            return -jnp.mean(action_log_probs)

        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Train for a few epochs over the dataset
    steps_per_epoch = source.num_samples // batch_size
    for epoch in range(1, 11):
        epoch_loss = 0.0
        for _ in tqdm.trange(steps_per_epoch, desc=f"Epoch {epoch}", leave=False):
            batch, loader_state, _ = loader.next(loader_state)
            policy_params, opt_state, loss = update(policy_params, opt_state, batch)
            epoch_loss += loss
            
        print(f"Epoch {epoch}: BC Loss = {epoch_loss / steps_per_epoch:.4f}")


def main() -> None:
    args = parse_args()
    
    if not args.train_only:
        collect_dataset(args)
        
    if not args.collect_only:
        train_offline(args)


if __name__ == "__main__":
    main()
