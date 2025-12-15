"""Policy-gradient training on Gymnax CartPole using cyreal."""
from __future__ import annotations

import argparse
from typing import Any, Tuple

import equinox as eqx
import gymnax
import jax
import jax.numpy as jnp
import optax
import tqdm
from gymnax.wrappers import purerl

from cyreal import BatchTransform, DataLoader, GymnaxSource
from cyreal.rl import set_loader_policy_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200, help="Optimization steps.")
    parser.add_argument("--rollout-length", type=int, default=2048, help="Steps per policy rollout.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Policy MLP width.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Reward discount factor.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for policy and environment.")
    return parser.parse_args()


class RNNPolicy(eqx.Module):
    input_layer: eqx.nn.Linear
    hidden_layer: eqx.nn.Linear
    head: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.input_layer = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
        self.hidden_layer = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=k2)
        self.head = eqx.nn.Linear(hidden_dim, output_dim, key=k3)
        self.norm = eqx.nn.LayerNorm((hidden_dim,), use_bias=False, use_weight=False)

    def __call__(self, obs: jax.Array, hidden: jax.Array) -> tuple[jax.Array, jax.Array]:
        preact = self.input_layer(obs) + self.hidden_layer(hidden)
        new_hidden = jnp.tanh(preact)
        logits = self.head(new_hidden)
        return logits, new_hidden


def reward_to_go(rewards: jax.Array, dones: jax.Array, gamma: float) -> jax.Array:
    dones_f = dones.astype(jnp.float32)

    def scan_fn(carry, inputs):
        reward, done = inputs
        new_carry = reward + gamma * carry * (1.0 - done)
        return new_carry, new_carry

    _, reversed_returns = jax.lax.scan(
        scan_fn,
        jnp.array(0.0, dtype=rewards.dtype),
        (rewards[::-1], dones_f[::-1]),
    )
    return reversed_returns[::-1]


def select_log_probs(logits: jax.Array, actions: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    indices = actions.astype(jnp.int32)[..., None]
    return jnp.take_along_axis(log_probs, indices, axis=-1)[..., 0]


def build_env() -> Tuple[gymnax.environments.environment.Environment, Any]:
    base_env = gymnax.environments.classic_control.cartpole.CartPole()
    env = purerl.LogWrapper(base_env)
    return env, env.default_params


def summarize_episode_metrics(info_tree: dict[str, jax.Array]) -> tuple[float | None, float | None]:
    returns = info_tree.get("returned_episode_returns")
    lengths = info_tree.get("returned_episode_lengths")
    flags = info_tree.get("returned_episode")
    if returns is None or lengths is None or flags is None:
        return None, None
    mask = jnp.asarray(flags, dtype=bool)
    if not jnp.any(mask):
        return None, None
    valid_returns = returns[mask]
    valid_lengths = lengths[mask]
    return float(jnp.mean(valid_returns)), float(jnp.mean(valid_lengths))


def main() -> None:
    args = parse_args()
    env, env_params = build_env()

    rng = jax.random.PRNGKey(args.seed)
    policy_key, loader_key = jax.random.split(rng)

    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)

    policy = RNNPolicy(obs_dim, args.hidden_size, action_dim, key=policy_key)
    policy_params, policy_static = eqx.partition(policy, eqx.is_array)

    def make_policy_state(params):
        return {
            "params": params,
            "hidden": jnp.zeros(args.hidden_size, dtype=jnp.float32),
        }

    def reset_hidden_if_needed(hidden, new_episode):
        flag = jnp.asarray(new_episode, dtype=jnp.bool_)
        return jnp.where(flag, jnp.zeros_like(hidden), hidden)

    def mask_cart_position(obs: jax.Array) -> jax.Array:
        obs = obs.at[0].set(0.0)
        obs = obs.at[2].set(0.0)
        return obs

    def policy_step(obs, policy_state, new_episode, key):
        model = eqx.combine(policy_state["params"], policy_static)
        hidden = reset_hidden_if_needed(policy_state["hidden"], new_episode)
        logits, new_hidden = model(mask_cart_position(obs), hidden)
        action = jax.random.categorical(key, logits=logits)
        next_state = {
            "params": policy_state["params"],
            "hidden": new_hidden,
        }
        return action, next_state

    def train_logits(params, observations, dones):
        model = eqx.combine(params, policy_static)
        done_flags = jnp.asarray(dones, dtype=jnp.bool_)
        new_episode = jnp.concatenate(
            [jnp.array([True], dtype=jnp.bool_), done_flags[:-1]],
            axis=0,
        )

        def step(hidden, inputs):
            obs, episode_reset = inputs
            hidden = reset_hidden_if_needed(hidden, episode_reset)
            logits, new_hidden = model(mask_cart_position(obs), hidden)
            return new_hidden, logits

        init_hidden = jnp.zeros(args.hidden_size, dtype=jnp.float32)
        _, logits = jax.lax.scan(step, init_hidden, (observations, new_episode))
        return logits

    source = GymnaxSource(
        env=env,
        env_params=env_params,
        policy_step_fn=policy_step,
        policy_state_template=make_policy_state(policy_params),
        steps_per_epoch=args.rollout_length,
    )
    pipeline = [
        source,
        BatchTransform(batch_size=args.rollout_length, drop_last=True),
    ]
    loader = DataLoader(pipeline=pipeline)
    loader_state = loader.init_state(loader_key)
    loader_state = set_loader_policy_state(loader_state, make_policy_state(policy_params))

    optimizer = optax.adamw(args.learning_rate)
    opt_state = optimizer.init(policy_params)

    def loss_fn(params, batch):
        logits = train_logits(params, batch["state"], batch["done"])
        log_probs = select_log_probs(logits, batch["action"])
        returns = reward_to_go(batch["reward"], batch["done"], args.gamma)
        centered = returns - jnp.mean(returns)
        return -jnp.mean(log_probs * centered)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    for epoch in tqdm.trange(1, args.epochs + 1):
        batch, loader_state, _ = loader.next(loader_state)
        loss, grads = loss_and_grad(policy_params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params=policy_params)
        policy_params = optax.apply_updates(policy_params, updates)
        loader_state = set_loader_policy_state(loader_state, make_policy_state(policy_params))

        mean_return, mean_length = summarize_episode_metrics(batch["info"])
        if mean_return is None:
            mean_return = float(jnp.sum(batch["reward"]))
        if mean_length is None:
            mean_length = float(args.rollout_length)
        tqdm.tqdm.write(
            f"Epoch {epoch}: loss={float(loss):.4f}, return={mean_return:.2f}, length={mean_length:.1f}"
        )


if __name__ == "__main__":
    main()
