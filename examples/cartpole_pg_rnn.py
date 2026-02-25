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

from cyreal.transforms import BatchTransform
from cyreal.loader import DataLoader
from cyreal.sources import GymnaxSource
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


class MaskObservationWrapper:
    """Zeroes selected observation indices after env reset/step."""

    def __init__(self, env: gymnax.environments.environment.Environment, mask_indices: tuple[int, ...]):
        self.env = env
        self.mask_indices = tuple(mask_indices)

    def _mask(self, obs: jax.Array) -> jax.Array:
        if not self.mask_indices:
            return obs
        for idx in self.mask_indices:
            obs = obs.at[idx].set(0.0)
        return obs

    def reset(self, key, params):
        obs, state = self.env.reset(key, params)
        return self._mask(obs), state

    def step(self, key, state, action, params):
        obs, next_state, reward, done, info = self.env.step(key, state, action, params)
        return self._mask(obs), next_state, reward, done, info

    def observation_space(self, params):
        return self.env.observation_space(params)

    def action_space(self, params):
        return self.env.action_space(params)

    def state_space(self, params):
        return self.env.state_space(params)

    @property
    def default_params(self):
        return self.env.default_params

    def num_actions(self, params):
        return self.env.num_actions(params)

    def reward(self, state, action, params):
        return self.env.reward(state, action, params)

    def discount(self, state, action, next_state, params):
        return self.env.discount(state, action, next_state, params)

    def done(self, state, action, params):
        return self.env.done(state, action, params)


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


def build_env() -> Tuple[gymnax.environments.environment.Environment, Any]:
    base_env = gymnax.environments.classic_control.cartpole.CartPole()
    env = purerl.LogWrapper(base_env)
    env = MaskObservationWrapper(env, mask_indices=(0, 2))
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


def reset_hidden_if_needed(hidden: jax.Array, new_episode: jax.Array) -> jax.Array:
    flag = jnp.asarray(new_episode, dtype=jnp.bool_)
    return jnp.where(flag, jnp.zeros_like(hidden), hidden)


def make_policy_state(params: Any, hidden_size: int) -> dict[str, jax.Array]:
    return {
        "params": params,
        "hidden": jnp.zeros(hidden_size, dtype=jnp.float32),
    }


def policy_logits(model: RNNPolicy, hidden: jax.Array, obs: jax.Array, new_episode: jax.Array):
    hidden = reset_hidden_if_needed(hidden, new_episode)
    return model(obs, hidden)


def train(args: argparse.Namespace) -> None:
    env, env_params = build_env()

    rng = jax.random.PRNGKey(args.seed)
    policy_key, loader_key = jax.random.split(rng)

    obs_dim = int(env.observation_space(env_params).shape[0])
    action_dim = int(env.action_space(env_params).n)

    policy = RNNPolicy(obs_dim, args.hidden_size, action_dim, key=policy_key)
    policy_params, policy_static = eqx.partition(policy, eqx.is_array)
    policy_state_template = make_policy_state(policy_params, args.hidden_size)

    def act(obs, policy_state, new_episode, key):
        model = eqx.combine(policy_state["params"], policy_static)
        logits, new_hidden = policy_logits(model, policy_state["hidden"], obs, new_episode)
        action = jax.random.categorical(key, logits=logits)
        next_state = {
            "params": policy_state["params"],
            "hidden": new_hidden,
        }
        return action, next_state

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

    optimizer = optax.adamw(args.learning_rate)
    opt_state = optimizer.init(policy_params)

    def loss_fn(params, batch):
        model = eqx.combine(params, policy_static)
        done_flags = jnp.asarray(batch["done"], dtype=jnp.bool_)
        new_episode = jnp.concatenate(
            [jnp.array([True], dtype=jnp.bool_), done_flags[:-1]],
            axis=0,
        )

        def scan_step(hidden, inputs):
            obs, episode_reset = inputs
            logits, new_hidden = policy_logits(model, hidden, obs, episode_reset)
            return new_hidden, logits

        init_hidden = jnp.zeros(args.hidden_size, dtype=jnp.float32)
        _, logits = jax.lax.scan(scan_step, init_hidden, (batch["state"], new_episode))
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        indices = batch["action"].astype(jnp.int32)[..., None]
        log_probs = jnp.take_along_axis(log_probs, indices, axis=-1)[..., 0]
        returns = reward_to_go(batch["reward"], batch["done"], args.gamma)
        centered = returns - jnp.mean(returns)
        return -jnp.mean(log_probs * centered)

    loss_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def update(params, opt_state, batch):
        loss, grads = loss_and_grad(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for epoch in tqdm.trange(1, args.epochs + 1):
        batch, loader_state, _ = loader.next(loader_state)
        policy_params, opt_state, loss = update(policy_params, opt_state, batch)
        loader_state = set_loader_policy_state(
            loader_state, make_policy_state(policy_params, args.hidden_size)
        )

        mean_return, mean_length = summarize_episode_metrics(batch["info"])
        if mean_return is None:
            mean_return = float(jnp.sum(batch["reward"]))
        if mean_length is None:
            mean_length = float(args.rollout_length)
        tqdm.tqdm.write(
            f"Epoch {epoch}: loss={float(loss):.4f}, return={mean_return:.2f}, length={mean_length:.1f}"
        )


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
