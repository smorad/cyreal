"""Policy-gradient training on Gymnax CartPole using cyreal with multiple environments."""
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
    parser.add_argument("--rollout-length", type=int, default=512, help="Steps per policy rollout.")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Policy MLP width.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Reward discount factor.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for policy and environment.")
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


def reward_to_go(rewards: jax.Array, dones: jax.Array, gamma: float) -> jax.Array:
    dones_f = dones.astype(jnp.float32)

    def scan_fn(carry, inputs):
        reward, done = inputs
        new_carry = reward + gamma * carry * (1.0 - done)
        return new_carry, new_carry

    _, reversed_returns = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(rewards[0]),
        (rewards[::-1], dones_f[::-1]),
    )
    return reversed_returns[::-1]


class VectorEnvWrapper:
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs
        self.v_reset = jax.vmap(env.reset, in_axes=(0, None))
        self.v_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def reset(self, key, params):
        keys = jax.random.split(key, self.num_envs)
        return self.v_reset(keys, params)

    def step(self, key, state, action, params):
        keys = jax.random.split(key, self.num_envs)
        return self.v_step(keys, state, action, params)


def build_env(num_envs: int) -> Tuple[VectorEnvWrapper, Any]:
    base_env = gymnax.environments.classic_control.cartpole.CartPole()
    env = purerl.LogWrapper(base_env)
    vector_env = VectorEnvWrapper(env, num_envs)
    return vector_env, base_env.default_params


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


def make_policy_state(params: Any) -> dict[str, Any]:
    return {"params": params}


def train(args: argparse.Namespace) -> None:
    env, env_params = build_env(args.num_envs)

    rng = jax.random.PRNGKey(args.seed)
    policy_key, loader_key = jax.random.split(rng)

    # We need the base env to get observation and action spaces
    base_env = gymnax.environments.classic_control.cartpole.CartPole()
    obs_dim = int(base_env.observation_space(env_params).shape[0])
    action_dim = int(base_env.action_space(env_params).n)

    policy = PolicyNetwork(obs_dim, args.hidden_size, action_dim, key=policy_key)
    policy_params, policy_static = eqx.partition(policy, eqx.is_array)
    policy_state_template = make_policy_state(policy_params)

    def act(obs, policy_state, new_episode, key):
        keys = jax.random.split(key, args.num_envs)
        
        def single_act(o, p_state, new_ep, k):
            model = eqx.combine(p_state["params"], policy_static)
            logits = model(o)
            action = jax.random.categorical(k, logits=logits)
            return action, p_state
            
        new_ep_axis = 0 if getattr(new_episode, "ndim", 0) > 0 else None
        action, _ = jax.vmap(single_act, in_axes=(0, None, new_ep_axis, 0))(
            obs, policy_state, new_episode, keys
        )
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

    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(policy_params)

    @jax.jit
    def update(params, opt_state, batch):
        def loss_fn(params, batch):
            model = eqx.combine(params, policy_static)
            # batch["state"] has shape (rollout_length, num_envs, obs_dim)
            # We vmap over both rollout_length and num_envs
            logits = jax.vmap(jax.vmap(model))(batch["state"])
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            indices = batch["action"].astype(jnp.int32)[..., None]
            log_probs = jnp.take_along_axis(log_probs, indices, axis=-1)[..., 0]
            
            # Calculate returns for each environment independently
            # vmap over the num_envs dimension (axis 1)
            v_reward_to_go = jax.vmap(reward_to_go, in_axes=(1, 1, None), out_axes=1)
            returns = v_reward_to_go(batch["reward"], batch["done"], args.gamma)
            
            centered = returns - jnp.mean(returns)
            return -jnp.mean(log_probs * centered)

        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for epoch in tqdm.trange(1, args.epochs + 1):
        batch, loader_state, _ = loader.next(loader_state)
        policy_params, opt_state, loss = update(policy_params, opt_state, batch)
        loader_state = set_loader_policy_state(loader_state, make_policy_state(policy_params))

        mean_return, mean_length = summarize_episode_metrics(batch["info"])
        if mean_return is None:
            mean_return = float(jnp.sum(batch["reward"]) / args.num_envs)
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
