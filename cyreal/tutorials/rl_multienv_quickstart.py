"""We provide utilities for interacting with `gymnax` environments. Take care to update the policy
state in the dataloader after updating your policy parameters.

```python
import gymnax
import jax
import jax.numpy as jnp

from cyreal.transforms import BatchTransform
from cyreal.loader import DataLoader
from cyreal.rl import set_loader_policy_state, set_source_policy_state
from cyreal.sources import GymnaxSource

base_env = gymnax.environments.classic_control.cartpole.CartPole()
env = base_env
env_params = base_env.default_params

num_envs = 8

# Create a wrapper to handle key splitting and vmapping
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

vector_env = VectorEnvWrapper(env, num_envs)

def act(obs, policy_state, new_episode, key):
    # Split key for each environment
    keys = jax.random.split(key, num_envs)
    
    def single_act(o, p_state, new_ep, k):
        # policy_state can hold nn parameters and recurrent states
        # new_episode can be used to reset recurrent states
        # within the policy_state if needed.
        logits = o @ p_state["params"]
        action = jax.random.categorical(k, logits=logits)
        return action, p_state
        
    # Handle scalar new_episode during shape inference
    new_ep_axis = 0 if getattr(new_episode, "ndim", 0) > 0 else None
    
    # vmap over batch dimensions: obs (0), policy_state (None), new_episode (new_ep_axis), keys (0)
    # Note: we do not vmap over policy_state, so it is returned unbatched.
    action, _ = jax.vmap(single_act, in_axes=(0, None, new_ep_axis, 0))(
        obs, policy_state, new_episode, keys
    )
    return action, policy_state

policy_state = {
    "params": jnp.zeros((4, 2)),
    "recurrent_state": jnp.zeros((3,)),
}

# GymnaxSource will call policy_step_fn to sample actions from the environment
source = GymnaxSource(
    env=vector_env,
    env_params=env_params,
    policy_step_fn=act,
    policy_state_template=policy_state,
    # Rollouts of length 32
    steps_per_epoch=32,
)
pipeline = [
    source,
    # Rollouts are length 32, batches are length 16
    # Two batches per epoch
    BatchTransform(batch_size=16),
]
loader = DataLoader(pipeline)
state = loader.init_state(jax.random.key(0))
state = set_loader_policy_state(state, policy_state)

# Perform training
for epoch in range(2):
    for _ in range(loader.steps_per_epoch):
        batch, state, mask = jax.jit(loader.next)(state)
        # Update the rollout policy parameters after each policy update
        policy_state.update({"params": jnp.ones((4, 2))})
        state = set_loader_policy_state(state, policy_state)
```
"""
