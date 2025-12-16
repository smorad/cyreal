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

env = gymnax.environments.classic_control.cartpole.CartPole()
env_params = env.default_params

def act(obs, policy_state, new_episode, key):
    # policy_state can hold nn parameters and recurrent states
    # new_episode can be used to reset recurrent states
    # within the policy_state if needed.
    logits = obs @ policy_state["params"]
    action = jax.random.categorical(key, logits=logits)
    return action, policy_state

policy_state = {
    "params": jnp.zeros((4, 2)),
    "recurrent_state": jnp.zeros((3,)),
}

# GymnaxSource will call policy_step_fn to sample actions from the environment
source = GymnaxSource(
    env=env,
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
