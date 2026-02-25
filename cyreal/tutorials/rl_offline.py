"""Offline RL tutorial using cyreal. We collect a dataset using a random policy, then use the dataset to train a policy using offline RL.
```
import gymnax
import jax
import jax.numpy as jnp

from cyreal.transforms import BatchTransform
from cyreal.loader import DataLoader
from cyreal.rl import set_loader_policy_state, set_source_policy_state
from cyreal.sources import GymnaxSource, ArraySource

env = gymnax.environments.classic_control.cartpole.CartPole()
env_params = env.default_params

def act(obs, policy_state, new_episode, key):
    # policy_state can hold nn parameters and recurrent states
    # new_episode can be used to reset recurrent states
    # within the policy_state if needed.
    logits = obs @ policy_state["params"]
    action = jax.random.categorical(key, logits=logits)
    return action, policy_state

behavior_policy_state = {
    "params": jnp.zeros((4, 2)),
    "recurrent_state": jnp.zeros((3,)),
}

# GymnaxSource will call policy_step_fn to sample actions from the environment
source = GymnaxSource(
    env=env,
    env_params=env_params,
    policy_step_fn=act,
    policy_state_template=behavior_policy_state,
    steps_per_epoch=32,
)
pipeline = [
    source,
    # Two batches per epoch
    BatchTransform(batch_size=16),
]
loader = DataLoader(pipeline)
state = loader.init_state(jax.random.key(0))
state = set_loader_policy_state(state, behavior_policy_state)

# Collect dataset
dataset = []
for epoch in range(2):
    for _ in range(loader.steps_per_epoch):
        batch, state, mask = jax.jit(loader.next)(state)
        dataset.append(batch)
        # Update the rollout policy parameters after each policy update
        behavior_policy_state.update({"params": jnp.ones((4, 2))})
        state = set_loader_policy_state(state, behavior_policy_state)

# Convert to dict, you can save this as HDF5 or numpy files for later use
dataset = jax.tree_util.tree_map(lambda *arrays: jnp.concatenate(arrays, axis=0), *dataset)

# Save to disk
import numpy as np
import os
np.savez_compressed("offline_dataset.npz", **dataset)

# Load from disk
loaded = np.load("offline_dataset.npz")
dataset = {key: jnp.array(loaded[key]) for key in loaded.files}
os.remove("offline_dataset.npz")

# Now do offline RL
learned_policy_state = {
    "params": jnp.zeros((4, 2)),
}

# We can reload the collected dataset as an ArraySource
source = ArraySource(
    data=dataset,
    ordering="shuffle",
)
pipeline = [
    source,
    BatchTransform(batch_size=4),
]
offline_loader = DataLoader(pipeline)
state = offline_loader.init_state(jax.random.key(0))

for epoch in range(2):
    for _ in range(offline_loader.steps_per_epoch):
        batch, state, mask = jax.jit(offline_loader.next)(state)
        learned_policy_state.update({"params": jnp.ones((4, 2))})
```"""