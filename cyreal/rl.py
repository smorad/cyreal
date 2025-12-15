"""Helpers for reinforcement learning utilities (e.g., Gymnax pipelines)."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax.numpy as jnp

from .loader import LoaderState

PyTree = Any


def _inject_policy_state(component: Any, policy_state: PyTree, new_episode) -> Any:
    if hasattr(component, "policy_state"):
        flag = new_episode
        if flag is None:
            flag = getattr(component, "new_episode", None)
            if flag is None:
                flag = jnp.array(True, dtype=jnp.bool_)
        else:
            flag = jnp.asarray(flag, dtype=jnp.bool_)
        return replace(component, policy_state=policy_state, new_episode=flag)
    if hasattr(component, "inner_state"):
        updated = _inject_policy_state(component.inner_state, policy_state, new_episode)
        return replace(component, inner_state=updated)
    raise AttributeError("No policy_state field found in provided component.")


def set_loader_policy_state(
    state: LoaderState,
    policy_state: PyTree,
    *,
    new_episode: bool | jnp.ndarray | None = None,
) -> LoaderState:
    """Return a LoaderState with policy_state replaced throughout the pipeline."""
    return LoaderState(inner_state=_inject_policy_state(state.inner_state, policy_state, new_episode))


def set_source_policy_state(state: Any, policy_state: PyTree, *, new_episode=None):
    """Return a source state (e.g., GymnaxSourceState) with a new policy state."""
    return _inject_policy_state(state, policy_state, new_episode)
