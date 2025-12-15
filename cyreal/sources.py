"""Streaming dataset sources inspired by Grain's paradigm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.experimental import io_callback

PyTree = Any
StateT = TypeVar("StateT")


class Source(Protocol[StateT]):
    """Interface for stateful, JIT-friendly data streams.

    A source exposes fixed-length epochs (`steps_per_epoch`) and provides
    methods for initializing internal state and iteratively producing samples
    plus boolean masks that denote whether a sample is valid (e.g. under
    padding).
    """

    steps_per_epoch: int

    def init_state(self, key: jax.Array | None = None) -> StateT:
        """Return an initial state for the source.

        Args:
            key: Optional PRNG key used for randomized behavior such as
                shuffling. Implementations should fall back to a default key
                when ``None`` is provided.

        Returns:
            Backend-specific state object that must be passed to ``next``.
        """
        ...

    def next(self, state: StateT) -> tuple[PyTree, jax.Array, StateT]:
        """Advance the stream and return the next value.

        Args:
            state: Previously-initialized source state.

        Returns:
            Tuple ``(value, mask, new_state)`` where ``value`` is a PyTree of
            arrays, ``mask`` is a boolean array indicating whether the sample
            is valid, and ``new_state`` should be provided to the next call.
        """
        ...

    def element_spec(self) -> PyTree:
        """PyTree of :class:`jax.ShapeDtypeStruct` describing emitted samples."""
        ...


@jax.tree_util.register_pytree_node_class
@dataclass
class ArraySourceState:
    indices: jax.Array
    mask: jax.Array
    position: jax.Array
    key: jax.Array
    epoch: jax.Array

    def tree_flatten(self):
        return (self.indices, self.mask, self.position, self.key, self.epoch), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        indices, mask, position, key, epoch = children
        return cls(indices=indices, mask=mask, position=position, key=key, epoch=epoch)


@jax.tree_util.register_pytree_node_class
@dataclass
class DiskSourceState:
    indices: jax.Array
    position: jax.Array
    key: jax.Array
    epoch: jax.Array
    buffer: PyTree
    buffer_pos: jax.Array
    buffer_count: jax.Array

    def tree_flatten(self):
        buffer_leaves, buffer_def = tree_util.tree_flatten(self.buffer)
        children = (
            self.indices,
            self.position,
            self.key,
            self.epoch,
            *buffer_leaves,
            self.buffer_pos,
            self.buffer_count,
        )
        return children, buffer_def

    @classmethod
    def tree_unflatten(cls, buffer_def, children):
        indices, position, key, epoch, *rest = children
        buffer_leaf_count = buffer_def.num_leaves if buffer_def is not None else 0
        buffer_leaves = rest[:buffer_leaf_count]
        buffer = (
            tree_util.tree_unflatten(buffer_def, buffer_leaves)
            if buffer_def is not None
            else None
        )
        buffer_pos, buffer_count = rest[buffer_leaf_count:]
        return cls(
            indices=indices,
            position=position,
            key=key,
            epoch=epoch,
            buffer=buffer,
            buffer_pos=buffer_pos,
            buffer_count=buffer_count,
        )


@dataclass
class ArraySource(Source[ArraySourceState]):
    """Sample-level stream over an in-memory PyTree of arrays.

    Args:
        data: PyTree whose leaves are arrays with a leading sample dimension.
        ordering: Either ``"sequential"`` or ``"shuffle"``.
    """

    data: PyTree
    ordering: Literal["sequential", "shuffle"] = "shuffle"

    def __post_init__(self) -> None:
        leaves, self._treedef = tree_util.tree_flatten(self.data)
        if not leaves:
            raise ValueError("Data tree must contain at least one array.")

        first = leaves[0]
        self._num_samples = int(first.shape[0])
        for leaf in leaves[1:]:
            if leaf.shape[0] != self._num_samples:
                raise ValueError("All leaves must share the leading dimension.")
        if self._num_samples == 0:
            raise ValueError("Dataset cannot be empty.")

        self.steps_per_epoch = self._num_samples
        self._mask_template = jnp.ones(self._num_samples, dtype=bool)
        self._element_spec = tree_util.tree_map(
            lambda leaf: jax.ShapeDtypeStruct(shape=leaf.shape[1:], dtype=leaf.dtype),
            self.data,
        )

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def element_spec(self) -> PyTree:
        """Shape/dtype metadata describing samples produced by the source."""
        return self._element_spec

    def _build_epoch_indices(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        base = jnp.arange(self._num_samples)
        if self.ordering == "shuffle":
            base = jax.random.permutation(key, base)
        elif self.ordering != "sequential":
            raise ValueError(f"Unknown ordering '{self.ordering}'.")

        return base, self._mask_template

    def init_state(self, key: jax.Array | None = None) -> ArraySourceState:
        """Create the initial iteration state.

        Args:
            key: Optional PRNG key. Defaults to ``jax.random.PRNGKey(0)`` when
                omitted.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        key, perm_key = jax.random.split(key)
        indices, mask = self._build_epoch_indices(perm_key)
        position = jnp.array(0, dtype=jnp.int32)
        epoch = jnp.array(0, dtype=jnp.int32)
        return ArraySourceState(indices=indices, mask=mask, position=position, key=key, epoch=epoch)

    def next(self, state: ArraySourceState) -> tuple[PyTree, jax.Array, ArraySourceState]:
        """Return the next sample (with mask) and the advanced state."""
        index = jax.lax.dynamic_index_in_dim(state.indices, state.position, axis=0, keepdims=False)
        mask_value = jax.lax.dynamic_index_in_dim(state.mask, state.position, axis=0, keepdims=False)
        sample = tree_util.tree_map(
            lambda arr: jax.lax.dynamic_index_in_dim(arr, index, axis=0, keepdims=False),
            self.data,
        )

        next_position = state.position + 1

        def _reset_epoch(_: None):
            new_key, perm_key = jax.random.split(state.key)
            indices, mask = self._build_epoch_indices(perm_key)
            return ArraySourceState(
                indices=indices,
                mask=mask,
                position=jnp.array(0, dtype=jnp.int32),
                key=new_key,
                epoch=state.epoch + 1,
            )

        def _advance(_: None):
            return ArraySourceState(
                indices=state.indices,
                mask=state.mask,
                position=next_position,
                key=state.key,
                epoch=state.epoch,
            )

        need_reset = next_position >= state.indices.shape[0]
        new_state = jax.lax.cond(need_reset, _reset_epoch, _advance, operand=None)
        return sample, mask_value, new_state


@dataclass
class DiskSource(Source[DiskSourceState]):
    """Sample-level stream that loads items via a Python callback (disk, RPC, etc.).

    Use this if your dataset will not fit in system memory.

    Args:
        length: Number of samples in the dataset.
        sample_fn: Python callable that takes an integer index and returns a PyTree of arrays.
        sample_spec: Optional PyTree of `jax.ShapeDtypeStruct` describing the shape and dtype of samples.
            If not provided, the first sample (index 0) will be used to infer the spec.
        ordering: Sample ordering strategy, either 'sequential' or 'shuffle'. The shuffling occurs over the entire dataset, not within the prefetch buffer.
        prefetch_size: Number of samples to prefetch into a JAX array buffer. Set this larger to achieve better throughput at the cost of more memory usage.
    """

    length: int
    sample_fn: Callable[[int], PyTree]
    sample_spec: PyTree | None = None
    ordering: Literal["sequential", "shuffle"] = "shuffle"
    prefetch_size: int = 64

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("Dataset cannot be empty.")
        if self.prefetch_size <= 0:
            raise ValueError("prefetch_size must be positive.")

        if self.sample_spec is None:
            example = self.sample_fn(0)

            def _to_spec(leaf):
                arr = np.asarray(leaf)
                return jax.ShapeDtypeStruct(shape=arr.shape, dtype=arr.dtype)

            self.sample_spec = tree_util.tree_map(_to_spec, example)

        leaves = tree_util.tree_leaves(self.sample_spec)
        if not leaves:
            raise ValueError("element_spec must include at least one leaf.")
        for leaf in leaves:
            if not isinstance(leaf, jax.ShapeDtypeStruct):
                raise TypeError("element_spec leaves must be jax.ShapeDtypeStruct instances.")

        self._num_samples = int(self.length)
        self.steps_per_epoch = self._num_samples
        self._element_spec = self.sample_spec
        self.prefetch_size = int(self.prefetch_size)

        def _zeros(spec: jax.ShapeDtypeStruct):
            return np.zeros(spec.shape, dtype=np.dtype(spec.dtype))

        def _buffer_shape(spec: jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(
                shape=(self.prefetch_size, *spec.shape),
                dtype=spec.dtype,
            )

        self._zero_sample = tree_util.tree_map(_zeros, self.sample_spec)
        self._chunk_spec = tree_util.tree_map(_buffer_shape, self.sample_spec)
        self._buffer_template = tree_util.tree_map(
            lambda spec: jnp.zeros((self.prefetch_size, *spec.shape), dtype=spec.dtype),
            self.sample_spec,
        )

    def element_spec(self) -> PyTree:
        """Shape/dtype metadata describing samples produced by the source."""
        return self._element_spec

    def _build_epoch_indices(self, key: jax.Array) -> jax.Array:
        base = jnp.arange(self._num_samples)
        if self.ordering == "shuffle":
            base = jax.random.permutation(key, base)
        elif self.ordering != "sequential":
            raise ValueError(f"Unknown ordering '{self.ordering}'.")
        return base

    def init_state(self, key: jax.Array | None = None) -> DiskSourceState:
        """Build the starting state, optionally seeding randomness with ``key``."""
        if key is None:
            key = jax.random.PRNGKey(0)
        key, perm_key = jax.random.split(key)
        indices = self._build_epoch_indices(perm_key)
        position = jnp.array(0, dtype=jnp.int32)
        epoch = jnp.array(0, dtype=jnp.int32)
        return DiskSourceState(
            indices=indices,
            position=position,
            key=key,
            epoch=epoch,
            buffer=self._buffer_template,
            buffer_pos=jnp.array(0, dtype=jnp.int32),
            buffer_count=jnp.array(0, dtype=jnp.int32),
        )

    def _chunk_callback(self, indices: np.ndarray, mask: np.ndarray) -> PyTree:
        idx_array = np.asarray(indices, dtype=np.int64)
        mask_array = np.asarray(mask, dtype=bool)
        samples: list[PyTree] = []
        for keep, idx in zip(mask_array, idx_array):
            if keep:
                samples.append(self.sample_fn(int(idx)))
            else:
                samples.append(self._zero_sample)
        return tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *samples)

    def _maybe_reset_epoch(self, state: DiskSourceState) -> DiskSourceState:
        def _reset(state: DiskSourceState):
            new_key, perm_key = jax.random.split(state.key)
            indices = self._build_epoch_indices(perm_key)
            return DiskSourceState(
                indices=indices,
                position=jnp.array(0, dtype=jnp.int32),
                key=new_key,
                epoch=state.epoch + 1,
                buffer=self._buffer_template,
                buffer_pos=jnp.array(0, dtype=jnp.int32),
                buffer_count=jnp.array(0, dtype=jnp.int32),
            )

        return jax.lax.cond(state.position >= self._num_samples, _reset, lambda s: s, state)

    def _maybe_refill_buffer(self, state: DiskSourceState) -> DiskSourceState:
        def _needs(state: DiskSourceState):
            return jnp.logical_or(state.buffer_count == 0, state.buffer_pos >= state.buffer_count)

        def _refill(state: DiskSourceState):
            refreshed = self._maybe_reset_epoch(state)
            remaining = self._num_samples - refreshed.position
            chunk = jnp.minimum(remaining, self.prefetch_size)
            chunk = jnp.maximum(chunk, 0)
            chunk = chunk.astype(jnp.int32)
            offsets = jnp.arange(self.prefetch_size, dtype=jnp.int32)
            gather_positions = jnp.minimum(
                refreshed.position + offsets,
                refreshed.indices.shape[0] - 1,
            )
            chunk_indices = jax.vmap(
                lambda idx: jax.lax.dynamic_index_in_dim(
                    refreshed.indices, idx, axis=0, keepdims=False
                )
            )(gather_positions)
            valid_mask = offsets < chunk
            buffer = io_callback(self._chunk_callback, self._chunk_spec, chunk_indices, valid_mask)
            new_position = refreshed.position + chunk
            return DiskSourceState(
                indices=refreshed.indices,
                position=new_position,
                key=refreshed.key,
                epoch=refreshed.epoch,
                buffer=buffer,
                buffer_pos=jnp.array(0, dtype=jnp.int32),
                buffer_count=chunk,
            )

        return jax.lax.cond(_needs(state), _refill, lambda s: s, state)

    def next(self, state: DiskSourceState) -> tuple[PyTree, jax.Array, DiskSourceState]:
        """Return buffered sample, all-True mask, and updated state."""
        state = self._maybe_refill_buffer(state)
        sample = tree_util.tree_map(
            lambda buf: jax.lax.dynamic_index_in_dim(
                buf, state.buffer_pos, axis=0, keepdims=False
            ),
            state.buffer,
        )
        mask_value = jnp.array(True, dtype=bool)
        new_state = DiskSourceState(
            indices=state.indices,
            position=state.position,
            key=state.key,
            epoch=state.epoch,
            buffer=state.buffer,
            buffer_pos=state.buffer_pos + 1,
            buffer_count=state.buffer_count,
        )
        return sample, mask_value, new_state




@jax.tree_util.register_pytree_node_class
@dataclass
class GymnaxSourceState:
    env_state: PyTree
    obs: PyTree
    key: jax.Array
    step: jax.Array
    epoch: jax.Array
    policy_state: PyTree | None = None
    new_episode: jax.Array | None = None

    def tree_flatten(self):
        return (
            self.env_state,
            self.obs,
            self.key,
            self.step,
            self.epoch,
            self.policy_state,
            self.new_episode,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        env_state, obs, key, step, epoch, policy_state, new_episode = children
        return cls(
            env_state=env_state,
            obs=obs,
            key=key,
            step=step,
            epoch=epoch,
            policy_state=policy_state,
            new_episode=new_episode,
        )


@dataclass
class GymnaxSource(Source[GymnaxSourceState]):
    """Stream transitions by rolling out a Gymnax environment with a policy.
    
    Useful for reinforcement learning.

    Args:
        env: Gymnax environment instance.
        env_params: Parameters to pass to the environment's reset and step functions.
        policy_step_fn: Callable that takes (observation, policy_state, new_episode, key) and
            returns (action, new_policy_state).
        policy_state_template: Example PyTree carrying everything required by
            ``policy_step_fn`` (for example, policy parameters and recurrent
            carries). This template is used only to infer the element spec; callers
            are responsible for injecting a real policy state into the loader
            state before calling ``next``.
        steps_per_epoch: Number of environment steps per epoch for a single environment.
    """

    env: Any
    env_params: Any
    policy_step_fn: Callable[[PyTree, PyTree, jax.Array, jax.Array], tuple[PyTree, PyTree]]
    policy_state_template: PyTree | None = None
    steps_per_epoch: int = 1024

    def __post_init__(self) -> None:
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive.")
        if self.policy_state_template is None:
            raise ValueError("GymnaxSource requires a policy_state_template for shape inference.")

        def _sample(key, policy_state):
            obs, env_state = self.env.reset(key, self.env_params)
            action, next_policy_state = self.policy_step_fn(
                obs,
                policy_state,
                jnp.array(True, dtype=jnp.bool_),
                key,
            )
            next_obs, _, reward, done, info = self.env.step(
                key,
                env_state,
                action,
                self.env_params,
            )
            transition = {
                "state": obs,
                "action": action,
                "reward": reward,
                "next_state": next_obs,
                "done": done,
                "info": info,
            }
            return transition, next_policy_state

        shaped, _ = jax.eval_shape(_sample, jax.random.PRNGKey(0), self.policy_state_template)
        self._element_spec = tree_util.tree_map(
            lambda arr: jax.ShapeDtypeStruct(shape=arr.shape, dtype=arr.dtype), shaped
        )
        self.policy_state_template = None

    def element_spec(self) -> PyTree:
        """Shape/dtype metadata describing Gymnax transitions."""
        return self._element_spec

    def init_state(self, key: jax.Array | None = None) -> GymnaxSourceState:
        """Return RNG-seeded environment + policy state for iteration."""
        if key is None:
            key = jax.random.PRNGKey(0)
        key, env_key = jax.random.split(key)
        obs, env_state = self.env.reset(env_key, self.env_params)
        return GymnaxSourceState(
            env_state=env_state,
            obs=obs,
            key=key,
            step=jnp.array(0, dtype=jnp.int32),
            epoch=jnp.array(0, dtype=jnp.int32),
            policy_state=None,
            new_episode=jnp.array(True, dtype=jnp.bool_),
        )

    def next(self, state: GymnaxSourceState) -> tuple[PyTree, jax.Array, GymnaxSourceState]:
        """Roll the environment forward one step and emit a transition."""
        key, policy_key, step_key, done_reset_key, epoch_reset_key = jax.random.split(state.key, 5)

        if state.policy_state is None:
            raise ValueError(
                "GymnaxSource state is missing `policy_state`; set it explicitly before calling `next`."
            )
        policy_state = state.policy_state

        if state.new_episode is None:
            raise ValueError("GymnaxSource state is missing `new_episode` flag.")

        action, updated_policy_state = self.policy_step_fn(
            state.obs,
            policy_state,
            state.new_episode,
            policy_key,
        )
        next_obs, next_env_state, reward, done, info = self.env.step(
            step_key,
            state.env_state,
            action,
            self.env_params,
        )

        transition = {
            "state": state.obs,
            "action": action,
            "reward": reward,
            "next_state": next_obs,
            "done": done,
            "info": info,
        }
        mask = jnp.array(True, dtype=bool)

        done_flag = jnp.asarray(done, dtype=bool)
        done_flag = jnp.reshape(done_flag, ())
        reset_obs, reset_env_state = self.env.reset(done_reset_key, self.env_params)

        cont_obs, cont_env_state = jax.lax.cond(
            done_flag,
            lambda _: (reset_obs, reset_env_state),
            lambda _: (next_obs, next_env_state),
            operand=None,
        )

        next_step = state.step + 1
        need_epoch_reset = next_step >= self.steps_per_epoch

        def _reset_epoch(_: None):
            epoch_obs, epoch_env_state = self.env.reset(epoch_reset_key, self.env_params)
            return GymnaxSourceState(
                env_state=epoch_env_state,
                obs=epoch_obs,
                key=key,
                step=jnp.array(0, dtype=jnp.int32),
                epoch=state.epoch + 1,
                policy_state=updated_policy_state,
                new_episode=jnp.array(True, dtype=jnp.bool_),
            )

        def _continue(_: None):
            return GymnaxSourceState(
                env_state=cont_env_state,
                obs=cont_obs,
                key=key,
                step=next_step,
                epoch=state.epoch,
                policy_state=updated_policy_state,
                new_episode=done_flag,
            )

        new_state = jax.lax.cond(need_epoch_reset, _reset_epoch, _continue, operand=None)
        return transition, mask, new_state