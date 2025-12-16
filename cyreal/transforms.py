"""Composable transforms that wrap sources to build pipelines."""
from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.experimental import io_callback

from .sources import Source

PyTree = Any


def _zeros_from_spec(spec_tree: PyTree, batch_size: int):
    def _make(spec):
        if isinstance(spec, jax.ShapeDtypeStruct):
            return jnp.zeros((batch_size, *spec.shape), dtype=spec.dtype)
        raise TypeError("Element spec must be jax.ShapeDtypeStruct instances.")

    return tree_util.tree_map(_make, spec_tree)


def _write_buffer_impl(buffer: PyTree, value: PyTree, index: jax.Array) -> PyTree:
    def _update(buf, val):
        val_expanded = jnp.expand_dims(val, axis=0)
        return jax.lax.dynamic_update_index_in_dim(buf, val_expanded, index, axis=0)

    return tree_util.tree_map(_update, buffer, value)


class SourceTransform(Source, Protocol):
    """Base class for Transform implementations."""
    inner: Source


@dataclass
class BatchTransform:
    """Batch elements emitted by a source."""

    batch_size: int
    """Number of elements per batch."""
    drop_last: bool = False
    """If True, drop the final batch if it is less than batch_size."""
    pad_last_batch: bool = False
    """If True, pad the final batch with zeros if it is less than batch_size. Useful to prevent a second jit recompile. Make sure you use the mask to ignore padded values."""
    element_spec_override: PyTree | None = None

    def __call__(self, inner: Source) -> Source:
        return _BatchTransformSource(
            inner=inner,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            pad_last_batch=self.pad_last_batch,
            element_spec_override=self.element_spec_override,
        )



@jax.tree_util.register_pytree_node_class
@dataclass
class _BatchTransformState:
    inner_state: Any
    position_in_epoch: jax.Array

    def tree_flatten(self):
        return (self.inner_state, self.position_in_epoch), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        inner_state, position_in_epoch = children
        return cls(inner_state=inner_state, position_in_epoch=position_in_epoch)


@dataclass
class _BatchTransformSource(SourceTransform):
    """Accumulate `batch_size` items from an upstream source."""

    inner: Source
    batch_size: int
    drop_last: bool = False
    pad_last_batch: bool = True
    element_spec_override: PyTree | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self._samples_per_epoch = int(self.inner.steps_per_epoch)
        if self._samples_per_epoch <= 0:
            raise ValueError("Inner source must emit at least one item per epoch.")
        if self.drop_last and self._samples_per_epoch < self.batch_size:
            raise ValueError("drop_last=True would discard every batch for this source.")

        spec = self.element_spec_override or self.inner.element_spec()
        self._buffer_template = _zeros_from_spec(spec, self.batch_size)
        self._pad_sample = tree_util.tree_map(
            lambda spec_leaf: jnp.zeros(spec_leaf.shape, dtype=spec_leaf.dtype),
            spec,
        )
        self._mask_template = jnp.zeros(self.batch_size, dtype=bool)
        self._element_spec = tree_util.tree_map(
            lambda spec_leaf: jax.ShapeDtypeStruct(
                shape=(self.batch_size, *spec_leaf.shape), dtype=spec_leaf.dtype
            ),
            spec,
        )

        if self.drop_last:
            self.steps_per_epoch = self._samples_per_epoch // self.batch_size
        else:
            self.steps_per_epoch = math.ceil(self._samples_per_epoch / self.batch_size)

    def element_spec(self) -> PyTree:
        return self._element_spec

    def init_state(self, key: jax.Array) -> _BatchTransformState:
        return _BatchTransformState(
            inner_state=self.inner.init_state(key),
            position_in_epoch=jnp.array(0, dtype=jnp.int32),
        )

    def _write_slice(self, buffer: PyTree, value: PyTree, index: int) -> PyTree:
        return tree_util.tree_map(lambda buf, val: buf.at[index].set(val), buffer, value)

    def _maybe_skip_incomplete_epoch(self, state: _BatchTransformState) -> _BatchTransformState:
        if not self.drop_last:
            return state

        remaining = self._samples_per_epoch - state.position_in_epoch
        remaining = jnp.maximum(remaining, 0)
        need_skip = jnp.logical_and(remaining < self.batch_size, remaining > 0)

        def _drain(operand):
            state, remaining = operand

            def drain_body(carry, idx):
                inner_state, position = carry

                def _advance(carry):
                    inner_state, position = carry
                    _, _, inner_state = self.inner.next(inner_state)
                    return inner_state, position + 1

                return jax.lax.cond(
                    idx < remaining,
                    _advance,
                    lambda c: c,
                    operand=(inner_state, position),
                ), None

            (inner_state, _), _ = jax.lax.scan(
                drain_body,
                (state.inner_state, state.position_in_epoch),
                jnp.arange(self.batch_size),
            )
            return _BatchTransformState(
                inner_state=inner_state,
                position_in_epoch=jnp.array(0, dtype=jnp.int32),
            )

        return jax.lax.cond(need_skip, _drain, lambda operand: operand[0], (state, remaining))

    def next(self, state: _BatchTransformState):
        state = self._maybe_skip_incomplete_epoch(state)

        def body(carry, i):
            inner_state, position, buffer, mask = carry
            remaining = self._samples_per_epoch - position
            remaining = jnp.maximum(remaining, 0)

            def _consume(_: None):
                value, value_mask, next_inner_state = self.inner.next(inner_state)
                sample_mask = jnp.all(jnp.asarray(value_mask, dtype=jnp.bool_))
                updated_buffer = self._write_slice(buffer, value, i)
                updated_mask = mask.at[i].set(sample_mask)
                return (next_inner_state, position + 1, updated_buffer, updated_mask), None

            def _pad(_: None):
                updated_buffer = self._write_slice(buffer, self._pad_sample, i)
                updated_mask = mask.at[i].set(False)
                return (inner_state, position, updated_buffer, updated_mask), None

            return jax.lax.cond(remaining > 0, _consume, _pad, operand=None)

        (inner_state, position, batch, mask), _ = jax.lax.scan(
            body,
            (state.inner_state, state.position_in_epoch, self._buffer_template, self._mask_template),
            jnp.arange(self.batch_size),
        )

        wrapped_position = jnp.where(
            position >= self._samples_per_epoch,
            jnp.array(0, dtype=jnp.int32),
            position,
        )
        return batch, mask, _BatchTransformState(
            inner_state=inner_state,
            position_in_epoch=wrapped_position,
        )


@dataclass
class BufferTransform:
    """Cache streaming samples for later randomized replay.

    This transform stores scalar samples emitted by ``inner`` and, once the
    buffer is "prefilled", serves elements from the cache.
    The elements can be consumed directly, or batching/minibatching can 
    be handled by ``BatchTransform`` or similar
    utilities placed after the buffer in the pipeline.
    """

    capacity: int
    """Maximum number of samples to store."""
    prefill: int 
    """Number of valid samples to observe before consuming from the buffer."""
    sample_size: int = 1
    """Number of buffered samples emitted per `next` call."""
    mode: Literal["sequential", "shuffled"] = "sequential"
    """Sampling mode. `sequential` iterates through the buffer in order, while `shuffled` draws uniform random indices each step."""
    write_mode: Literal["fifo", "reservoir"] = "fifo"
    """Buffer write mode. `fifo` behaves like a ring buffer, `reservoir` performs uniform replacement akin to reservoir sampling once the buffer is full."""

    def __call__(self, inner: Source) -> Source:
        return _BufferTransformSource(
            inner=inner,
            capacity=self.capacity,
            prefill=self.prefill,
            sample_size=self.sample_size,
            mode=self.mode,
            write_mode=self.write_mode,
        )


@jax.tree_util.register_pytree_node_class
@dataclass
class _BufferState:
    inner_state: Any
    buffer: PyTree
    count: jax.Array
    write_index: jax.Array
    read_index: jax.Array
    seen: jax.Array
    rng: jax.Array

    def tree_flatten(self):
        buffer_leaves, buffer_def = tree_util.tree_flatten(self.buffer)
        children = (
            self.inner_state,
            *buffer_leaves,
            self.count,
            self.write_index,
            self.read_index,
            self.seen,
            self.rng,
        )
        return children, buffer_def

    @classmethod
    def tree_unflatten(cls, buffer_def, children):
        inner_state = children[0]
        leaf_count = buffer_def.num_leaves if buffer_def is not None else 0
        buffer_leaves = children[1 : 1 + leaf_count]
        buffer = (
            tree_util.tree_unflatten(buffer_def, buffer_leaves)
            if buffer_def is not None
            else None
        )
        count, write_index, read_index, seen, rng = children[1 + leaf_count :]
        return cls(
            inner_state=inner_state,
            buffer=buffer,
            count=count,
            write_index=write_index,
            read_index=read_index,
            seen=seen,
            rng=rng,
        )


@dataclass
class _BufferTransformSource(SourceTransform):
    inner: Source
    capacity: int
    prefill: int 
    sample_size: int
    mode: Literal["sequential", "shuffled"]
    write_mode: Literal["fifo", "reservoir"]

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive.")
        self._capacity = int(self.capacity)
        self._warmup = int(self.prefill)
        if self._warmup <= 0:
            raise ValueError("prefill must be positive.")
        if self._warmup > self._capacity:
            raise ValueError("prefill cannot exceed capacity.")
        self._sample_size = int(self.sample_size)
        if self._sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        if self._sample_size > self._capacity:
            raise ValueError("sample_size cannot exceed capacity.")
        if self._warmup < self._sample_size:
            raise ValueError("prefill must be at least sample_size.")
        if self.mode not in ("sequential", "shuffled"):
            raise ValueError("mode must be either 'sequential' or 'shuffled'.")
        self._mode = self.mode
        self._mode_is_sequential = self._mode == "sequential"
        if self.write_mode not in ("fifo", "reservoir"):
            raise ValueError("write_mode must be either 'fifo' or 'reservoir'.")
        self._write_mode_is_fifo = self.write_mode == "fifo"

        self.steps_per_epoch = self.inner.steps_per_epoch
        spec = self.inner.element_spec()
        leaves = tree_util.tree_leaves(spec)
        if not leaves:
            raise ValueError("BufferTransform requires at least one spec leaf.")
        for leaf in leaves:
            if not isinstance(leaf, jax.ShapeDtypeStruct):
                raise TypeError("element_spec leaves must be jax.ShapeDtypeStruct instances.")
        if self._sample_size == 1:
            self._element_spec = spec
        else:
            self._element_spec = tree_util.tree_map(
                lambda leaf: jax.ShapeDtypeStruct(
                    shape=(self._sample_size, *leaf.shape),
                    dtype=leaf.dtype,
                ),
                spec,
            )
        self._buffer_template = tree_util.tree_map(
            lambda leaf: jnp.zeros((self._capacity, *leaf.shape), dtype=leaf.dtype),
            spec,
        )
        self._chunk_template = tree_util.tree_map(
            lambda leaf: jnp.zeros((self._sample_size, *leaf.shape), dtype=leaf.dtype),
            spec,
        )
        self._mask_template = jnp.zeros(self._sample_size, dtype=jnp.bool_)
        self._write_buffer = jax.jit(_write_buffer_impl, donate_argnums=(0,))

    def element_spec(self) -> PyTree:
        return self._element_spec

    def init_state(self, key: jax.Array | None = None) -> _BufferState:
        if key is None:
            key = jax.random.PRNGKey(0)
        inner_state = self.inner.init_state(key)
        rng = jax.random.fold_in(key, 1)
        return _BufferState(
            inner_state=inner_state,
            buffer=self._buffer_template,
            count=jnp.array(0, dtype=jnp.int32),
            write_index=jnp.array(0, dtype=jnp.int32),
            read_index=jnp.array(0, dtype=jnp.int32),
            seen=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )

    def _gather_many(self, buffer: PyTree, indices: jax.Array) -> PyTree:
        def gather_leaf(buf):
            return jax.vmap(
                lambda idx: jax.lax.dynamic_index_in_dim(buf, idx, axis=0, keepdims=False)
            )(indices)

        return tree_util.tree_map(gather_leaf, buffer)

    def _format_chunk(self, chunk: PyTree) -> PyTree:
        if self._sample_size == 1:
            return tree_util.tree_map(lambda arr: jnp.squeeze(arr, axis=0), chunk)
        return chunk

    def _format_mask(self, mask: jax.Array) -> jax.Array:
        if self._sample_size == 1:
            return mask[0]
        return mask

    def next(self, state: _BufferState):
        value, mask, inner_state = self.inner.next(state.inner_state)
        mask_scalar = jnp.all(jnp.asarray(mask, dtype=jnp.bool_))

        one = jnp.array(1, dtype=jnp.int32)
        zero = jnp.array(0, dtype=jnp.int32)
        increment = jnp.where(mask_scalar, one, zero)
        rng, write_key, sample_key = jax.random.split(state.rng, 3)
        new_seen = state.seen + increment

        def _write(_: None):
            def _fifo(_: None):
                buffer = self._write_buffer(state.buffer, value, state.write_index)
                next_write = (state.write_index + 1) % self._capacity
                return buffer, next_write, jnp.array(True, dtype=jnp.bool_)

            def _reservoir(_: None):
                def _fill(_: None):
                    buffer = self._write_buffer(state.buffer, value, state.write_index)
                    next_write = (state.write_index + 1) % self._capacity
                    return buffer, next_write, jnp.array(True, dtype=jnp.bool_)

                def _replace(_: None):
                    maxval = jnp.maximum(new_seen, one)
                    rand_idx = jax.random.randint(write_key, (), minval=0, maxval=maxval)

                    def _commit(_: None):
                        buffer = self._write_buffer(state.buffer, value, rand_idx)
                        return buffer, state.write_index, jnp.array(True, dtype=jnp.bool_)

                    def _skip(_: None):
                        return state.buffer, state.write_index, jnp.array(False, dtype=jnp.bool_)

                    return jax.lax.cond(rand_idx < self._capacity, _commit, _skip, operand=None)

                return jax.lax.cond(state.count < self._capacity, _fill, _replace, operand=None)

            return jax.lax.cond(self._write_mode_is_fifo, _fifo, _reservoir, operand=None)

        def _skip(_: None):
            return state.buffer, state.write_index, jnp.array(False, dtype=jnp.bool_)

        updated_buffer, new_write, wrote_sample = jax.lax.cond(
            mask_scalar,
            _write,
            _skip,
            operand=None,
        )

        new_count = jnp.minimum(
            state.count
            + jnp.where(wrote_sample, one, zero),
            jnp.array(self._capacity, dtype=jnp.int32),
        )

        buffer_ready = state.count >= self._warmup

        def _from_buffer(_: None):
            def _sequential(_: None):
                idxs = (
                    state.read_index
                    + jnp.arange(self._sample_size, dtype=jnp.int32)
                ) % jnp.maximum(new_count, one)
                chunk = self._gather_many(updated_buffer, idxs)
                mask_vec = jnp.ones(self._sample_size, dtype=jnp.bool_)
                next_read = (
                    state.read_index + jnp.array(self._sample_size, dtype=jnp.int32)
                ) % jnp.maximum(new_count, one)
                return chunk, mask_vec, next_read

            def _shuffled(_: None):
                idxs = jax.random.randint(
                    sample_key,
                    (self._sample_size,),
                    minval=0,
                    maxval=jnp.maximum(new_count, one),
                )
                chunk = self._gather_many(updated_buffer, idxs)
                mask_vec = jnp.ones(self._sample_size, dtype=jnp.bool_)
                return chunk, mask_vec, state.read_index

            return jax.lax.cond(self._mode_is_sequential, _sequential, _shuffled, operand=None)

        def _passthrough(_: None):
            chunk = tree_util.tree_map(
                lambda template, val: template.at[0].set(val),
                self._chunk_template,
                value,
            )
            mask_vec = self._mask_template.at[0].set(mask_scalar)
            return chunk, mask_vec, state.read_index

        chunk, mask_vec, next_read_index = jax.lax.cond(
            buffer_ready,
            _from_buffer,
            _passthrough,
            operand=None,
        )

        formatted_chunk = self._format_chunk(chunk)
        formatted_mask = self._format_mask(mask_vec)

        next_state = _BufferState(
            inner_state=inner_state,
            buffer=updated_buffer,
            count=new_count,
            write_index=new_write,
            read_index=next_read_index,
            seen=new_seen,
            rng=rng,
        )
        return formatted_chunk, formatted_mask, next_state


@dataclass
class TimeSeriesBatchTransform:
    """Reshape batched sequences for time-series models."""

    sequence_key: str = "context"
    """Key within the batch mapping that stores time-series tensors."""
    mode: Literal["batched", "packed"] = "batched"
    """Either ``"batched"`` for (B, T, F) outputs or ``"packed"`` for flattened (B * T, F) representations."""
    sequence_start_key: str = "sequence_start"
    """Key used to store the "new sequence" flags when ``mode="packed"`` is selected."""

    def __call__(self, inner: Source) -> Source:
        return _TimeSeriesBatchTransformSource(
            inner=inner,
            sequence_key=self.sequence_key,
            mode=self.mode,
            sequence_start_key=self.sequence_start_key,
        )


@dataclass
class _TimeSeriesBatchTransformSource(SourceTransform):
    inner: Source
    sequence_key: str
    mode: Literal["batched", "packed"]
    sequence_start_key: str

    def __post_init__(self) -> None:
        if self.mode not in ("batched", "packed"):
            raise ValueError("mode must be either 'batched' or 'packed'.")

        self.steps_per_epoch = self.inner.steps_per_epoch
        spec = self.inner.element_spec()
        if not isinstance(spec, Mapping):
            raise TypeError("TimeSeriesBatchTransform expects mapping element specs.")
        if self.sequence_key not in spec:
            raise KeyError(f"Key '{self.sequence_key}' missing from element spec.")

        seq_spec = spec[self.sequence_key]
        if not isinstance(seq_spec, jax.ShapeDtypeStruct):
            raise TypeError("Sequence key spec must be a jax.ShapeDtypeStruct.")
        if len(seq_spec.shape) < 2:
            raise ValueError("Sequence tensors must include batch and time axes.")

        self._batch = int(seq_spec.shape[0])
        self._time = int(seq_spec.shape[1])
        feature_shape = seq_spec.shape[2:]
        if not feature_shape:
            feature_shape = (1,)
        self._feature_size = int(np.prod(feature_shape))
        self._sequence_dtype = seq_spec.dtype
        self._batched_shape = (self._batch, self._time, self._feature_size)
        self._packed_shape = (self._batch * self._time, self._feature_size)
        start = jnp.zeros(self._batch * self._time, dtype=jnp.bool_)
        start_indices = jnp.arange(0, self._batch * self._time, self._time)
        self._start_template = start.at[start_indices].set(True)

        updated_spec: dict[str, Any] = dict(spec)
        if self.mode == "batched":
            updated_spec[self.sequence_key] = jax.ShapeDtypeStruct(
                shape=self._batched_shape,
                dtype=self._sequence_dtype,
            )
        else:
            updated_spec[self.sequence_key] = jax.ShapeDtypeStruct(
                shape=self._packed_shape,
                dtype=self._sequence_dtype,
            )
            updated_spec[self.sequence_start_key] = jax.ShapeDtypeStruct(
                shape=(self._batch * self._time,),
                dtype=jnp.bool_,
            )
        self._element_spec = updated_spec

    def element_spec(self) -> PyTree:
        return self._element_spec

    def init_state(self, key: jax.Array | None = None):
        return self.inner.init_state(key)

    def _reshape_to_batched(self, sequence: jax.Array) -> jax.Array:
        return jnp.reshape(sequence, self._batched_shape)

    def next(self, state):
        batch, mask, inner_state = self.inner.next(state)
        if self.sequence_key not in batch:
            raise KeyError(f"Batch is missing '{self.sequence_key}'.")

        batched_sequence = self._reshape_to_batched(batch[self.sequence_key])
        if self.mode == "batched":
            updated_batch = _replace_mapping_item(batch, self.sequence_key, batched_sequence)
            return updated_batch, mask, inner_state

        packed_sequence = jnp.reshape(batched_sequence, self._packed_shape)
        repeated_mask = jnp.repeat(mask, self._time)
        sequence_start = self._start_template & repeated_mask
        updated_batch = _replace_mapping_item(batch, self.sequence_key, packed_sequence)
        updated_batch = _replace_mapping_item(updated_batch, self.sequence_start_key, sequence_start)
        return updated_batch, repeated_mask, inner_state


@dataclass
class MapTransform:
    """Apply a pure function to the input for map-style transforms."""

    fn: Callable[[PyTree, jax.Array], PyTree]
    """Function to apply to each batch. Receives (batch, mask) as arrays."""

    def __call__(self, inner: Source) -> Source:
        return _MapTransformSource(inner=inner, fn=self.fn)


@dataclass
class _MapTransformSource(SourceTransform):
    inner: Source
    fn: Callable[[PyTree, jax.Array], PyTree]

    def __post_init__(self) -> None:
        self.steps_per_epoch = self.inner.steps_per_epoch

    def element_spec(self) -> PyTree:
        return self.inner.element_spec()

    def init_state(self, key: jax.Array | None = None):
        return self.inner.init_state(key)

    def next(self, state):
        batch, mask, inner_state = self.inner.next(state)
        transformed = self.fn(batch, mask)
        return transformed, mask, inner_state


@dataclass
class HostCallbackTransform:
    """Invoke a host callback between jittable pipeline stages. Useful
    for logging or visualization.
    """

    fn: Callable[[PyTree, jax.Array], PyTree | None]
    """Function to invoke on host. Receives (batch, mask) as arrays."""
    element_spec_override: PyTree | None = None

    def __call__(self, inner: Source) -> Source:
        return _HostCallbackTransformSource(
            inner=inner,
            fn=self.fn,
            element_spec_override=self.element_spec_override,
        )


@dataclass
class _HostCallbackTransformSource(SourceTransform):
    inner: Source
    fn: Callable[[PyTree, jax.Array], PyTree | None]
    element_spec_override: PyTree | None = None

    def __post_init__(self) -> None:
        self.steps_per_epoch = self.inner.steps_per_epoch
        spec = self.element_spec_override or self.inner.element_spec()
        self._element_spec = spec
        self._callback_spec = tree_util.tree_map(
            lambda leaf: jax.ShapeDtypeStruct(shape=leaf.shape, dtype=leaf.dtype),
            spec,
        )

    def element_spec(self) -> PyTree:
        return self._element_spec

    def init_state(self, key: jax.Array | None = None):
        return self.inner.init_state(key)

    def next(self, state):
        batch, mask, inner_state = self.inner.next(state)

        def _callback_fn(batch_arg, mask_arg):
            batch_np = tree_util.tree_map(np.asarray, batch_arg)
            mask_np = np.asarray(mask_arg)
            result = self.fn(batch_np, mask_np)
            return batch_np if result is None else result

        transformed = io_callback(_callback_fn, self._callback_spec, batch, mask)
        return transformed, mask, inner_state


def _resolve_device(device: jax.Device | str) -> jax.Device:
    if isinstance(device, jax.Device):
        return device
    device_str = device
    index = None
    if ":" in device_str:
        device_str, idx_str = device_str.split(":", 1)
        index = int(idx_str)
    matching = [d for d in jax.devices() if d.platform == device_str]
    if not matching:
        raise ValueError(f"No JAX devices found for platform '{device}'.")
    if index is not None:
        if index >= len(matching):
            raise ValueError(
                f"Requested device '{device}' but only {len(matching)} devices available."
            )
        return matching[index]
    return matching[0]


@dataclass
class DevicePutTransform:
    """Move batches onto a target device.
    
    Use this to move your data onto accelerators (e.g., GPU/TPU) as part of the data pipeline.
    """

    device: jax.Device | str | None = None
    """Target device or device string (e.g., 'cpu', 'gpu:0', 'tpu:1'). If None, defaults to the default jax device."""

    def __call__(self, inner: Source) -> Source:
        return _DevicePutTransformSource(inner=inner, device=self.device)


@dataclass
class _DevicePutTransformSource(SourceTransform):
    inner: Source
    device: jax.Device | str | None = None

    def __post_init__(self) -> None:
        if self.device is None:
            devices = jax.devices()
            if not devices:
                raise ValueError("DevicePutTransform requires at least one JAX device.")
            target = devices[0]
        else:
            target = self.device
        self._device = _resolve_device(target)
        self.steps_per_epoch = self.inner.steps_per_epoch

    def element_spec(self) -> PyTree:
        return self.inner.element_spec()

    def init_state(self, key: jax.Array | None = None):
        return self.inner.init_state(key)

    def next(self, state):
        batch, mask, inner_state = self.inner.next(state)
        batch = tree_util.tree_map(lambda arr: jax.device_put(arr, self._device), batch)
        return batch, mask, inner_state


def _replace_mapping_item(obj: Mapping[str, Any], key: str, value: Any) -> Mapping[str, Any]:
    if isinstance(obj, MutableMapping):
        clone = obj.copy()
        clone[key] = value
        return clone
    if isinstance(obj, dict):
        clone = dict(obj)
        clone[key] = value
        return clone
    raise TypeError(
        "Transforms expect batches that behave like mutable mappings (dict/FrozenDict)."
    )


def _require_spec_mapping(spec: PyTree, key: str) -> dict[str, jax.ShapeDtypeStruct]:
    if not isinstance(spec, Mapping):
        raise TypeError("Element spec must be a mapping when using keyed transforms.")
    if key not in spec:
        raise KeyError(f"Key '{key}' not found in element spec.")
    return dict(spec)


@dataclass
class NormalizeImageTransform:
    """Scale uint8 image tensors to [0, 1] (or custom range)."""

    data_key: str = "image"
    """Key in the batch mapping corresponding to image tensors."""
    dtype: jnp.dtype = jnp.float32
    """Target dtype for normalized images."""
    scale: float = 1.0 / 255.0
    """Scale factor applied to uint8 images."""
    offset: float = 0.0
    """Offset added after scaling."""

    def __call__(self, inner: Source) -> Source:
        return _NormalizeImageTransformSource(
            inner=inner,
            data_key=self.data_key,
            dtype=self.dtype,
            scale=self.scale,
            offset=self.offset,
        )


@dataclass
class _NormalizeImageTransformSource(SourceTransform):
    inner: Source
    data_key: str = "image"
    dtype: jnp.dtype = jnp.float32
    scale: float = 1.0 / 255.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        self.steps_per_epoch = self.inner.steps_per_epoch
        spec = _require_spec_mapping(self.inner.element_spec(), self.data_key)
        image_spec = spec[self.data_key]
        spec[self.data_key] = jax.ShapeDtypeStruct(
            shape=image_spec.shape,
            dtype=self.dtype,
        )
        self._element_spec = spec

    def element_spec(self) -> PyTree:
        return self._element_spec

    def init_state(self, key: jax.Array | None = None):
        return self.inner.init_state(key)

    def next(self, state):
        batch, mask, inner_state = self.inner.next(state)

        def _normalize(img):
            return img.astype(self.dtype) * self.scale + self.offset

        normalized = _replace_mapping_item(batch, self.data_key, _normalize(batch[self.data_key]))
        return normalized, mask, inner_state


@dataclass
class FlattenTransform:
    """Flattens per-sample tensors."""

    data_key: str 
    """Key in the batch mapping corresponding to tensors to be flattened."""
    start_index: int = 1
    """Index at which to start flattening (default 0 flattens all dimensions)."""

    def __call__(self, inner: Source) -> Source:
        return _FlattenTransformSource(inner=inner, data_key=self.data_key, start_index=self.start_index)


@dataclass
class _FlattenTransformSource(SourceTransform):
    inner: Source
    data_key: str 
    start_index: int

    def __post_init__(self) -> None:
        self.steps_per_epoch = self.inner.steps_per_epoch
        spec = _require_spec_mapping(self.inner.element_spec(), self.data_key)
        image_spec = spec[self.data_key]
        rank = len(image_spec.shape)

        if not 0 <= self.start_index < rank:
            raise ValueError(
                "start_index must be within the tensor rank; got "
                f"start_index={self.start_index}, rank={rank}"
            )

        leading_shape = image_spec.shape[: self.start_index]
        trailing_shape = image_spec.shape[self.start_index :]
        flat_dim = math.prod(trailing_shape)
        spec[self.data_key] = jax.ShapeDtypeStruct(
            shape=leading_shape + (flat_dim,),
            dtype=image_spec.dtype,
        )
        self._element_spec = spec

    def element_spec(self) -> PyTree:
        return self._element_spec

    def init_state(self, key: jax.Array | None = None):
        return self.inner.init_state(key)

    def next(self, state):
        batch, mask, inner_state = self.inner.next(state)
        tensor = batch[self.data_key]
        desired_shape = self._element_spec[self.data_key].shape
        flattened = _replace_mapping_item(batch, self.data_key, jnp.reshape(tensor, desired_shape))
        return flattened, mask, inner_state