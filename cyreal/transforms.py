"""Composable transforms that wrap sources to build pipelines."""
from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

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


class SourceTransform(Source, Protocol):
    inner: Source


@dataclass
class BatchTransform:
    """Batch elements emitted by a source.
    Args:
        batch_size: Number of elements per batch.
        drop_last: If True, drop the final batch if it is less than batch_size.
        pad_last_batch: If True, pad the final batch with zeros if it is less than batch_size. Useful to prevent a second jit recompile. Make sure you use the mask to ignore padded values.
    
    """

    batch_size: int
    drop_last: bool = False
    pad_last_batch: bool = False
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
class BatchTransformState:
    inner_state: Any
    position_in_epoch: jax.Array

    def tree_flatten(self):
        return (self.inner_state, self.position_in_epoch), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
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

    def init_state(self, key: jax.Array | None = None) -> BatchTransformState:
        return BatchTransformState(
            inner_state=self.inner.init_state(key),
            position_in_epoch=jnp.array(0, dtype=jnp.int32),
        )

    def _write_slice(self, buffer: PyTree, value: PyTree, index: int) -> PyTree:
        return tree_util.tree_map(lambda buf, val: buf.at[index].set(val), buffer, value)

    def _maybe_skip_incomplete_epoch(self, state: BatchTransformState) -> BatchTransformState:
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
            return BatchTransformState(
                inner_state=inner_state,
                position_in_epoch=jnp.array(0, dtype=jnp.int32),
            )

        return jax.lax.cond(need_skip, _drain, lambda operand: operand[0], (state, remaining))

    def next(self, state: BatchTransformState):
        state = self._maybe_skip_incomplete_epoch(state)

        def body(carry, i):
            inner_state, position, buffer, mask = carry
            remaining = self._samples_per_epoch - position
            remaining = jnp.maximum(remaining, 0)

            def _consume(_: None):
                value, value_mask, next_inner_state = self.inner.next(inner_state)
                updated_buffer = self._write_slice(buffer, value, i)
                updated_mask = mask.at[i].set(value_mask)
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
        return batch, mask, BatchTransformState(
            inner_state=inner_state,
            position_in_epoch=wrapped_position,
        )


@dataclass
class MapTransform:
    """Apply a pure function to the input for map-style transforms."""

    fn: Callable[[PyTree, jax.Array], PyTree]

    def __call__(self, inner: Source) -> Source:
        return _MapTransformSource(inner=inner, fn=self.fn)


@dataclass
class _MapTransformSource(SourceTransform):
    """Apply a pure function to each batch emitted by the inner source."""

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
    
    Args:
        fn: Function to invoke on host. Receives (batch, mask) as arrays.
    """

    fn: Callable[[PyTree, jax.Array], PyTree | None]
    element_spec_override: PyTree | None = None

    def __call__(self, inner: Source) -> Source:
        return _HostCallbackTransformSource(
            inner=inner,
            fn=self.fn,
            element_spec_override=self.element_spec_override,
        )


@dataclass
class _HostCallbackTransformSource(SourceTransform):
    """Invoke a host callback between jittable pipeline stages."""

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

    Args:
        device: Target device or device string (e.g., 'cpu', 'gpu:0', 'tpu:1'). If None, defaults to the default jax device.
    """

    device: jax.Device | str | None = None

    def __call__(self, inner: Source) -> Source:
        return _DevicePutTransformSource(inner=inner, device=self.device)


@dataclass
class _DevicePutTransformSource(SourceTransform):
    """Move batches emitted by `inner` onto a target device."""

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
    """Scale uint8 image tensors to [0, 1] (or custom range).
    
    Args:
        data_key: Key in the batch mapping corresponding to image tensors.
        dtype: Target dtype for normalized images.
        scale: Scale factor applied to uint8 images.
        offset: Offset added after scaling."""

    data_key: str = "image"
    dtype: jnp.dtype = jnp.float32
    scale: float = 1.0 / 255.0
    offset: float = 0.0

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
    """Scale uint8 image tensors to [0, 1] (or custom range)."""

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
    """Flattens per-sample tensors.
    
    Args:
        data_key: Key in the batch mapping corresponding to image tensors."""

    data_key: str 
    start_index: int = 1

    def __call__(self, inner: Source) -> Source:
        return _FlattenTransformSource(inner=inner, data_key=self.data_key, start_index=self.start_index)


@dataclass
class _FlattenTransformSource(SourceTransform):
    """Flatten incoming tensors.
    
    Args:
        data_key: Key in the batch mapping corresponding to image tensors.
        start_index: Index at which to start flattening (default 0 flattens all dimensions).
    """

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