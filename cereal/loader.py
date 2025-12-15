"""High-level helpers for building jittable data pipelines."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np

from .dataset_protocol import DatasetProtocol
from .sources import Source

PyTree = Any


def _leaf_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _item_to_numpy(sample: Any) -> Any:
    return tree_util.tree_map(_leaf_to_numpy, sample)


def dataset_to_jax(
    dataset: DatasetProtocol,
    *,
    storage_device: jax.Device | str | None = "cpu",
) -> PyTree:
    """Materialize a finite dataset into host-resident JAX arrays."""

    length = len(dataset)
    if length == 0:
        raise ValueError("Dataset must contain at least one element.")

    records = [_item_to_numpy(dataset[i]) for i in range(length)]
    stacked = tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *records)

    target_device = None
    if storage_device is not None:
        if isinstance(storage_device, jax.Device):
            target_device = storage_device
        else:
            device_str = storage_device
            index = None
            if ":" in device_str:
                device_str, idx_str = device_str.split(":", 1)
                index = int(idx_str)
            matching = [d for d in jax.devices() if d.platform == device_str]
            if not matching:
                raise ValueError(f"No JAX devices found for platform '{storage_device}'.")
            if index is not None:
                if index >= len(matching):
                    raise ValueError(
                        f"Requested device '{storage_device}' but only {len(matching)} devices available."
                    )
                target_device = matching[index]
            else:
                target_device = matching[0]

    def _to_jax(arr: np.ndarray) -> jax.Array:
        if target_device is None:
            return jnp.asarray(arr)
        with jax.default_device(target_device):
            return jnp.asarray(arr)

    return tree_util.tree_map(_to_jax, stacked)


@jax.tree_util.register_pytree_node_class
@dataclass
class LoaderState:
    inner_state: Any

    def tree_flatten(self):
        return (self.inner_state,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (inner_state,) = children
        return cls(inner_state=inner_state)


@dataclass
class DataLoader:
    """Composable pipeline constructed from explicit stages."""

    pipeline: Source | Sequence[Any]

    def __post_init__(self) -> None:
        self._source = self._coerce_pipeline(self.pipeline)
        self.steps_per_epoch = self._source.steps_per_epoch

    def _coerce_pipeline(self, pipeline: Source | Sequence[Any]) -> Source:
        if self._looks_like_source(pipeline):
            return pipeline  # type: ignore[return-value]

        if isinstance(pipeline, Sequence):
            if not pipeline:
                raise ValueError("Pipeline sequence must contain at least one stage.")
            stages = list(pipeline)
            head = stages[0]
            if not self._looks_like_source(head):
                raise TypeError("First pipeline stage must implement the Source protocol.")
            current: Source = head  # type: ignore[assignment]
            for stage in stages[1:]:
                current = self._attach_stage(current, stage)
            return current

        raise TypeError("`pipeline` must be a Source or a sequence of stages.")

    @staticmethod
    def _looks_like_source(candidate: Any) -> bool:
        return all(hasattr(candidate, attr) for attr in ("init_state", "next", "steps_per_epoch"))

    def _attach_stage(self, current: Source, stage: Any) -> Source:
        radd = getattr(stage, "__radd__", None)
        if callable(radd):
            bound = radd(current)
            if bound is not NotImplemented:
                if not self._looks_like_source(bound):
                    raise TypeError("Stage did not return a valid Source when composed via '+'.")
                return bound

        if callable(stage):
            bound = stage(current)
            if not self._looks_like_source(bound):
                raise TypeError("Callable pipeline stage must return a Source.")
            return bound

        raise TypeError("Pipeline entries after the first must be transforms or callables returning Sources.")

    def init_state(self, key: jax.Array | None = None) -> LoaderState:
        inner_state = self._source.init_state(key)
        return LoaderState(inner_state=inner_state)

    def next_batch(self, state: LoaderState) -> Tuple[PyTree, LoaderState, jax.Array]:
        batch, mask, inner_state = self._source.next(state.inner_state)
        return batch, LoaderState(inner_state=inner_state), mask

    def scan_epoch(
        self,
        state: LoaderState,
        carry: Any,
        body_fn: Callable[[Any, PyTree, jax.Array], Tuple[Any, Any]],
    ):
        """Run a full epoch via `jax.lax.scan` with constant-shape batches."""

        def _body(loop_state, _):
            loader_state, loop_carry = loop_state
            batch, loader_state, mask = self.next_batch(loader_state)
            new_carry, output = body_fn(loop_carry, batch, mask)
            return (loader_state, new_carry), output

        (loader_state, final_carry), outputs = jax.lax.scan(
            _body, (state, carry), jnp.arange(self.steps_per_epoch)
        )
        return loader_state, final_carry, outputs