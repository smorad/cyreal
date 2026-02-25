"""High-level helpers for building jittable data pipelines."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from .sources import Source

PyTree = Any

@jax.tree_util.register_pytree_node_class
@dataclass
class _LoaderState:
    inner_state: Any

    def tree_flatten(self):
        return (self.inner_state,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (inner_state,) = children
        return cls(inner_state=inner_state)


class _LoaderIterator:
    """Python iterator that walks a loader state for a fixed number of steps."""

    def __init__(self, loader: DataLoader, state: _LoaderState, steps: int | None) -> None:
        self._loader = loader
        self._state = state
        self._remaining = steps

    def __iter__(self):
        return self

    def __next__(self):
        if self._remaining is not None:
            if self._remaining <= 0:
                raise StopIteration
            self._remaining -= 1

        batch, self._state, mask = self._loader.next(self._state)
        return batch, mask

    @property
    def state(self) -> _LoaderState:
        return self._state


@dataclass
class DataLoader:
    """Composable pipeline constructed from explicit stages."""

    pipeline: Source | Sequence[Any]
    """Either a Source or a sequence of stages to compose into a data pipeline."""

    def __post_init__(self) -> None:
        self._source = self._coerce_pipeline(self.pipeline)
        self.steps_per_epoch = self._source.steps_per_epoch
        # Very important to avoid JIT recompilation due to method rebinding!
        # Otherwise, calling `loader.next` performs dynamic lookup on the method. 
        # Since `next` is not a staticmethod, this lookup returns a new bound method object on every call.
        # This causes a cache miss in the JIT cache because the function object is different! 
        self.next = self.next  # type: ignore[assignment]

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

    def init_state(self, key: jax.Array) -> _LoaderState:
        """Returns a new loader state using the given random key."""
        inner_state = self._source.init_state(key)
        return _LoaderState(inner_state=inner_state)

    def next(self, state: _LoaderState) -> Tuple[PyTree, _LoaderState, jax.Array]:
        """Run the pipeline for one step, returning the batch, new loader state, and mask.
        
        This function is jittable and can be used in `jax.lax.scan` or jitted directly for fast iteration.

        ```python
        state = loader.init_state(jax.random.key(0))
        for _ in range(loader.steps_per_epoch):
            batch, state, mask = loader.next(state)
            ...
        ```
        """
        batch, mask, inner_state = self._source.next(state.inner_state)
        return batch, _LoaderState(inner_state=inner_state), mask

    def iterate(self, state: _LoaderState, *, steps: int | None = None) -> _LoaderIterator:
        """Return a Python iterator over loader outputs.

        ```python
        for batch, mask in loader.iterate(state):
            ...
        ```
        WARNING: This method is slow compared to using `loader.scan_epoch` or jitting `loader.next`.
        """

        if steps is None:
            steps = self.steps_per_epoch
        elif steps < 0:
            raise ValueError("steps must be non-negative or None.")
        return _LoaderIterator(self, state, steps)

    def scan_epoch(
        self,
        state: _LoaderState,
        carry: Any,
        body_fn: Callable[[Any, PyTree, jax.Array], Tuple[Any, Any]],
        unroll: bool | int = False,
    ):
        """Run a full epoch via `jax.lax.scan` with constant-shape batches.
        
        ```python
        def body_fn(model_state, batch, mask):
            ...
            return new_model_state, None

        state, batch = loader.scan_epoch(state, model_state, body_fn)
        ```
        """
        def _body(loop_state, _):
            loader_state, loop_carry = loop_state
            batch, loader_state, mask = self.next(loader_state)
            new_carry, output = body_fn(loop_carry, batch, mask)
            return (loader_state, new_carry), output

        (loader_state, final_carry), outputs = jax.lax.scan(
            _body, (state, carry), length=self.steps_per_epoch, unroll=unroll
        )
        return loader_state, final_carry, outputs