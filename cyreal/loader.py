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
class LoaderState:
    inner_state: Any

    def tree_flatten(self):
        return (self.inner_state,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (inner_state,) = children
        return cls(inner_state=inner_state)


class _LoaderIterator:
    """Python iterator that walks a loader state for a fixed number of steps."""

    def __init__(self, loader: DataLoader, state: LoaderState, steps: int | None) -> None:
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
    def state(self) -> LoaderState:
        return self._state


@dataclass
class DataLoader:
    """Composable pipeline constructed from explicit stages.
    
    Args:
        pipeline: Either a Source or a sequence of stages to compose into a data pipeline."""

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

    def next(self, state: LoaderState) -> Tuple[PyTree, LoaderState, jax.Array]:
        batch, mask, inner_state = self._source.next(state.inner_state)
        return batch, LoaderState(inner_state=inner_state), mask

    def iterate(self, state: LoaderState, *, steps: int | None = None) -> _LoaderIterator:
        """Return a Python iterator over loader outputs.

        Args:
            state: Starting loader state.
            steps: Number of steps (updates) to iterate; defaults to a single epoch.
                Pass ``None`` to iterate indefinitely.
        """

        if steps is None:
            steps = self.steps_per_epoch
        elif steps < 0:
            raise ValueError("steps must be non-negative or None.")
        return _LoaderIterator(self, state, steps)

    def scan_epoch(
        self,
        state: LoaderState,
        carry: Any,
        body_fn: Callable[[Any, PyTree, jax.Array], Tuple[Any, Any]],
    ):
        """Run a full epoch via `jax.lax.scan` with constant-shape batches."""

        def _body(loop_state, _):
            loader_state, loop_carry = loop_state
            batch, loader_state, mask = self.next(loader_state)
            new_carry, output = body_fn(loop_carry, batch, mask)
            return (loader_state, new_carry), output

        (loader_state, final_carry), outputs = jax.lax.scan(
            _body, (state, carry), jnp.arange(self.steps_per_epoch)
        )
        return loader_state, final_carry, outputs