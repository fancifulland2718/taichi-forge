"""Explicit NVTX ranges, independent of the CUDA Toolkit and device telemetry."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from taichi_forge._lib import core as _ti_core

_CATEGORIES = {
    "task": 1,
    "search": 2,
    "stage": 3,
    "recipe": 4,
    "materialization": 5,
    "trial": 6,
    "user": 7,
}
_recipe_trace_enabled = ContextVar("forge_recipe_search_trace", default=False)


@contextmanager
def external_range(message, *, category="user", payload=0):
    """Annotate an explicit, same-thread scope for an attached NVTX tool.

    This does not enable a kernel profiler, collect GPU counters, synchronize,
    or load a CUDA library. Without an attached tool the native range is a no-op.
    ``payload`` is an unsigned 64-bit integer; categories are stable trace labels.
    """
    if not isinstance(message, str) or "\0" in message:
        raise ValueError("external_range message must be a NUL-free string")
    if category not in _CATEGORIES:
        raise ValueError("unknown external_range category")
    if isinstance(payload, bool) or not isinstance(payload, int) or not 0 <= payload < (1 << 64):
        raise ValueError("external_range payload must be an unsigned 64-bit integer")
    _ti_core._push_external_profiler_range(message, _CATEGORIES[category], payload)
    try:
        yield
    finally:
        _ti_core._pop_external_profiler_range()


@contextmanager
def recipe_search_trace(label="Forge recipe search"):
    """Annotate searches run inside this scope with stage/recipe/trial identity.

    Place ``session.run(evaluator)`` inside the scope. Only the search adapter is
    instrumented: materialized Graphs retain their ordinary replay path. Trace
    overhead is diagnostic and must not be presented as uninstrumented timing.
    The context is nestable and thread-local; do not suspend it across threads.
    """
    token = _recipe_trace_enabled.set(True)
    try:
        with external_range(label, category="search"):
            yield
    finally:
        _recipe_trace_enabled.reset(token)


def _trace_materializer(materialize):
    @wraps(materialize)
    def traced(recipe, **kwargs):
        with external_range(recipe.recipe_id, category="materialization"):
            return materialize(recipe, **kwargs)

    return traced


def _trace_trial(evaluate):
    @wraps(evaluate)
    def traced(request):
        # The full measurement key and observation index join directly to the
        # CompileIQ trial records; never invent an unrelated local trial ID.
        message = (
            f"measurement={request.measurement_key} "
            f"observation={request.observation_index} "
            f"fidelity={request.fidelity_name}"
        )
        with external_range(request.recipe_id, category="recipe"):
            with external_range(message, category="trial", payload=request.observation_index):
                return evaluate(request)

    return traced


def _trace_stage(submit):
    @wraps(submit)
    def traced(batch):
        with external_range(batch.stage_fingerprint, category="stage", payload=batch.stage_index):
            return submit(batch)

    return traced


__all__ = ["external_range", "recipe_search_trace"]
