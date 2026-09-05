"""Explicit search tracing must preserve identity, nesting and failure cleanup."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from taichi_forge.profiler import external_trace as trace


def test_external_trace_restores_nested_context_on_failure_and_is_thread_local(monkeypatch):
    events = []
    monkeypatch.setattr(trace._ti_core, "_push_external_profiler_range", lambda *args: events.append(args))
    monkeypatch.setattr(trace._ti_core, "_pop_external_profiler_range", lambda: events.append("pop"))
    assert not trace._recipe_trace_enabled.get()
    with pytest.raises(RuntimeError, match="objective failed"):
        with trace.recipe_search_trace("outer"):
            assert trace._recipe_trace_enabled.get()
            with ThreadPoolExecutor(max_workers=1) as worker:
                assert worker.submit(trace._recipe_trace_enabled.get).result() is False
            with trace.recipe_search_trace("inner"):
                assert trace._recipe_trace_enabled.get()
                raise RuntimeError("objective failed")
    assert not trace._recipe_trace_enabled.get()
    assert events == [("outer", 2, 0), ("inner", 2, 0), "pop", "pop"]


def test_trace_materialization_failure_preserves_trial_key_and_exception(monkeypatch):
    events = []
    monkeypatch.setattr(trace._ti_core, "_push_external_profiler_range", lambda *args: events.append(args))
    monkeypatch.setattr(trace._ti_core, "_pop_external_profiler_range", lambda: events.append("pop"))
    recipe = SimpleNamespace(recipe_id="recipe:stable")
    request = SimpleNamespace(
        recipe_id=recipe.recipe_id, measurement_key="measurement:stable", observation_index=7, fidelity_name="full"
    )
    failure = RuntimeError("materialization failed")

    def materialize(selected, *, context):
        assert selected is recipe
        assert context == "owned context"
        raise failure

    materializer = trace._trace_materializer(materialize)
    evaluator = trace._trace_trial(lambda req: materializer(recipe, context="owned context"))
    with pytest.raises(RuntimeError) as caught:
        evaluator(request)
    assert caught.value is failure
    assert events == [
        (recipe.recipe_id, 4, 0),
        ("measurement=measurement:stable observation=7 fidelity=full", 6, 7),
        (recipe.recipe_id, 5, 0),
        "pop",
        "pop",
        "pop",
    ]
