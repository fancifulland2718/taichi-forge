"""A retained definition retires native caches after its Graph wrapper dies."""

import gc
import weakref

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils
from tests.python.test_aot_component_control import _ControlModel


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_reset_retires_spec_retained_without_graph_wrapper(monkeypatch):
    model = _ControlModel(7, 5)
    builder = ti.graph.GraphBuilder()
    model.record_control(builder, budget=9, group=1)
    graph = builder.compile()
    definition = graph.definition
    spec = graph._spec
    graph_ref = weakref.ref(graph)
    graph.run({})
    ti.sync()
    assert model.y[0] == 5
    runtime = impl.get_runtime()
    runtime.collect_ready_runtime_submission_owners(_limit=None)
    retired = []
    invalidate = spec.invalidate_runtime

    def observe_retirement(*args, **kwargs):
        assert runtime.prog is not None
        retired.append(True)
        return invalidate(*args, **kwargs)

    monkeypatch.setattr(spec, "invalidate_runtime", observe_retirement)
    del graph
    gc.collect()
    assert graph_ref() is None
    ti.reset()
    assert retired
    # The semantic definition remains caller-owned after runtime retirement.
    assert definition is not None
    del builder, definition, model
    gc.collect()
