"""Explicit recipe scratch must be classified before generic state snapshots."""

import json
import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from tests import test_utils


class _RetainedGraph(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, graph):
        self.graph, self.bindings = graph, graph.bind({})

    def compile(self):
        return self

    def _export_aot(self, context):
        context.bound_graph(self.graph, self.bindings)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_snapshot_preserves_unique_scratch_role_and_parent(tmp_path, monkeypatch):
    keys, output = [ti.ndarray(ti.i32, 17) for _ in range(2)]
    count = ti.ndarray(ti.i32, 1)
    keys.from_numpy(np.arange(17, dtype=np.int32) // 3)
    selected = ti.algorithms.prepare_unique(keys, output, count)
    builder = ti.graph.GraphBuilder()
    builder.append_native(selected.record())
    graph = builder.compile()
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_RetainedGraph(graph))
    parent = parent_builder.compile()
    sources = dict(keys=keys, output=output, count=count)
    module = ti.aot.Module()

    def no_snapshot(*args, **kwargs):
        raise AssertionError("Explicit scratch must not be sampled")

    with monkeypatch.context() as patch:
        patch.setattr(type(ti.lang.impl.get_runtime().prog), "copy_ndarrays_to_host", no_snapshot)
        for name, owner in (("step", graph), ("parent", parent)):
            plan = module.add_execution_plan(name, owner, source_bindings=sources, snapshot_state=True)
            resources = json.loads(plan.to_json())["resources"]
            assert resources and all("initial_data" not in r for r in resources)
    assert not module._data_payloads
    artifact = tmp_path / "unique.tcm"
    module.archive(str(artifact))
    parent.close()
    graph.close()
    selected.close()
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        arguments = {name: runtime.ndarray("i32", 1 if name == "count" else 17) for name in sources}
        for name in ("step", "parent"):
            with loaded.prepare_execution(name, arguments) as call:
                for data in (np.arange(17, dtype=np.int32) // 3, np.zeros(17, np.int32)):
                    arguments["keys"].write(data.tobytes())
                    call()
                    length = int(np.frombuffer(arguments["count"].read(), np.int32)[0])
                    np.testing.assert_array_equal(
                        np.frombuffer(arguments["output"].read(), np.int32)[:length], np.unique(data))
        loaded.close()


@pytest.mark.parametrize("capacity", [1, 17])
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prefix_sort_preserves_selected_dispatches_and_scratch(tmp_path, monkeypatch, capacity):
    keys = ti.ndarray(ti.f32, capacity)
    values, count = ti.ndarray(ti.i32, capacity), ti.ndarray(ti.i32, 1)
    selected = ti.algorithms.prepare_sort(keys, values, count=count)
    action = selected._executable.recordable_action
    physical_id = selected.report()["physical_plan_id"]
    builder = ti.graph.GraphBuilder()
    builder.append_native(selected.record())
    graph = builder.compile()
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_RetainedGraph(graph))
    parent = parent_builder.compile()
    sources = dict(keys=keys, values=values, count=count)
    module = ti.aot.Module()

    def no_snapshot(*args, **kwargs):
        raise AssertionError("Prefix workspace must be reconstructed without producer contents")

    from taichi_forge.aot._graph_export import _ExportContext

    emitted = []
    original_kernel = _ExportContext.kernel

    def emit(context, compiled, arguments, *args, **kwargs):
        emitted.append(compiled)
        return original_kernel(context, compiled, arguments, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(type(ti.lang.impl.get_runtime().prog), "copy_ndarrays_to_host", no_snapshot)
        patch.setattr(_ExportContext, "kernel", emit)
        for name, owner in (("step", graph), ("parent", parent), ("public_status", graph)):
            emitted.clear()
            bindings = dict(sources, status=selected.status) if name == "public_status" else sources
            plan = module.add_execution_plan(name, owner, source_bindings=bindings, snapshot_state=name == "step")
            assert emitted == [compiled for compiled, _ in action.dispatches]
            document = json.loads(plan.to_json())
            assert [op["kind"] for op in document["operations"]] == ["kernel"] * len(action.dispatches)
            workspace = [r for r in document["resources"] if r["kind"] == "ndarray"]
            assert len(workspace) == (4 if capacity == 1 else 5) - (name == "public_status")
            assert all("initial_data" not in r for r in workspace)
    assert selected.report()["physical_plan_id"] == physical_id
    assert not module._data_payloads
    artifact = tmp_path / "prefix-sort.tcm"
    module.archive(str(artifact))
    parent.close()
    graph.close()
    selected.close()
    with ti.aot.borrow_runtime() as runtime:
        with runtime.load_module(artifact) as loaded:
            arguments = {
                "keys": runtime.ndarray("f32", capacity),
                "values": runtime.ndarray("i32", capacity),
                "count": runtime.ndarray("i32", 1),
            }
            initial = ((np.arange(capacity) * 11 % 7) - 3).astype(np.float32)
            status = runtime.ndarray("i32", 2)
            for name in ("step", "parent", "public_status"):
                bindings = dict(arguments, status=status) if name == "public_status" else arguments
                with loaded.prepare_execution(name, bindings) as call:
                    for n in (0, 1, capacity, capacity // 2, -1, capacity + 1):
                        arguments["keys"].write(initial.tobytes())
                        arguments["values"].write(np.arange(capacity, dtype=np.int32).tobytes())
                        arguments["count"].write(np.array([n], np.int32).tobytes())
                        call()
                        order = np.arange(capacity)
                        if 0 <= n <= capacity:
                            order[:n] = np.argsort(initial[:n], kind="stable")
                        np.testing.assert_array_equal(
                            np.frombuffer(arguments["keys"].read(), np.float32), initial[order]
                        )
                        np.testing.assert_array_equal(np.frombuffer(arguments["values"].read(), np.int32), order)
                        if name == "public_status":
                            valid = 0 <= n <= capacity
                            np.testing.assert_array_equal(
                                np.frombuffer(status.read(), np.int32), [n if valid else 0, not valid]
                            )
