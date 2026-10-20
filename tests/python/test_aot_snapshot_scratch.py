"""Explicit recipe scratch must be classified before generic state snapshots."""

import json
import numpy as np
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
