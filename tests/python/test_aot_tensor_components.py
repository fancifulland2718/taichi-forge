"""Tensor scalar lanes preserve the typed parent and selected native strategy."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import taichi_forge as ti
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from tests import test_utils


class _RetainedGraph(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, graph):
        self.graph, self.bindings = graph, graph.bind({})

    def compile(self):
        return self

    def run(self):
        self.graph.run(self.bindings)

    def _export_aot(self, context):
        context.bound_graph(self.graph, self.bindings)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_tensor_components_public_private_parent_prefix_and_gradient(tmp_path):
    record = ti.types.struct(tag=ti.i32, inner=ti.types.struct(
        matrix=ti.types.matrix(2, 3, ti.f32), vector=ti.types.vector(3, ti.f32)), tail=ti.i32)
    records = ti.ndarray(record, 17)
    output = ti.ndarray(ti.f32, 17)
    host = np.zeros(17, dtype=records.numpy_dtype)
    host["tag"], host["tail"] = 37, 91
    host["inner"]["matrix"] = np.arange(17 * 6, dtype=np.float32).reshape(17, 2, 3) * .5
    host["inner"]["vector"][:] = [-7, -9, -11]
    records.from_numpy(host)
    lane = records.field("inner.vector", component=1)
    sequence = ti.algorithms.primitive_sequence()
    sequence.transform(records.field("inner.matrix", component=(1, 2)), lane, scale=3, bias=7)
    sequence.transform(lane, lane, scale=2, bias=-1)
    sequence.transform(lane, output, scale=1, bias=0)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    sources = dict(records=records, output=output)
    cold = module.add_execution_plan("cold", graph, source_bindings=sources)
    document = json.loads(cold.to_json())
    assert document["version"] == 25
    assert len(document["operations"]) == 3
    assert all(op["parameters"]["strategy"] == "strided_scalar" for op in document["operations"])
    assert {(r["path"], tuple(r["component"])) for r in document["resources"]} == {
        ("inner.matrix", (1, 2)), ("inner.vector", (1,))}
    assert not module._exported_kernels
    np.testing.assert_array_equal(records.to_numpy(), host)
    graph.run({})
    expected = 6 * host["inner"]["matrix"][:, 1, 2] + 13
    np.testing.assert_array_equal(output.to_numpy(), expected)
    actual = records.to_numpy()
    host_after = host.copy()
    host_after["inner"]["vector"][:, 1] = expected
    np.testing.assert_array_equal(actual, host_after)
    module.add_execution_plan("warm", graph, source_bindings=sources)
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_RetainedGraph(graph))
    parent = parent_builder.compile()
    module.add_execution_plan("parent", parent, source_bindings=sources)
    records.from_numpy(host)
    module.add_execution_plan("private", graph, source_bindings=dict(output=output), snapshot_state=True)
    assert not module._exported_kernels
    prefix = json.loads(cold.to_json())
    for operation in prefix["operations"]:
        operation["parameters"]["size"] = 11
    module.add_execution_plan("prefix", ti.aot.ExecutionPlan.from_json(json.dumps(prefix)))

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            y[i] = .5 * x[i] * x[i]

    @ti.kernel
    def extent(n: ti.i32, output: ti.types.ndarray(ti.i32, 1)):
        output[0] = n

    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="reverse")
    module.add_kernel(extent)
    module.add_execution_plan("suffix", ti.aot.ExecutionPlan(
        document["operations"] + [
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            dict(kind="kernel", entry="energy", arguments=["output", "energy"]),
            dict(kind="kernel", entry="reverse", arguments=["output", "energy"])],
        resources=document["resources"], constraints=document["constraints"],
        gradients=dict(output="dx", energy="dy")))
    member = dict(kind="ndarray_member", name="lane", array="records", path="inner.matrix", component=[1, 2])
    # Rank/bounds are checked against the actual consumer layout, even for
    # shape-only references that do not allocate or execute the member.
    for name, change in (
        ("empty", {}), ("bad_rank", dict(component=[1])),
        ("bad_index", dict(component=[2, 0])), ("bad_scalar", dict(path="tag", component=[0])),
    ):
        module.add_execution_plan(name, ti.aot.ExecutionPlan(
            [dict(kind="kernel", entry="extent", arguments=["n", "out"])],
            resources=[dict(member, **change), dict(kind="extent", name="n", dtype="i32", array="lane", axis=0)]))
    module.add_execution_plan("bad_copy", ti.aot.ExecutionPlan(
        [dict(kind="copy", source="lane", destination="output")], resources=[member]))
    module.add_ndarray_layout("wrong", ti.types.struct(
        tag=ti.f32, inner=ti.types.struct(matrix=ti.types.matrix(2, 3, ti.f32), vector=ti.types.vector(3, ti.f32)),
        tail=ti.i32))
    artifact = tmp_path / "components.tcm"
    module.archive(str(artifact))
    layout_name = next(c["layout"] for c in document["constraints"] if c["kind"] == "struct_array_layout")
    artifact.with_suffix(".layout").write_text(layout_name)
    parent.close()
    graph.close()
    sequence.clear()
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(__file__).with_name("aot_tensor_components_probe.py")),
             mode, arch, str(artifact), layout_name],
            env=os.environ.copy(), capture_output=True, text=True, timeout=90)
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert not any(m in text for m in ("Validation Error", "SYNC-HAZARD", "cleanup failed")), text
