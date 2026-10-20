"""AOT reductions retain scalar member/Field layouts and selected algorithms."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import taichi_forge as ti
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from tests import test_utils


class _Parent(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, graph, arguments):
        self.graph, self.bindings = graph, graph.bind(arguments)

    def compile(self):
        return self

    def run(self):
        self.graph.run(self.bindings)

    def _export_aot(self, context):
        context.bound_graph(self.graph, self.bindings)


def consume(artifact, kind):
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(__file__).with_name("aot_strided_reduce_probe.py")),
             mode, arch, str(artifact), kind],
            env=os.environ.copy(), capture_output=True, text=True, timeout=90)
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert not any(marker in text for marker in ("Validation Error", "SYNC-HAZARD", "cleanup failed")), text


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_member_reduce_cold_warm_parent_private_prefix_and_gradient(tmp_path):
    records = ti.ndarray(ti.types.struct(
        tag=ti.i32, matrix=ti.types.matrix(2, 3, ti.f32), leaf=ti.f32, result=ti.f32, tail=ti.i32), 17)
    host = np.zeros(17, dtype=records.numpy_dtype)
    host["tag"], host["tail"], host["result"] = 37, 91, -9
    host["matrix"] = np.arange(102, dtype=np.float32).reshape(17, 2, 3) * .5
    host["leaf"] = 1 - np.arange(17, dtype=np.float32)
    records.from_numpy(host)
    frames = dict(out=ti.ndarray(ti.f32, 17), low=ti.ndarray(ti.f32, 3), high=ti.ndarray(ti.f32, 3))
    for value in frames.values():
        value.fill(-13)
    sequence = ti.algorithms.primitive_sequence()
    sequence.reduce(records.field("matrix", component=(1, 2)), records.field("result"), op="sum")
    sequence.reduce(records.field("leaf"), frames["low"], op="min")
    sequence.reduce(records.field("leaf"), frames["high"], op="max")
    sequence.transform(records.field("result"), frames["out"], scale=1, bias=0)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    bindings = dict(records=records, **frames)
    cold = module.add_execution_plan("cold", graph, source_bindings=bindings)
    document = json.loads(cold.to_json())
    assert all(op["parameters"]["strategy"] == "strided_scalar" for op in document["operations"])
    assert not module._exported_kernels
    np.testing.assert_array_equal(records.to_numpy(), host)
    graph.run({})
    expected = host.copy()
    expected["result"][0] = np.sum(host["matrix"][:, 1, 2])
    np.testing.assert_array_equal(records.to_numpy(), expected)
    np.testing.assert_array_equal(frames["out"].to_numpy(), expected["result"])
    np.testing.assert_array_equal(frames["low"].to_numpy(), [-15, -13, -13])
    np.testing.assert_array_equal(frames["high"].to_numpy(), [1, -13, -13])
    assert all("strided_ndarray" in call.plan.method_name for call in sequence._calls[:3])
    module.add_execution_plan("warm", graph, source_bindings=bindings)
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_Parent(graph, {}))
    parent = parent_builder.compile()
    module.add_execution_plan("parent", parent, source_bindings=bindings)
    records.from_numpy(host)
    module.add_execution_plan("private", graph, source_bindings=frames, snapshot_state=True)
    assert not module._exported_kernels
    prefix = json.loads(cold.to_json())
    for operation in prefix["operations"][:3]:
        operation["parameters"]["size"] = 11
    module.add_execution_plan("prefix", ti.aot.ExecutionPlan.from_json(json.dumps(prefix)))
    for name, update in (("bad_strategy", dict(strategy="auto")),
                         ("bad_extent", dict(size=18))):
        bad = json.loads(cold.to_json())
        bad["operations"][0]["parameters"].update(update)
        module.add_execution_plan(name, ti.aot.ExecutionPlan.from_json(json.dumps(bad)))
    bad = json.loads(cold.to_json())
    bad["operations"][0]["bindings"]["output"] = bad["operations"][0]["bindings"]["values"]
    module.add_execution_plan("bad_alias", ti.aot.ExecutionPlan.from_json(json.dumps(bad)))

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            y[i] = .5 * x[i] * x[i]

    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="reverse")
    module.add_execution_plan("suffix", ti.aot.ExecutionPlan(
        document["operations"] + [
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            dict(kind="kernel", entry="energy", arguments=["out", "energy"]),
            dict(kind="kernel", entry="reverse", arguments=["out", "energy"])],
        resources=document["resources"], constraints=document["constraints"],
        gradients=dict(out="dx", energy="dy")))
    module.add_ndarray_layout("wrong", ti.types.struct(value=ti.f32))
    artifact = tmp_path / "member-reduce.tcm"
    module.archive(str(artifact))
    layout_name = next(c["layout"] for c in document["constraints"] if c["kind"] == "struct_array_layout")
    artifact.with_suffix(".json").write_text(json.dumps(dict(layout=layout_name)))
    parent.close()
    graph.close()
    sequence.clear()
    consume(artifact, "members")


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_field_reduce_preserves_interleaved_roots_and_first_output(tmp_path):
    state = ti.Struct.field(dict(value=ti.i32, tag=ti.i32), shape=17, offset=5)
    result = ti.Struct.field(dict(total=ti.i32, low=ti.i32, high=ti.i32, tag=ti.i32), shape=())
    state.value.from_numpy(np.arange(17, dtype=np.int32) - 8)
    state.tag.fill(37)
    result.total.fill(-9)
    result.low.fill(-9)
    result.high.fill(-9)
    result.tag.fill(91)
    output = ti.ndarray(ti.i32, 37)

    @ti.kernel
    def observe(out: ti.types.ndarray(ti.i32, 1)):
        out[0], out[1], out[2] = result[None].total, result[None].low, result[None].high
        for i in range(17):
            out[3 + i] = state[i + 5].value
            out[20 + i] = state[i + 5].tag + result[None].tag

    sequence = ti.algorithms.primitive_sequence()
    for op, dst in (("sum", result.total), ("min", result.low), ("max", result.high)):
        sequence.reduce(state.value, dst, op=op)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(observe, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1))
    graph = builder.compile()
    module = ti.aot.Module()
    cold = module.add_execution_plan("cold", graph, snapshot_state=True)
    document = json.loads(cold.to_json())
    assert len(module._exported_kernels) == 1
    assert all(op["parameters"]["strategy"] == "scalar_field" for op in document["operations"][:3])
    graph.run(dict(out=output))
    np.testing.assert_array_equal(output.to_numpy(), [0, -8, 8] + list(range(-8, 9)) + [128] * 17)
    assert all("dense_field" in call.plan.method_name for call in sequence._calls)
    for dst in (result.total, result.low, result.high):
        dst.fill(-9)
    module.add_execution_plan("warm", graph, snapshot_state=True)
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_Parent(graph, dict(out=output)))
    parent = parent_builder.compile()
    module.add_execution_plan("parent", parent, snapshot_state=True, source_bindings=dict(out=output))
    module.add_execution_plan("read", ti.aot.ExecutionPlan([document["operations"][-1]]))
    prefix = json.loads(cold.to_json())
    for operation in prefix["operations"][:3]:
        operation["parameters"]["size"] = 11
    module.add_execution_plan("prefix", ti.aot.ExecutionPlan.from_json(json.dumps(prefix)))
    artifact = tmp_path / "field-reduce.tcm"
    module.archive(str(artifact))
    parent.close()
    graph.close()
    sequence.clear()
    consume(artifact, "fields")
