"""Frozen Graph members retain their parent ABI and selected native strategy."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from tests import test_utils


class _RetainedGraph(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, graph):
        self.graph = graph
        self.bindings = graph.bind({})

    def compile(self):
        return self

    def run(self):
        self.graph.run(self.bindings)

    def _export_aot(self, context):
        context.bound_graph(self.graph, self.bindings)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_graph_scalar_members_cold_warm_parent_and_suffix_gradient(tmp_path):
    record_type = ti.types.struct(tag=ti.i32, inner=ti.types.struct(value=ti.f32, other=ti.f32))
    records = ti.ndarray(record_type, shape=17)
    output = ti.ndarray(ti.f32, shape=17)
    host = records.to_numpy()
    host["tag"] = -7
    host["inner"]["value"] = np.arange(17, dtype=np.float32) * .5
    host["inner"]["other"] = -9
    records.from_numpy(host)
    output.fill(-13)
    sequence = ti.algorithms.primitive_sequence()
    sequence.transform(records.field("inner.value"), records.field("inner.other"), scale=3, bias=7)
    sequence.transform(records.field("inner.other"), output, scale=2, bias=-1)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    source_bindings = dict(records=records, output=output)
    cold = module.add_execution_plan("cold", graph, source_bindings=source_bindings)
    document = json.loads(cold.to_json())
    assert document["version"] == 24
    assert len(document["operations"]) == 2
    assert all(op["parameters"]["strategy"] == "strided_scalar" for op in document["operations"])
    assert all(r["kind"] == "ndarray_member" for r in document["resources"])
    assert len(module._ndarray_layouts) == 1 and not module._exported_kernels
    np.testing.assert_array_equal(records.to_numpy(), host)
    np.testing.assert_array_equal(output.to_numpy(), np.full(17, -13, dtype=np.float32))
    sequence.run()
    expected = 6 * host["inner"]["value"] + 13
    np.testing.assert_array_equal(output.to_numpy(), expected)
    module.add_execution_plan("warm", graph, source_bindings=source_bindings)
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_RetainedGraph(graph))
    parent = parent_builder.compile()
    module.add_execution_plan("parent", parent, source_bindings=source_bindings)
    assert len(module._ndarray_layouts) == 1 and not module._exported_kernels

    # Failed preflight must not publish the pending structure layout.
    failed = ti.aot.Module()
    with pytest.raises((ValueError, ti.TaichiRuntimeError)):
        failed.add_execution_plan("missing_output", graph, source_bindings=dict(records=records))
    assert not failed._ndarray_layouts and not failed._exported_kernels

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)):
        for i in x:
            y[i] = .5 * x[i] * x[i]

    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="reverse")
    module.add_execution_plan("suffix", ti.aot.ExecutionPlan(
        document["operations"] + [
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            dict(kind="kernel", entry="energy", arguments=["output", "energy"]),
            dict(kind="kernel", entry="reverse", arguments=["output", "energy"]),
        ], resources=document["resources"], constraints=document["constraints"],
        gradients=dict(output="dx", energy="dy"),
    ))
    artifact = tmp_path / "graph-members.tcm"
    module.archive(str(artifact))
    layout_name = next(c["layout"] for c in document["constraints"] if c["kind"] == "struct_array_layout")
    parent.close()
    graph.close()
    sequence.clear()
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(__file__).with_name("aot_graph_struct_members_probe.py")),
             mode, arch, str(artifact), layout_name],
            env=os.environ.copy(), capture_output=True, text=True, timeout=90,
        )
        output_text = result.stdout + result.stderr
        assert result.returncode == 0, output_text
        assert not any(m in output_text for m in ("Validation Error", "SYNC-HAZARD", "cleanup failed")), output_text
