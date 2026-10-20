"""Export public buffer recordings without introducing a second executor."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from tests import test_utils


class _Parent(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, graph, bindings):
        self.graph, self.bindings = graph, graph.bind(bindings)

    def compile(self):
        return self

    def run(self):
        self.graph.run(self.bindings)

    def _export_aot(self, context):
        context.bound_graph(self.graph, self.bindings)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_buffer_command_backend_isolation():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.native_operation_capabilities("vulkan_buffer_commands")


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_buffer_recording_parent_gradient_and_retirement(tmp_path):
    command = ti.graph.VulkanBufferCommand
    recording = ti.graph.VulkanBufferCommandRecording((
        command.fill_u32("output", 16, 0, offset=8),
        command.buffer_barrier("output"),
        command.copy("output", "input", 16, destination_offset=8, source_offset=4),
        command.memory_barrier(),
    ))
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    copy = builder.compile()
    source, output = ti.ndarray(ti.f32, 8), ti.ndarray(ti.f32, 8)
    source.from_numpy(np.arange(8, dtype=np.float32))
    output.fill(-99)
    bindings = dict(input=source, output=output)
    padding, values = ti.field(ti.f32), ti.field(ti.f32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 7).place(padding)
    fields.dense(ti.i, 16).place(values)
    tree = fields.finalize()
    padding.fill(77)
    values.from_numpy(np.arange(16, dtype=np.float32))
    module = ti.aot.Module()
    start = time.perf_counter()
    plan = module.add_execution_plan("copy", copy, source_bindings=bindings)
    export_ms = (time.perf_counter() - start) * 1000
    doc = json.loads(plan.to_json())
    assert len(doc["operations"]) == 1
    op = doc["operations"][0]
    assert op["provider"] == "vulkan_rhi"
    assert op["parameters"]["buffer_count"] == 2
    assert [c["kind"] for c in op["parameters"]["commands"]] == [
        "fill_u32", "buffer_barrier", "copy", "memory_barrier"
    ]
    np.testing.assert_array_equal(output.to_numpy(), [-99] * 8)
    copy.run(bindings)
    np.testing.assert_array_equal(output.to_numpy(), [-99, -99, 1, 2, 3, 4, -99, -99])

    @ti.kernel
    def seed(doutput: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1),
             dloss: ti.types.ndarray(ti.f32, 1)):
        for i in doutput:
            doutput[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(output: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(2, 6):
            loss[0] += 0.5 * output[i] * output[i]

    arg = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_Parent(copy, bindings))
    parent_builder.dispatch(seed, arg("doutput"), arg("loss"), arg("dloss"))
    parent_builder.dispatch(energy, arg("output"), arg("loss"))
    parent_builder.dispatch(energy.grad, arg("output"), arg("loss"))
    parent = parent_builder.compile()
    module.add_execution_plan("parent", parent, source_bindings=bindings,
                              gradients=dict(output="doutput", loss="dloss"))
    barrier_builder = ti.graph.GraphBuilder()
    barrier_builder.append_native(ti.graph.VulkanBufferCommandRecording((command.memory_barrier(),)))
    barrier = barrier_builder.compile()
    module.add_execution_plan("barrier", barrier)

    @ti.kernel
    def observe(out: ti.types.ndarray(ti.f32, 1)):
        for i in range(16):
            out[i] = values[i]
        for i in range(7):
            out[16 + i] = padding[i]

    field_builder = ti.graph.GraphBuilder()
    field_builder._bind_internal_runtime_arg(arg("src"), ti.experimental.ndarray_view(values, slices=slice(1, 5)))
    field_builder._bind_internal_runtime_arg(arg("dst"), ti.experimental.ndarray_view(values, slices=slice(9, 13)))
    field_builder.append_native(ti.graph.VulkanBufferCommandRecording((
        command.copy("dst", "src", 16), command.memory_barrier(),
    )))
    field_builder.dispatch(observe, arg("out"))
    field_graph = field_builder.compile()
    field_plan = module.add_execution_plan("field", field_graph, snapshot_state=True)
    field_doc = json.loads(field_plan.to_json())
    assert len([r for r in field_doc["resources"] if r["kind"] == "ndarray_view"]) == 2
    module.add_execution_plan("field_read", ti.aot.ExecutionPlan([field_doc["operations"][-1]]))
    np.testing.assert_array_equal(values.to_numpy(), np.arange(16, dtype=np.float32))

    # Explicit subclass opt-in remains required, even for inherited commands.
    class Unqualified(ti.graph.VulkanBufferCommandRecording):
        pass

    bad_builder = ti.graph.GraphBuilder()
    bad_builder.append_native(Unqualified((command.memory_barrier(),)))
    bad = bad_builder.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="reconstruction is unavailable"):
        module.add_execution_plan("unqualified", bad)
    assert "unqualified" not in module._execution_plans
    for graph in (bad, field_graph, barrier, parent, copy):
        graph.close()
    artifact = tmp_path / "buffer-commands.tcm"
    module.archive(str(artifact))
    tree.destroy()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        run = subprocess.run([
            sys.executable, *(["-S"] if mode == "standalone" else []),
            str(Path(__file__).with_name("aot_buffer_commands_probe.py")), mode,
            str(artifact), str(tmp_path / f"{mode}.json"),
        ], env=os.environ.copy(), capture_output=True, text=True, timeout=90)
        assert run.returncode == 0, run.stdout + run.stderr
        assert not any(marker in run.stdout + run.stderr for marker in (
            "VUID-", "Validation Error", "SYNC-HAZARD", "Failed to retire"))
    (tmp_path / "producer.json").write_text(json.dumps(dict(
        export_ms=export_ms, artifact_bytes=artifact.stat().st_size,
        plan_bytes=len(plan.to_json().encode()),
    ), indent=2))
