"""Affine module roots preserve interleaved Field storage without packing."""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
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


@pytest.mark.parametrize("shape", [(17,), (3, 7)])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_affine_field_cold_warm_prefix_parent_and_gradient(tmp_path, shape):
    unused = ti.field(ti.i32, shape=7)
    offset = (5,) if len(shape) == 1 else (2, 5)
    state = ti.Struct.field(dict(value=ti.f32, other=ti.f32, tag=ti.i32), shape=shape, offset=offset)
    n = int(np.prod(shape))
    values = np.arange(n, dtype=np.float32).reshape(shape) * .5
    state.value.from_numpy(values)
    state.other.fill(-9)
    state.tag.fill(37)
    frames = {name: ti.ndarray(dtype, n) for name, dtype in
              (("out", ti.f32), ("source", ti.f32), ("tags", ti.i32))}

    @ti.kernel
    def observe(out: ti.types.ndarray(ti.f32, 1), source: ti.types.ndarray(ti.f32, 1),
                tags: ti.types.ndarray(ti.i32, 1)):
        for index in ti.grouped(state):
            linear = 0
            for axis in ti.static(range(len(shape))):
                linear = linear * shape[axis] + index[axis] - offset[axis]
            out[linear] = state[index].other
            source[linear] = state[index].value
            tags[linear] = state[index].tag

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            y[i] = .5 * x[i] * x[i]

    sequence = ti.algorithms.primitive_sequence()
    sequence.transform(state.value, state.other, scale=3, bias=7)
    sequence.transform(state.other, state.other, scale=2, bias=-1)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(observe, *[
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, array.dtype, ndim=1)
        for name, array in frames.items()
    ])
    graph = builder.compile()
    module = ti.aot.Module()
    cold = module.add_execution_plan("cold", graph, snapshot_state=True)
    document = json.loads(cold.to_json())
    assert len(document["resources"]) == 2
    assert all(r["kind"] == "module_array" for r in document["resources"])
    assert len(module._exported_kernels) == 1  # Only observe, no packing helper.
    assert all(op["parameters"]["strategy"] == "strided_scalar" for op in document["operations"][:2])
    np.testing.assert_array_equal(state.other.to_numpy(), np.full(shape, -9))
    graph.run(frames)
    np.testing.assert_array_equal(frames["out"].to_numpy(), 6 * values.ravel() + 13)
    assert all("dense_field" in call.plan.method_name for call in sequence._calls)
    state.other.fill(-9)
    module.add_execution_plan("warm", graph, snapshot_state=True)
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_Parent(graph, frames))
    parent = parent_builder.compile()
    module.add_execution_plan("parent", parent, snapshot_state=True, source_bindings=frames)
    assert len(module._exported_kernels) == 1
    module.add_execution_plan("read", ti.aot.ExecutionPlan([document["operations"][-1]]))
    prefix = json.loads(cold.to_json())
    for op in prefix["operations"][:2]:
        op["parameters"]["size"] = 11
    module.add_execution_plan("prefix", ti.aot.ExecutionPlan.from_json(json.dumps(prefix)))
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="reverse")
    module.add_execution_plan("suffix", ti.aot.ExecutionPlan(
        document["operations"] + [
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            dict(kind="kernel", entry="energy", arguments=["out", "energy"]),
            dict(kind="kernel", entry="reverse", arguments=["out", "energy"]),
        ], resources=document["resources"], gradients=dict(out="dx", energy="dy")))
    member = document["operations"][0]["bindings"]["src"]
    module.add_execution_plan("bad_copy", ti.aot.ExecutionPlan(
        [dict(kind="copy", source=member, destination="out")], resources=document["resources"]))
    module.add_execution_plan("bad_view", ti.aot.ExecutionPlan(
        [dict(kind="copy", source="tail", destination="out")],
        resources=document["resources"] + [
            dict(kind="ndarray_view", name="tail", array=member, offset=1, shape=[n - 1])]))
    artifact = tmp_path / "affine-field.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(json.dumps(dict(size=n, shape=shape)))
    invalid_artifact = tmp_path / "invalid-affine-field.tcm"
    with zipfile.ZipFile(artifact) as archive, zipfile.ZipFile(invalid_artifact, "w") as invalid:
        metadata_name = next(name for name in archive.namelist() if name.endswith("root_storage.json"))
        metadata = json.loads(archive.read(metadata_name))
        assert metadata["version"] == 2 and len(metadata["views"]) == 2
        views = tuple(metadata["views"].values())
        assert all(v["shape"] == list(shape) and v["index_strides_bytes"][-1] == 12 for v in views)
        assert views[0]["byte_offset"] != views[1]["byte_offset"]
        for item in archive.infolist():
            data = archive.read(item.filename)
            if item.filename == metadata_name:
                for view in metadata["views"].values():
                    view["index_strides_bytes"] = [stride * (1 << 30) for stride in view["index_strides_bytes"]]
                data = json.dumps(metadata).encode()
            invalid.writestr(item, data)
    parent.close()
    graph.close()
    sequence.clear()
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(__file__).with_name("aot_affine_field_probe.py")),
             mode, arch, str(artifact), str(invalid_artifact)],
            env=os.environ.copy(), capture_output=True, text=True, timeout=90)
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert not any(marker in text for marker in ("Validation Error", "SYNC-HAZARD", "cleanup failed")), text
