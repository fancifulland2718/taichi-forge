"""Qualify raster device preparation separately from graphics execution."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.aot._graph_export_calls import export_array_template_call
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from taichi_forge.hardware._raster_draw import PreparedRasterDraw
from tests import test_utils


@ti.kernel
def gather(
    mesh: ti.types.ndarray(ti.types.vector(12, ti.f32), 1),
    particles: ti.types.ndarray(ti.types.vector(12, ti.f32), 1),
    transforms: ti.types.ndarray(ti.f32, 3),
    indices: ti.types.ndarray(ti.i32, 1),
    output: ti.types.ndarray(ti.f32, 1),
):
    for i in range(36):
        output[i] = mesh[i // 12][i % 12]
    for i in range(24):
        output[36 + i] = particles[i // 12][i % 12]
    for i in range(16):
        output[60 + i] = transforms[0, i // 4, i % 4]
    for i in range(3):
        output[76 + i] = indices[i]


class PreparationExecutable(NativeGraphExecutable):
    def __init__(self, draws, output):
        self.draws = draws
        mesh, particles = draws
        self.arguments = (
            mesh._buffers["vertices"],
            particles._buffers["vertices"],
            mesh._buffers["transforms"],
            mesh._buffers["indices"],
            output,
        )

    def run(self, runtime_args=None):
        for draw in self.draws:
            for function, arguments in draw._updates:
                function(*arguments)
        gather(*self.arguments)

    def _export_aot(self, context):
        for draw in self.draws:
            draw._export_aot_preparation(context)
        export_array_template_call(context, gather, self.arguments)


class PreparationNode(NativeGraphNode):
    def __init__(self, executable):
        self.executable = executable

    def compile(self):
        return self.executable


@pytest.mark.skipif(not core.GGUI_AVAILABLE, reason="PreparedRasterDraw uses native FieldInfo")
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_raster_preparation_arrays_templates_and_scratch(tmp_path):
    # Template Fields use native root layouts, not the packed ndarray ABI.
    vertices = ti.Vector.field(3, ti.f32, 3, layout=ti.Layout.SOA)
    indices = ti.field(ti.i32, 3)
    transforms = ti.Matrix.field(4, 4, ti.f32, 1)
    vertices.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32))
    indices.from_numpy(np.array([0, 1, 2], np.int32))
    transforms.from_numpy(np.eye(4, dtype=np.float32)[None])
    centers, colors = (ti.Vector.ndarray(3, ti.f32, 2) for _ in range(2))
    radius = ti.ndarray(ti.f32, 2)
    centers.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], np.float32))
    colors.from_numpy(np.array([[0.25, 0.5, 0.75], [1, 0, 0.5]], np.float32))
    radius.from_numpy(np.array([0.125, 0.25], np.float32))
    # Only the preparation is under test: no Window, camera or draw execution.
    scene = SimpleNamespace(scene=SimpleNamespace(mesh_instance=lambda *args: None, particles=lambda *args: None))
    start = time.perf_counter()
    draws = (
        PreparedRasterDraw(scene, "mesh_instance", dict(vertices=vertices, indices=indices, transforms=transforms)),
        PreparedRasterDraw(
            scene, "particles", dict(centers=centers, radius=0.5, per_vertex_color=colors, per_vertex_radius=radius)
        ),
    )
    prepare_ms = (time.perf_counter() - start) * 1000
    output = ti.ndarray(ti.f32, 79)
    executable = PreparationExecutable(draws, output)
    executable.run()
    expected = output.to_numpy().tolist()
    reference = np.zeros(79, np.float32)
    for i, position in enumerate(((0, 0, 0), (1, 0, 0), (0, 1, 0))):
        reference[12 * i : 12 * i + 3] = position
        reference[12 * i + 5] = 1
    for i, (position, color, r) in enumerate((((1, 2, 3), (0.25, 0.5, 0.75), 0.125), ((4, 5, 6), (1, 0, 0.5), 0.25))):
        base = 36 + 12 * i
        reference[base : base + 3] = position
        reference[base + 3] = r
        reference[base + 8 : base + 12] = (*color, 1)
    reference[60:76] = np.eye(4).ravel()
    reference[76:] = (0, 1, 2)
    np.testing.assert_array_equal(expected, reference)

    @ti.kernel
    def seed(dout: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i in dout:
            dout[i] = 0
        loss[0] = 0
        dloss[0] = 1

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            loss[0] += 0.5 * x[i] ** 2

    builder = ti.graph.GraphBuilder()
    for _ in range(2):
        builder.append_native(PreparationNode(executable))
    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("output", "dout", "loss", "dloss")
    }
    builder.dispatch(seed, args["dout"], args["loss"], args["dloss"])
    builder.dispatch(energy, args["output"], args["loss"])
    builder.dispatch(energy.grad, args["output"], args["loss"])
    graph = builder.compile()
    module = ti.aot.Module()
    options = dict(
        source_bindings=dict(centers=centers, colors=colors, radius=radius, output=output),
        gradients=dict(output="dout", loss="dloss"),
    )
    with pytest.raises(ti.TaichiRuntimeError, match="snapshot_state"):
        module.add_execution_plan("invalid", graph, **options)
    assert not module._exported_kernels and not module._graph_export_failed
    start = time.perf_counter()
    plan = module.add_execution_plan("prepare", graph, snapshot_state=True, **options)
    export_ms = (time.perf_counter() - start) * 1000
    document = json.loads(plan.to_json())
    assert len(document["initialization"]) == 2, "One-time VBO zeroing must be shared by repeated calls"
    kernels = len(module._exported_kernels)
    module.add_execution_plan("again", graph, snapshot_state=True, **options)
    assert len(module._exported_kernels) == kernels
    artifact = tmp_path / "preparation.tcm"
    module.archive(str(artifact))
    (tmp_path / "reference.json").write_text(json.dumps(expected))
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(prepare_ms=prepare_ms, export_ms=export_ms, kernels=kernels, bytes=artifact.stat().st_size))
    )
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    graph.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_raster_preparation_probe.py")),
                mode,
                arch,
                str(tmp_path),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
