"""Direct mesh/task raster exports share the existing graphics execution core."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_graphics_export import selected_graphics
from tests.python.test_hardware_graphics import _spirv_header


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_graphics_mesh_backend_isolation():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.native_resource_capabilities("vulkan_graphics_pipeline", version=2)
        assert not runtime.native_operation_capabilities("vulkan_graphics_pass_direct", version=4)


@pytest.mark.parametrize("task,recipe", [(False, "ordinary"), (True, "selected"), (False, "specialized")])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_graphics_mesh_export_and_gradients(tmp_path, task, recipe):
    if not ti.hardware.graphics.is_mesh_shader_available(task_shader=task):
        pytest.skip("Required Vulkan mesh/task shaders are unavailable")

    @ti.kernel
    def copy_color(choice: ti.types.ndarray(ti.f32, 1), color: ti.types.ndarray(ti.f32, 1)):
        for i in color:
            color[i] = choice[i]

    @ti.kernel
    def sample(image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        value = image.fetch(ti.Vector([4, 4]), 0)
        for i in ti.static(range(4)):
            output[i] = value[i]

    @ti.kernel
    def seed(dout: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i in dout:
            dout[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in output:
            loss[0] += 0.5 * output[i] * output[i]

    started = time.perf_counter()
    graphics, arg, kind = ti.hardware.graphics, ti.graph.Arg, ti.graph.ArgKind
    assets = Path(__file__).parent / "assets"
    mesh_shader = (assets / "aot_graphics_color.mesh.spv").read_bytes()
    if recipe == "specialized":
        mesh_shader = graphics.SpirvShader.from_file(
            assets / "hardware_specialization.mesh.spv", stage="mesh"
        ).specialize({0: 1, 1: True})
    pipeline = graphics.VulkanMeshPipeline(
        mesh_shader,
        _spirv_header("2_triangle.frag.spv.h"),
        task_spirv=(assets / "hardware_graphics_mesh.task.spv").read_bytes() if task else None,
        shader_buffer_bindings=(graphics.ShaderBufferBinding(0, 0, "storage"),),
    )
    recording = pipeline.record_pass(
        (pipeline.pass_draw(graphics.MeshDraw(), shader_buffers={(0, 0): "color"}),), color="target"
    )
    target = ti.Texture(ti.Format.rgba32f, (8, 8))
    builder = ti.graph.GraphBuilder()
    builder.dispatch(copy_color, arg(kind.NDARRAY, "choice", ti.f32, ndim=1), arg(kind.NDARRAY, "color", ti.f32, ndim=1))
    builder.append_native(recording, admission="auto")
    output, dout, loss = [arg(kind.NDARRAY, name, ti.f32, ndim=1) for name in ("output", "dout", "loss")]
    builder.dispatch(sample, arg(kind.TEXTURE, "target", ndim=2), output)
    builder.dispatch(seed, dout, loss, builder.private_ndarray("dloss", ti.f32, 1))
    builder.dispatch(energy, output, loss)
    builder.dispatch(energy.grad, output, loss)
    module = ti.aot.Module()
    shader_inputs = {f"shader_{i}": shader for i, shader in enumerate(pipeline._shader_artifacts)}
    for name, shader in shader_inputs.items():
        (tmp_path / (name + ".spv")).write_bytes(shader.code)
    (tmp_path / "pipeline.json").write_text(json.dumps(pipeline._definition.native.parameter_dict()))

    def export(graph):
        for public in (False, True):
            plan = module.add_execution_plan(
                "public" if public else "private",
                graph,
                source_bindings=dict(target=target),
                resource_bindings={pipeline: "pipeline", target: "target"} if public else {pipeline: shader_inputs},
                gradients=dict(output="dout", loss="dloss"),
            )
            document = json.loads(plan.to_json())
            draw = next(op for op in document["operations"] if op.get("provider") == "vulkan_graphics")
            assert draw["version"] == 4 and draw["parameters"]["draws"][0]["mesh"] == [1, 1, 1]
            if not public:
                resource = next(r for r in document["resources"] if r.get("provider") == "vulkan_graphics")
                assert resource["version"] == 2

    if recipe == "selected":
        with selected_graphics(builder.freeze()) as graph:
            export(graph)
    else:
        graph = builder.compile()
        try:
            export(graph)
        finally:
            graph.close()
    artifact = tmp_path / "graphics-mesh.tcm"
    module.archive(str(artifact))
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(compile_ms=(time.perf_counter() - started) * 1000, artifact_bytes=artifact.stat().st_size))
    )
    pipeline.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_graphics_mesh_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
