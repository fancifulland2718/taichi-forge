"""Export fixed raster descriptor arrays, retaining aliases and logical views."""

import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_graphics_export import selected_graphics
from tests.python.test_hardware_graphics import _spirv_header


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_graphics_arrays_backend_isolation():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.native_operation_capabilities("vulkan_graphics_pass_direct", version=3)


@pytest.mark.parametrize("recipe", ["ordinary", "selected"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_graphics_arrays_views_and_gradients(tmp_path, recipe):
    if not ti.hardware.graphics.is_bindless_buffer_available():
        pytest.skip("Fixed storage descriptor arrays are unavailable")

    @ti.kernel
    def select(choice: ti.types.ndarray(ti.u32, 1), selector: ti.types.ndarray(ti.u32, 1)):
        selector[0] = choice[0]

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
    pipeline = graphics.VulkanGraphicsPipeline(
        (Path(__file__).parent / "assets/hardware_graphics_array_ranges.vert.spv").read_bytes(),
        _spirv_header("2_triangle.frag.spv.h"),
        vertex_bindings=(graphics.VertexBinding(0, 20),),
        vertex_attributes=(graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),),
        shader_buffer_bindings=(
            graphics.ShaderBufferArrayBinding(0, 0, 4),
            graphics.ShaderBufferBinding(0, 1, "uniform"),
        ),
    )
    recording = pipeline.record_pass(
        (
            pipeline.pass_draw(
                graphics.Draw(3),
                vertex_buffers={0: "vertices"},
                shader_buffers={(0, 0): ("c0", "c1", "c2", "c0"), (0, 1): "selector"},
            ),
        ),
        color="target",
    )
    target = ti.Texture(ti.Format.rgba32f, (8, 8))
    builder = ti.graph.GraphBuilder()
    builder.dispatch(select, arg(kind.NDARRAY, "choice", ti.u32, ndim=1), arg(kind.NDARRAY, "selector", ti.u32, ndim=1))
    builder.append_native(recording, admission="auto")
    output = arg(kind.NDARRAY, "output", ti.f32, ndim=1)
    dout, loss = [arg(kind.NDARRAY, name, ti.f32, ndim=1) for name in ("dout", "loss")]
    builder.dispatch(sample, arg(kind.TEXTURE, "target", ndim=2), output)
    builder.dispatch(seed, dout, loss, builder.private_ndarray("dloss", ti.f32, 1))
    builder.dispatch(energy, output, loss)
    builder.dispatch(energy.grad, output, loss)
    module = ti.aot.Module()

    def export(graph):
        return module.add_execution_plan(
            "export",
            graph,
            source_bindings=dict(target=target),
            resource_bindings={pipeline: dict(zip(("shader_0", "shader_1"), pipeline._shader_artifacts))},
            gradients=dict(output="dout", loss="dloss"),
        )

    if recipe == "selected":
        with selected_graphics(builder.freeze()) as graph:
            plan = export(graph)
    else:
        graph = builder.compile()
        try:
            plan = export(graph)
        finally:
            graph.close()
    document = json.loads(plan.to_json())
    raster = next(op for op in document["operations"] if op.get("provider") == "vulkan_graphics")
    table = raster["parameters"]["draws"][0]["buffers"][0]["arrays"]
    assert raster["version"] == 3 and len(table) == 4 and table[0] == table[-1]
    # 256-byte offsets satisfy Vulkan's required maximum storage alignment.
    # Padding is deliberately observable if a descriptor loses its offset.
    bank = [-91.0] * 256
    colors = ([1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1])
    for i, color in enumerate(colors):
        bank[64 * (i + 1) : 64 * (i + 1) + 4] = color
    for private in (False, True):
        derived = copy.deepcopy(document)
        derived["resources"] += [
            dict(kind="ndarray_view", name=f"c{i}", array="bank", offset=64 * (i + 1), shape=[4]) for i in range(3)
        ]
        if private:
            derived["resources"].append(
                dict(
                    kind="ndarray",
                    name="bank",
                    dtype="f32",
                    shape=[256],
                    initial_data=module.add_data(struct.pack("<256f", *bank)),
                )
            )
        module.add_execution_plan(
            "private_views" if private else "public_views", ti.aot.ExecutionPlan.from_json(json.dumps(derived))
        )
    artifact = tmp_path / "graphics-arrays.tcm"
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
                str(Path(__file__).with_name("aot_graphics_arrays_probe.py")),
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
