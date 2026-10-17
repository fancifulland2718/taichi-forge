"""Reconstruct classic indirect draws with finite command/count views."""

import copy
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
def test_aot_graphics_indirect_backend_isolation():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.native_operation_capabilities("vulkan_graphics_pass_indirect")


@pytest.mark.parametrize("indexed,recipe", [(False, "ordinary"), (True, "selected")])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_graphics_indirect_views_and_gradients(tmp_path, indexed, recipe):
    if not ti.hardware.graphics.is_indirect_available(count_buffer=indexed):
        pytest.skip("Required Vulkan indirect command features are unavailable")

    @ti.kernel
    def publish(
        active: ti.types.ndarray(ti.u32, 1), commands: ti.types.ndarray(ti.u32, 1), count: ti.types.ndarray(ti.u32, 1)
    ):
        commands[1] = 3 if active[0] else 0
        commands[2], commands[3], commands[4] = 1, 0, 0
        if ti.static(indexed):
            commands[1], commands[5] = 3, 0
        count[1] = active[0]

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
        _spirv_header("2_triangle.vert.spv.h"),
        _spirv_header("2_triangle.frag.spv.h"),
        vertex_bindings=(graphics.VertexBinding(0, 20),),
        vertex_attributes=(
            graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
            graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
        ),
    )
    draw = graphics.IndirectDraw(
        1, 3, index_element_limit=3 if indexed else 0, command_offset=4, count_offset=4 if indexed else None
    )
    recording = pipeline.record_pass(
        (
            pipeline.pass_draw(graphics.Draw(3), vertex_buffers={0: "background"}),
            pipeline.pass_draw(
                draw,
                vertex_buffers={0: "vertices"},
                index_buffer="indices" if indexed else None,
                indirect_buffer="commands",
                count_buffer="count" if indexed else None,
            ),
        ),
        color="target",
        clear_color=(0, 0, 1, 1),
    )
    target = ti.Texture(ti.Format.rgba32f, (8, 8))
    builder = ti.graph.GraphBuilder()
    builder.dispatch(publish, *(arg(kind.NDARRAY, name, ti.u32, ndim=1) for name in ("active", "commands", "count")))
    builder.append_native(recording, admission="auto")
    output, dout, loss = [arg(kind.NDARRAY, name, ti.f32, ndim=1) for name in ("output", "dout", "loss")]
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
    operation = next(op for op in document["operations"] if op.get("provider") == "vulkan_graphics")
    assert operation["operation"] == "pass_indirect" and operation["version"] == 1
    indirect = operation["parameters"]["draws"][1]["indirect"]
    assert indirect["command_offset"] == 4 and ("count" in indirect) == indexed
    for private in (False, True):
        derived = copy.deepcopy(document)
        derived["resources"] += [
            dict(kind="ndarray_view", name="commands", array="bank", offset=4, shape=[6 if indexed else 5]),
            dict(kind="ndarray_view", name="count", array="bank", offset=24, shape=[2]),
        ]
        if private:
            derived["version"] = max(18, derived["version"])
            derived["resources"].append(
                dict(
                    kind="ndarray",
                    name="bank",
                    dtype="u32",
                    shape=[32],
                    usage=["storage", "indirect"],
                    initial_data=module.add_data(bytes(128)),
                )
            )
        module.add_execution_plan(
            "private_views" if private else "public_views", ti.aot.ExecutionPlan.from_json(json.dumps(derived))
        )
    artifact = tmp_path / "graphics-indirect.tcm"
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
                str(Path(__file__).with_name("aot_graphics_indirect_probe.py")),
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
