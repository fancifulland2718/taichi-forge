"""Compute/graphics buffer dependencies, including vertex/index and SSBO writes.

Run with Khronos synchronization validation and its shader-access heuristic to
check visibility as well as numeric results. Consumers reject validation errors.
"""

from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_graphics_export import selected_graphics


def _storage_write_pipeline():
    graphics = ti.hardware.graphics
    assets = Path(__file__).parent / "assets"
    try:
        return graphics.VulkanGraphicsPipeline(
            (assets / "graphics_sync.vert.spv").read_bytes(),
            (assets / "graphics_sync.frag.spv").read_bytes(),
            vertex_bindings=(graphics.VertexBinding(0, 8),),
            vertex_attributes=(graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),),
            shader_buffer_bindings=(
                graphics.ShaderBufferBinding(0, 0, "uniform"),
                graphics.ShaderBufferBinding(0, 1, "storage", "write"),
            ),
        )
    except (ValueError, RuntimeError) as exc:
        if "requires fragmentStoresAndAtomics" in str(exc):
            pytest.skip("Device lacks fragment storage writes")
        raise


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_indirect_rejects_same_pass_shader_write_alias(tmp_path):
    graphics = ti.hardware.graphics
    pipeline = _storage_write_pipeline()
    try:
        draw = pipeline.pass_draw(
            graphics.IndirectDraw(1, vertex_record_limit=3),
            vertex_buffers={0: "vertices"},
            indirect_buffer="commands",
            shader_buffers={(0, 0): "parameters", (0, 1): "state"},
        )
        recording = pipeline.record_pass((draw,), color="target")
        commands = ti.ndarray(ti.u32, 64)
        commands.from_numpy(np.array([3, 1, 0, 0] + [0] * 60, dtype=np.uint32))
        frame = dict(
            target=ti.Texture(ti.Format.rgba32f, (8, 8)),
            vertices=ti.ndarray(ti.f32, 6),
            parameters=ti.ndarray(ti.u32, 4),
            commands=commands,
            state=commands,
        )
        # Different Graph names are deliberately bound to the same allocation.
        with pytest.raises(RuntimeError, match="overlap writable shader storage"):
            recording.prepare_graph_execute(frame)
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="auto")
        module = ti.aot.Module()
        with closing(builder.compile()) as graph:
            module.add_execution_plan(
                "alias",
                graph,
                source_bindings=dict(target=frame["target"]),
                resource_bindings={pipeline: dict(zip(("shader_0", "shader_1"), pipeline._shader_artifacts))},
            )
        artifact = tmp_path / "indirect-alias.tcm"
        module.archive(str(artifact))
        with ti.aot.borrow_runtime() as runtime, ExitStack() as owners:
            loaded = owners.enter_context(closing(runtime.load_module(str(artifact))))
            inputs = {
                name: owners.enter_context(closing(runtime.ndarray(dtype, count, usage=usage)))
                for name, dtype, count, usage in (
                    ("vertices", "f32", 6, ("vertex",)),
                    ("parameters", "u32", 4, ("uniform",)),
                    ("commands", "u32", 64, ("storage", "indirect")),
                )
            }
            inputs["state"] = inputs["commands"]
            with pytest.raises(RuntimeError, match="overlap writable shader storage"):
                loaded.prepare_execution("alias", inputs)
    finally:
        pipeline.close()


@pytest.mark.parametrize("recipe", ["ordinary", "selected", "segmented", "independent"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_graphics_buffer_dependencies(tmp_path, recipe):
    independent = recipe == "independent"

    @ti.kernel
    def publish_independent(base: ti.types.ndarray(ti.u32, 1), scratch: ti.types.ndarray(ti.f32, 1)):
        scratch[0] = base[0]

    @ti.kernel
    def consume_independent(
        state: ti.types.ndarray(ti.f32, 1), scratch: ti.types.ndarray(ti.f32, 1),
        output: ti.types.ndarray(ti.f32, 1),
    ):
        for i in output:
            output[i] = 2 * (state[i] + scratch[0])

    @ti.kernel
    def publish(
        base: ti.types.ndarray(ti.u32, 1),
        parameters: ti.types.ndarray(ti.u32, 1),
        vertices: ti.types.ndarray(ti.f32, 1),
        indices: ti.types.ndarray(ti.u32, 1),
    ):
        parameters[0] = base[0]
        vertices[0], vertices[1] = -1, -1
        vertices[2], vertices[3] = 3, -1
        vertices[4], vertices[5] = -1, 3
        indices[0], indices[1], indices[2] = 0, 1, 2

    @ti.kernel
    def consume(state: ti.types.ndarray(ti.f32, 1), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = 2 * state[i]

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

    graphics, arg, kind = ti.hardware.graphics, ti.graph.Arg, ti.graph.ArgKind
    pipeline = _storage_write_pipeline()
    recording = pipeline.record_pass(
        (
            pipeline.pass_draw(
                graphics.Draw(3, index_bounds=(0, 2)),
                vertex_buffers={0: "vertices"},
                index_buffer="indices",
                shader_buffers={(0, 0): "parameters", (0, 1): "state"},
            ),
        ),
        color="target",
    )
    target = ti.Texture(ti.Format.rgba32f, (8, 8))
    names = ("base", "parameters", "vertices", "indices", "state", "output", "dout", "loss")
    dtypes = (ti.u32, ti.u32, ti.f32, ti.u32, ti.f32, ti.f32, ti.f32, ti.f32)
    arguments = {name: arg(kind.NDARRAY, name, dtype, ndim=1) for name, dtype in zip(names, dtypes)}
    scratch = arg(kind.NDARRAY, "scratch", ti.f32, ndim=1)

    def make_graph(adjoint):
        builder = ti.graph.GraphBuilder()
        if independent:
            builder.dispatch(publish_independent, arguments["base"], scratch)
        else:
            builder.dispatch(publish, *(arguments[name] for name in names[:4]))
        builder.append_native(recording, admission="auto")
        if independent:
            builder.dispatch(consume_independent, arguments["state"], scratch, arguments["output"])
        else:
            builder.dispatch(consume, arguments["state"], arguments["output"])
        if adjoint:
            builder.dispatch(seed, arguments["dout"], arguments["loss"], builder.private_ndarray("dloss", ti.f32, 1))
            builder.dispatch(energy, arguments["output"], arguments["loss"])
            builder.dispatch(energy.grad, arguments["output"], arguments["loss"])
        choice = {
            "selected": "graphics-queue-argument-images",
            "segmented": "immutable-argument-images",
            "independent": "independent-graphics-argument-images",
        }.get(recipe)
        return selected_graphics(builder.freeze(), choice) if choice else closing(builder.compile())

    frame = {name: ti.ndarray(dtype, count) for name, dtype, count in zip(names, dtypes, (1, 4, 6, 3, 64, 64, 64, 1))}
    if independent:
        frame["scratch"] = ti.ndarray(ti.f32, 1)
        frame["parameters"].from_numpy(np.zeros(4, dtype=np.uint32))
        frame["vertices"].from_numpy(np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32))
        frame["indices"].from_numpy(np.array([0, 1, 2], dtype=np.uint32))
    module = ti.aot.Module()
    # JIT Graph's Python entry does not bind ndarray gradients; AOT does below.
    with make_graph(False) as graph:
        for base in (1, 3, 0):
            frame["base"].from_numpy(np.array([base], dtype=np.uint32))
            bindings = dict({name: frame[name] for name in names[:6]}, target=target)
            if independent:
                bindings["scratch"] = frame["scratch"]
            graph.run(bindings)
            ti.sync()
            expected = 2 * (np.arange(64) + base)
            np.testing.assert_array_equal(frame["output"].to_numpy(), expected)
    with make_graph(True) as graph:
        plan = module.add_execution_plan(
            "step",
            graph,
            source_bindings=dict(target=target),
            resource_bindings={pipeline: dict(zip(("shader_0", "shader_1"), pipeline._shader_artifacts))},
            gradients=dict(output="dout", loss="dloss"),
        )
        if recipe in ("segmented", "independent"):
            selected = json.loads(plan.to_json())["submission_recipe"]
            assert selected["kind"] == ("vulkan_independent_graphics" if independent else "vulkan_segmented")
            if not independent:
                assert selected["ends"] == [1, 2, 6]
    artifact = tmp_path / "graphics-sync.tcm"
    module.archive(str(artifact))
    pipeline.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_graphics_sync_probe.py")),
                mode,
                str(artifact),
                recipe,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
