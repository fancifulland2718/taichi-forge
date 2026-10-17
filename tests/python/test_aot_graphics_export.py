"""Reconstruct Graph raster recordings with shared owners and physical ordering."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_hardware_graphics import _spirv_header
from tests.python.aot_vulkan_graphics_probe import pipeline_spec
from tests.python.test_graph_binding_frame_recipe import _catalog


@contextmanager
def selected_graphics(definition, choice="graphics-queue-argument-images"):
    catalog = _catalog(definition)
    recipe = next(
        entry.recipe
        for entry in catalog.entries()
        if any(fragment.fragment_key.endswith(":" + choice) for fragment in entry.recipe.fragments)
    )
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            yield materialized.executor


@pytest.mark.parametrize("recipe", ["ordinary", "selected"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_graphics_recording_export_and_gradients(tmp_path, recipe):
    @ti.kernel
    def sample(image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for i in range(64):
            value = image.fetch(ti.Vector([i % 8, i // 8]), 0)
            for j in ti.static(range(4)):
                output[4 * i + j] = value[j]

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

    graphics = ti.hardware.graphics
    start = time.perf_counter()
    pipelines = []
    for instanced in (False, True):
        spec = pipeline_spec(instanced)
        shaders = [_spirv_header(f"{2 if instanced else 3}_triangle.{stage}.spv.h") for stage in ("vert", "frag")]
        for index, code in enumerate(shaders):
            (tmp_path / f"shader_{int(instanced)}_{index}.spv").write_bytes(code)
        pipeline = graphics.VulkanGraphicsPipeline(
            *shaders,
            vertex_bindings=tuple(graphics.VertexBinding(**item) for item in spec["vertices"]),
            vertex_attributes=tuple(
                graphics.VertexAttribute(**dict(item, format=ti.Format.__members__[item["format"]]))
                for item in spec["attributes"]
            ),
            shader_buffer_bindings=() if instanced else (graphics.ShaderBufferBinding(0, 3, kind="uniform"),),
            shader_image_bindings=() if instanced else (graphics.ShaderImageBinding(0, 5),),
            color_targets=(graphics.ColorTarget(),),
        )
        pipelines.append(pipeline)
        (tmp_path / f"pipeline_{int(instanced)}.json").write_text(pipeline._aot_pipeline_spec)
    pipeline, instance = pipelines
    first = pipeline.record_pass(
        (
            pipeline.pass_draw(
                graphics.Draw(3),
                vertex_buffers={0: "vertices"},
                shader_buffers={(0, 3): "uniform"},
                shader_images={(0, 5): "source"},
            ),
        ),
        color="color",
        depth="depth",
        clear_color=(0, 0, 1, 1),
        viewport=(0, 0, 4, 8),
    )
    last = instance.record_pass(
        (
            instance.pass_draw(
                graphics.Draw(3, first_instance=1, index_bounds=(0, 2)),
                vertex_buffers={0: "positions", 1: "colors"},
                index_buffer="indices",
            ),
        ),
        color="color",
        depth="depth",
        color_load_op="load",
        depth_load_op="load",
        viewport=(4, 0, 4, 8),
    )
    sources = dict(
        color=ti.Texture(ti.Format.rgba32f, (8, 8)),
        depth=ti.Texture(ti.Format.depth32f, (8, 8)),
        source=ti.Texture(ti.Format.rgba32f, (1, 1)),
    )
    sources["result"] = sources["color"]
    inputs = {
        owner: {f"shader_{i}": artifact for i, artifact in enumerate(owner._shader_artifacts)} for owner in pipelines
    }
    inputs[sources["source"]] = dict(pixels="pixels")
    arg, kind = ti.graph.Arg, ti.graph.ArgKind
    output, dout, loss = [arg(kind.NDARRAY, name, ti.f32, ndim=1) for name in ("features", "dfeatures", "loss")]
    module = ti.aot.Module()
    costs = []

    for early_read in (False, True):
        builder = ti.graph.GraphBuilder()
        if early_read:
            builder.dispatch(sample, arg(kind.TEXTURE, "result", ndim=2), output)
        builder.append_native(first, admission="auto")
        builder.append_native(last, admission="auto")
        builder.dispatch(sample, arg(kind.TEXTURE, "result", ndim=2), output)
        dloss = builder.private_ndarray("dloss", ti.f32, 1)
        builder.dispatch(seed, dout, loss, dloss)
        builder.dispatch(energy, output, loss)
        builder.dispatch(energy.grad, output, loss)

        def export(graph):
            start = time.perf_counter()
            resource_inputs = dict(inputs)
            if early_read:
                resource_inputs[sources["color"]] = dict(pixels="target_pixels")
            plan = module.add_execution_plan(
                "seeded" if early_read else "step",
                graph,
                source_bindings=sources,
                resource_bindings=resource_inputs,
                gradients=dict(features="dfeatures", loss="dloss"),
            )
            document = json.loads(plan.to_json())
            images = [value for value in document["resources"] if value.get("provider") == "image"]
            assert len(images) == 3 and sum(x["parameters"].get("attachment", False) for x in images) == 2
            assert len(document["initialization"]) == (2 if early_read else 1)
            native = [value for value in document["operations"] if value.get("provider") == "vulkan_graphics"]
            assert len(native) == 2
            assert native[0]["bindings"]["color_0"] == native[1]["bindings"]["color_0"]
            if not early_read:
                count = len(module._exported_kernels)
                public_inputs = {owner: f"pipeline_{i}" for i, owner in enumerate(pipelines)}
                public_inputs.update({sources[name]: name for name in ("color", "depth", "source")})
                module.add_execution_plan(
                    "public",
                    graph,
                    source_bindings=sources,
                    resource_bindings=public_inputs,
                    gradients=dict(features="dfeatures", loss="dloss"),
                )
                assert len(module._exported_kernels) == count
            costs.append(dict(seeded=early_read, export_ms=(time.perf_counter() - start) * 1000))

        if recipe == "selected":
            definition = builder.freeze()
            with selected_graphics(definition) as graph:
                export(graph)
            with selected_graphics(definition, "immutable-argument-images") as graph:
                with pytest.raises(ti.TaichiRuntimeError, match="ordered Vulkan compute"):
                    ti.aot.Module().add_execution_plan("bad", graph)
        else:
            graph = builder.compile()
            try:
                export(graph)
            finally:
                graph.close()

    # The legacy single-draw recording shares the same explicit pass factory.
    single = ti.graph.GraphBuilder()
    single.append_native(
        instance.record(
            graphics.Draw(3, first_instance=1), color="color", vertex_buffers={0: "positions", 1: "colors"}
        ),
        admission="auto",
    )
    single.dispatch(sample, arg(kind.TEXTURE, "color", ndim=2), output)
    graph = single.compile()
    try:
        module.add_execution_plan(
            "single",
            graph,
            source_bindings=dict(color=sources["color"]),
            resource_bindings={instance: inputs[instance]},
        )
        bad = ti.aot.Module()
        wrong = dict(inputs[instance], shader_0=pipeline._shader_artifacts[0])
        with pytest.raises(ti.TaichiRuntimeError, match="shader asset differs"):
            bad.add_execution_plan(
                "bad", graph, source_bindings=dict(color=sources["color"]), resource_bindings={instance: wrong}
            )
        assert not bad._exported_kernels and not bad._graph_export_failed
        mipped = ti.Texture(ti.Format.rgba32f, (8, 8), mip_levels=2)
        with pytest.raises(ti.TaichiRuntimeError, match="single-mip"):
            bad.add_execution_plan(
                "bad", graph, source_bindings=dict(color=mipped), resource_bindings={instance: inputs[instance]}
            )
    finally:
        graph.close()

    # Mixed integer attachments preserve clear bits and shader values. Explicit
    # blend/depth state exercises the source/native reconstruction certificate.
    mrt = graphics.VulkanGraphicsPipeline(
        _spirv_header("2_triangle.vert.spv.h"),
        (Path(__file__).parent / "assets/hardware_graphics_mrt.frag.spv").read_bytes(),
        vertex_bindings=(graphics.VertexBinding(0, 20),),
        vertex_attributes=(
            graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
            graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
        ),
        color_targets=(
            graphics.ColorTarget(
                blending=True,
                src_color_factor="one",
                dst_color_factor="zero",
                src_alpha_factor="one",
                dst_alpha_factor="zero",
            ),
            graphics.ColorTarget(),
            graphics.ColorTarget(),
        ),
        depth_compare="less_equal",
        depth_bias_constant=0.1,
        depth_bias_slope=-0.25,
    )
    pipelines.append(mrt)
    mrt_sources = {
        name: ti.Texture(fmt, (8, 4))
        for name, fmt in (("color", ti.Format.rgba32f), ("unsigned", ti.Format.r32u), ("signed", ti.Format.r32i))
    }
    recording = mrt.record_pass(
        (mrt.pass_draw(graphics.Draw(3), vertex_buffers={0: "mrt_vertices"}),),
        colors=(
            graphics.ColorAttachment("color", clear_value=(0.25, 0.5, 0.75, 1)),
            graphics.ColorAttachment("unsigned", clear_value=(0xFEDCBA98, 0, 0, 0)),
            graphics.ColorAttachment("signed", clear_value=(-987654321, 0, 0, 0)),
        ),
        viewport=(0, 0, 4, 4),
    )

    @ti.kernel
    def consume(
        unsigned: ti.types.rw_texture(2, fmt=ti.Format.r32u),
        signed: ti.types.rw_texture(2, fmt=ti.Format.r32i),
        ids: ti.types.ndarray(ti.u32, 1),
        primitives: ti.types.ndarray(ti.i32, 1),
    ):
        for i in ids:
            xy = ti.Vector([i % 8, i // 8])
            ids[i], primitives[i] = unsigned.load(xy).x, signed.load(xy).x

    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    builder.dispatch(
        consume,
        arg(kind.RWTEXTURE, "unsigned", ndim=2, fmt=ti.Format.r32u),
        arg(kind.RWTEXTURE, "signed", ndim=2, fmt=ti.Format.r32i),
        arg(kind.NDARRAY, "ids", ti.u32, ndim=1),
        arg(kind.NDARRAY, "primitives", ti.i32, ndim=1),
    )
    graph = builder.compile()
    try:
        plan = module.add_execution_plan(
            "mrt",
            graph,
            source_bindings=mrt_sources,
            resource_bindings={mrt: dict(zip(("shader_0", "shader_1"), mrt._shader_artifacts))},
        )
        assert not json.loads(plan.to_json()).get("initialization")
    finally:
        graph.close()

    indirect = ti.graph.GraphBuilder()
    indirect.append_native(
        instance.record_pass(
            (
                instance.pass_draw(
                    graphics.IndirectDraw(1, 3),
                    vertex_buffers={0: "positions", 1: "colors"},
                    indirect_buffer="commands",
                ),
            ),
            color="color",
        ),
        admission="auto",
    )
    graph = indirect.compile()
    try:
        with pytest.raises(ti.TaichiRuntimeError, match="direct/indexed"):
            ti.aot.Module().add_execution_plan("bad", graph, source_bindings=dict(color=sources["color"]))
    finally:
        graph.close()

    if recipe == "selected":
        fork = ti.graph.GraphBuilder()
        fork.dispatch(seed, dout, loss, fork.private_ndarray("dloss", ti.f32, 1))
        fork.append_native(last, admission="auto")
        fork.dispatch(sample, arg(kind.TEXTURE, "color", ndim=2), output)
        with selected_graphics(fork.freeze(), "independent-graphics-argument-images") as graph:
            with pytest.raises(ti.TaichiRuntimeError, match="selected storage/binding recipe"):
                ti.aot.Module().add_execution_plan("bad", graph)

    artifact = tmp_path / "graphics-export.tcm"
    module.archive(str(artifact))
    (tmp_path / "producer.json").write_text(
        json.dumps(
            dict(
                producer_ms=(time.perf_counter() - start) * 1000, export=costs, artifact_bytes=artifact.stat().st_size
            ),
            indent=2,
        )
    )
    for pipeline in pipelines:
        pipeline.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_graphics_export_probe.py")),
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
