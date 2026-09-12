from pathlib import Path

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from taichi_forge.lang import impl
from tests import test_utils


def _sampled_pipeline():
    graphics = ti.hardware.graphics
    shaders = Path(__file__).parents[2] / "python" / "taichi_forge" / "shaders"
    return graphics.VulkanGraphicsPipeline(
        (shaders / "SetImage_vk_vert.spv").read_bytes(),
        (shaders / "SetImage_vk_frag.spv").read_bytes(),
        vertex_bindings=(graphics.VertexBinding(0, 48),),
        vertex_attributes=(
            graphics.VertexAttribute(0, 0, ti.Format.rgb32f, 0),
            graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 12),
            graphics.VertexAttribute(2, 0, ti.Format.rg32f, 24),
            graphics.VertexAttribute(3, 0, ti.Format.rgba32f, 32),
        ),
        shader_buffer_bindings=(graphics.ShaderBufferBinding(0, 1, "uniform", "read"),),
        shader_image_bindings=(graphics.ShaderImageBinding(0, 0),),
    )


def _quad_resources():
    data = np.zeros((6, 12), dtype=np.float32)
    data[:, :2] = [(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)]
    data[:, 6:8] = (data[:, :2] + 1) * 0.5
    vertices = ti.ndarray(ti.f32, shape=data.size)
    vertices.from_numpy(data.reshape(-1))
    uniform = ti.ndarray(ti.f32, shape=8)
    # SetImage UBO: lower, upper, x/y scale, transpose (int zero), padding.
    uniform.from_numpy(np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=np.float32))
    return vertices, uniform


def _recording(pipeline):
    draw = pipeline.pass_draw(
        ti.hardware.graphics.Draw(6),
        vertex_buffers={0: "vertices"},
        shader_buffers={(0, 1): "uniform"},
        shader_images={(0, 0): "source"},
    )
    return pipeline.record_pass((draw,), color="target")


@pytest.mark.parametrize("binding_recipe", [False, True])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_sampled_graphics_device_producer_consumer_and_prepared_rebind(monkeypatch, binding_recipe):
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")

    @ti.kernel
    def produce(
        image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8),
        value: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for x, y in image:
            image.store(ti.Vector([x, y]), ti.Vector([value[0], 0.25, 0.5, 1.0]))

    @ti.kernel
    def consume(
        image: ti.types.texture(num_dimensions=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=3),
    ):
        for x, y in ti.ndrange(result.shape[0], result.shape[1]):
            sampled = image.fetch(ti.Vector([x, y]), 0)
            for channel in ti.static(range(4)):
                result[x, y, channel] = sampled[channel]

    with _sampled_pipeline() as pipeline:
        vertices, uniform = _quad_resources()
        recording = _recording(pipeline)
        assert ("source", GraphAccess.READ) in tuple(
            (effect.resource, effect.access) for effect in recording.resource_effects
        )
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            produce,
            ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "source", ndim=2, fmt=ti.Format.rgba8),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "value", ti.f32, ndim=1),
        )
        builder.append_native(recording, admission="auto")
        builder.dispatch(
            consume,
            ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "target", ndim=2),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=3),
        )
        materialization = None
        if binding_recipe:
            from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
            from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider

            definition = builder.freeze()
            catalog = definition.recipe_catalog(
                providers=(GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider())
            )
            assert len(catalog.entries()) == 2
            recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
            materialization = definition.materialization_context(provider_set=catalog.provider_set)
            materialized = materialization.materialize(recipe)
            graph = materialized.executor
            assert materialized.manifest.submissions[0].replay_mode == "vulkan_secondary_frames_with_ordered_graphics"
            assert recipe.planned_physical_id != catalog.baseline.recipe.planned_physical_id
        else:
            graph = builder.compile()
        source = ti.Texture(ti.Format.rgba8, (64, 32))
        target = ti.Texture(ti.Format.rgba8, (32, 16))
        value = ti.ndarray(ti.f32, shape=1)
        result = ti.ndarray(ti.f32, shape=(32, 16, 4))
        bindings = graph.bind(
            dict(
                source=source,
                target=target,
                vertices=vertices,
                uniform=uniform,
                value=value,
                result=result,
            )
        )
        frame = bindings._version.execution_frame
        if binding_recipe:
            assert frame is not None
            assert len(frame._frames) == 2
            assert frame.argument_bytes() > 0
            executor = graph._instance._backend_executable
            native_prepare = executor._prepare

            def unexpected_native_prepare(*args, **kwargs):
                raise AssertionError("Bound execution rebuilt compute argument frames")

            monkeypatch.setattr(executor, "_prepare", unexpected_native_prepare)
        prepare = recording._prepare_packet

        def unexpected_prepare(*args, **kwargs):
            raise AssertionError("Bound execution rebuilt graphics descriptors")

        monkeypatch.setattr(recording, "_prepare_packet", unexpected_prepare)
        for red in (0.2, 0.8):
            value.fill(red)
            ti.sync()  # Exclude the input upload from the Graph queue counts.
            program = impl.get_runtime().prog
            before = dict(program._debug_vulkan_queue_submission_stats())
            graph.submit(bindings).wait()
            after = dict(program._debug_vulkan_queue_submission_stats())
            # Producer, graphics, consumer plus the immutable bridge command;
            # caching must preserve the original queue dependency shape.
            assert after["queue_submit_calls"] - before["queue_submit_calls"] == 3
            assert after["submitted_command_buffers"] - before["submitted_command_buffers"] == 4
            expected = np.broadcast_to([red, 0.25, 0.5, 1.0], (32, 16, 4))
            np.testing.assert_allclose(result.to_numpy(), expected, atol=1 / 255)
        if binding_recipe:
            stats = graph.execution_stats()
            assert stats.memory.persistent_argument_bytes == frame.argument_bytes()
            assert [s.last_path for s in stats.segments if s.kind == "cgraph"] == [
                "vulkan_prepared_compute_with_ordered_graphics"
            ] * 2
            assert stats.compiled_task_count == 2
            assert stats.counters_complete is False

        monkeypatch.setattr(recording, "_prepare_packet", prepare)
        if binding_recipe:
            monkeypatch.setattr(executor, "_prepare", native_prepare)
        revision = bindings.revision
        with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="alias|attachment"):
            bindings.update(source=target)
        assert bindings.revision == revision
        replacement = ti.Texture(ti.Format.rgba8, (48, 24))
        bindings.update(source=replacement)
        monkeypatch.setattr(recording, "_prepare_packet", unexpected_prepare)
        graph.run(bindings)
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1 / 255)
        if binding_recipe:
            # Preparation of a later compute segment may fail after the first
            # owns native resources. Roll back only the new publication.
            previous_version = bindings._version
            partial = []

            def fail_second(*args, **kwargs):
                if partial:
                    raise RuntimeError("injected later segment preparation failure")
                prepared = native_prepare(*args, **kwargs)
                partial.append(prepared)
                return prepared

            monkeypatch.setattr(recording, "_prepare_packet", prepare)
            monkeypatch.setattr(executor, "_prepare", fail_second)
            with pytest.raises(RuntimeError, match="later segment"):
                bindings.update(source=source)
            assert bindings._version is previous_version
            assert partial[0].argument_bytes() == 0
            graph.submit(bindings).wait()
            np.testing.assert_allclose(result.to_numpy(), expected, atol=1 / 255)
        graph.close()
        if binding_recipe:
            assert frame.argument_bytes() == 0
            assert previous_version.execution_frame.argument_bytes() == 0
            with pytest.raises(RuntimeError, match="closed"):
                frame.run()
            materialized.close()
            materialization.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_sampled_graphics_prepared_packet_lifetime_and_reset():
    if not ti.hardware.graphics.is_available():
        pytest.skip("Vulkan graphics commands are unavailable")
    pipeline = _sampled_pipeline()
    vertices, uniform = _quad_resources()
    source = ti.Texture(ti.Format.rgba8, (16, 8))
    target = ti.Texture(ti.Format.rgba8, (16, 8))
    recording = _recording(pipeline)
    program = impl.get_runtime().prog
    before = dict(program._debug_vulkan_graphics_resource_stats())
    queue_before = dict(program._debug_vulkan_queue_submission_stats())
    execute = recording.prepare_graph_execute(dict(source=source, target=target, vertices=vertices, uniform=uniform))
    queue_prepared = dict(program._debug_vulkan_queue_submission_stats())
    assert queue_prepared["queue_submit_calls"] == queue_before["queue_submit_calls"]
    assert queue_prepared["submitted_command_buffers"] == queue_before["submitted_command_buffers"]
    prepared = dict(program._debug_vulkan_graphics_resource_stats())
    assert prepared["prepared_resource_leases"] == before["prepared_resource_leases"] + 1
    assert prepared["prepared_draw_resources"] == before["prepared_draw_resources"] + 1
    assert prepared["prepared_descriptor_sets"] == before["prepared_descriptor_sets"] + 1
    assert prepared["prepared_raster_resources"] == before["prepared_raster_resources"] + 1
    execute()
    execute()
    replayed = dict(program._debug_vulkan_graphics_resource_stats())
    for key in (
        "prepared_resource_leases",
        "prepared_draw_resources",
        "prepared_descriptor_sets",
        "prepared_raster_resources",
    ):
        assert replayed[key] == prepared[key]
    pipeline.close()
    closed = dict(program._debug_vulkan_graphics_resource_stats())
    assert closed["prepared_resource_leases"] == before["prepared_resource_leases"]
    assert closed["prepared_draw_resources"] == before["prepared_draw_resources"]
    assert closed["prepared_descriptor_sets"] == before["prepared_descriptor_sets"]
    assert closed["prepared_raster_resources"] == before["prepared_raster_resources"]
    ti.sync()
    with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="closed|stale"):
        execute()
    stats = dict(program._debug_vulkan_graphics_resource_stats())
    assert stats["retiring"] == 0
    # Retained host packets must not own a device pipeline beyond reset.
    ti.reset()
    with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="runtime|generation"):
        execute()


@pytest.mark.parametrize("retirement", ["pipeline_close", "reset"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_graphics_binding_recipe_search_resolve_and_retirement(retirement):
    from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
    from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider

    @ti.kernel
    def write(image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8)):
        for x, y in image:
            image.store(ti.Vector([x, y]), ti.Vector([0.75, 0.25, 0.5, 1.0]))

    @ti.kernel
    def read(image: ti.types.texture(num_dimensions=2), output: ti.types.ndarray(dtype=ti.f32, ndim=3)):
        for x, y in ti.ndrange(output.shape[0], output.shape[1]):
            value = image.fetch(ti.Vector([x, y]), 0)
            for c in ti.static(range(4)):
                output[x, y, c] = value[c]

    pipeline = _sampled_pipeline()
    vertices, uniform = _quad_resources()
    recording = _recording(pipeline)

    def definition():
        builder = ti.graph.GraphBuilder()
        # Both leading and trailing ordered boundaries must be kept, not just
        # graphics found between two reusable compute segments.
        builder.append_native(recording, admission="auto")
        builder.dispatch(
            read,
            ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "target", ndim=2),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=3),
        )
        builder.append_native(recording, admission="auto")
        return builder.freeze()

    providers = (GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider())
    frozen = definition()
    only_graphics = ti.graph.GraphBuilder()
    only_graphics.append_native(recording, admission="auto")
    assert GraphBindingFrameRecipeProvider().fragments(only_graphics.freeze()) == ()
    source, target = (ti.Texture(ti.Format.rgba8, (16, 8)) for _ in range(2))
    output = ti.ndarray(ti.f32, (16, 8, 4))
    write(source)
    args = dict(source=source, target=target, output=output, vertices=vertices, uniform=uniform)
    expected = np.broadcast_to([0.75, 0.25, 0.5, 1.0], (16, 8, 4))
    observed = set()

    def evaluate(graph, request):
        bindings = graph.bind(args)
        graph.submit(bindings).wait()
        np.testing.assert_allclose(output.to_numpy(), expected, atol=1 / 255)
        observed.add(request.recipe_id)
        # Deterministic search-contract test, not an acceleration measurement.
        return {"prepared_candidate": float(bindings._version.execution_frame is not None)}

    session = frozen.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(objectives=(("prepared_candidate", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=4, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    )
    decision = session.run(evaluate)
    assert decision.status == "selected", decision.report.results
    assert decision.report.search_complete
    assert len(observed) == 2
    fresh = definition()
    assert fresh.semantic_graph_id == frozen.semantic_graph_id
    selection = fresh.resolve_recipe(decision.selection_artifact, providers=providers)
    with fresh.materialize(selection) as materialized:
        graph = materialized.executor
        bindings = graph.bind(args)
        frame = bindings._version.execution_frame
        assert len(frame._frames) == 1
        assert len(frame._actions) == 3
        graph.run(args)  # Raw mappings explicitly prepare and retire a frame.
        np.testing.assert_allclose(output.to_numpy(), expected, atol=1 / 255)
        assert len(graph._instance._backend_executable._frames) == 1
        ticket = graph.submit(bindings)
        if retirement == "pipeline_close":
            pipeline.close()
            ticket.wait()
            np.testing.assert_allclose(output.to_numpy(), expected, atol=1 / 255)
            with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="closed|stale"):
                graph.submit(bindings)
            graph.close()
        else:
            ti.reset()
            with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="runtime|generation"):
                graph.submit(bindings)
        assert frame.argument_bytes() == 0
        with pytest.raises(RuntimeError, match="closed"):
            frame.run()
    pipeline.close()
