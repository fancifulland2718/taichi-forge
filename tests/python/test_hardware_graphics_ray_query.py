"""Compute refit -> raster fragment ray query -> compute, with fixed AS binding."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from tests.python.test_hardware_graphics_shadow import _pipeline


def _scene(custom_index=17):
    vertices = ti.ndarray(ti.f32, (3, 3))
    indices = ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array([[-1, -1, 0.5], [1, -1, 0.5], [0, 1, 0.5]], np.float32))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    blas = ti.hardware.ray.TriangleBLAS(vertices, indices)
    scene = ti.hardware.ray.InstanceTLAS([ti.hardware.ray.RayInstance(blas, custom_index=custom_index)])
    return blas, scene


@pytest.mark.parametrize("binding_recipe", [False, "segmented", "graphics_queue"])
@test_utils.test(arch=ti.vulkan, offline_cache=False, gfx_cmdlist_lazy_submit=True, gfx_cmdlist_max_dispatches=10000)
def test_graphics_ray_query_refit_fixed_binding_and_owner_close(binding_recipe, monkeypatch):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query is unavailable")
    gfx = ti.hardware.graphics
    blas, scene = _scene()
    pipeline = _pipeline(
        "hardware_graphics_ray_query.frag.spv",
        shader_acceleration_structure_bindings=(gfx.ShaderAccelerationStructureBinding(0, 0),),
    )
    # Indexed receiver geometry; the varying carries world-space position.
    raster_vertices = ti.ndarray(ti.f32, 24)
    data = np.zeros((4, 6), np.float32)
    data[:, :2] = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    data[:, 3:5] = data[:, :2]
    raster_vertices.from_numpy(data.reshape(-1))
    raster_indices = ti.ndarray(ti.u32, 6)
    raster_indices.from_numpy(np.array([0, 1, 2, 0, 2, 3], np.uint32))
    image = ti.Texture(ti.Format.rgba32f, (32, 24))
    result = ti.Vector.ndarray(4, ti.f32, (32, 24))
    transforms = ti.ndarray(ti.f32, (1, 12))
    shift = ti.ndarray(ti.f32, 1)

    @ti.kernel
    def transform(values: ti.types.ndarray(ti.f32, ndim=2), offset: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(12):
            values[0, i] = 0.0
            if i == 0 or i == 5 or i == 10:
                values[0, i] = 1.0
            if i == 3:
                values[0, i] = offset[0]

    @ti.kernel
    def consume(target: ti.types.texture(2), output: ti.types.ndarray()):
        for i, j in output:
            output[i, j] = target.fetch(ti.Vector([i, j]), 0)

    recording = pipeline.record_pass(
        (
            pipeline.pass_draw(
                gfx.Draw(6, index_bounds=(0, 3)),
                vertex_buffers={0: "vertices"},
                index_buffer="indices",
                shader_acceleration_structures={(0, 0): "scene"},
            ),
        ),
        color="target",
    )
    builder = ti.graph.GraphBuilder()
    refit = scene.record_refit_transforms()
    if binding_recipe != "graphics_queue":
        builder.dispatch(
            transform,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "transforms", ti.f32, ndim=2),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "shift", ti.f32, ndim=1),
        )
        builder.append_native(refit, admission="auto")
    builder.append_native(recording, admission="auto")
    builder.dispatch(
        consume,
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "target", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.types.vector(4, ti.f32), ndim=2),
    )
    context = materialized = None
    if binding_recipe:
        from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
        from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider

        definition = builder.freeze()
        catalog = definition.recipe_catalog(
            providers=(GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider())
        )
        suffix = (
            ":graphics-queue-argument-images" if binding_recipe == "graphics_queue" else ":immutable-argument-images"
        )
        recipe = next(
            entry.recipe
            for entry in catalog.entries()
            if any(fragment.fragment_key.endswith(suffix) for fragment in entry.recipe.fragments)
        )
        context = definition.materialization_context(provider_set=catalog.provider_set)
        materialized = context.materialize(recipe)
        graph = materialized.executor
    else:
        graph = builder.compile()
    arguments = dict(
        vertices=raster_vertices,
        indices=raster_indices,
        scene=scene,
        target=image,
        transforms=transforms,
        shift=shift,
        output=result,
    )
    if binding_recipe == "graphics_queue":
        arguments.pop("transforms")
        arguments.pop("shift")
    bound = graph.bind(arguments)
    prepared_refit = refit.prepare_graph_execute({"transforms": transforms})

    # A repeated frame must not rebuild the descriptor or traverse Python bindings.
    def unexpected(*args, **kwargs):
        raise AssertionError("graphics AS packet was prepared during replay")

    monkeypatch.setattr(type(recording), "_prepare_packet", unexpected)
    for displacement in (0, 4, 0):
        shift.fill(displacement)
        if binding_recipe == "graphics_queue":
            # Refit is an existing ordered native boundary, not an inline
            # graphics command. Its result feeds a fully recorded draw/consumer.
            transform(transforms, shift)
            prepared_refit()
        graph.run(bound)
        pixels = result.to_numpy()
        np.testing.assert_array_equal(pixels[16, 12], [1, 0, 17, 1] if not displacement else [0, -1, -1, 1])
        np.testing.assert_array_equal(pixels[0, 23], [0, -1, -1, 1])
    # Existing compute query sees the very same TLAS and typed hit metadata.
    rays, hits, ids = ti.ndarray(ti.f32, (1, 8)), ti.ndarray(ti.f32, (1, 4)), ti.ndarray(ti.i32, (1, 4))
    rays.from_numpy(np.array([[0, 0, 0, 0.001, 0, 0, 1, 1]], np.float32))
    scene.trace_typed(rays, hits, ids)
    np.testing.assert_array_equal(ids.to_numpy()[0], [0, 0, 17, 1])
    monkeypatch.undo()
    if binding_recipe == "graphics_queue":
        replacement_blas, replacement = _scene(custom_index=31)
        bound.update(scene=replacement)
        scene.close()
        graph.run(bound)
        np.testing.assert_array_equal(result.to_numpy()[16, 12], [1, 0, 31, 1])
        replacement.close()
        replacement_blas.close()
        with pytest.raises(RuntimeError, match="closed|retired"):
            graph.run(bound)
    graph.close()
    if materialized is not None:
        materialized.close()
        context.close()
    monkeypatch.undo()
    if scene.closed:
        blas.close()
        blas, scene = _scene()
        arguments["scene"] = scene

    # A separately prepared draw shares the same TLAS. Close clears its native
    # descriptor payload; retaining the callable must not retain allocations past reset.
    prepared = recording.prepare_graph_execute({key: arguments[key] for key in recording.binding_names})
    prepared()
    ti.sync()
    scene.close()
    with pytest.raises(RuntimeError, match="stale|closed"):
        prepared()
    pipeline.close()
    blas.close()
    ti.reset()
    with pytest.raises(RuntimeError, match="runtime|generation"):
        prepared()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_graphics_ray_query_rejects_missing_shader_declaration_before_submission():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query is unavailable")
    pipeline = _pipeline("hardware_graphics_ray_query.frag.spv")
    vertices = ti.ndarray(ti.f32, 18)
    color = ti.Texture(ti.Format.rgba32f, (32, 24))
    record = pipeline.record_pass(
        (pipeline.pass_draw(ti.hardware.graphics.Draw(3), vertex_buffers={0: "vertices"}),), color="target"
    )
    with pytest.raises(RuntimeError, match="AS.*missing"):
        record.prepare_graph_execute(dict(vertices=vertices, target=color))
    pipeline.close()
