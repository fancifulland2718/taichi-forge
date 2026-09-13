import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _triangle_blas(z=0):
    vertices = ti.ndarray(ti.f32, (3, 3))
    indices = ti.ndarray(ti.i32, (1, 3))
    vertices.from_numpy(np.array([[0, 0, z], [2, 0, z], [0, 2, z]], np.float32))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    return ti.hardware.ray.TriangleBLAS(vertices, indices)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
@pytest.mark.parametrize("storage_kind", ["ndarray", "field_range"])
@pytest.mark.parametrize("binding_recipe", [False, True])
def test_device_tlas_transforms_compose_without_repreparing_and_retain_owners(
    storage_kind, binding_recipe, monkeypatch
):
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query is unavailable")
    count = 131  # Two dispatch groups and a partial tail.
    blases = (_triangle_blas(), _triangle_blas(0.25))
    instances = [
        ti.hardware.ray.RayInstance(
            blases[i % 2],
            transform=(1, 0, 0, 4 * i, 0, 1, 0, 0, 0, 0, 1, 0),
            mask=0 if i % 5 == 0 else 0xFF,
            custom_index=0xFFFF00 + i,
        )
        for i in range(count)
    ]
    tlas = ti.hardware.ray.InstanceTLAS(instances)
    memory_before = dict(
        impl.get_runtime().prog._vulkan_ray_resource_memory_stats(tlas._handle)
    )
    if storage_kind == "field_range":
        padding = ti.field(ti.i32)
        source = ti.field(ti.f32)
        fields = ti.FieldsBuilder()
        fields.dense(ti.i, 7).place(padding)
        fields.dense(ti.ij, (count + 2, 12)).place(source)
        tree = fields.finalize()
        padding.fill(917)
        source.fill(19)
        transforms = ti.experimental.ndarray_view(
            source, slices=(slice(1, count + 1), slice(None))
        )
        assert transforms.descriptor.byte_offset % 16 != 0
    else:
        source = transforms = ti.ndarray(ti.f32, (count, 12))
        source.fill(19)
    shift = ti.ndarray(ti.f32, (1,))
    rays = ti.ndarray(ti.f32, (count, 8))
    hits = ti.ndarray(ti.f32, (count, 4))
    hit_indices = ti.ndarray(ti.i32, (count, 4))
    selected = ti.ndarray(ti.i32, (count,))
    ray_values = np.tile([0.5, 0.5, 2, 0.001, 0, 0, -1, 100], (count, 1))
    ray_values[:, 0] += 4 * np.arange(count)
    rays.from_numpy(ray_values.astype(np.float32))
    shift.fill(0)

    @ti.kernel
    def produce(
        values: ti.types.ndarray(ti.f32, ndim=2),
        offset: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in range(values.shape[0]):
            for word in ti.static(range(12)):
                values[i, word] = 0
            values[i, 0] = 1
            values[i, 5] = 1
            values[i, 10] = 1
            values[i, 3] = 4 * i + offset[0]

    @ti.kernel
    def consume(
        ids: ti.types.ndarray(ti.i32, ndim=2),
        output: ti.types.ndarray(ti.i32, ndim=1),
    ):
        for i in output:
            output[i] = ids[i, 2]

    recording = tlas.record_refit_transforms()
    assert recording.command_count == 2
    assert recording.no_host_readback
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        produce,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "transforms", ti.f32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "shift", ti.f32, ndim=1),
    )
    builder.append_native(recording, admission="auto")
    builder.append_native(tlas.record_typed(count), admission="auto")
    builder.dispatch(
        consume,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "hit_indices", ti.i32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "selected", ti.i32, ndim=1),
    )
    context = materialized = None
    if binding_recipe:
        from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
        from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider

        definition = builder.freeze()
        catalog = definition.recipe_catalog(
            providers=(GraphRuntimeAssemblyProvider(), GraphBindingFrameRecipeProvider())
        )
        candidates = [entry.recipe for entry in catalog.entries() if entry.recipe.fragments]
        assert len(candidates) == 1
        context = definition.materialization_context(provider_set=catalog.provider_set)
        materialized = context.materialize(candidates[0])
        graph = materialized.executor
        assert graph._instance.physical_submission_mode == "vulkan_secondary_frames_with_ordered_native_published"
    else:
        graph = builder.compile()
    arguments = dict(
        transforms=transforms,
        shift=shift,
        rays=rays,
        hits=hits,
        hit_indices=hit_indices,
        selected=selected,
    )
    if binding_recipe and storage_kind == "field_range":
        # Prepared native actions accept qualified field views, but secondary
        # compute argument images still require Program ndarray owners. Do not
        # silently stage the view or claim the ordinary route was retained.
        with pytest.raises(RuntimeError, match="Program ndarray owners"):
            graph.bind(arguments)
        graph.close()
        materialized.close()
        context.close()
        tlas.close()
        for blas in blases:
            blas.close()
        return
    bindings = graph.bind(arguments)
    assert bindings.fast_path_qualified, bindings.statistics()
    np.testing.assert_array_equal(source.to_numpy(), 19)
    memory_after = dict(
        impl.get_runtime().prog._vulkan_ray_resource_memory_stats(tlas._handle)
    )
    assert memory_after == memory_before  # Packing reuses the 64*N input buffer.
    assert memory_after["geometry_input_requested_bytes"] == count * 64

    # Independent BLAS Python handles can close; the TLAS retains their storage.
    for blas in blases:
        blas.close()

    def unexpected_cold_work(*args, **kwargs):
        raise AssertionError("TLAS transform validation or host packing entered replay")

    expected_ids = np.arange(count, dtype=np.int32) + 0xFFFF00
    expected_ids[::5] = -1
    with monkeypatch.context() as patched:
        for name in ("_prepare_packet", "_binding_description"):
            patched.setattr(type(recording), name, unexpected_cold_work)
        patched.setattr(ti.hardware.ray.RayInstance, "_to_core", unexpected_cold_work)
        for displacement in (0, 2, 0):
            shift.fill(displacement)
            program = impl.get_runtime().prog
            if binding_recipe:
                # Flush the producer before observing the recipe's own queue
                # publication. A later array read must not be what submits it.
                ti.sync()
                before_submit = program._debug_vulkan_queue_submission_stats()
            graph.run(bindings)
            if binding_recipe:
                after_submit = program._debug_vulkan_queue_submission_stats()
                assert (
                    after_submit["queue_submit_calls"]
                    == before_submit["queue_submit_calls"] + 1
                )
            expected = expected_ids if displacement == 0 else np.full(count, -1)
            np.testing.assert_array_equal(selected.to_numpy(), expected)
            if displacement == 0:
                visible = np.arange(count) % 5 != 0
                np.testing.assert_array_equal(
                    hit_indices.to_numpy()[visible, 1], np.arange(count)[visible]
                )
                np.testing.assert_allclose(
                    hits.to_numpy()[visible, 0],
                    (2 - 0.25 * (np.arange(count) % 2))[visible],
                )
        if binding_recipe:
            # The same publisher must join Graph.submit's existing transaction,
            # not submit each compute/native segment independently.
            ti.sync()
            before_submit = program._debug_vulkan_queue_submission_stats()
            ticket = graph.submit(bindings)
            after_submit = program._debug_vulkan_queue_submission_stats()
            assert (
                after_submit["queue_submit_calls"]
                == before_submit["queue_submit_calls"] + 1
            )
            ticket.wait()
            np.testing.assert_array_equal(selected.to_numpy(), expected_ids)
    revision = bindings.revision
    for bad, reason in (
        (ti.ndarray(ti.i32, (count, 12)), "dtype"),
        (ti.ndarray(ti.f32, (count - 1, 12)), "instance count"),
        (
            ti.experimental.ndarray_view(
                ti.ndarray(ti.f32, (2 * count, 12)),
                slices=(slice(0, 2 * count, 2), slice(None)),
            ),
            "compact",
        ),
    ):
        with pytest.raises(RuntimeError, match=reason):
            bindings.update(transforms=bad)
        assert bindings.revision == revision
    if binding_recipe:
        # Raw mappings intentionally prepare temporary frames; they still
        # publish the complete recipe before returning, then retire safely.
        ti.sync()
        before_submit = program._debug_vulkan_queue_submission_stats()
        graph.run(arguments)
        after_submit = program._debug_vulkan_queue_submission_stats()
        assert (
            after_submit["queue_submit_calls"]
            == before_submit["queue_submit_calls"] + 1
        )
        np.testing.assert_array_equal(selected.to_numpy(), expected_ids)
    graph.run(bindings)
    np.testing.assert_array_equal(selected.to_numpy(), expected_ids)
    if storage_kind == "field_range":
        np.testing.assert_array_equal(source.to_numpy()[[0, count + 1]], 19)
        np.testing.assert_array_equal(padding.to_numpy(), 917)
        prepared = recording.prepare_graph_execute({"transforms": transforms})
        tree.destroy()
        with pytest.raises(RuntimeError, match="retired|destroyed|generation"):
            prepared()
        with pytest.raises(RuntimeError, match="retired|destroyed|generation"):
            graph.run(bindings)
    else:
        previous = source.to_numpy()
        replacement = ti.ndarray(ti.f32, (count, 12))
        bindings.update(transforms=replacement)
        shift.fill(2)
        graph.run(bindings)
        np.testing.assert_array_equal(source.to_numpy(), previous)
        np.testing.assert_array_equal(selected.to_numpy(), -1)
        prepared = recording.prepare_graph_execute({"transforms": replacement})
        tlas.close()
        with pytest.raises(RuntimeError, match="closed"):
            prepared()
        with pytest.raises(RuntimeError, match="closed"):
            graph.run(bindings)
    graph.close()
    if context is not None:
        materialized.close()
        context.close()
    tlas.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_device_tlas_matrix_layout_host_metadata_and_reset_contract():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query is unavailable")
    blas = _triangle_blas()
    tlas = ti.hardware.ray.InstanceTLAS(
        [ti.hardware.ray.RayInstance(blas, custom_index=71)]
    )
    matrix = ti.Matrix.ndarray(3, 4, ti.f32, shape=(1,))
    value = np.array([[[2, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0]]], np.float32)
    matrix.from_numpy(value)
    scalar = ti.ndarray(ti.f32, (1, 3, 4))
    scalar.from_numpy(value)
    rays = ti.ndarray(ti.f32, (1, 8))
    hits = ti.ndarray(ti.f32, (1, 4))
    ids = ti.ndarray(ti.i32, (1, 4))
    rays.from_numpy(np.array([[5, 0.5, 2, 0.001, 0, 0, -1, 100]], np.float32))
    for transforms in (matrix, scalar):
        tlas.refit_transforms(transforms)
        tlas.trace_typed(rays, hits, ids)
        np.testing.assert_array_equal(ids.to_numpy(), [[0, 0, 71, 1]])
        np.testing.assert_allclose(hits.to_numpy(), [[2, 0.25, 0.25, 0]])
    # Explicit host refit refreshes metadata, and later device packing preserves it.
    tlas.refit([ti.hardware.ray.RayInstance(blas, custom_index=37)])
    tlas.refit_transforms(matrix)
    tlas.trace_typed(rays, hits, ids)
    np.testing.assert_array_equal(ids.to_numpy(), [[0, 0, 37, 1]])
    recording = tlas.record_refit_transforms()
    ti.reset()
    ti.init(arch=ti.vulkan, offline_cache=False)
    with pytest.raises(RuntimeError, match="previous Taichi runtime generation"):
        recording.execute({"transforms": matrix})
    tlas.close()
    blas.close()
