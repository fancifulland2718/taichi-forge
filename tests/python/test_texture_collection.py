import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.lang import impl
from tests import test_utils


def _make_constant_texture(value):
    source = ti.ndarray(dtype=ti.f32, shape=(2, 2))
    source.fill(value)
    texture = ti.Texture(ti.Format.r32f, (2, 2))
    texture.from_ndarray(source)
    return texture


def _make_collection(values):
    textures = tuple(_make_constant_texture(value) for value in values)
    try:
        return ti.TextureCollection(textures), textures
    except RuntimeError as exc:
        if "non-uniform indexing" in str(exc):
            pytest.skip("Vulkan sampled-image non-uniform indexing is unavailable")
        raise


def _collection_definition():
    @ti.kernel
    def fetch_selected(
        textures: ti.types.texture_collection(ndim=2, capacity=3),
        selectors: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for lane in output:
            # selectors is the producer of a lane-dependent index. The public
            # contract requires every value to be in [0, capacity).
            slot = selectors[lane]
            output[lane] = (
                textures[slot].fetch(ti.Vector([0, 0]), 0).x
                + textures[slot].sample_lod(ti.Vector([0.5, 0.5]), 0.0).x
            )

    table = ti.graph.Arg(
        ti.graph.ArgKind.TEXTURE_COLLECTION,
        "textures",
        ndim=2,
        capacity=3,
    )
    selectors = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "selectors", ti.i32, ndim=1
    )
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fetch_selected, table, selectors, output)
    return builder.freeze(), fetch_selected


def _collection_graph():
    definition, fetch_selected = _collection_definition()
    return definition.compile(), fetch_selected


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_texture_collection_type_contract_and_aot_rejection():
    annotation = ti.types.texture_collection(ndim=2, capacity=4)
    assert annotation.num_dimensions == 2
    assert annotation.capacity == 4
    with pytest.raises(ti.TaichiCompilationError, match="ndim"):
        ti.types.texture_collection(ndim=0, capacity=4)
    with pytest.raises(ti.TaichiCompilationError, match="capacity"):
        ti.types.texture_collection(ndim=2, capacity=0)

    @ti.kernel
    def unsupported(table: ti.types.texture_collection(ndim=2, capacity=1)):
        pass

    module = ti.aot.Module()
    with pytest.raises(ti.TaichiCompilationError, match="JIT-only"):
        module.add_kernel(unsupported)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_texture_collection_dynamic_index_and_explicit_graph_rebind():
    collection, textures = _make_collection((2.0, 5.0, 11.0))
    replacement, replacement_textures = _make_collection((13.0, 17.0, 19.0))
    assert collection.members == textures
    assert replacement.snapshot_id > collection.snapshot_id
    with pytest.raises(TypeError):
        collection.members[0] = textures[1]
    integer_texture = ti.Texture(ti.Format.r32u, (2, 2))
    with pytest.raises(ValueError, match="normalized sampled formats"):
        ti.TextureCollection((integer_texture,))
    integer_texture._delete_runtime_texture()

    selectors = ti.ndarray(dtype=ti.i32, shape=6)
    selectors.from_numpy(np.asarray([2, 0, 1, 2, 1, 0], dtype=np.int32))
    output = ti.ndarray(dtype=ti.f32, shape=6)
    graph, fetch_selected = _collection_graph()
    module = ti.aot.Module()
    with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="JIT-only"):
        module.add_graph("texture_collection", graph)

    # Direct launch and Graph dispatch exercise the same hardware descriptor
    # array path; no Python-side slot dispatch participates in either result.
    fetch_selected(collection, selectors, output)
    np.testing.assert_array_equal(output.to_numpy(), [22, 4, 10, 22, 10, 4])

    bindings = graph.bind(
        {"textures": collection, "selectors": selectors, "output": output}
    )
    graph.run(bindings)
    np.testing.assert_array_equal(output.to_numpy(), [22, 4, 10, 22, 10, 4])

    bindings.update(textures=replacement)
    graph.run(bindings)
    np.testing.assert_array_equal(output.to_numpy(), [38, 26, 34, 38, 34, 26])

    wrong_capacity, wrong_textures = _make_collection((23.0, 29.0))
    revision = bindings.revision
    with pytest.raises(ti.TaichiRuntimeError, match="capacity"):
        bindings.update(textures=wrong_capacity)
    assert bindings.revision == revision
    graph.run(bindings)
    np.testing.assert_array_equal(output.to_numpy(), [38, 26, 34, 38, 34, 26])
    graph.close()
    assert replacement_textures and wrong_textures


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_texture_collection_above_legacy_pool_capacity():
    # Cross the historical fixed pool capacity and include one ordinary
    # sampled texture so both pipeline-limit validation and pool sizing see the
    # aggregate combined-image-sampler demand.
    capacity = 257
    source = ti.ndarray(dtype=ti.f32, shape=(1, 1))
    textures = []
    for value in range(capacity):
        source.fill(float(value + 1))
        texture = ti.Texture(ti.Format.r32f, (1, 1))
        texture.from_ndarray(source)
        textures.append(texture)
    try:
        collection = ti.TextureCollection(tuple(textures))
    except RuntimeError as exc:
        if "descriptor limit" in str(exc):
            pytest.skip("Vulkan sampled-image descriptor limit is below 257")
        raise
    bonus = _make_constant_texture(1000.0)
    selectors = ti.ndarray(dtype=ti.i32, shape=3)
    output = ti.ndarray(dtype=ti.f32, shape=3)

    @ti.kernel
    def produce_indices(indices: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for lane in indices:
            indices[lane] = lane * 128

    @ti.kernel
    def consume_indices(
        table: ti.types.texture_collection(ndim=2, capacity=capacity),
        extra: ti.types.texture(2),
        indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        result: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for lane in result:
            slot = indices[lane]
            result[lane] = table[slot].fetch(ti.Vector([0, 0]), 0).x + extra.fetch(
                ti.Vector([0, 0]), 0
            ).x

    produce_indices(selectors)
    try:
        consume_indices(collection, bonus, selectors, output)
    except ti.TaichiCompilationError as exc:
        if "descriptor" in str(exc) and "limit" in str(exc):
            pytest.skip("Vulkan aggregate sampled-image descriptor limit is below 258")
        raise
    np.testing.assert_array_equal(output.to_numpy(), [1001, 1129, 1257])


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_texture_collection_public_binding_frame_recipe_rebind_and_retirement():
    definition, _ = _collection_definition()
    providers = (
        GraphRuntimeAssemblyProvider(),
        GraphBindingFrameRecipeProvider(),
    )
    catalog = definition.recipe_catalog(providers=providers)
    recipes = [
        entry.recipe for entry in catalog.entries() if entry.recipe.fragments
    ]
    assert len(recipes) == 1
    collection, textures = _make_collection((3.0, 7.0, 9.0))
    replacement, replacement_textures = _make_collection((13.0, 17.0, 19.0))
    selectors = ti.ndarray(dtype=ti.i32, shape=3)
    selectors.from_numpy(np.asarray([2, 1, 0], dtype=np.int32))
    output = ti.ndarray(dtype=ti.f32, shape=3)
    program = impl.get_runtime().prog

    with definition.materialization_context(
        provider_set=catalog.provider_set
    ) as context:
        with context.materialize(recipes[0]) as materialized:
            graph = materialized.executor
            assert (
                materialized.manifest.submissions[0].replay_mode
                == "vulkan_secondary_immutable_argument_frames_published"
            )
            bindings = graph.bind(
                {
                    "textures": collection,
                    "selectors": selectors,
                    "output": output,
                }
            )
            first_frame = bindings._version.execution_frame
            assert first_frame is not None
            assert first_frame.uses_secondary_commands()

            # Published frames own the immutable descriptor table and leases.
            # Retiring the source Texture views must not revalidate or scan the
            # collection on replay.
            for texture in textures:
                texture._delete_runtime_texture()
            gc.collect()
            graph.run(bindings)
            np.testing.assert_array_equal(output.to_numpy(), [18, 14, 6])
            assert graph._graph_stats[0]["last_path"] == (
                "vulkan_prepared_binding_plan"
            )

            # Replacement is a new immutable collection/frame publication.
            bindings.update(textures=replacement)
            assert bindings._version.execution_frame is not first_frame
            for texture in replacement_textures:
                texture._delete_runtime_texture()
            gc.collect()
            graph.run(bindings)
            np.testing.assert_array_equal(output.to_numpy(), [38, 34, 26])
            assert graph._graph_stats[0]["last_path"] == (
                "vulkan_prepared_binding_plan"
            )

    ti.sync()
    stats = dict(program._debug_texture_resource_stats())
    assert stats["retiring"] == 0
    assert stats["inflight"] == 0
    assert stats["release_errors"] == 0


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_texture_collection_reset_invalidates_kernel_and_graph_bindings():
    collection, _ = _make_collection((1.0, 2.0, 3.0))
    selectors = ti.ndarray(dtype=ti.i32, shape=1)
    selectors.fill(0)
    output = ti.ndarray(dtype=ti.f32, shape=1)
    graph, fetch_selected = _collection_graph()
    bindings = graph.bind(
        {"textures": collection, "selectors": selectors, "output": output}
    )
    ti.reset()
    assert collection.collection is None
    with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="reset|runtime"):
        fetch_selected(collection, selectors, output)
    with pytest.raises((RuntimeError, ti.TaichiRuntimeError), match="reset|runtime"):
        graph.run(bindings)
