"""Fixed dense roots composed with images/AS, including native retirement."""

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.lang import impl
from tests import test_utils


def _dense_consumer(resource_kind):
    if resource_kind == "as" and not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray query features are unavailable")
    source = ti.field(ti.f32)
    source_builder = ti.FieldsBuilder()
    source_builder.dense(ti.ij, (8, 4)).place(source)
    source_tree = source_builder.finalize()
    result = ti.field(ti.f32)
    result_builder = ti.FieldsBuilder()
    result_builder.dense(ti.ij, (8, 4)).place(result)
    result_tree = result_builder.finalize()
    source.fill(2)
    result.fill(-31)
    output = ti.ndarray(ti.f32, (8, 4))
    output.fill(-53)
    builder = ti.graph.GraphBuilder()
    gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "gain", ti.f32)

    @ti.kernel
    def publish(output: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i, j in result:
            output[i, j] = result[i, j] + source[i, j]

    if resource_kind == "image":
        image = ti.Texture(ti.Format.r32f, (8, 4))

        @ti.kernel
        def write(image: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0), gain: ti.f32):
            for i, j in source:
                image.store(ti.Vector([i, j]), ti.Vector([source[i, j] * gain, 0.0, 0.0, 1.0]))

        @ti.kernel
        def sample(image: ti.types.texture(num_dimensions=2)):
            for i, j in result:
                result[i, j] = image.fetch(ti.Vector([i, j]), 0).x

        builder.dispatch(write, ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "write_image", fmt=ti.Format.r32f, ndim=2), gain)
        builder.dispatch(sample, ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "read_image", ndim=2))
        bindings = dict(write_image=image, read_image=image, output=output, gain=3.0)
        retire_resources = image._delete_runtime_texture
    else:
        vertices = ti.ndarray(ti.f32, (3, 3))
        triangles = ti.ndarray(ti.i32, (1, 3))
        vertices.from_numpy(np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32))
        triangles.from_numpy(np.array([[0, 1, 2]], np.int32))
        blas = ti.hardware.ray.TriangleBLAS(vertices, triangles)
        scene = ti.hardware.ray.InstanceTLAS([ti.hardware.ray.RayInstance(blas)])

        @ti.kernel
        def query(scene: ti.types.acceleration_structure(), gain: ti.f32):
            for i, j in result:
                hit = scene.trace_closest(
                    ti.Vector([0.0, 0.0, source[i, j]]), ti.Vector([0.0, 0.0, -1.0])
                )
                result[i, j] = hit.t * gain if hit.hit else -71.0

        builder.dispatch(query, ti.graph.Arg(ti.graph.ArgKind.ACCELERATION_STRUCTURE, "scene"), gain)
        bindings = dict(scene=scene, output=output, gain=3.0)

        def retire_resources():
            scene.close()
            blas.close()

    builder.dispatch(publish, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2))
    return builder.freeze(), bindings, source, (source_tree, result_tree), retire_resources


@pytest.mark.parametrize("resource_kind", ["image", "as"])
@pytest.mark.parametrize("retirement", ["tree", "close", "reset"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_dense_resource_whole_graph_frames(resource_kind, retirement, monkeypatch):
    definition, arguments, source, trees, retire_resources = _dense_consumer(resource_kind)
    provider = GraphBindingFrameRecipeProvider()
    # Old wheels must not offer a recipe that cannot retain its roots.
    with monkeypatch.context() as patch:
        patch.setattr(core._VulkanFixedGraphRecording, "supports_snode_tree_dependencies", None)
        assert provider.fragments(definition) == ()
    catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(), provider))
    assert len(catalog.entries()) == 2
    candidate = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(candidate) as materialized:
            graph = materialized.executor
            bindings = [graph.bind(dict(arguments, gain=gain)) for gain in (3.0, 5.0)]
            frames = [binding._version.execution_frame for binding in bindings]
            assert all(frame.uses_secondary_commands() for frame in frames)
            np.testing.assert_array_equal(arguments["output"].to_numpy(), np.full((8, 4), -53))

            def unexpected(*args, **kwargs):
                raise AssertionError("Replay must not repeat preparation or dependency discovery")

            with monkeypatch.context() as patch:
                patch.setattr(core, "_prepare_vulkan_graph_recording", unexpected)
                patch.setattr(core._VulkanFixedGraphRecording, "supports_snode_tree_dependencies", unexpected)
                for index in (0, 1, 0):
                    graph.run(bindings[index])
                    np.testing.assert_array_equal(arguments["output"].to_numpy(), np.full((8, 4), (8, 12)[index]))
                source.fill(4)
                graph.run(bindings[1])
                np.testing.assert_array_equal(arguments["output"].to_numpy(), np.full((8, 4), 24))

            # Published frames own closed images/TLAS/BLAS as well as the roots.
            retire_resources()
            graph.run(bindings[0])
            if retirement == "tree":
                trees[0].destroy()
                np.testing.assert_array_equal(arguments["output"].to_numpy(), np.full((8, 4), 16))
                with pytest.raises(ti.TaichiRuntimeError, match="destroyed SNodeTree"):
                    graph.run(bindings[0])
            elif retirement == "close":
                graph.close()
                # close precedes submission/completion; the parent command
                # must still retain every dense/image/AS allocation it uses.
                np.testing.assert_array_equal(arguments["output"].to_numpy(), np.full((8, 4), 16))
            else:
                ti.reset()
            assert all(frame.argument_bytes() == 0 for frame in frames)
            with pytest.raises(RuntimeError, match="closed|retired|finaliz|runtime"):
                frames[0].run()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_native_dense_frame_tree_retirement_and_id_reuse():
    definition, arguments, _, trees, retire_resources = _dense_consumer("image")
    program = impl.get_runtime().prog
    sources = [node.compiled_graph for node in definition._runtime_spec.nodes]
    native_arguments = dict(
        write_image=arguments["write_image"].tex,
        read_image=arguments["read_image"].tex,
        output=arguments["output"].arr,
        gain=3.0,
    )
    frame = core._prepare_vulkan_graph_recording(program, sources, native_arguments)
    assert frame.uses_secondary_commands()
    # An unrelated root retirement must leave this native registration active.
    unrelated = ti.field(ti.i32)
    unrelated_builder = ti.FieldsBuilder()
    unrelated_builder.dense(ti.i, 4).place(unrelated)
    unrelated_tree = unrelated_builder.finalize()
    unrelated_tree.destroy()
    frame.run()
    identity = (trees[0].id, trees[0].generation)
    # No Python materialized Graph exists to close this independent frame.
    trees[0].destroy()
    np.testing.assert_array_equal(arguments["output"].to_numpy(), np.full((8, 4), 8))
    replacement = ti.field(ti.f32)
    replacement_builder = ti.FieldsBuilder()
    replacement_builder.dense(ti.ij, (8, 4)).place(replacement)
    replacement_tree = replacement_builder.finalize()
    assert replacement_tree.id == identity[0]
    assert replacement_tree.generation > identity[1]
    with pytest.raises(RuntimeError, match="destroyed SNodeTree|retired"):
        frame.run()
    with pytest.raises(RuntimeError, match="stale SNodeTree"):
        core._prepare_vulkan_graph_recording(program, sources, native_arguments)
    assert frame.argument_bytes() == 0
    frame.close()
    frame.close()
    retire_resources()


@test_utils.test(arch=ti.vulkan, require=ti.extension.sparse, offline_cache=False)
def test_fixed_frames_reject_sparse_tree_even_when_dispatch_uses_dense_sibling():
    dense = ti.field(ti.i32)
    sparse = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 8).place(dense)
    fields.bitmasked(ti.i, 8).place(sparse)
    tree = fields.finalize()

    @ti.kernel
    def advance():
        for i in range(8):
            dense[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance)
    definition = builder.freeze()
    assert GraphBindingFrameRecipeProvider().fragments(definition) == ()
    with pytest.raises(RuntimeError, match="fixed dense SNodeTree"):
        core._prepare_vulkan_graph_recording(
            impl.get_runtime().prog, [node.compiled_graph for node in definition._runtime_spec.nodes], {}
        )
    np.testing.assert_array_equal(dense.to_numpy(), np.zeros(8))
    tree.destroy()
