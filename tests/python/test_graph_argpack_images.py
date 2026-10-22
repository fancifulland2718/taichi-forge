"""Pack images reuse the ordinary image reconstruction and alias contracts."""

import array
import gc
import json

import pytest
import taichi_forge as ti
from tests import test_utils


def read(value):
    values = array.array("f")
    values.frombytes(value.read())
    return list(values)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_pack_image_domain_borrows_alias_and_keeps_peer_live():
    packet = ti.types.argpack(image=ti.types.texture(2), scale=ti.f32)
    textures = [ti.Texture(ti.Format.r32f, (2, 1)) for _ in range(2)]
    source = ti.ndarray(ti.f32, (2, 1))
    for texture, value in zip(textures, (3, 7)):
        source.fill(value)
        texture.from_ndarray(source)
    pack = packet(image=textures[0], scale=2)

    @ti.kernel
    def sample(p: packet, image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            xy = ti.Vector([i, 0])
            output[i] = p.scale * p.image.fetch(xy, 0).x + image.fetch(xy, 0).x

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        sample,
        ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", packet),
        ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "alias", ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1),
    )
    graph = builder.compile()
    left, right = ti.ndarray(ti.f32, 2), ti.ndarray(ti.f32, 2)
    with (
        ti.graph.execution_domain(instance_capacity=1) as first,
        ti.graph.execution_domain(instance_capacity=1) as second,
    ):
        a = graph.prepare(dict(pack=pack, alias=textures[0], output=left), execution_domain=first)
        a.initialize()
        pack.image, pack.scale = textures[1], 4
        b = graph.prepare(dict(pack=pack, alias=textures[1], output=right), execution_domain=second)
        b.initialize()
        graph.close()
        del graph, pack, textures, texture
        gc.collect()
        a.submit().wait()
        b.submit().wait()
        assert left.to_numpy().tolist() == [9, 9]
        assert right.to_numpy().tolist() == [35, 35]
        a.close()
        first.close()
        b.submit().wait()
        assert right.to_numpy().tolist() == [35, 35]
        b.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_frozen_pack_sampled_image_and_downstream_adjoint(tmp_path):
    child = ti.types.argpack(image=ti.types.texture(2))
    packet = ti.types.argpack(scale=ti.f32, child=child)
    image = ti.Texture(ti.Format.r32f, (2, 1))
    pack = packet(scale=2, child=child(image=image))

    @ti.kernel
    def sample(p: packet, output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = p.scale * p.child.image.fetch(ti.Vector([i, 0]), 0).x

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in output:
            loss[0] += 0.5 * output[i] ** 2

    builder = ti.graph.GraphBuilder()
    p = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", packet)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    loss = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    builder.dispatch(sample, p, output)
    builder.dispatch(energy, output, loss)
    builder.dispatch(energy.grad, output, loss)
    graph = builder.compile()
    source = ti.ndarray(ti.f32, (2, 1))
    source.fill(3)
    image.from_ndarray(source)
    live_output = ti.ndarray(ti.f32, 2, needs_grad=True)
    live_loss = ti.ndarray(ti.f32, 1, needs_grad=True)
    live_loss.fill(0)
    live_loss.grad.fill(1)
    live_output.grad.fill(0)
    sample(pack, live_output)
    ordinary = ti.graph.GraphBuilder()
    ordinary.dispatch(sample, p, output)
    ordinary.compile().run(dict(pack=pack, output=live_output))
    energy(live_output, live_loss)
    energy.grad(live_output, live_loss)
    assert live_output.to_numpy().tolist() == live_output.grad.to_numpy().tolist() == [6, 6]
    assert live_loss.to_numpy().tolist() == [36]
    module = ti.aot.Module()
    module.add_execution_plan(
        "step",
        graph,
        source_bindings=dict(pack=pack),
        resource_bindings={image: dict(pixels="pixels")},
        gradients=dict(output="dout", loss="dloss"),
    )
    module.add_execution_plan(
        "public_image",
        graph,
        source_bindings=dict(pack=pack),
        resource_bindings={image: "image"},
        gradients=dict(output="dout", loss="dloss"),
    )
    module.add_graph("cgraph", graph)
    artifact = tmp_path / "image.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        pixels = runtime.ndarray("f32", (2, 1))
        pixels.write(array.array("f", [3, 7]).tobytes())
        imported = runtime.native_resource(
            "image",
            "sampled",
            parameters=dict(shape=[2, 1], format="r32f", filter="nearest"),
            bindings=dict(pixels=pixels),
        )
        entry = loaded.graph("cgraph")
        packet = entry.argpack("pack", dict(scale=2, child=dict(image=imported)))
        for name, resources in (
            ("step", dict(pixels=pixels)),
            ("public_image", dict(image=imported)),
            ("cgraph", dict(pack=packet)),
        ):
            arrays = {
                key: runtime.ndarray("f32", count)
                for key, count in (("output", 2), ("dout", 2), ("loss", 1), ("dloss", 1))
            }
            for key, data in dict(output=[-1, -1], dout=[0, 0], loss=[0], dloss=[1]).items():
                arrays[key].write(array.array("f", data).tobytes())
            if name == "cgraph":
                owner = entry.prepare(
                    dict(output=arrays["output"], loss=arrays["loss"], **resources),
                    gradients=dict(output=arrays["dout"], loss=arrays["dloss"]),
                )
            else:
                owner = loaded.prepare_execution(name, dict(arrays, **resources))
            assert read(arrays["output"]) == [-1, -1]
            owner.run()
            runtime.wait()
            assert read(arrays["output"]) == read(arrays["dout"]) == [6, 14]
            assert read(arrays["loss"]) == [116]
            owner.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_pack_storage_use_qualifies_earlier_sample_alias(tmp_path):
    packet = ti.types.argpack(image=ti.types.rw_texture(2, fmt=ti.Format.r32f))
    image = ti.Texture(ti.Format.r32f, (2, 1))
    pack = packet(image=image)

    @ti.kernel
    def sample(image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1), offset: ti.i32):
        for i in range(2):
            output[offset + i] = image.fetch(ti.Vector([i, 0]), 0).x

    @ti.kernel
    def update(p: packet):
        for i in range(2):
            xy = ti.Vector([i, 0])
            p.image.store(xy, ti.Vector([p.image.load(xy).x + 1, 0, 0, 0]))

    builder = ti.graph.GraphBuilder()
    source = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    first = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "first", ti.i32)
    last = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "last", ti.i32)
    builder.dispatch(sample, source, output, first)
    builder.dispatch(update, ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", packet))
    builder.dispatch(sample, source, output, last)
    graph = builder.compile()
    module = ti.aot.Module()
    wrong = ti.Texture(ti.Format.r32u, (2, 1))
    pack.image = wrong
    with pytest.raises(ti.TaichiRuntimeError, match="format"):
        module.add_execution_plan("bad", graph, source_bindings=dict(pack=pack, image=image))
    assert not module._exported_kernels and not module._graph_export_failed
    pack.image = image
    plan = module.add_execution_plan(
        "step",
        graph,
        source_bindings=dict(pack=pack, image=image),
        resource_bindings={image: dict(pixels="pixels")},
    )
    document = json.loads(plan.to_json())
    assert sum(resource.get("provider") == "image" for resource in document["resources"]) == 1
    artifact = tmp_path / "storage.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        pixels, output = runtime.ndarray("f32", (2, 1)), runtime.ndarray("f32", 4)
        pixels.write(array.array("f", [3, 7]).tobytes())
        owner = loaded.prepare_execution("step", dict(pixels=pixels, output=output, first=0, last=2))
        owner.run()
        runtime.wait()
        assert read(output) == [3, 7, 4, 8]
        owner.run()
        runtime.wait()
        assert read(output) == [4, 8, 5, 9]
        owner.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_pack_image_collection_preserves_member_aliases(tmp_path):
    packet = ti.types.argpack(table=ti.types.texture_collection(ndim=2, capacity=3), bias=ti.f32)
    first, second = [ti.Texture(ti.Format.r32f, (1, 1)) for _ in range(2)]
    table = ti.TextureCollection([second, first, second])
    pack = packet(table=table, bias=1)

    @ti.kernel
    def sample(p: packet, output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = p.table[i].fetch(ti.Vector([0, 0]), 0).x + p.bias

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        sample,
        ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", packet),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1),
    )
    graph = builder.compile()
    source = ti.ndarray(ti.f32, (1, 1))
    source.fill(3)
    first.from_ndarray(source)
    source.fill(7)
    second.from_ndarray(source)
    live_output = ti.ndarray(ti.f32, 3)
    graph.run(dict(pack=pack, output=live_output))
    assert live_output.to_numpy().tolist() == [8, 4, 8]
    module = ti.aot.Module(
        caps=["spirv_has_sampled_image_array_non_uniform_indexing", "max_sampled_texture_collection_size=3"]
    )
    plan = module.add_execution_plan(
        "step",
        graph,
        source_bindings=dict(pack=pack),
        resource_bindings={first: dict(pixels="first"), second: dict(pixels="second")},
    )
    resources = json.loads(plan.to_json())["resources"]
    assert sum(resource.get("operation") == "collection" for resource in resources) == 1
    assert sum(resource.get("operation") == "sampled" for resource in resources) == 2
    artifact = tmp_path / "collection.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        a, b, output = runtime.ndarray("f32", (1, 1)), runtime.ndarray("f32", (1, 1)), runtime.ndarray("f32", 3)
        for target, value in ((a, 3), (b, 7)):
            target.write(array.array("f", [value]).tobytes())
        owner = loaded.prepare_execution("step", dict(first=a, second=b, output=output))
        owner.run()
        runtime.wait()
        assert read(output) == [8, 4, 8]
        owner.close()
