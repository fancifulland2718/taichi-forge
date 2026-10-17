"""Export ordinary/selected image graphs through the common resource contract."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_binding_recipe_export import selected


@pytest.mark.parametrize("recipe", ["ordinary", "selected"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_rwtexture_graph_export_aliases_and_gradients(tmp_path, recipe):
    @ti.kernel
    def before(image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for i in range(2):
            output[i] = image.fetch(ti.Vector([i, 0]), 0).x

    @ti.kernel
    def update(
        image: ti.types.rw_texture(2, fmt=ti.Format.r32f),
        unsigned: ti.types.rw_texture(2, fmt=ti.Format.r32u),
        ids: ti.types.ndarray(ti.u32, 1),
    ):
        for i in ids:
            xy = ti.Vector([i % 3, i // 3])
            image.store(xy, ti.Vector([image.load(xy).x + 1, 0, 0, 0]))
            value = unsigned.load(xy).x + 3
            unsigned.store(xy, ti.Vector([value, 0, 0, 0]))
            ids[i] = value

    @ti.kernel
    def after(image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for i in range(2):
            output[2 + i] = image.fetch(ti.Vector([i, 0]), 0).x

    @ti.kernel
    def seed(dout: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i in dout:
            dout[i] = 0
        loss[0] = 0
        dloss[0] = 1

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in output:
            loss[0] += 0.5 * output[i] ** 2

    @ti.kernel
    def mip(image: ti.types.rw_texture(2, fmt=ti.Format.r32f, lod=1), output: ti.types.ndarray(ti.f32, 1)):
        output[0], output[1] = image.shape[0], image.shape[1]
        for y in range(2):
            xy = ti.Vector([0, y])
            value = image.load(xy).x + 1
            image.store(xy, ti.Vector([value, 0, 0, 0]))
            output[2 + y] = value

    arg, kind = ti.graph.Arg, ti.graph.ArgKind
    producer_start = time.perf_counter()
    builder = ti.graph.GraphBuilder()
    image = arg(kind.TEXTURE, "image", ndim=2)
    writable = arg(kind.RWTEXTURE, "writable", ndim=2, fmt=ti.Format.r32f)
    unsigned = arg(kind.RWTEXTURE, "unsigned", ndim=2, fmt=ti.Format.r32u)
    output, dout, loss = [arg(kind.NDARRAY, name, ti.f32, ndim=1) for name in ("output", "dout", "loss")]
    ids = arg(kind.NDARRAY, "ids", ti.u32, ndim=1)
    dloss = builder.private_ndarray("dloss", ti.f32, 1)
    builder.dispatch(before, image, output)
    builder.dispatch(update, writable, unsigned, ids)
    builder.dispatch(after, image, output)
    builder.dispatch(seed, dout, loss, dloss)
    builder.dispatch(energy, output, loss)
    builder.dispatch(energy.grad, output, loss)
    texture = ti.Texture(ti.Format.r32f, (3, 5))
    integer = ti.Texture(ti.Format.r32u, (3, 5))
    sources = dict(image=texture, writable=texture, unsigned=integer)
    module = ti.aot.Module()

    def export(graph):
        wrong = ti.Texture(ti.Format.r32i, (3, 5))
        with pytest.raises(ti.TaichiRuntimeError, match="format"):
            module.add_execution_plan("bad", graph, source_bindings=dict(sources, unsigned=wrong))
        assert not module._exported_kernels and not module._graph_export_failed
        start = time.perf_counter()
        plan = module.add_execution_plan(
            "step",
            graph,
            source_bindings=sources,
            resource_bindings={texture: dict(pixels="pixels"), integer: dict(pixels="upixels")},
            gradients=dict(output="dout", loss="dloss"),
        )
        document = json.loads(plan.to_json())
        images = [r for r in document["resources"] if r.get("provider") == "image"]
        assert len(images) == len(document["initialization"]) == 2
        sampled = next(r for r in images if r["operation"] == "sampled")
        storage = next(r for r in images if r["operation"] == "storage")
        assert sampled["parameters"]["storage"] is True
        assert "sampler" not in storage["parameters"] and "filter" not in storage["parameters"]
        assert document["operations"][0]["arguments"][0] == document["operations"][1]["arguments"][0] == sampled["name"]
        count = len(module._exported_kernels)
        public = module.add_execution_plan(
            "public",
            graph,
            source_bindings=sources,
            resource_bindings={texture: "public_image", integer: "public_ids"},
            gradients=dict(output="dout", loss="dloss"),
        )
        assert not any(r.get("provider") == "image" for r in json.loads(public.to_json())["resources"])
        assert len(module._exported_kernels) == count
        return dict(export_ms=(time.perf_counter() - start) * 1000, kernels=count)

    if recipe == "selected":
        with selected(builder.freeze()) as graph:
            costs = export(graph)
    else:
        graph = builder.compile()
        costs = export(graph)
        graph.close()
    mipped = ti.Texture(ti.Format.r32f, (3, 5), mip_levels=2)
    mip_builder = ti.graph.GraphBuilder()
    mip_builder.dispatch(
        mip, arg(kind.RWTEXTURE, "image", ndim=2, fmt=ti.Format.r32f), arg(kind.NDARRAY, "mip_output", ti.f32, ndim=1)
    )

    def export_mip(graph):
        module.add_execution_plan(
            "mip",
            graph,
            source_bindings=dict(image=mipped),
            resource_bindings={mipped: dict(pixels_0="m0", pixels_1="m1")},
        )

    if recipe == "selected":
        with selected(mip_builder.freeze()) as graph:
            export_mip(graph)
    else:
        graph = mip_builder.compile()
        export_mip(graph)
        graph.close()
    costs["producer_ms"] = (time.perf_counter() - producer_start) * 1000
    artifact = tmp_path / "rw-images.tcm"
    module.archive(str(artifact))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_rwtexture_export_probe.py")),
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
