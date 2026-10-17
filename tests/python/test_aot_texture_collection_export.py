"""Graph tables share member reconstruction, usage qualification and aliases."""

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
def test_aot_texture_collection_graph_reconstruction(tmp_path, recipe):
    @ti.kernel
    def sample(
        table: ti.types.texture_collection(ndim=2, capacity=3),
        alias: ti.types.texture(2),
        output: ti.types.ndarray(ti.f32, 1),
    ):
        for i in output:
            output[i] = table[i % 3].fetch(ti.Vector([0, 0]), 0).x + alias.fetch(ti.Vector([0, 0]), 0).x

    @ti.kernel
    def paint(image: ti.types.rw_texture(2, fmt=ti.Format.r32f), gain: ti.f32):
        for i, j in ti.ndrange(2, 2):
            image.store(ti.Vector([i, j]), ti.Vector([gain, 0.0, 0.0, 0.0]))

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
    def analytic(
        output: ti.types.ndarray(ti.f32, 1), dout: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1)
    ):
        for i in output:
            loss[0] += 0.5 * output[i] ** 2
            dout[i] = output[i]

    first, second = [ti.Texture(ti.Format.r32f, (2, 2)) for _ in range(2)]
    pixels = ti.ndarray(ti.f32, (2, 2))
    pixels.fill(99)  # Must never become an implicit serialized texture snapshot.
    for image in (first, second):
        image.from_ndarray(pixels)
    table = ti.TextureCollection([second, first, second])
    builder = ti.graph.GraphBuilder()
    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("before", "after", "dout", "loss")
    }
    textures = ti.graph.Arg(ti.graph.ArgKind.TEXTURE_COLLECTION, "table", ndim=2, capacity=3)
    alias = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "alias", ndim=2)
    writable = ti.graph.Arg(ti.graph.ArgKind.RWTEXTURE, "writable", ndim=2, fmt=ti.Format.r32f)
    gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "gain", ti.f32)
    dloss = builder.private_ndarray("dloss", ti.f32, 1)
    if recipe == "ordinary":
        builder._bind_internal_runtime_arg(textures, table)
    builder.dispatch(sample, textures, alias, args["before"])
    builder.dispatch(paint, writable, gain)
    builder.dispatch(sample, textures, alias, args["after"])
    builder.dispatch(seed, args["dout"], args["loss"], dloss)
    if recipe == "ordinary":
        builder.dispatch(energy, args["after"], args["loss"])
        builder.dispatch(energy.grad, args["after"], args["loss"])
    else:
        builder.dispatch(analytic, args["after"], args["dout"], args["loss"])
    module = ti.aot.Module(
        caps=["spirv_has_sampled_image_array_non_uniform_indexing", "max_sampled_texture_collection_size=4"]
    )

    def export(graph):
        source = dict(alias=second, writable=second)
        if recipe == "selected":
            source["table"] = table
            frame = {name: ti.ndarray(ti.f32, 1 if name == "loss" else 6) for name in args}
            graph.submit(graph.bind(dict(frame, table=table, alias=second, writable=second, gain=99.0))).wait()
        options = dict(source_bindings=source)
        if recipe == "ordinary":
            options["gradients"] = dict(after="dout", loss="dloss")
        else:
            wrong = ti.TextureCollection([first, second])
            with pytest.raises(ti.TaichiRuntimeError, match="dimension/capacity"):
                module.add_execution_plan("wrong_capacity", graph, source_bindings=dict(source, table=wrong))
            assert not module._exported_kernels and not module._graph_export_failed
        start = time.perf_counter()
        plan = module.add_execution_plan(
            "private", graph, **options, resource_bindings={first: dict(pixels="seed_a"), second: dict(pixels="seed_b")}
        )
        elapsed = (time.perf_counter() - start) * 1000
        document = json.loads(plan.to_json())
        images = [r for r in document["resources"] if r.get("operation") == "sampled"]
        tables = [r for r in document["resources"] if r.get("operation") == "collection"]
        assert len(images) == 2 and len(tables) == 1
        assert sum(r["parameters"].get("storage", False) for r in images) == 1
        assert tables[0]["bindings"]["image_0"] == tables[0]["bindings"]["image_2"]
        assert len(document["initialization"]) == 2
        assert not module._data_payloads
        count = len(module._exported_kernels)
        module.add_execution_plan(
            "mixed", graph, **options, resource_bindings={first: dict(pixels="seed_a"), second: "public_second"}
        )
        module.add_execution_plan(
            "public", graph, **options, resource_bindings={table: "public_table", second: "public_second"}
        )
        assert len(module._exported_kernels) == count
        public = json.loads(module._execution_plans["public"].to_json())
        assert any(c["kind"] == "resource_alias" for c in public["constraints"])
        assert not any(r.get("provider") == "image" for r in public.get("resources", []))
        with pytest.raises(ValueError, match="rebuilt privately"):
            module.add_execution_plan(
                "bad_shared", graph, **options, resource_bindings={table: "public_table", second: dict(pixels="seed_b")}
            )
        assert not module._graph_export_failed
        return dict(export_ms=elapsed, kernels=count, plan_bytes=len(plan.to_json().encode()))

    if recipe == "selected":
        with selected(builder.freeze()) as graph:
            costs = export(graph)
    else:
        graph = builder.compile()
        costs = export(graph)
        graph.close()
    artifact = tmp_path / "collection-graph.tcm"
    module.archive(str(artifact))
    costs["bytes"] = artifact.stat().st_size
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_texture_collection_export_probe.py")),
                mode,
                str(artifact),
                str(tmp_path / f"{mode}.json"),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
