"""Extended Vulkan sampler state survives AOT reconstruction and borrowing."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_binding_recipe_export import selected
from tests.python.aot_image_sampler_probe import PIXELS, sampler_parameters, expected


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_cuda_rejects_unqualified_sampler_options():
    with ti.aot.borrow_runtime() as runtime:
        for sampler, message in (
            (dict(mip_filter="linear"), "basic sampler"),
            (dict(min_filter="nearest", mag_filter="linear"), "matching min and mag"),
            (dict(compare_op="less"), "basic sampler"),
        ):
            with pytest.raises(RuntimeError, match=message):
                runtime.native_resource(
                    "image", "sampled", parameters=dict(shape=[2, 2], format="r32f", sampler=sampler)
                )
        with runtime.native_resource("image", "sampled", parameters=dict(shape=[2, 2], format="r32f")):
            pass


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_extended_image_sampler_and_gradients(tmp_path):
    @ti.kernel
    def color(image: ti.types.texture(num_dimensions=2), result: ti.types.ndarray(ti.f32, 1)):
        result[0] = image.sample_lod(ti.Vector([0.5, 0.25]), 0.0).x
        result[1] = image.sample_lod(ti.Vector([0.125, 0.75]), 0.0).x
        result[2] = image.sample_lod(ti.Vector([1.125, 0.25]), 0.0).x
        result[3] = image.sample_lod(ti.Vector([-0.125, 0.75]), 0.0).x

    @ti.kernel
    def compare(image: ti.types.texture(num_dimensions=2), result: ti.types.ndarray(ti.f32, 1)):
        for i in result:
            result[i] = image.sample_compare(ti.Vector([(i + 0.5) / 4.0, 0.25]), 0.375)

    @ti.kernel
    def clear(
        dresult: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)
    ):
        for i in dresult:
            dresult[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(
        result: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in result:
            loss[0] += result[i] * result[i] * 0.5

    start = time.perf_counter()
    module = ti.aot.Module()
    for kind, query in (("color", color), ("compare", compare)):
        sampler = ti.hardware.sampling.SamplerConfig(**sampler_parameters(kind))
        texture = ti.Texture(ti.Format.depth32f if kind == "compare" else ti.Format.r32f, (4, 2), sampler=sampler)
        pixels = ti.ndarray(ti.f32, (4, 2))
        pixels.from_numpy(np.array(PIXELS, np.float32).reshape(4, 2))
        texture.from_ndarray(pixels)
        result = ti.ndarray(ti.f32, 4, needs_grad=True)
        loss = ti.ndarray(ti.f32, 1, needs_grad=True)
        query(texture, result)
        np.testing.assert_allclose(result.to_numpy(), expected(kind), rtol=0, atol=1e-6)
        result.grad.fill(0)
        loss.fill(0)
        loss.grad.fill(1)
        energy(result, loss)
        energy.grad(result, loss)
        np.testing.assert_allclose(result.grad.to_numpy(), expected(kind), rtol=0, atol=1e-6)

        def builder():
            b = ti.graph.GraphBuilder()

            def arg(name):
                return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)

            b.dispatch(query, ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2), arg("result"))
            dloss = b.private_ndarray("dloss", ti.f32, 1)
            b.dispatch(clear, arg("dresult"), arg("loss"), dloss)
            b.dispatch(energy, arg("result"), arg("loss"))
            b.dispatch(energy.grad, arg("result"), arg("loss"))
            return b

        def export(graph, recipe):
            options = dict(source_bindings=dict(image=texture), gradients=dict(result="dresult", loss="dloss"))
            plan = module.add_execution_plan(
                kind + "_" + recipe + "_private", graph, **options, resource_bindings={texture: dict(pixels="pixels")}
            )
            resource = next(r for r in json.loads(plan.to_json())["resources"] if r.get("operation") == "sampled")
            assert resource["parameters"]["sampler"] == asdict(sampler)
            module.add_execution_plan(
                kind + "_" + recipe + "_public", graph, **options, resource_bindings={texture: "public_image"}
            )

        graph = builder().compile()
        export(graph, "ordinary")
        graph.close()
        with selected(builder().freeze()) as graph:
            export(graph, "selected")
    costs = dict(producer_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "sampler.tcm"
    module.archive(str(artifact))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_image_sampler_probe.py")),
                mode,
                str(artifact),
                str(tmp_path / (mode + ".json")),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
