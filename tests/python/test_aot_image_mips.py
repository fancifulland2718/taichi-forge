"""Complete Vulkan mip initialization survives Graph export and runtime-only use."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_binding_recipe_export import selected
from tests.python.aot_image_mips_probe import SHAPES, VALUES, expected, sampler_parameters


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_mip_chain_export_and_gradients(tmp_path):
    @ti.kernel
    def sample(image: ti.types.texture(num_dimensions=2), result: ti.types.ndarray(ti.f32, 1)):
        for i in range(3):
            result[i] = image.sample_lod(ti.Vector([0.5, 0.5]), ti.cast(i, ti.f32)).x
            result[i + 3] = image.fetch(ti.Vector([0, 0]), i).x
        result[6] = image.sample_lod(ti.Vector([0.5, 0.5]), 0.5).x

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
    texture = ti.Texture(
        ti.Format.r32f, SHAPES[0], mip_levels=3, sampler=ti.hardware.sampling.SamplerConfig(**sampler_parameters())
    )
    for level, values in enumerate(VALUES):
        pixels = ti.ndarray(ti.f32, len(values))
        pixels.from_numpy(np.array(values, np.float32))
        ti.hardware.image.VulkanBufferToImageRecording(
            image_region=ti.hardware.image.VulkanImageRegion(mip_level=level)
        ).execute(dict(source=pixels, destination=texture))
    output = ti.ndarray(ti.f32, 7)
    sample(texture, output)
    np.testing.assert_allclose(output.to_numpy(), expected(), atol=1e-6, rtol=0)
    module = ti.aot.Module()

    def builder():
        b = ti.graph.GraphBuilder()

        def arg(name):
            return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)

        b.dispatch(sample, ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2), arg("result"))
        dloss = b.private_ndarray("dloss", ti.f32, 1)
        b.dispatch(clear, arg("dresult"), arg("loss"), dloss)
        b.dispatch(energy, arg("result"), arg("loss"))
        b.dispatch(energy.grad, arg("result"), arg("loss"))
        return b

    def export(graph, recipe):
        options = dict(source_bindings=dict(image=texture), gradients=dict(result="dresult", loss="dloss"))
        plan = module.add_execution_plan(
            recipe + "_private",
            graph,
            **options,
            resource_bindings={texture: {f"pixels_{i}": f"seed{i}" for i in range(3)}},
        )
        document = json.loads(plan.to_json())
        image = next(r for r in document["resources"] if r.get("operation") == "sampled")
        assert image["parameters"]["mip_levels"] == 3
        assert len(document["initialization"]) == 1
        assert document["initialization"][0]["operation"] == "upload_mips"
        module.add_execution_plan(recipe + "_public", graph, **options, resource_bindings={texture: "public_image"})

    graph = builder().compile()
    export(graph, "ordinary")
    graph.close()
    with selected(builder().freeze()) as graph:
        export(graph, "selected")
    costs = dict(producer_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "mips.tcm"
    module.archive(str(artifact))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_image_mips_probe.py")),
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
