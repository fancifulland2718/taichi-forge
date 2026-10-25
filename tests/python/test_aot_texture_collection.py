"""Vulkan AOT collection metadata and shared ordinary execution packets."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_texture_collection_consumption_and_downstream_gradients(tmp_path):
    from tests.python.aot_texture_collection_probe import plan_parts

    @ti.kernel
    def sample(
        table: ti.types.texture_collection(ndim=2, capacity=3),
        selectors: ti.types.ndarray(ti.i32, 1),
        output: ti.types.ndarray(ti.f32, 1),
    ):
        for i in output:
            slot = selectors[i]
            output[i] = table[slot].fetch(ti.Vector([0, 0]), 0).x + table[slot].sample_lod(ti.Vector([0.5, 0.5]), 0.0).x

    @ti.kernel
    def paint(
        scale: ti.types.ndarray(ti.f32, 1), pixels: ti.types.ndarray(ti.f32, 2), selectors: ti.types.ndarray(ti.i32, 1)
    ):
        for i, j in pixels:
            pixels[i, j] = scale[0] * (i * 2 + j + 1)
        for i in selectors:
            selectors[i] = i % 3

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in output:
            loss[0] += 0.5 * output[i] * output[i]

    start = time.perf_counter()
    module = ti.aot.Module(
        caps=["spirv_has_sampled_image_array_non_uniform_indexing", "max_sampled_texture_collection_size=3"]
    )
    # Exercise injection through explicit template_args as well as ordinary
    # inferred specialization; texture tables do not allocate N dummy images.
    module.add_kernel(sample, template_args=dict(selectors=ti.ndarray(ti.i32, 6), output=ti.ndarray(ti.f32, 6)))
    module.add_kernel(sample, name="sample_ordinary")
    module.add_kernel(paint)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    for policy in ("private", "public"):
        module.add_execution_plan(policy, ti.aot.ExecutionPlan(**plan_parts(policy == "private")))
    build_ms = (time.perf_counter() - start) * 1000
    folder = tmp_path / "collection"
    module.save(str(folder))
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["metadata_version"] == 13
    kernel = next(k for k in metadata["kernels"] if k["name"] == "sample")
    parameter = next(a["value"] for a in kernel["ctx_attribs"]["arg_attribs_vec_"] if a["key"] == [0])
    assert parameter["element_shape"] == [3] and parameter["field_dim"] == 1
    artifact = tmp_path / "collection.tcm"
    module.archive(str(artifact))

    # Independent JIT reference uses the same physical Vulkan texel ordering.
    pixels = ti.ndarray(ti.f32, (3, 2))
    seed = ti.ndarray(ti.f32, (2, 2))
    seed.fill(2)
    scale, selectors = ti.ndarray(ti.f32, 1), ti.ndarray(ti.i32, 6)
    output, loss = ti.ndarray(ti.f32, 6, needs_grad=True), ti.ndarray(ti.f32, 1, needs_grad=True)
    first, second = ti.Texture(ti.Format.r32f, (2, 2)), ti.Texture(ti.Format.r32f, (3, 2))
    first.from_ndarray(seed)
    scale.fill(1)
    paint(scale, pixels, selectors)
    second.from_ndarray(pixels)
    table = ti.TextureCollection([second, first, second])
    sample(table, selectors, output)
    output.grad.fill(0)
    loss.fill(0)
    loss.grad.fill(1)
    energy(output, loss)
    energy.grad(output, loss)
    np.testing.assert_allclose(output.to_numpy(), [4.5, 4, 4.5] * 2)
    np.testing.assert_allclose(output.grad.to_numpy(), [4.5, 4, 4.5] * 2)
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_texture_collection_probe.py")),
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
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(build_ms=build_ms, bytes=artifact.stat().st_size), indent=2)
    )
