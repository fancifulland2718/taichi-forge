"""Managed image owners bind ordinary AOT kernels without public raw handles."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_sampled_kernel_images_and_gradients(tmp_path):
    from tests.python.aot_kernel_image_probe import plan_parts

    arch = ti.cfg.arch.name

    @ti.kernel
    def sample(
        one: ti.types.texture(num_dimensions=1),
        linear: ti.types.texture(num_dimensions=2),
        nearest: ti.types.texture(num_dimensions=2),
        three: ti.types.texture(num_dimensions=3),
        output: ti.types.ndarray(ti.f32, 1),
    ):
        output[0] = one.fetch(ti.Vector([2]), 0).x
        output[1] = one.sample_lod(ti.Vector([0.875]), 0.0).x
        output[2] = linear.fetch(ti.Vector([0, 4]), 0).x
        output[3] = linear.fetch(ti.Vector([2, 1]), 0).x
        output[4] = linear.sample_lod(ti.Vector([0.5, 0.25]), 0.0).x
        output[5] = nearest.sample_lod(ti.Vector([0.5, 0.25]), 0.0).x
        output[6] = three.fetch(ti.Vector([0, 2, 3]), 0).x
        output[7] = three.fetch(ti.Vector([1, 0, 2]), 0).x
        output[8] = three.sample_lod(ti.Vector([0.5, 0.5, 0.5]), 0.0).x

    @ti.kernel
    def paint(scale: ti.types.ndarray(ti.f32, 1), pixels: ti.types.ndarray(ti.f32, 2)):
        for i, j in pixels:
            pixels[i, j] = scale[0] * (i * 5 + j)

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in output:
            loss[0] += 0.5 * output[i] * output[i]

    start = time.perf_counter()
    module = ti.aot.Module()
    for kernel in (sample, paint, energy):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    policies = ["seeded", "updated"] + (["family"] if arch == "cuda" else [])
    if arch == "vulkan":
        policies += ["depth", "depth_seeded"]
    adapter = (
        os.environ.get("TAICHI_FORGE_TEST_OPTIX_PROVIDER") if arch == "cuda" else None
    )
    if adapter:
        policies.append("ordered")
    for policy in policies:
        base = (
            "seeded"
            if policy == "depth_seeded"
            else "updated" if policy == "depth" else policy
        )
        parts = plan_parts(base)
        if policy.startswith("depth"):
            next(r for r in parts["resources"] if r["name"] == "nearest")["parameters"][
                "format"
            ] = "depth32f"
        module.add_execution_plan(policy, ti.aot.ExecutionPlan(**parts))
    costs = dict(build_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "kernel-images.tcm"
    module.archive(str(artifact))
    costs["bytes"] = artifact.stat().st_size
    # CUDA also has an independent JIT reference; Vulkan uses the physical
    # X-contiguous image oracle in the consumer (not CUDA's reversed axes).
    if arch == "cuda":
        jit_reference(sample, energy)
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_kernel_image_probe.py")),
                mode,
                str(artifact),
                adapter or "",
                str(tmp_path / f"{mode}.json"),
                arch,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))


def jit_reference(sample, energy):
    textures = []
    for fmt, shape, dtype, values, nearest in (
        (ti.Format.r8, (4,), ti.u8, [0, 64, 128, 255], False),
        (ti.Format.r32f, (3, 5), ti.f32, range(15), False),
        (ti.Format.r32f, (3, 5), ti.f32, range(15), True),
        (ti.Format.r32f, (2, 3, 4), ti.f32, range(24), False),
    ):
        pixels = ti.ndarray(dtype, shape)
        pixels.from_numpy(
            np.array(values, np.uint8 if dtype == ti.u8 else np.float32).reshape(shape)
        )
        sampler = (
            ti.hardware.sampling.SamplerConfig(
                min_filter="nearest", mag_filter="nearest"
            )
            if nearest
            else None
        )
        texture = ti.Texture(fmt, shape, sampler=sampler)
        texture.from_ndarray(pixels)
        textures.append(texture)
    output = ti.ndarray(ti.f32, 9, needs_grad=True)
    loss = ti.ndarray(ti.f32, 1, needs_grad=True)
    sample(*textures, output)
    output.grad.fill(0)
    loss.fill(0)
    loss.grad.fill(1)
    energy(output, loss)
    energy.grad(output, loss)
    expected = [128 / 255, 1, 4, 11, 5.75, 6, 11, 14, 11.5]
    np.testing.assert_allclose(output.to_numpy(), expected, atol=2e-3)
    np.testing.assert_allclose(output.grad.to_numpy(), expected, atol=2e-3)
