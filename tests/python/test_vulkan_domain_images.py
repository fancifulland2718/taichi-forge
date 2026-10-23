"""Private images share an instance boundary across preparation and replay."""

import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils
from tests.python.aot_kernel_image_probe import native


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_private_domain_images_and_downstream_adjoint(tmp_path):
    @ti.kernel
    def update(image: ti.types.rw_texture(2, fmt=ti.Format.r32f), delta: ti.types.ndarray(ti.f32, 1)):
        for x, y in ti.ndrange(2, 2):
            xy = ti.Vector([x, y])
            image.store(xy, ti.Vector([image.load(xy).x + delta[0], 0, 0, 0]))

    @ti.kernel
    def sample(image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = image.fetch(ti.Vector([i % 2, i // 2]), 0).x

    @ti.kernel
    def energy(output: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in output:
            loss[0] += output[i] ** 2 * 0.5

    module = ti.aot.Module()
    for kernel in (update, sample, energy):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    operations = [
        dict(kind="kernel", entry="update", arguments=["image", "delta"]),
        dict(kind="kernel", entry="sample", arguments=["image", "output"]),
        dict(kind="fill32", destination="doutput", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["output", "loss"]),
        dict(kind="kernel", entry="backward", arguments=["output", "loss"]),
    ]
    for policy in ("seeded", "uploaded"):
        image = dict(native("image", "sampled", dict(pixels="seed") if policy == "seeded" else {},
                            shape=[2, 2], format="r32f", storage=True), name="image")
        initialization = ([native("image", "upload", dict(image="image", pixels="seed"))]
                          if policy == "uploaded" else [])
        module.add_execution_plan(policy, ti.aot.ExecutionPlan(
            operations, resources=[image], initialization=initialization or None,
            gradients=dict(output="doutput", loss="dloss")))
    archive = tmp_path / "private-images.tcm"
    module.archive(str(archive))
    ti.reset()
    result = subprocess.run(
        [sys.executable, "-S", str(Path(__file__).with_name("aot_domain_image_probe.py")), str(archive)],
        env=os.environ.copy(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VUID-" not in result.stdout + result.stderr
    assert "SYNC-HAZARD" not in result.stdout + result.stderr
    assert "private_image_instances=4" in result.stdout
