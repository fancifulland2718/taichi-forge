"""Private image layout cycles span selected control recipes and replay exits."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_kernel_image_probe import native


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_private_images_in_selected_device_control(tmp_path):
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
    def condition(image: ti.types.texture(2), count: ti.types.ndarray(ti.i32, 1),
                  predicate: ti.types.ndarray(ti.i32, 1), limit: ti.types.ndarray(ti.i32, 1)):
        predicate[0] = int(count[0] < limit[0] and image.fetch(ti.Vector([0, 0]), 0).x < 1000)

    @ti.kernel
    def increment(count: ti.types.ndarray(ti.i32, 1)):
        count[0] += 1

    @ti.kernel
    def zero(count: ti.types.ndarray(ti.i32, 1)):
        count[0] = 0

    @ti.kernel
    def energy(output: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in output:
            loss[0] += output[i] ** 2 * 0.5

    @ti.kernel
    def alias(image: ti.types.texture(2), writable: ti.types.rw_texture(2, fmt=ti.Format.r32f)):
        writable.store(ti.Vector([0, 0]), image.fetch(ti.Vector([0, 0]), 0))

    module = ti.aot.Module()
    for kernel in (update, sample, condition, increment, zero, energy, alias):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")

    def call(entry, *arguments):
        return dict(kind="kernel", entry=entry, arguments=list(arguments))

    test = call("condition", "image", "count", "predicate", "target")
    outer_test = call("condition", "image", "outer", "outer_predicate", "outer_target")
    update_image = call("update", "image", "delta")
    body = [update_image, update_image, call("increment", "count")]
    resources = [dict(native("image", "sampled", dict(pixels="seed"),
                             shape=[2, 2], format="r32f", storage=True), name="image")]
    resources += [dict(kind="ndarray", name=name, dtype="i32", shape=[1])
                  for name in ("count", "predicate", "outer", "outer_predicate")]
    suffix = [call("sample", "image", "output"),
              dict(kind="fill32", destination="doutput", value=0),
              dict(kind="fill32", destination="loss", value=0),
              dict(kind="fill32", destination="dloss", value=0x3F800000),
              call("energy", "output", "loss"), call("backward", "output", "loss")]
    for name in ("zero", "chunks", "conditional", "coarse_conditional", "nested", "alias"):
        budget = 0 if name == "zero" else 3
        chunks = [dict(iterations=1, strategy="compact"), dict(iterations=2, strategy="chained")]
        if name in ("conditional", "coarse_conditional"):
            chunks = [dict(iterations=budget, strategy=name)]
        loop = dict(kind="while", predicate="predicate", max_iterations=budget,
                    condition=[test], body=body,
                    lowering=dict(kind="vulkan_structured", version=1, counter="count",
                                  chunks=chunks if budget else []))
        if name == "nested":
            loop = dict(kind="while", predicate="outer_predicate", max_iterations=2,
                        condition=[outer_test],
                        body=[call("zero", "count"), test, *body, test,
                              call("increment", "outer"), outer_test],
                        lowering=dict(kind="vulkan_nested", version=1, counter="outer", inners=[dict(
                            predicate="predicate", counter="count", condition_begin=2, body_begin=3,
                            end=7, max_iterations=3, chunk_size=3)]))
        if name == "alias":
            loop["body"] = [call("alias", "image", "image"), call("increment", "count")]
        module.add_execution_plan(name, ti.aot.ExecutionPlan(
            [call("zero", "count"), call("zero", "outer"), loop] + suffix,
            resources=resources, gradients=dict(output="doutput", loss="dloss")))
    archive = tmp_path / "private-image-control.tcm"
    module.archive(str(archive))
    ti.reset()
    result = subprocess.run(
        [sys.executable, "-S", str(Path(__file__).with_name("aot_domain_image_control_probe.py")), str(archive)],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True, text=True, timeout=90)
    output = result.stdout + result.stderr
    print(output)
    if result.returncode == 77:
        pytest.skip(output)
    assert result.returncode == 0, output
    assert not any(token in output for token in ("VUID-", "SYNC-HAZARD", "Validation Error"))
    assert "private_image_control_instances=10" in output
