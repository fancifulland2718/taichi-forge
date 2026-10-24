"""Private offscreen attachments and their compute consumer share one domain."""

import os
from pathlib import Path
import struct
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils
from tests.python.aot_vulkan_graphics_probe import native, pipeline_spec, pass_spec
from tests.python.test_hardware_graphics import _spirv_header


def build_fixture(folder):
    @ti.kernel
    def sample(color: ti.types.texture(2), depth: ti.types.texture(2),
               output: ti.types.ndarray(ti.f32, 1), scale: ti.u64):
        for i in range(64):
            xy = ti.Vector([i % 8, i // 8])
            output[i] = color.fetch(xy, 0).x * ti.cast(scale, ti.f32)
            output[i + 64] = depth.fetch(xy, 0).x * ti.cast(scale, ti.f32)

    @ti.kernel
    def energy(output: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in output:
            loss[0] += output[i] * output[i] * 0.5

    module = ti.aot.Module(caps=["spirv_has_int64"])
    module.add_kernel(sample)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    resources = []
    for i, stage in enumerate(("vert", "frag")):
        code = _spirv_header(f"3_triangle.{stage}.spv.h")
        resources.append(dict(kind="host_blob", name=f"shader_{i}",
                              data=module.add_data(code), byte_size=len(code)))
    spec = pipeline_spec()
    spec.update(depth_test=True, depth_write=True, depth_compare="less")
    resources.append(dict(native("vulkan_graphics", "pipeline",
                                 dict(shader_0="shader_0", shader_1="shader_1"), **spec), name="pipeline"))
    for name, values, usage in (
        ("vertices", [v for xy in ((-1, -1), (3, -1), (-1, 3))
                      for v in (*xy, 1, 0, 0, 0.5, 0.5)], ["vertex"]),
        ("uniform", [1, 0, 0, 0], ["uniform"]),
        ("pixels", [0.5] * 4, ["storage"]),
    ):
        resources.append(dict(kind="ndarray", name=name, dtype="f32",
                              shape=[1, 1, 4] if name == "pixels" else [len(values)],
                              usage=usage, initial_data=module.add_data(struct.pack(f"<{len(values)}f", *values))))
    resources += [
        dict(native("image", "sampled", {}, shape=[8, 8], format="rgba32f", attachment=True), name="target"),
        dict(native("image", "sampled", {}, shape=[8, 8], format="depth32f", attachment=True), name="depth"),
        dict(native("image", "sampled", dict(pixels="pixels"), shape=[1, 1], format="rgba32f"), name="source"),
    ]
    draw = pass_spec()
    draw["parameters"]["clear_depth"] = 1.0
    plan = ti.aot.ExecutionPlan([
        draw,
        dict(kind="kernel", entry="sample", arguments=["target", "depth", "output", "scale"]),
        dict(kind="fill32", destination="doutput", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["output", "loss"]),
        dict(kind="kernel", entry="backward", arguments=["output", "loss"]),
    ], resources=resources, gradients=dict(output="doutput", loss="dloss"))
    module.add_execution_plan("offscreen", plan)
    module.save(folder)
    # The native qualification consumes exactly the producer's persisted plan.
    (folder / "offscreen.json").write_text(plan.to_json(), encoding="utf8")


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_offscreen_domain_sampling_parameters_and_adjoint(tmp_path):
    # The ordinary JIT entry exposes the same cold queue selection contract.
    with ti.graph.execution_domain(queue="graphics") as domain:
        assert domain.info["queue_family_index"] >= 0
    folder = tmp_path / "module"
    build_fixture(folder)
    ti.reset()
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-S", str(Path(__file__).with_name("aot_domain_graphics_probe.py")), str(folder)],
        env=env, capture_output=True, text=True, timeout=90)
    (tmp_path / "consumer.log").write_text(result.stdout + result.stderr, encoding="utf8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VUID-" not in result.stdout + result.stderr
    assert "SYNC-HAZARD" not in result.stdout + result.stderr
    assert "offscreen_domain_ok" in result.stdout
    executable = env.get("TI_TEST_DOMAIN_EXECUTABLE")
    if executable:
        env["TAICHI_VULKAN_OFFSCREEN_AOT_DIR"] = str(folder)
        result = subprocess.run(
            [executable, "--gtest_filter=VulkanExecutionOwner.PrivateOffscreenIgnoresDefaultQueue"],
            env=env, capture_output=True, text=True, timeout=60)
        (tmp_path / "native.log").write_text(result.stdout + result.stderr, encoding="utf8")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[  PASSED  ] 1 test." in result.stdout
        assert "[  SKIPPED ]" not in result.stdout
        assert "VUID-" not in result.stdout + result.stderr
        assert "SYNC-HAZARD" not in result.stdout + result.stderr
