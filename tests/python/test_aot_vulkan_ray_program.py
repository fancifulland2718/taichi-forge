"""Program pipelines and launches reconstruct without the producer runtime."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_vulkan_ray_program_probe import native, pipeline_spec, launch_spec, operations, private_resources
from tests.python.test_hardware_vulkan_ray_program import _shader


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_vulkan_ray_program_resources_and_gradients(tmp_path):
    if not ti.hardware.ray.is_program_available():
        pytest.skip("Vulkan RT pipeline unavailable")

    @ti.kernel
    def decode(values: ti.types.ndarray(ti.u32, 2), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = ti.bit_cast(values[i, 4], ti.f32)

    @ti.kernel
    def sample(image: ti.types.texture(num_dimensions=2), output: ti.types.ndarray(ti.f32, 1)):
        for i in range(6):
            value = image.fetch(ti.Vector([i % 3, i // 3]), 0)
            for j in ti.static(range(4)):
                output[4 * i + j] = value[j]

    @ti.kernel
    def energy(
        values: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in values:
            if values[i] > 0:
                loss[0] += 0.5 * values[i] * values[i]

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(decode)
    module.add_kernel(sample)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    for images in (False, True):
        spec = pipeline_spec(images)
        code_resources = []
        filenames = (
            ["program_images.rgen"]
            if images
            else ["instance_mapping.rgen", "instance_mapping.rmiss", "instance_mapping.rchit", "instance_mapping.rahit"]
        )
        for i, (filename, metadata) in enumerate(zip(filenames, spec["shaders"])):
            shader = _shader(filename, metadata["stage"])
            code = shader.code
            (tmp_path / f"{'image' if images else 'ray'}_{i}.spv").write_bytes(code)
            code_resources.append(
                dict(kind="host_blob", name=f"code_{i}", data=module.add_data(code), byte_size=len(code))
            )
        _, payload = launch_spec(images)
        blob = dict(kind="host_blob", name="payload", data=module.add_data(payload), byte_size=len(payload))
        pipeline = dict(
            native(
                "vulkan_ray_program",
                "pipeline",
                {f"shader_{i}": f"code_{i}" for i in range(len(code_resources))},
                **spec,
            ),
            name="pipeline",
        )
        for private in (False, True):
            resources = [blob, dict(kind="ndarray", name="dloss", dtype="f32", shape=[1])]
            if private:
                resources += code_resources + [pipeline] + private_resources(images)
            module.add_execution_plan(
                ("images" if images else "rays") + ("_private" if private else "_public"),
                ti.aot.ExecutionPlan(
                    operations(images), resources=resources, gradients=dict(features="dfeatures", loss="dloss")
                ),
            )
    costs = dict(compile_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "ray_program.tcm"
    module.archive(str(artifact))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_vulkan_ray_program_probe.py")),
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
        assert "cleanup failed" not in result.stdout + result.stderr
