"""AOT image region transfer factories preserve buffer and resource contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_image_transfer_backend_isolation():
    with ti.aot.borrow_runtime() as runtime:
        for kind in ("image_copy_region", "image_blit", "buffer_to_image", "image_to_buffer"):
            assert not runtime.native_operation_capabilities(kind)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_image_transfer_regions_views_and_downstream_gradients(tmp_path):
    from tests.python.aot_image_transfer_probe import plan_parts

    @ti.kernel
    def sample(image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = image.fetch(ti.Vector([i % 4, i // 4]), 0).x

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in output:
            loss[0] += 0.5 * output[i] * output[i]

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(sample)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    for policy in ("private", "public"):
        module.add_execution_plan(policy, ti.aot.ExecutionPlan(**plan_parts(policy == "private")))
    build_ms = (time.perf_counter() - start) * 1000
    artifact = tmp_path / "image-transfer.tcm"
    module.archive(str(artifact))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        run = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_image_transfer_probe.py")),
                mode,
                str(artifact),
                str(tmp_path / f"{mode}.json"),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        assert "Validation Error" not in run.stdout + run.stderr
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(build_ms=build_ms, artifact_bytes=artifact.stat().st_size), indent=2)
    )
