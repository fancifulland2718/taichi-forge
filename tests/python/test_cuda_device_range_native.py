"""Real LLVM artifacts consumed by the native selected-task recorder."""

import os
from pathlib import Path
import subprocess

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_native_range_recording_instances_and_explicit_adjoint(tmp_path):
    executable = Path(os.environ.get("TI_TEST_CPP_EXECUTABLE", "build/taichi_cpp_tests.exe"))
    if not executable.is_file():
        pytest.skip("Requires the native CUDA range test executable")

    @ti.kernel
    def stage(bounds: ti.types.ndarray(ti.i32, 1), out: ti.types.ndarray(ti.i32, 1)):
        bounds[1] = bounds[0]
        ti.loop_config(block_dim=32)
        for i in range(-3, bounds[1] - 3):
            out[i + 3] = i + 4
        bounds[1] = bounds[1] // 2
        ti.loop_config(block_dim=32)
        for i in range(bounds[1]):
            out[i] += 100
        out[65] = 1000 + bounds[1]

    @ti.kernel
    def square(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               bounds: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(block_dim=32)
        for i in range(bounds[0]):
            y[i] = x[i] * x[i]

    module = ti.aot.Module()
    module.add_kernel(stage)
    module.add_kernel(square)
    module.add_kernel(square.grad, name="square_grad")
    module.save(tmp_path)
    result = subprocess.run(
        [str(executable.resolve()), "--gtest_filter=CudaRangeGraph.*"],
        env={**os.environ, "TAICHI_CUDA_RANGE_AOT_FOLDER": str(tmp_path)},
        capture_output=True, text=True, timeout=90,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
    assert "[  PASSED  ] 4 tests." in result.stdout
    assert "[  SKIPPED ]" not in result.stdout
