"""Validate retained unreachable IR before submitting any shader to a driver."""

import shutil
import subprocess

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("tier", ["fast", "balanced", "full"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_continue_unreachable_spirv_and_jit(tier, tmp_path):
    validator = shutil.which("spirv-val")
    if validator is None:
        pytest.skip("spirv-val is required to validate shaders before execution")
    ti.init(
        arch=ti.vulkan,
        enable_fallback=False,
        offline_cache=False,
        compile_tier=tier,
        advanced_optimization=tier != "fast",
    )
    x = ti.field(ti.i32, shape=67)
    y = ti.field(ti.i32, shape=67)

    @ti.kernel
    def static_continue():
        for _ in range(1):
            for j in range(67):
                if x[j] == 2:
                    if ti.static(True):
                        continue
                    y[j] = x[j] + 1
                elif x[j] == 4:
                    y[j] = x[j] + 2

    @ti.kernel
    def nested_continue():
        for j in range(67):
            if x[j] < 3:
                if x[j] == 2:
                    if ti.static(True):
                        continue
                    y[j] = 99
                else:
                    continue
                y[j] = 8
            else:
                y[j] = x[j] + 2

    @ti.kernel
    def while_continue_break():
        for j in range(67):
            k = 0
            while k < 4:
                k += 1
                if k == 2:
                    if ti.static(True):
                        continue
                    y[j] = 99
                if k == 4:
                    break
                y[j] += k

    @ti.kernel
    def dynamic_continue():
        for _ in range(1):
            for j in range(67):
                if x[j] == 2:
                    continue
                y[j] = x[j] + 1

    @ti.kernel
    def unconditional_continue():
        for j in range(67):
            if ti.static(True):
                continue
            y[j] = 99

    @ti.kernel
    def serial_unconditional_continue():
        for _ in range(1):
            for j in range(67):
                if ti.static(True):
                    continue
                y[j] = 99

    kernels = (
        static_continue,
        nested_continue,
        while_continue_break,
        dynamic_continue,
        unconditional_continue,
        serial_unconditional_continue,
    )
    module = ti.aot.Module(ti.vulkan)
    for kernel in kernels:
        module.add_kernel(kernel)
    module.save(str(tmp_path))
    shaders = list(tmp_path.glob("*.spv"))
    # Higher tiers can remove the entirely empty unconditional loop.
    assert len(shaders) >= 4
    for shader in shaders:
        result = subprocess.run(
            [validator, "--target-env", "vulkan1.2", str(shader)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{shader.name}: {result.stderr}"

    inputs = np.arange(67, dtype=np.int32) % 5
    x.from_numpy(inputs)
    expected = (
        np.where(inputs == 4, 6, 0),
        np.where(inputs >= 3, inputs + 2, 0),
        np.full(67, 4),
        np.where(inputs == 2, 0, inputs + 1),
        np.zeros(67),
        np.zeros(67),
    )
    for kernel, reference in zip(kernels, expected):
        y.fill(0)
        kernel()
        np.testing.assert_array_equal(y.to_numpy(), reference)
