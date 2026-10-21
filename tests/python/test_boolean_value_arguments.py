"""Uniform argument storage and logical bool expressions use distinct types."""

import shutil
import subprocess

import pytest
import taichi_forge as ti

from tests import test_utils


def boolean_kernel():
    vector = ti.types.vector(2, ti.u1)
    inner = ti.types.struct(flag=ti.u1, bits=vector)
    record = ti.types.struct(first=ti.u1, last=ti.u1, inner=inner)

    @ti.kernel
    def evaluate(scalar: ti.u1, bits: vector, value: record) -> ti.i32:
        return (
            ti.cast(scalar, ti.i32)
            + 2 * ti.cast(bits[0], ti.i32)
            + 4 * ti.cast(bits[1], ti.i32)
            + 8 * ti.cast(value.first, ti.i32)
            + 16 * ti.cast(value.last, ti.i32)
            + 32 * ti.cast(value.inner.flag, ti.i32)
            + 64 * ti.cast(value.inner.bits[0], ti.i32)
            + 128 * ti.cast(value.inner.bits[1], ti.i32)
        )

    return evaluate, vector, inner, record


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_boolean_value_arguments():
    evaluate, vector, inner, record = boolean_kernel()
    for mask in (0, 255, 85, 170):
        bits = [bool(mask & (1 << i)) for i in range(8)]
        value = record(first=bits[3], last=bits[4],
                       inner=inner(flag=bits[5], bits=vector(bits[6:])))
        assert evaluate(bits[0], bits[1:3], value) == mask


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_boolean_value_argument_spirv(tmp_path):
    validator = shutil.which("spirv-val")
    if validator is None:
        pytest.skip("SPIR-V Tools validator is required for the artifact check")
    evaluate, _, _, _ = boolean_kernel()
    module = ti.aot.Module()
    module.add_kernel(evaluate)
    module.save(str(tmp_path))
    shaders = list(tmp_path.glob("*.spv"))
    assert shaders
    for shader in shaders:
        result = subprocess.run(
            [validator, "--target-env", "vulkan1.3", str(shader)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
