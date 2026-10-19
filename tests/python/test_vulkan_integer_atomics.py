"""Native integer atomics must declare and require their actual SPIR-V capability."""

import shutil
import struct
import subprocess

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("dtype", [ti.i32, ti.i64, ti.u64])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_integer_atomic_native_operations(dtype, tmp_path):
    arch = ti.lang.impl.current_cfg().arch
    wide = dtype != ti.i32
    if arch == ti.vulkan and wide:
        if not ti.lang.impl.get_runtime().prog.get_device_caps().get("spirv_has_atomic_int64"):
            pytest.skip("Device has no 64-bit integer atomics")
    size = 17
    state = ti.field(dtype, shape=7)
    previous = ti.field(dtype, shape=(size, 2))
    low = (1 << 62) if dtype == ti.u64 else -(1 << (40 if wide else 20))
    high = low + size

    @ti.kernel
    def update():
        for i in range(size):
            previous[i, 0] = ti.atomic_add(state[0], ti.cast(1, dtype))
            previous[i, 1] = ti.atomic_sub(state[1], ti.cast(1, dtype))
            ti.atomic_min(state[2], dtype(low) + i)
            ti.atomic_max(state[3], dtype(low) + i)
            ti.atomic_or(state[4], ti.cast(1, dtype) << i)
            ti.atomic_and(state[5], ~(ti.cast(1, dtype) << i))
            ti.atomic_xor(state[6], ti.cast(1, dtype) << i)

    if arch == ti.vulkan:
        caps = ["spirv_has_int64", "spirv_has_atomic_int64"] if wide else []
        module = ti.aot.Module(caps=caps)
        module.add_kernel(update)
        module.save(str(tmp_path))
        shaders = list(tmp_path.glob("*.spv"))
        assert shaders
        atomic_opcodes = {234, 235, 236, 237, 238, 239, 240, 241, 242}
        seen = set()
        validator = shutil.which("spirv-val")
        for shader in shaders:
            code = shader.read_bytes()
            words = struct.unpack(f"<{len(code) // 4}I", code)
            caps_used = set()
            pos = 5
            while pos < len(words):
                count, op = words[pos] >> 16, words[pos] & 0xFFFF
                assert count > 0
                if op == 17:  # OpCapability
                    caps_used.add(words[pos + 1])
                if op in atomic_opcodes:
                    seen.add(op)
                pos += count
            assert (12 in caps_used) == wide  # Int64Atomics
            if validator:
                result = subprocess.run(
                    [validator, "--target-env", "vulkan1.2", str(shader)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert result.returncode == 0, result.stderr
        # Subtraction may be lowered to IAdd with a negative operand.
        assert {234, 240, 241, 242} <= seen
        assert ({237, 239} if dtype == ti.u64 else {236, 238}) <= seen

    mask = (1 << size) - 1
    state.from_numpy(
        np.asarray(
            [0, size, high, low - 1, 0, mask, 0], dtype={ti.i32: np.int32, ti.i64: np.int64, ti.u64: np.uint64}[dtype]
        )
    )
    update()
    assert state.to_numpy().tolist() == [size, 0, low, high - 1, mask, 0, mask]
    old = previous.to_numpy()
    assert sorted(old[:, 0].tolist()) == list(range(size))
    assert sorted(old[:, 1].tolist()) == list(range(1, size + 1))


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_integer_atomic_requires_int64_atomics():
    value = ti.field(ti.u64, shape=1)

    @ti.kernel
    def update():
        for i in range(17):
            ti.atomic_min(value[0], ti.cast(i, ti.u64))

    module = ti.aot.Module(caps=["spirv_has_int64"])
    with pytest.raises(RuntimeError, match="spirv_has_atomic_int64"):
        module.add_kernel(update)
