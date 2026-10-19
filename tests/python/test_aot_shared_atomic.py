"""Shared float CAS storage must preserve SPIR-V types and native fast paths."""

import array
from contextlib import closing
import struct
from zipfile import ZipFile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Scalar
from tests import test_utils


def _atomic_opcodes(artifact):
    opcodes = []
    with ZipFile(artifact) as archive:
        for name in archive.namelist():
            if not name.endswith(".spv"):
                continue
            data = archive.read(name)
            words = struct.unpack(f"<{len(data) // 4}I", data)
            types, values, atomics = {}, {}, []
            offset = 5
            while offset < len(words):
                size, opcode = words[offset] >> 16, words[offset] & 0xFFFF
                assert size and offset + size <= len(words)
                args = words[offset + 1 : offset + size]
                offset += size
                opcodes.append(opcode)
                if opcode == 32:  # OpTypePointer: result, storage class, element.
                    types[args[0]] = args[2]
                if opcode in (59, 65):  # OpVariable / OpAccessChain.
                    values[args[1]] = args[0]
                if opcode in (227, 230, 5614, 5615, 6035):
                    # AtomicLoad / CompareExchange / FMin / FMax / FAdd.
                    atomics.append((opcode, args[0], args[2]))
            for opcode, result_type, pointer in atomics:
                assert result_type == types[values[pointer]], (
                    name,
                    opcode,
                    result_type,
                    pointer,
                )
    return opcodes


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64], ids=["f32", "f64"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_shared_float_atomic_storage(tmp_path, dtype):
    @ti.kernel
    def reduce(output: ti.types.ndarray(dtype, 1), scale: dtype):
        ti.loop_config(block_dim=32)
        for i in range(32):
            total = ti.simt.block.SharedArray((1,), dtype)
            bounds = ti.simt.block.SharedArray((2,), dtype)
            if i == 0:
                total[0] = 0
                bounds[0] = 100
                bounds[1] = -100
            ti.simt.block.sync()
            ti.atomic_add(total[0], ti.cast(i + 1, dtype) * scale)
            ti.atomic_sub(total[0], scale)
            # This allocation mixes operations with independently selected caps.
            ti.atomic_add(bounds[0], scale)
            ti.atomic_sub(bounds[0], scale)
            ti.simt.block.sync()
            ti.atomic_min(bounds[0], ti.cast(i - 16, dtype) * scale)
            ti.atomic_max(bounds[1], ti.cast(i - 16, dtype) * scale)
            ti.simt.block.sync()
            if i == 0:
                output[0] = total[0]
                output[1] = bounds[0]
                output[2] = bounds[1]

    with ti.aot.borrow_runtime() as runtime:
        caps = runtime.device_capabilities()
        base = (
            []
            if dtype == ti.f32
            else [
                "spirv_has_float64",
                "spirv_has_int64",
                "spirv_has_shared_atomic_int64",
            ]
        )
        if any(not caps.get(cap, 0) for cap in base):
            pytest.skip("f64 shared CAS requires float64 and int64 atomics")
        add = "spirv_has_shared_atomic_float_add" if dtype == ti.f32 else "spirv_has_shared_atomic_float64_add"
        minmax = "spirv_has_shared_atomic_float_minmax" if dtype == ti.f32 else "spirv_has_shared_atomic_float64_minmax"
        profiles = [base]
        # Buffer-only native capability must not enable workgroup FAdd.
        buffer_add = "spirv_has_atomic_float_add" if dtype == ti.f32 else "spirv_has_atomic_float64_add"
        if caps.get(buffer_add, 0):
            profiles.append(base + [buffer_add])
        if caps.get(add, 0):
            profiles.append(base + [add])
            if caps.get(minmax, 0):
                profiles.append(base + [add, minmax])
        for index, selected in enumerate(profiles):
            module = ti.aot.Module(caps=[getattr(ti.DeviceCapability, name) for name in selected])
            module.add_kernel(reduce)
            artifact = tmp_path / f"shared-{index}.tcm"
            module.archive(str(artifact))
            opcodes = _atomic_opcodes(artifact)
            # The add-only allocation retains native FAdd even if the other
            # allocation needs CAS for min/max; no global fast-path regression.
            assert (6035 in opcodes) == (add in selected)
            assert (230 in opcodes) == (minmax not in selected)
            with (
                runtime.load_module(artifact) as loaded,
                closing(runtime.ndarray(str(dtype), [3])) as output,
            ):
                for scale in (1, 0, 2):
                    scalar = Scalar(str(dtype), scale)
                    with closing(loaded.kernel("reduce").prepare(output, scalar)) as call:
                        call()
                        result = array.array("f" if dtype == ti.f32 else "d")
                        result.frombytes(output.read())
                        np.testing.assert_array_equal(result, np.array([496, -16, 15]) * scale)
        if dtype == ti.f64:
            # Buffer int64 support cannot satisfy the shared CAS contract.
            missing_shared = ti.aot.Module(caps=["spirv_has_float64", "spirv_has_int64", "spirv_has_atomic_int64"])
            with pytest.raises(RuntimeError, match="spirv_has_shared_atomic_int64"):
                missing_shared.add_kernel(reduce)
