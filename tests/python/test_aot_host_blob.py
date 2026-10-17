"""Public immutable host inputs use the common native resource lifecycle."""

import os

import pytest
import taichi_forge as ti
from tests import test_utils
from taichi_forge_runtime.aot import Runtime


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_public_host_blob_borrowed_owner_and_input_boundaries():
    arch = ti.cfg.arch.name
    with ti.aot.borrow_runtime() as runtime:
        for data in (b"", b"\0\xff\x11\0", bytearray([0, 255, 17]), memoryview(b"ab01")[::2]):
            blob = runtime.host_blob(data)
            blob._check()
            if isinstance(data, bytearray):
                data[:] = b"\0" * len(data)
            blob.close()
            blob.close()  # Python owners provide idempotent retirement.
        with pytest.raises(TypeError):
            runtime.host_blob("not a byte buffer")
        with pytest.raises(TypeError):
            runtime.host_blob(123)
        blob = runtime.host_blob(b"owned until reset")
        ti.reset()
        with pytest.raises(RuntimeError):
            blob._check()
        blob.close()
    with Runtime(
        arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0
    ) as runtime:
        blob = runtime.host_blob(b"independent owner")
        with pytest.raises(RuntimeError):
            runtime.native_resource(
                "image", "sampled", parameters=dict(format="r32f", shape=[1]), bindings=dict(pixels=blob)
            )
    with pytest.raises(RuntimeError):
        blob._check()
