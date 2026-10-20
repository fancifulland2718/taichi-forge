"""No-kernel layout catalog with the same native module lifetime as entries."""

import importlib.abc
import os
from pathlib import Path
import struct
import sys


class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError(fullname)


def reject(error, callback):
    try:
        callback()
    except error:
        return
    raise AssertionError(f"Expected {error}")


def main(mode, arch, artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
    from taichi_forge_runtime import aot

    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = aot.Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0
        )
    with runtime:
        modules = [runtime.load_module(artifact), runtime.load_module(artifact)]
        layout = modules[0].ndarray_layout("record")
        assert modules[0].ndarray_layout("record") is layout
        assert modules[0].ndarray_layout("alias")._handle == layout._handle
        assert layout.member("state.v").element_shape == (3,)
        record = runtime.ndarray(layout, 17)
        data = bytearray(record.nbytes)
        for i in range(17):
            struct.pack_into("=f", data, i * layout.element_stride + layout.member("state.value").offset, i + 0.5)
        record.write(data)
        assert record.read() == data
        reject(aot.AotError, lambda: modules[0].ndarray_layout("missing"))
        # Catalog handles remain checked against the actual runtime's owner set.
        with aot.Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0
        ) as other:
            reject(ValueError, lambda: other.ndarray(layout, 1))
            reject(
                aot.AotError,
                lambda: other._api.call(
                    other._api.ti_get_aot_module_ndarray_layout_ext, other._handle, modules[0]._handle, b"record"
                ),
            )
        invalid = runtime.load_module(Path(artifact).with_name("invalid.tcm"))
        for _ in range(2):
            reject(aot.AotError, lambda: invalid.ndarray_layout("record"))
        invalid.close()
        old_handle = modules[0]._handle
        modules[0].close()
        reject(RuntimeError, lambda: runtime.ndarray(layout, 1))
        reject(
            aot.AotError,
            lambda: runtime._api.call(
                runtime._api.ti_get_aot_module_ndarray_layout_ext, runtime._handle, old_handle, b"record"
            ),
        )
        sibling = runtime.ndarray(modules[1].ndarray_layout("record"), 17)
        sibling.write(data)
        assert sibling.read() == record.read() == data
        record.close()
        sibling.close()
        modules[1].close()
    print("Kernel-free named layouts, aliases, malformed metadata and owners passed", mode, arch)


if __name__ == "__main__":
    main(*sys.argv[1:])
