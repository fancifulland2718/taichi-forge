"""Source-free result checks; the borrowed mode also checks Program retirement."""

import array
import ctypes as C
import importlib.abc
import json
import os
from pathlib import Path
import statistics
import sys
import time


class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError(f"Unexpected frontend dependency: {fullname}")


def reject(exception, action):
    try:
        action()
    except exception:
        return
    raise AssertionError(f"Expected {exception}")


def read(value, code):
    result = array.array(code)
    result.frombytes(value.read())
    return result.tolist()


def main(mode, arch, artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime import aot

        runtime = aot.Runtime(
            arch,
            device_index=(
                int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0))
                if arch == "vulkan"
                else 0
            ),
        )
    else:
        import taichi_forge as ti
        from taichi_forge_runtime import aot

        ti.init(
            arch=getattr(ti, arch),
            offline_cache=False,
            debug=True,
            cpu_max_num_threads=4,
        )
        runtime = ti.aot.borrow_runtime()
    from taichi_forge_runtime.aot import _c_api as abi

    timings = {}
    with runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        timings["load_seconds"] = time.perf_counter() - start
        effects = runtime.ndarray("i32", 1)
        effects.write(array.array("i", [0]).tobytes())
        start = time.perf_counter()
        scalar = module.kernel("scalar")
        bound = scalar.bind(5, effects)
        timings["lookup_bind_seconds"] = time.perf_counter() - start
        assert read(effects, "i") == [0]  # Layout queries and binding do not execute.
        start = time.perf_counter()
        assert bound() == 13
        assert not runtime._pending
        timings["first_result_seconds"] = time.perf_counter() - start
        assert scalar(-2, effects) == -8
        assert scalar(0, effects) == -2

        # Invalid output capacity must fail before the kernel changes effects.
        poisoned = C.create_string_buffer(b"\xa5" * 16, 16)
        before = read(effects, "i")
        reject(
            aot.AotError,
            lambda: runtime._api.call(
                runtime._api.ti_launch_kernel_with_result_ext,
                runtime._handle,
                scalar._handle,
                len(bound._arguments),
                bound._arguments,
                len(bound._bindings),
                bound._bindings,
                16,
                poisoned,
            ),
        )
        assert poisoned.raw == b"\xa5" * 16 and read(effects, "i") == before
        assert bound() == 13
        # Legacy C callers may discard returns; their readback still completes.
        runtime._api.call(
            runtime._api.ti_launch_kernel_with_bindings_ext,
            runtime._handle,
            scalar._handle,
            len(bound._arguments),
            bound._arguments,
            len(bound._bindings),
            bound._bindings,
        )
        assert read(effects, "i") == [before[0] + 2]

        assert module.kernel("vec")(2.0) == (2.0, 2.5, -2.0)
        assert module.kernel("col")(2.0) == ((2.0,), (2.5,), (-2.0,))
        assert module.kernel("multiple")(-4) == (-7, -2.0)
        widths = module.kernel("widths")
        for value in (-3, 2):
            assert widths(value) == dict(
                flag=value > 0,
                flags=(value > 0, value < 0),
                half=value * 0.5,
                precise=value + 2.0**-40,
            )
        structured = module.kernel("structured")
        for x in (1.5, -2.5):
            result = structured(x)
            assert result == dict(
                tag=-7,
                wide=0xFEDCBA9876543210,
                inner=dict(code=513, weight=x),
                values=tuple(x + i for i in range(65)),
                basis=tuple(tuple(x + i * 9 + j for j in range(9)) for i in range(9)),
            )
        if arch != "vulkan":
            checked = module.kernel("checked")
            failed = checked.bind(-1)
            sentinel = C.create_string_buffer(
                b"\xa5" * checked._return_layout.size, checked._return_layout.size
            )
            reject(
                aot.AotError,
                lambda: runtime._api.call(
                    runtime._api.ti_launch_kernel_with_result_ext,
                    runtime._handle,
                    checked._handle,
                    len(failed._arguments),
                    failed._arguments,
                    len(failed._bindings),
                    failed._bindings,
                    checked._return_layout.size,
                    sentinel,
                ),
            )
            assert sentinel.raw == b"\xa5" * checked._return_layout.size
            assert checked(3) == 8  # Failed readback remains owned until completion.

        x, dx = runtime.ndarray("f32", 4), runtime.ndarray("f32", 4)
        y, dy = runtime.ndarray("f32", 1), runtime.ndarray("f32", 1)
        x.write(array.array("f", [1, 2, -3, 0.5]).tobytes())
        dx.write(bytes(16))
        y.write(bytes(4))
        dy.write(array.array("f", [1]).tobytes())
        energy = module.kernel("energy").bind(x, y, gradients={0: dx, 1: dy})
        backward = module.kernel("backward").bind(x, y, gradients={0: dx, 1: dy})
        result_kernel = module.kernel("read_energy")
        query, launch_result = (
            runtime._api.ti_get_kernel_return_info_ext,
            runtime._api.ti_launch_kernel_with_result_ext,
        )

        def unexpected(*args):
            raise AssertionError("Void replay queried or consumed a result")

        runtime._api.ti_get_kernel_return_info_ext = unexpected
        runtime._api.ti_launch_kernel_with_result_ext = unexpected
        assert energy() is None
        assert backward() is None
        runtime._api.ti_get_kernel_return_info_ext = query
        runtime._api.ti_launch_kernel_with_result_ext = launch_result
        assert result_kernel(y) == 14.25
        assert read(dx, "f") == [2, 4, -6, 1]

        samples = []
        runtime._api.ti_get_kernel_return_info_ext = unexpected
        for _ in range(12):
            start = time.perf_counter()
            assert bound() == 13
            samples.append(time.perf_counter() - start)
        runtime._api.ti_get_kernel_return_info_ext = query
        timings["warm_result_median_seconds"] = statistics.median(samples)
        timings["scalar_result_bytes"] = scalar._return_layout.size
        timings["structured_result_bytes"] = structured._return_layout.size
        reject(
            aot.AotError, bound.prepare
        )  # Prepared result frames are a separate delivery.

        sibling = runtime.load_module(artifact)
        sibling_call = sibling.kernel("scalar").bind(9, effects)
        module.close()
        reject(RuntimeError, bound)
        info = abi.KernelReturnInfo()
        reject(
            aot.AotError,
            lambda: runtime._api.call(
                query, runtime._handle, scalar._handle, C.byref(info)
            ),
        )
        assert sibling_call() == 25
        # Close a module directly after a failed return, without another launch.
        if arch != "vulkan":
            reject(aot.AotError, lambda: sibling.kernel("checked")(-1))
        sibling.close()
        reject(RuntimeError, sibling_call)
    if mode == "borrowed":
        late = ti.aot.borrow_runtime()
        late_module = late.load_module(artifact)
        late_call = late_module.kernel("vec").bind(3.0)
        assert late_call() == (3, 3.5, -3)
        ti.reset()
        reject(RuntimeError, late_call)
        late.close()
    print(json.dumps(dict(mode=mode, arch=arch, timings=timings)))
    Path(artifact, mode + "-result.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
