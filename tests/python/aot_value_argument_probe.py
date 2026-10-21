"""No-DSL aggregate binding checks, with a separate borrowed owner mode."""

import array
import copy
import ctypes as C
import gc
import importlib.abc
import json
import os
import statistics
import sys
import time
import weakref


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


def read(value, code="f"):
    result = array.array(code)
    result.frombytes(value.read())
    return result.tolist()


def write(value, data, code="f"):
    value.write(array.array(code, data).tobytes())


def record():
    return dict(tag=-3, inner=dict(scale=2.0, code=7), bias=[1.0, -2.0, 3.0],
                basis=[[4.0, 9.0], [8.0, 5.0]], column=[[10.0], [20.0], [30.0]])


def expected(v, r):
    return [r["inner"]["scale"] * v[i] + r["bias"][i] + r["basis"][i % 2][i % 2]
            + r["column"][i][0] + r["tag"] + r["inner"]["code"] for i in range(3)]


def main(mode, arch, artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime import aot
        runtime = aot.Runtime(arch, device_index=(int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0))
                                                 if arch == "vulkan" else 0))
    else:
        import taichi_forge as ti
        from taichi_forge_runtime import aot
        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
        runtime = ti.aot.borrow_runtime()
    from taichi_forge_runtime.aot import _c_api as abi

    timings = {}
    with runtime:
        started = time.perf_counter()
        module = runtime.load_module(artifact)
        timings["load_seconds"] = time.perf_counter() - started
        out = runtime.ndarray("f32", 3)
        write(out, [-123] * 3)
        v, r = [1.0, 2.0, 3.0], record()
        reference = expected(v, r)
        started = time.perf_counter()
        entry = module.kernel("write_values")
        bound = entry.bind(v, r, out)
        timings["lookup_bind_seconds"] = time.perf_counter() - started
        assert read(out) == [-123] * 3
        reject(ValueError, lambda: entry.bind(v, r))
        reject((TypeError, ValueError), lambda: entry.bind([[1], [2], [3]], r, out))
        invalid = copy.deepcopy(r)
        del invalid["inner"]["code"]
        reject(ValueError, lambda: entry.bind(v, invalid, out))
        invalid = copy.deepcopy(r)
        invalid["extra"] = 1
        reject(ValueError, lambda: entry.bind(v, invalid, out))
        invalid = copy.deepcopy(r)
        invalid["tag"] = 1 << 40
        reject(ValueError, lambda: entry.bind(v, invalid, out))

        # C callers get the same capacity/type checks before device submission.
        bad = (abi.Argument * 3)(*bound._arguments)
        size = bad[0].value.value.size
        bad[0].value.value.size = size - 1
        reject(aot.AotError, lambda: runtime._api.call(
            runtime._api.ti_launch_kernel_with_bindings_ext,
            runtime._handle, entry._handle, 3, bad, 0, None))
        reject(aot.AotError, lambda: runtime._api.call(
            runtime._api.ti_prepare_kernel_ext,
            runtime._handle, entry._handle, 3, bad, 0, None))
        bad[0].value.value.size = size
        bad[0].value.value.data = None
        reject(aot.AotError, lambda: runtime._api.call(
            runtime._api.ti_launch_kernel_with_bindings_ext,
            runtime._handle, entry._handle, 3, bad, 0, None))
        assert read(out) == [-123] * 3
        del bad
        # Mutating the Python input cannot change its existing immutable binding.
        v[0], r["inner"]["scale"] = 99, 99
        started = time.perf_counter()
        bound.run()
        runtime.wait()
        timings["ordinary_first_seconds"] = time.perf_counter() - started
        assert read(out) == reference
        write(out, [-123] * 3)
        started = time.perf_counter()
        prepared = bound.prepare()
        timings["prepare_seconds"] = time.perf_counter() - started
        assert read(out) == [-123] * 3
        payloads = [weakref.ref(buffer) for buffer in bound._value_buffers]
        for buffer in bound._value_buffers:
            C.memset(buffer, 0xA5, C.sizeof(buffer))
        del buffer, bound
        gc.collect()
        assert all(payload() is None for payload in payloads)
        prepared.run()
        runtime.wait()
        assert read(out) == reference
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            prepared.run()
            runtime.wait()
            samples.append(time.perf_counter() - started)
        timings["prepared_warm_complete_seconds"] = statistics.median(samples)
        timings["prepared_argument_bytes"] = prepared.info["argument_bytes"]

        other_out = runtime.ndarray("f32", 3)
        other = entry.prepare([-1, -2, -3], record(), other_out)
        other.run()
        runtime.wait()
        assert read(other_out) == expected([-1, -2, -3], record())
        assert read(out) == reference
        returned = module.kernel("value_return")
        assert returned([1, 2, 3], [[10], [20], [30]]) == (31.0, -18.0, 30.0)
        reject((TypeError, ValueError), lambda: returned.bind([1, 2, 3], [10, 20, 30]))

        large_out = runtime.ndarray("f32", 1)
        large = [i * 2 for i in range(65)]
        matrix = [[i * 9 + j for j in range(9)] for i in range(9)]
        wide = module.kernel("wide").prepare(large, matrix, large_out)
        wide.run()
        runtime.wait()
        assert read(large_out) == [312.0]
        integers, floats, unsigned = (runtime.ndarray("i64", 4), runtime.ndarray("f64", 1),
                                     runtime.ndarray("u64", 1))
        widths = dict(flag=True, other=False, flags=[False, True], half=1.5,
                      precise=2.0 + 2.0**-40, tag=-7, code=513, wide=0xFEDCBA9876543210)
        mixed = module.kernel("mixed_widths").prepare(widths, integers, floats, unsigned)
        mixed.run()
        runtime.wait()
        assert read(integers, "q") == [-7, 513, 1, 2]
        assert read(floats, "d") == [3.5 + 2.0**-40]
        assert read(unsigned, "Q") == [0xFEDCBA9876543210]

        # Values are constants for AD; ndarray gradients use existing bindings.
        x, dx = runtime.ndarray("f32", 3), runtime.ndarray("f32", 3)
        loss, dloss = runtime.ndarray("f32", 1), runtime.ndarray("f32", 1)
        write(x, [1, -2, 3])
        write(dx, [0, 0, 0])
        write(loss, [0])
        write(dloss, [1])
        args = (x, [1.0, 2.0, 0.5], record(), loss)
        gradients = {0: dx, 3: dloss}
        energy = module.kernel("energy").prepare(*args, gradients=gradients)
        backward = module.kernel("backward").prepare(*args, gradients=gradients)
        energy.run()
        backward.run()
        runtime.wait()
        assert read(loss) == [41.0]
        assert read(dx) == [5.0, -18.0, 9.0]
        second = runtime.load_module(artifact)
        independent = second.kernel("write_values").prepare([1, 2, 3], record(), other_out)
        ticket = prepared.submit()
        module.close()
        ticket.close()
        reject(aot.AotError, prepared.run)
        independent.run()
        runtime.wait()
        assert read(other_out) == reference
        for value in (prepared, other, wide, mixed, energy, backward):
            value.close()
        if mode == "borrowed":
            independent.run()
            ti.reset()
            reject(RuntimeError, independent.run)
        else:
            independent.close()
            second.close()
    print("VALUE_ARGUMENTS_PASSED", json.dumps(dict(mode=mode, arch=arch, timings=timings)))


if __name__ == "__main__":
    main(*sys.argv[1:])
