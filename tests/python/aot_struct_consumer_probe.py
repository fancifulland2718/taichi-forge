"""Fixed-AoS consumer: no DSL or NumPy required for the standalone route."""

import array
import importlib.abc
import math
import os
import struct
import sys


def reject(exception, action):
    try:
        action()
    except exception:
        return
    raise AssertionError(f"Expected {exception}")


class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError(f"Unexpected frontend dependency: {fullname}")


def read(value):
    result = array.array("f")
    result.frombytes(value.read())
    return result.tolist()


def records_bytes(layout, n):
    # Poison padding, then fill only the ABI's exported scalar/tensor members.
    payload = bytearray([0xA5]) * (n * layout.element_stride)
    codes = dict(i8="b", u16="H", f32="f")
    for i in range(n):
        values = {
            "tag": [-3],
            "inner.code": [300 + i],
            "inner.value": [i * 0.5],
            "velocity": [1, 2, 3],
            "basis": [4, 5, 6, 7],
        }
        for member in layout.members:
            count = math.prod(member.element_shape)
            assert count == len(values[member.path])
            struct.pack_into(
                "=" + codes[member.dtype] * count,
                payload,
                i * layout.element_stride + member.offset,
                *values[member.path],
            )
    return payload


def main(mode, arch, artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime import aot

        runtime = aot.Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0)) if arch == "vulkan" else 0
        )
    else:
        import taichi_forge as ti
        from taichi_forge_runtime import aot

        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
        runtime = ti.aot.borrow_runtime()
    n = 17
    with runtime:
        modules = [runtime.load_module(artifact), runtime.load_module(artifact)]
        kernel = modules[0].kernel("update")
        graph = modules[0].graph("step")
        layout = kernel.ndarray_layout(0)
        named_layout = modules[0].ndarray_layout("records")
        assert named_layout._handle == layout._handle
        assert named_layout.members == layout.members
        assert kernel.ndarray_layout(0) is layout
        assert graph.ndarray_layout("records").members == layout.members
        assert layout.member("basis").element_shape == (2, 2)
        assert layout.member("velocity").element_shape == (3,)
        reject(KeyError, lambda: layout.member("missing"))
        reject((AttributeError, TypeError), lambda: setattr(layout, "element_stride", 1))
        reject(ValueError, lambda: kernel.ndarray_layout(-1))
        reject(aot.AotError, lambda: kernel.ndarray_layout(1))
        reject(ValueError, lambda: runtime.ndarray(layout, n, element_shape=(2,)))
        data = records_bytes(layout, n)
        records = runtime.ndarray(layout, n)
        result = runtime.ndarray("f32", n)
        expected = [305 + 1.5 * i for i in range(n)]
        updated = bytearray(data)
        for i in range(n):
            struct.pack_into(
                "=f", updated, i * layout.element_stride + layout.member("inner.value").offset, i * 0.5 + 1
            )
        bound = kernel.bind(records, result)
        graph_bound = graph.bind(dict(records=records, result=result))
        prepared = [bound.prepare(), graph_bound.prepare()]
        for call in (bound, graph_bound, *prepared):
            records.write(data)
            call()
            assert read(result) == expected
            assert records.read() == updated  # Includes all unrelated members/padding.
        frame = dict(
            records=records,
            result=result,
            energy=runtime.ndarray("f32", n),
            dx=runtime.ndarray("f32", n),
            dy=runtime.ndarray("f32", n),
        )
        fixed = modules[0].prepare_execution("records", frame)
        records.write(data)
        fixed()
        assert read(result) == expected
        assert read(frame["energy"]) == [x * x for x in expected]
        assert read(frame["dx"]) == [2 * x for x in expected]
        # Explicit ordinary Graph bindings carry structure ABI and float gradients together.
        records.write(data)
        frame["dx"].write(array.array("f", [0] * n))
        frame["dy"].write(array.array("f", [1] * n))
        differentiate = (
            modules[0]
            .graph("differentiate")
            .bind(
                dict(records=records, result=result, x=result, y=frame["energy"]),
                gradients=dict(x=frame["dx"], y=frame["dy"]),
            )
        )
        differentiate()
        assert read(frame["dx"]) == [2 * x for x in expected]
        reject(ValueError, lambda: kernel.bind(records, result, gradients={0: records}))
        reject(ValueError, lambda: modules[0].prepare_execution("records", frame, gradients={"records": records}))
        wrong_layout = modules[0].kernel("other").ndarray_layout(0)
        wrong = runtime.ndarray(wrong_layout, n)
        rank = runtime.ndarray(layout, (n, 1))
        for bad in (wrong, rank):
            records.write(data)
            result.write(array.array("f", [123] * n))
            reject(aot.AotError, lambda: kernel.prepare(bad, result))
            reject(aot.AotError, lambda: graph.run(dict(records=bad, result=result)))
            assert records.read() == data
            assert read(result) == [123] * n  # mark must not run before validation.
        empty, empty_result = runtime.ndarray(layout, 0), runtime.ndarray("f32", 0)
        with kernel.prepare(empty, empty_result) as empty_call:
            empty_call()
            runtime.wait()
        assert empty.read() == empty_result.read() == b""
        empty.close()
        empty_result.close()
        # Layout provenance is checked before allocation; no foreign handles escape.
        with aot.Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0)) if arch == "vulkan" else 0
        ) as foreign:
            reject(ValueError, lambda: foreign.ndarray(layout, n))
        sibling_kernel = modules[1].kernel("update")
        sibling = runtime.ndarray(sibling_kernel.ndarray_layout(0), n)
        sibling.write(data)
        sibling_call = sibling_kernel.prepare(sibling, result)
        records.write(data)
        ticket = fixed.submit()
        modules[0].close()
        ticket.wait()
        ticket.close()
        reject(RuntimeError, lambda: runtime.ndarray(layout, n))
        reject(RuntimeError, bound)
        reject((RuntimeError, aot.AotError), fixed)
        sibling_call()
        assert read(result) == expected
        # Explicit memory close retires dependent commands, independently of code ownership.
        ticket = sibling_call.submit()
        sibling.close()
        ticket.wait()
        ticket.close()
        reject((RuntimeError, aot.AotError), sibling_call)
        for call in [fixed, sibling_call, *prepared]:
            call.close()
        modules[1].close()
    print("AoS layouts, ordinary/prepared/plan calls, float AD and retirement passed", mode, arch)


if __name__ == "__main__":
    main(*sys.argv[1:])
