"""Run by test_aot_runtime_consumer.py in a fresh consumer process."""

import array
import gc
import importlib.abc
import sys
import weakref


class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError(f"AOT consumer imported {fullname}")


def floats(values):
    return array.array("f", values)


def read(value):
    result = floats([])
    result.frombytes(value.read())
    return result.tolist()


def raises(error, function):
    try:
        function()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def main(arch, module_path, order):
    import_guard = NoFrontend()
    sys.meta_path.insert(0, import_guard)
    from taichi_forge_runtime import aot
    from taichi_forge_runtime import _native

    assert _native.loaded_path() is None  # Import alone loads no native core.
    sys.meta_path.remove(import_guard)

    def start_jit():
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        output = ti.ndarray(ti.i32, 4)

        @ti.kernel
        def fill(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
            for i in out:
                out[i] = i + 7

        def check():
            fill(output)
            assert output.to_numpy().tolist() == [7, 8, 9, 10]

        check()
        return ti, check

    if order == "jit-first":
        ti, check_jit = start_jit()
    else:
        sys.meta_path.insert(0, import_guard)

    runtime = aot.Runtime(arch)
    module = runtime.load_module(module_path)
    x, y, dx, dy = [runtime.ndarray("f32", 4) for _ in range(4)]
    x.write(floats([1, 2, 3, 4]))
    forward, backward = module.kernel("forward"), module.kernel("backward")
    graph = module.graph("differentiate")
    for graph_mode in (False, True):
        seed = -0.5 if graph_mode else 1
        dx.write(floats([0] * 4))
        dy.write(floats([seed] * 4))
        if graph_mode:
            call = graph.bind({"x": x, "y": y}, gradients={"x": dx, "y": dy})
            call()
        else:
            forward(x, y, gradients={0: dx, 1: dy})
            backward(x, y, gradients={0: dx, 1: dy})
        assert read(y) == [3, 8, 15, 24]
        assert read(dx) == [seed * value for value in [4, 6, 8, 10]]

    tensor = runtime.ndarray("f32", 3, element_shape=(2,))
    module.kernel("adjust")(aot.Scalar("f32", 0.5), 2, tensor)
    assert read(tensor) == [2, 1, 2.5, 1.5, 3, 2]
    tensor.write(floats([9] * 6).tobytes())
    assert read(tensor) == [9] * 6
    raises(ValueError, lambda: tensor.write(b"short"))
    raises(ValueError, lambda: tensor.read_into(bytes(tensor.nbytes)))
    raises(ValueError, lambda: tensor.write(memoryview(bytearray(48))[::2]))
    raises(ValueError, lambda: forward.bind(x, y, gradients={3: dx}))
    raises(aot.AotError, lambda: module.kernel("adjust")(0.5, 2))
    raises(aot.AotError, lambda: module.kernel("adjust")(aot.Scalar("f64", 0.5), 2, tensor))
    raises(aot.AotError, lambda: forward(x, y))  # Missing explicit adjoints.
    wrong_gradient = runtime.ndarray("f32", 3)
    raises(aot.AotError, lambda: forward(x, y, gradients={0: wrong_gradient, 1: dy}))
    with aot.Runtime(arch) as other:
        foreign = other.ndarray("f32", 4)
        raises(ValueError, lambda: forward.bind(foreign, y))
    # Rebinding and runtime retirement never permit the old native handle.
    saved = module.kernel("adjust").bind(2.0, -1, tensor)
    saved()
    tensor.close()
    raises(RuntimeError, saved)
    with runtime.ndarray("u8", 0)._memory as empty_memory:
        assert empty_memory.size == 0
    empty = runtime.ndarray("f32", 0)
    module.kernel("adjust")(0.5, 2, runtime.ndarray("f32", 0, element_shape=(2,)))
    empty.write(b"")
    assert empty.read() == b""
    if arch == "cpu":
        scalars = runtime.ndarray("f64", 4)
        module.kernel("scalar_widths")(
            aot.Scalar("i8", -12),
            aot.Scalar("u64", 1 << 40),
            aot.Scalar("f16", 1.5),
            aot.Scalar("f64", -2.25),
            scalars,
        )
        values = array.array("d")
        values.frombytes(scalars.read())
        assert values.tolist() == [-12, 1 << 40, 1.5, -2.25]
        scalar_call = module.graph("scalar_graph")
        args = dict(
            a=aot.Scalar("i8", -7),
            b=aot.Scalar("u64", 1 << 41),
            c=aot.Scalar("f16", -0.5),
            d=aot.Scalar("f64", 0.125),
            out=scalars,
        )
        scalar_call(args)
        values = array.array("d")
        values.frombytes(scalars.read())
        assert values.tolist() == [-7, 1 << 41, -0.5, 0.125]
        args["a"] = aot.Scalar("i16", -7)
        raises(aot.AotError, lambda: scalar_call(args))

    if order != "jit-first":
        assert not any(name.split(".")[0] in ("taichi", "taichi_forge", "numpy") for name in sys.modules)
        sys.meta_path.remove(import_guard)
        if order == "aot-first":
            ti, check_jit = start_jit()
            call()
    runtime.close()
    runtime.close()
    raises(RuntimeError, call)
    assert not runtime._resources
    if order != "pure":
        check_jit()
        ti.reset()

    # Pending native work retains both code and arrays, including during GC.
    runtime = aot.Runtime(arch)
    module = runtime.load_module(module_path)
    x, y, dx, dy = [runtime.ndarray("f32", 4) for _ in range(4)]
    x.write(floats([1, 2, 3, 4]))
    dx.write(floats([0] * 4))
    dy.write(floats([1] * 4))
    module.graph("differentiate")({"x": x, "y": y}, gradients={"x": dx, "y": dy})
    reference = weakref.ref(runtime)
    del runtime, module, x, y, dx, dy
    gc.collect()
    assert reference() is None
    print(f"AOT consumer passed: {arch} {order}")


if __name__ == "__main__":
    main(*sys.argv[1:])
