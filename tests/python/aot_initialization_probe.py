"""No frontend needed to initialize a snapshot and replay its explicit AD."""

import struct
import sys
import weakref
import gc

from taichi_forge_runtime import aot


def write(value, data):
    value.write(struct.pack("<17f", *data))


def read(value):
    return list(struct.unpack("<17f", value.read()))


def retired(function):
    try:
        function()
    except aot.AotError as error:
        assert "retired dependency" in str(error), str(error)
    else:
        raise AssertionError("Retired dependency was accepted")


def frame(runtime, module, scale):
    bindings = {name: runtime.ndarray("f32", 17) for name in ("x", "y", "gradient")}
    write(bindings["x"], [scale * (i - 8) for i in range(17)])
    write(bindings["y"], [-91] * 17)
    prepared = module.prepare_execution("snapshot", bindings)
    assert read(bindings["y"]) == [-91] * 17
    assert prepared.info["resource_bytes"] == 4 * 17 * 4
    assert prepared.info["launches"] == 0
    return bindings, prepared


def result(bindings, scale):
    assert read(bindings["y"]) == [4 * (scale * (i - 8)) ** 2 for i in range(17)]
    assert read(bindings["gradient"]) == [4 * scale * (i - 8) for i in range(17)]


def check(runtime, path):
    module = runtime.load_module(path)
    first, one = frame(runtime, module, 1)
    second, two = frame(runtime, module, 3)
    waiting = module.prepare_execution("snapshot", first)
    one.initialize()
    assert one.info["launches"] == 0
    assert one.info["resource_bytes"] == 3 * 17 * 4
    assert read(first["y"]) == [-91] * 17
    write(first["x"], [123] * 17)
    one.initialize()
    first["x"].close()
    retired(waiting.initialize)
    retired(waiting.run)
    waiting.close()
    waiting.close()
    for _ in range(3):
        one()
        result(first, 1)
    # C API first launch initializes too; Python's later explicit call is
    # idempotent and releases its initialization-only owner references.
    runtime._api.call(runtime._api.ti_launch_prepared_ext, runtime._handle, two._handle)
    result(second, 3)
    write(second["x"], [99] * 17)
    two()
    result(second, 3)
    source = weakref.ref(second.pop("x")._memory)
    runtime.wait()
    gc.collect()
    assert source() is None, "Initialization-only Python allocation is retained"
    assert two.info["resource_bytes"] == 3 * 17 * 4
    two()
    result(second, 3)
    # Different slot names aliasing one allocation must retain its replay edge.
    alias = {"x": second["y"], "y": second["y"], "gradient": second["gradient"]}
    aliased = module.prepare_execution("snapshot", alias)
    aliased.initialize()
    second["y"].close()
    retired(aliased.run)
    retired(two.run)
    aliased.close()
    one()
    result(first, 1)
    # Closing a module retires initialized and uninitialized recordings alike.
    pending_values, pending = frame(runtime, module, 2)
    closing_values, closing = frame(runtime, module, 4)
    closing_source = weakref.ref(closing_values.pop("x")._memory)
    assert closing_source() is not None
    closing.close()
    gc.collect()
    assert closing_source() is None, "Close retained an initialization input"
    for value in closing_values.values():
        value.close()
    module.close()
    retired(one.run)
    retired(pending.initialize)
    for prepared in (one, two, pending):
        prepared.close()
    for bindings in (first, second, pending_values):
        for value in bindings.values():
            value.close()


if __name__ == "__main__":
    mode, arch, path = sys.argv[1:]
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        with ti.aot.borrow_runtime() as runtime:
            check(runtime, path)
            module = runtime.load_module(path)
            bindings, prepared = frame(runtime, module, 1)
            prepared.initialize()
            prepared()
            ti.reset()
            try:
                prepared()
            except (RuntimeError, aot.AotError):
                pass
            else:
                raise AssertionError("Borrowed reset retained a live recording")
            prepared.close()
    else:
        with aot.Runtime(arch) as runtime:
            check(runtime, path)
        assert "taichi_forge" not in sys.modules
