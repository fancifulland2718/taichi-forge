"""Source-free replay of selected Graph while code and private controls."""

import struct
import sys

from taichi_forge_runtime import aot


def frame(runtime, module, name, target):
    arrays = {n: runtime.ndarray("f32", 17) for n in ("x", "y", "dx")}
    arrays["terminal"] = runtime.ndarray("i32", 4)
    if name == "step":
        arrays["state"] = runtime.ndarray("f32", 17)
        arrays.update({n: runtime.ndarray("i32", ()) for n in ("counter", "predicate", "checks", "status")})
    bindings = dict(arrays, target=aot.Scalar("i32", target))
    arrays["terminal"].write(struct.pack("<4i", -1, -1, -1, -1))
    prepared = module.prepare_execution(name, bindings)
    assert struct.unpack("<4i", arrays["terminal"].read()) == (-1,) * 4
    return arrays, prepared, bindings


def check(runtime, arch, path):
    module = runtime.load_module(path)
    cases = [("step", 2), ("private", 0), ("private", 10), ("zero", 2)]
    if arch == "cpu":
        cases.append(("nested", 2))
    if arch == "vulkan":
        for name in ("chained", "conditional", "coarse_conditional"):
            try:
                module.execution_plan(name)
            except (KeyError, FileNotFoundError):
                continue
            cases.append((name, 3))
    for name, target in cases:
        arrays, prepared, bindings = frame(runtime, module, name, target)
        count = 0 if name == "zero" else min(target, 4)
        active = int(count < target)
        for scale in (1, 3):
            values = [scale * (i - 8) for i in range(17)]
            arrays["x"].write(struct.pack("<17f", *values))
            program = None
            # First use may also perform the plan's separate one-time private
            # resource initialization. Count the warmed complete step itself.
            if arch == "vulkan" and name != "zero" and scale == 3 and "taichi_forge" in sys.modules:
                program = sys.modules["taichi_forge"].lang.impl.get_runtime().prog
                before = program._debug_vulkan_queue_submission_stats()["queue_submit_calls"]
            prepared()
            if program is not None:
                after = program._debug_vulkan_queue_submission_stats()["queue_submit_calls"]
                assert after - before == 1, (name, before, after)
            assert struct.unpack("<4i", arrays["terminal"].read()) == (
                count,
                active,
                4 if name == "nested" else count + 1,
                2 - active,
            )
            assert list(struct.unpack("<17f", arrays["y"].read())) == [(x * (1 << count)) ** 2 for x in values]
            assert list(struct.unpack("<17f", arrays["dx"].read())) == [2 * x * (1 << (2 * count)) for x in values]
        if arch == "cuda":
            assert prepared.info["execution_kind"] == "cuda_graph"
        if arch == "vulkan" and name != "zero":
            assert prepared.info["execution_kind"] == "vulkan_commands"
            assert prepared.info["workspace_bytes"] > 0
        if name == "step":
            try:
                module.prepare_execution(name, dict(bindings, predicate=arrays["counter"]))
            except aot.AotError as error:
                assert "disjoint" in str(error).lower(), str(error)
            else:
                raise AssertionError("Graph control alias contract was lost")
            invalid = runtime.ndarray("i32", 2)
            try:
                module.prepare_execution(name, dict(bindings, status=invalid))
            except aot.AotError:
                pass
            else:
                raise AssertionError("Graph control scalar contract was lost")
            invalid.close()
        prepared.close()
        for value in arrays.values():
            value.close()
    # Two independent bindings may remain in flight together; the second
    # preparation cannot overwrite the first one's frozen scalar/descriptor.
    first, one, _ = frame(runtime, module, "private", 1)
    second, two, _ = frame(runtime, module, "private", 3)
    for arrays in (first, second):
        arrays["x"].write(struct.pack("<17f", *([1] * 17)))
    one()
    two()
    one()
    assert struct.unpack("<17f", first["dx"].read()) == (8,) * 17
    assert struct.unpack("<17f", second["dx"].read()) == (128,) * 17
    one()
    first["x"].close()
    try:
        one()
    except aot.AotError as error:
        assert "retired dependency" in str(error), str(error)
    else:
        raise AssertionError("Closed input did not retire control")
    one.close()
    two()
    module.close()
    two.close()
    for arrays in (first, second):
        for value in arrays.values():
            value.close()


if __name__ == "__main__":
    mode, arch, path = sys.argv[1:]
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        with ti.aot.borrow_runtime() as runtime:
            check(runtime, arch, path)
            module = runtime.load_module(path)
            arrays, prepared, _ = frame(runtime, module, "private", 2)
            prepared()
            ti.reset()
            try:
                prepared()
            except (RuntimeError, aot.AotError):
                pass
            else:
                raise AssertionError("Owner reset did not retire control")
            prepared.close()
    else:
        with aot.Runtime(arch) as runtime:
            check(runtime, arch, path)
        assert "taichi_forge" not in sys.modules
