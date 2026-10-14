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
    for name, target in cases:
        arrays, prepared, bindings = frame(runtime, module, name, target)
        count = 0 if name == "zero" else min(target, 4)
        active = int(count < target)
        for scale in (1, 3):
            values = [scale * (i - 8) for i in range(17)]
            arrays["x"].write(struct.pack("<17f", *values))
            prepared()
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
    module.close()


if __name__ == "__main__":
    mode, arch, path = sys.argv[1:]
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        with ti.aot.borrow_runtime() as runtime:
            check(runtime, arch, path)
    else:
        with aot.Runtime(arch) as runtime:
            check(runtime, arch, path)
        assert "taichi_forge" not in sys.modules
