"""Source-free registered recipes, consecutive unique and explicit gather AD."""

import array
import math
import sys

from aot_resources_probe import NoFrontend, raises


def write(value, data):
    value.write(array.array({"i32": "i", "u32": "I", "i64": "q", "u64": "Q", "f32": "f"}[value.dtype], data))


def read(value):
    result = array.array({"i32": "i", "u32": "I", "i64": "q", "u64": "Q", "f32": "f"}[value.dtype])
    result.frombytes(value.read())
    return result.tolist()


def check(arch, path):
    sys.meta_path.insert(0, NoFrontend())
    from taichi_forge_runtime import aot

    with aot.Runtime(arch) as runtime:
        module = runtime.load_module(path)
        # The consumer sees the exact kernel + compact expansion. There is no
        # second unique backend, frontend callback or vendor lookup at replay.
        assert not runtime.native_operation_capabilities("unique")
        for dtype in ("i32", "u32", "i64", "u64"):
            base = {"i32": -(1 << 31), "u32": 1 << 31, "i64": -(1 << 62), "u64": 1 << 63}[dtype]
            data = [base, base, base + 1, base + 2, base + 2, base]
            bindings = {key: runtime.ndarray(dtype, 6) for key in ("keys", "output")}
            bindings["count"] = runtime.ndarray("i32", 1)
            write(bindings["keys"], data)
            with module.execution_plan(dtype).prepare(module, bindings) as prepared:
                assert prepared.info["resource_bytes"] == 24
                prepared()
                assert read(bindings["count"]) == [4]
                assert read(bindings["output"])[:4] == [base, base + 1, base + 2, base]
            for value in bindings.values():
                value.close()

        frames = []
        for n in (0, 17, 4097):
            bindings = {key: runtime.ndarray("i32", n) for key in ("keys", "output", "selected")}
            bindings.update({key: runtime.ndarray("f32", n) for key in ("x", "y", "dx", "dy")})
            bindings["count"] = runtime.ndarray("i32", 1)
            write(bindings["count"], [-99])
            prepared = module.execution_plan("gradient").prepare(module, bindings)
            assert read(bindings["count"]) == [-99]
            assert prepared.info["resource_bytes"] == 2 * max(1, n * 4)
            for step in range(3):
                # Mixed runs, one run, then every item distinct. Reuse the
                # recording with fresh data so stale tails/count cannot pass.
                keys = [(i // 3) % 11 if step == 0 else (0 if step == 1 else n - i) for i in range(n)]
                selected = [i for i in range(n) if i == 0 or keys[i] != keys[i - 1]]
                xs = [(i % 19) * 0.25 - 2 for i in range(n)]
                seeds = [(i % 5 + 1) * 0.125 for i in range(n)]
                for name, values in (("keys", keys), ("x", xs), ("dy", seeds)):
                    write(bindings[name], values)
                for name in ("output", "selected"):
                    write(bindings[name], [-91] * n)
                prepared()
                assert read(bindings["count"]) == [len(selected)]
                assert read(bindings["selected"]) == selected + [-91] * (n - len(selected))
                assert read(bindings["output"]) == [keys[i] for i in selected] + [-91] * (n - len(selected))
                expected_y = [xs[i] ** 2 for i in selected] + [0] * (n - len(selected))
                expected_dx = [0] * n
                for j, i in enumerate(selected):
                    expected_dx[i] = 2 * xs[i] * seeds[j]
                for name, expected in (("y", expected_y), ("dx", expected_dx)):
                    assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(read(bindings[name]), expected))
            frames.append((bindings, prepared))

        n = 17
        bindings = {key: runtime.ndarray("i32", n) for key in ("keys", "output")}
        bindings.update({key: runtime.ndarray("f32", n, element_shape=(3,)) for key in ("values", "unique_values")})
        bindings["count"] = runtime.ndarray("i32", 1)
        write(bindings["keys"], [i // 3 for i in range(n)])
        write(bindings["values"], [float(i) for i in range(3 * n)])
        with module.execution_plan("payload").prepare(module, bindings) as payload:
            payload()
            assert read(bindings["count"]) == [6]
            assert read(bindings["unique_values"])[:18] == [
                float(3 * i + lane) for i in range(0, n, 3) for lane in range(3)
            ]

        plan = module.execution_plan("i32")
        basic = {key: bindings[key] for key in ("keys", "output", "count")}
        for invalid in (
            {"output": bindings["keys"]},
            {"keys": runtime.ndarray("f32", n)},
            {"output": runtime.ndarray("i32", n - 1)},
        ):
            write(bindings["count"], [-71])
            raises(aot.AotError, lambda: plan.prepare(module, {**basic, **invalid}))
            assert read(bindings["count"]) == [-71]
        frames[1][1]()
        frames[1][0]["keys"].close()
        assert frames[1][1].info["resource_bytes"] == 0
        raises(aot.AotError, frames[1][1])
        frames[1][1].close()
        frames[2][1]()
        module.close()
        raises(aot.AotError, frames[2][1])
        assert frames[2][1].info["resource_bytes"] == 0
        frames[2][1].close()
        frames[0][1].close()
    assert not {"taichi", "taichi_forge", "numpy"} & set(sys.modules)


if __name__ == "__main__":
    check(*sys.argv[1:])
