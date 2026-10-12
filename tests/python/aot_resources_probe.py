"""Exercise private resource ownership without importing the DSL or NumPy."""

import array
import gc
import json
import math
import sys
import weakref


class NoFrontend:
    def find_spec(self, fullname, *_):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError(f"AOT consumer imported {fullname}")


def write(target, values):
    target.write(array.array("f" if target.dtype == "f32" else "i", values))


def read(source):
    result = array.array("f" if source.dtype == "f32" else "i")
    result.frombytes(source.read())
    return result.tolist()


def raises(kind, function):
    try:
        function()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def check(arch, path):
    sys.meta_path.insert(0, NoFrontend())
    from taichi_forge_runtime import aot

    with aot.Runtime(arch) as runtime:
        module = runtime.load_module(path)
        plan = module.execution_plan("square_with_resources")
        frames = []
        for n in (0, 17, 257):
            bindings = {name: runtime.ndarray("f32", n) for name in ("x", "y", "dx", "dy")}
            write(bindings["dx"], [-91] * n)
            prepared = plan.prepare(module, bindings)
            assert read(bindings["dx"]) == [-91] * n
            assert prepared.info["resource_bytes"] == 2 * max(1, 4 * n)
            assert prepared.info["workspace_bytes"] == 0
            for step in range(3):
                xs = [(i % 19) - 7 + step * 0.125 for i in range(n)]
                seeds = [(i % 4 + 1) * 0.25 for i in range(n)]
                write(bindings["x"], xs)
                write(bindings["dy"], seeds)
                prepared()
                assert all(math.isclose(a, x * x, rel_tol=1e-6, abs_tol=1e-6) for a, x in zip(read(bindings["y"]), xs))
                assert all(
                    math.isclose(a, 2 * x * seed, rel_tol=1e-6, abs_tol=1e-6)
                    for a, x, seed in zip(read(bindings["dx"]), xs, seeds)
                )
            frames.append((bindings, prepared))

        # Shape-only external bindings are snapshotted, not read by replay.
        source = runtime.ndarray("i32", 17)
        output = runtime.ndarray("i32", 17)
        shape_plan = aot.ExecutionPlan(
            [
                {"kind": "fill32", "destination": "temp", "value": 11},
                {"kind": "copy", "source": "temp", "destination": "out"},
            ],
            resources=[
                {"kind": "ndarray", "name": "temp", "dtype": "i32", "shape": [{"array": "shape_source", "axis": 0}]}
            ],
        )
        with shape_plan.prepare(module, {"shape_source": source, "out": output}) as shaped:
            source_owner = weakref.ref(source._memory)
            del source
            gc.collect()
            assert source_owner() is None
            shaped()
            assert read(output) == [11] * 17

        bindings, prepared = frames[1]
        duplicate = runtime.ndarray("f32", 17)
        raises(aot.AotError, lambda: plan.prepare(module, bindings, gradients={"y": duplicate}))
        duplicate.close()
        document = json.loads(plan.to_json())
        document["operations"].insert(0, {"kind": "fill32", "destination": "dx", "value": 0})
        document["operations"][3]["entry"] = "missing_export"
        invalid = aot.ExecutionPlan.from_json(json.dumps(document))
        write(bindings["dx"], [-93] * 17)
        raises(aot.AotError, lambda: invalid.prepare(module, bindings))
        assert read(bindings["dx"]) == [-93] * 17
        bad_shape = json.loads(plan.to_json())
        bad_shape["resources"][1]["shape"][0]["axis"] = 9
        raises(aot.AotError, lambda: aot.ExecutionPlan.from_json(json.dumps(bad_shape)).prepare(module, bindings))

        # Retire one instance's borrowed input and private storage in flight.
        prepared()
        bindings["x"].close()
        assert prepared.info["resource_bytes"] == 0
        raises(aot.AotError, prepared)
        prepared.close()
        frames[2][1]()
        module.close()
        assert frames[2][1].info["resource_bytes"] == 0
        raises(aot.AotError, frames[2][1])
        frames[2][1].close()
        frames[0][1].close()
    assert not {"taichi", "taichi_forge", "numpy"} & set(sys.modules)


if __name__ == "__main__":
    check(*sys.argv[1:])
