"""Consume forward/transpose operator code and fixed data without the DSL."""

import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_binding_constraints_probe import failure, imported
from tests.python.aot_cutensor_probe import near, raises
from tests.python.aot_graph_recording_probe import read, write


def check(arrays, x):
    matrix = ((2, -1, 0.5), (3, 4, -2))
    y = [sum(a * b for a, b in zip(row, x)) for row in matrix]
    dx = [sum(matrix[row][column] * y[row] for row in range(2)) for column in range(3)]
    near(read(arrays["y"]), y)
    near(read(arrays["dx"]), dx)


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(artifact)
        assert set(module.execution_plan("step").argument_names) == {"x", "y", "dx"}
        frames = [{name: runtime.ndarray("f32", (n,)) for name, n in (("x", 3), ("y", 2), ("dx", 3))} for _ in range(2)]
        calls = [module.prepare_execution("step", arrays) for arrays in frames]
        assert all(call.info["resource_bytes"] == 48 for call in calls)
        for step in range(3):
            for index, (arrays, call) in enumerate(zip(frames, calls)):
                x = [0.25 + step, -0.5, 2 + index]
                write(arrays["x"], x)
                call()
                check(arrays, x)
        arrays = frames[0]
        write(arrays["y"], [99, 99])
        for extent in (2, 4):
            wrong = runtime.ndarray("f32", (extent,))
            failure(lambda: module.prepare_execution("step", dict(arrays, x=wrong)), "exact shape")
            wrong.close()
        # A 2-element alias is valid as an output but overlaps the 3-element x.
        alias = imported(runtime, arrays["x"], arch, size=8)
        failure(lambda: module.prepare_execution("step", dict(arrays, y=alias)), "disjoint")
        alias.close()
        assert read(arrays["y"]) == [99, 99]
        with module.prepare_execution("again", arrays) as replacement:
            replacement()
            check(arrays, [2.25, -0.5, 2])
        with module.prepare_execution("private", {name: arrays[name] for name in ("x", "dx")}) as private:
            assert private.info["resource_bytes"] == 56
            private()
            check(arrays, [2.25, -0.5, 2])
        calls[0].close()
        calls[1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(calls[1])
        calls[1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
