"""Standard-library consumer of nested operator/parameter generations."""

import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_binding_constraints_probe import failure, imported
from tests.python.aot_cutensor_probe import near, raises
from tests.python.aot_graph_recording_probe import read, write


def transpose(matrix):
    return list(map(list, zip(*matrix)))


def apply(matrix, x):
    return [sum(a * b for a, b in zip(row, x)) for row in matrix]


def matrix(epoch):
    a = [[2, -1, 0.5], [3, 4, -2]]
    b = [[1, 0.25, -2], [0.5, -1, 3]]
    alpha, beta = (0.5, -0.25) if epoch != "current" else (2.5, 0.5)
    affine = [[alpha * a[i][j] + beta * b[i][j] for j in range(3)] for i in range(2)]
    if epoch == "rectangle":
        return affine
    return [
        [0.5 * sum(affine[k][i] * affine[k][j] for k in range(2)) + ((i + 1.25) if i == j else 0) for j in range(3)]
        for i in range(3)
    ]


def check(arrays, coefficients, x):
    y = apply(coefficients, x)
    near(read(arrays["y"]), y)
    near(read(arrays["dx"]), apply(transpose(coefficients), y))


def main(mode, arch, artifact, scratch_bytes):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(artifact)
        frames = [{name: runtime.ndarray("f32", (3,)) for name in ("x", "y", "dx")} for _ in range(2)]
        calls = [module.prepare_execution(epoch, arrays) for epoch, arrays in zip(("old", "current"), frames)]
        assert all(call.info["resource_bytes"] == 120 + int(scratch_bytes) for call in calls)
        for step in range(3):
            for epoch, arrays, call in zip(("old", "current"), frames, calls):
                x = [0.25 + step, -0.5, 2]
                write(arrays["x"], x)
                call()
                check(arrays, matrix(epoch), x)
        arrays = frames[0]
        failure(lambda: module.prepare_execution("old", dict(arrays, y=arrays["x"])), "disjoint")
        alias = imported(runtime, arrays["x"], arch)
        failure(lambda: module.prepare_execution("old", dict(arrays, y=alias)), "disjoint")
        alias.close()
        with module.prepare_execution("old", arrays) as again:
            again()
            check(arrays, matrix("old"), [2.25, -0.5, 2])
        rectangle = dict(arrays, y=runtime.ndarray("f32", (2,)))
        with module.prepare_execution("rectangle", rectangle) as rectangular:
            rectangular()
            check(rectangle, matrix("rectangle"), [2.25, -0.5, 2])
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
