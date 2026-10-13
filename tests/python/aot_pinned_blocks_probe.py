"""Consume nested block and private-workspace views without the producer DSL."""

import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cutensor_probe import near, raises
from tests.python.aot_graph_recording_probe import read, write


def multiply(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def main(mode, arch, path):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    a = ((2, -1, 0.5), (3, 4, -2))
    matrix = [[0] * 7 for _ in range(5)]
    for i in range(2):
        matrix[i][:3] = a[i]
        matrix[i + 2][3:6] = a[i]
    matrix[4][6] = 2
    transpose = list(zip(*matrix))
    gram = [[0.5 * sum(matrix[k][i] * matrix[k][j] for k in range(5)) for j in range(7)] for i in range(7)]
    with runtime:
        module = runtime.load_module(path)
        frames = [{name: runtime.ndarray("f32", (n,)) for name, n in (("x", 7), ("y", 5), ("dx", 7))} for _ in range(2)]
        calls = [module.prepare_execution("private", {name: frame[name] for name in ("x", "dx")}) for frame in frames]
        assert all(call.info["resource_bytes"] == 108 for call in calls)
        for iteration in range(3):
            for index, (frame, call) in enumerate(zip(frames, calls)):
                values = [i * 0.25 - 0.5 + iteration + index for i in range(7)]
                write(frame["x"], values)
                call()
                near(read(frame["dx"]), multiply(transpose, multiply(matrix, values)))
        frame = frames[0]
        values = [i * 0.25 - 0.5 for i in range(7)]
        write(frame["x"], values)
        for name in ("old", "repeated"):
            with module.prepare_execution(name, frame) as call:
                assert call.info["resource_bytes"] == 88
                call()
                near(read(frame["y"]), multiply(matrix, values))
                near(read(frame["dx"]), multiply(transpose, multiply(matrix, values)))
        updated = [row[:] for row in matrix]
        for i in range(2):
            updated[i] = [2 * v for v in updated[i]]
        with module.prepare_execution("current", frame) as call:
            call()
            near(read(frame["dx"]), multiply(list(zip(*updated)), multiply(updated, values)))
        square = dict(frame, y=runtime.ndarray("f32", (7,)))
        with module.prepare_execution("normal", square) as call:
            assert call.info["resource_bytes"] == 128
            call()
            near(read(square["y"]), multiply(gram, values))
            near(read(square["dx"]), multiply(gram, multiply(gram, values)))
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
