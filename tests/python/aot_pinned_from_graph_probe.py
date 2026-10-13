"""Read graph-provider state without producer source or a frontend installation."""

import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cutensor_probe import near, raises
from tests.python.aot_graph_recording_probe import read, write


def check(arrays, value, *, legacy=False, forward_scale=1, adjoint_scale=1, numeric=1):
    matrix = ((2, -1), (3, 4)) if legacy else ((2, -1, 0.5), (3, 4, -2))
    y = [forward_scale * numeric * sum(a * b for a, b in zip(row, value)) for row in matrix]
    near(read(arrays["y"]), y)
    if not legacy:
        dx = [adjoint_scale * numeric * sum(matrix[row][col] * y[row] for row in range(2)) for col in range(3)]
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
        frames = [{name: runtime.ndarray("f32", (n,)) for name, n in (("x", 3), ("y", 2), ("dx", 3))} for _ in range(2)]
        calls = [module.prepare_execution("stateful", frame) for frame in frames]
        assert all(call.info["resource_bytes"] == 40 for call in calls)
        # Interleaved preparations mutate their own shared forward/adjoint state.
        for iteration in range(3):
            for index, (frame, call) in enumerate(zip(frames, calls)):
                value = [0.25 + index, -0.5, 2 + iteration]
                write(frame["x"], value)
                call()
                check(frame, value, forward_scale=26 + 12 * iteration, adjoint_scale=32 + 12 * iteration)
        arrays = frames[0]
        value = [1, -1, 0.5]
        write(arrays["x"], value)
        for entry, scale, numeric in (("stateful", 26, 1), ("current", 26, 2), ("plain", 1, 1)):
            with module.prepare_execution(entry, arrays) as replacement:
                replacement()
                check(
                    arrays,
                    value,
                    forward_scale=scale,
                    adjoint_scale=scale + 6 if entry != "plain" else 1,
                    numeric=numeric,
                )
        legacy = {name: runtime.ndarray("f32", (2,)) for name in ("x", "y")}
        write(legacy["x"], [1, -2])
        with module.prepare_execution("legacy", legacy) as call:
            assert call.info["resource_bytes"] == 32
            for iteration in range(2):
                call()
                check(legacy, [1, -2], legacy=True, forward_scale=14 + 6 * iteration)
        square = dict(arrays, y=runtime.ndarray("f32", (3,)))
        gram = ((13, 10, -5), (10, 17, -8.5), (-5, -8.5, 4.25))
        y = [sum(a * b for a, b in zip(row, value)) for row in gram]
        dx = [sum(a * b for a, b in zip(row, y)) for row in gram]
        with module.prepare_execution("composition", square) as call:
            call()
            near(read(square["y"]), y)
            near(read(square["dx"]), dx)
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
