"""Consume frozen ordinary and selected triangular recordings without the DSL."""

import sys
from taichi_forge_runtime.aot import Runtime
from tests.python.aot_dot_probe import near, read, write
from tests.python.aot_triangular_probe import solve


def main(artifact, rhs):
    rhs = int(rhs)
    with Runtime("cuda") as runtime:
        with runtime.load_module(artifact) as module:
            shape = (3,) if rhs == 1 else (3, rhs)
            frames = {name: runtime.ndarray("f32", shape) for name in ("x", "y", "dx", "scratch", "result")}
            for name in ("cold", "warm", "selected_0", *(("selected_1",) if rhs == 1 else ()), "changed"):
                args = {key: frames[key] for key in module.execution_plan(name).argument_names}
                with module.prepare_execution(name, args) as call:
                    assert call.info["execution_kind"] == "cuda_graph"
                    factor = 2 if name == "changed" else 1
                    values = [v * factor for v in (2, 0, 0, 1, 4, 0, 0, -1, 2)]
                    for step in range(2):
                        x = [(i - 2 + step) / 4 for i in range(3 * rhs)]
                        write(frames["x"], x)
                        call()
                        y = solve(values, x, rhs, "lower", False, False)
                        dx = solve(values, y, rhs, "lower", False, True)
                        near(read(frames["y"]), y)
                        near(read(frames["dx"]), dx)
                        near(read(frames["result"]), [2 * v + 0.25 for v in dx])
            for value in frames.values():
                value.close()
    assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
