"""Consume frozen ordinary/selected cuDSS recordings without importing the DSL."""

import sys
from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cudss_probe import near, raises, read, solve, write


def main(artifact, adapter, vendor, phase):
    with Runtime("cuda") as runtime:
        runtime.configure_provider("cudss", adapter_path=adapter, library_path=vendor)
        runtime.configure_provider("unused", adapter_path="absent-adapter")
        with runtime.load_module(artifact) as module:
            frames = {name: runtime.ndarray("f32", (4,)) for name in
                      ("source", "rhs", "rhs2", "solution", "solution2", "adjoint_rhs", "adjoint", "result")}
            if phase != "solve":
                frames["values"] = runtime.ndarray("f32", (10,))
            for name in ("cold", "warm", "selected", "changed"):
                plan = module.execution_plan(name)
                args = {key: frames[key] for key in plan.argument_names}
                # The original capture contract forbids aliases even between
                # read-only RHS bindings; low-level native groups are broader.
                raises(lambda: plan.prepare(module, dict(args, rhs2=frames["rhs"])))
                assert not any(key in args for key in ("row", "column", "seed_values"))
                with plan.prepare(module, args) as call:
                    assert call.info["execution_kind"] == "cuda_graph"
                    for step in range(2):
                        source = [(i + 1 + step) / 2 for i in range(4)]
                        factor = 1 if phase == "solve" else 2 + step
                        write(frames["source"], source)
                        if phase != "solve":
                            write(frames["values"], [factor * v for v in (4, -1, -1, 4, -1, -1, 4, -1, -1, 3)])
                        call()
                        x = solve([2 * v for v in source], factor)
                        y = solve([-v for v in source], factor)
                        adjoint = solve(x, factor)
                        near(read(frames["solution"]), x)
                        near(read(frames["solution2"]), y)
                        near(read(frames["adjoint"]), adjoint)
                        near(read(frames["result"]), [a + b / 4 for a, b in zip(x, adjoint)])
            for array in frames.values():
                array.close()
    assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
