"""Consume ordinary solve snapshots, including real kernel gradients, without source."""

import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cudss_probe import near, raises, read, solve, write


def main(mode, artifact, adapter, vendor):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("cudss", adapter_path=adapter, library_path=vendor)
        module = runtime.load_module(artifact)
        retained = []
        for name, scale in (("original", 1), ("updated", 2)):
            plan = module.execution_plan(name)
            assert set(plan.argument_names) == {"rhs", "x", "dx", "db", "loss", "dloss"}
            for factor in (1, -0.5):
                frame = {
                    key: runtime.ndarray("f32", 1 if key in ("loss", "dloss") else 4) for key in plan.argument_names
                }
                write(frame["x"], [-99] * 4)
                call = plan.prepare(module, frame)
                assert call.info["execution_kind"] == "cuda_graph"
                near(read(frame["x"]), [-99] * 4)  # Prepare must not execute the user chain.
                for step in range(2):
                    rhs = [factor * (i + 1 + step) for i in range(4)]
                    write(frame["rhs"], rhs)
                    call()
                    x = solve(rhs, scale)
                    near(read(frame["x"]), x)
                    near(read(frame["dx"]), x)
                    near(read(frame["db"]), solve(x, scale))
                    near(read(frame["loss"]), [0.5 * sum(value * value for value in x)])
                retained.append((frame, call))
        # Retiring one binding cannot invalidate other reconstructed owners.
        retained[0][1]()
        retained[0][1].close()
        retained[-1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for frame, call in retained[1:]:
            raises(call)
            call.close()
        for frame, _call in retained:
            for value in frame.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
