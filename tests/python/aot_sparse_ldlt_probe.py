"""Source-free consumer of selected symmetric-indefinite factors and their adjoint."""

import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cudss_probe import near, raises, read, write


def reference(rhs):
    # Analytic inverse of [[2, 3, 0], [3, 2, 0], [0, 0, -2]].
    return [-0.4 * rhs[0] + 0.6 * rhs[1], 0.6 * rhs[0] - 0.4 * rhs[1], -0.5 * rhs[2]]


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
        plan = module.execution_plan("ldlt")
        frames = []
        for scale in (1, -0.5):
            frame = {
                name: runtime.ndarray("f32", 1 if name in ("loss", "dloss") else 3)
                for name in plan.argument_names
            }
            write(frame["x"], [-99] * 3)
            call = plan.prepare(module, frame)
            assert call.info["execution_kind"] == "cuda_graph"
            near(read(frame["x"]), [-99] * 3)
            for entries in ([1, 2, 3], [-1, 3, 0.5]):
                rhs = [scale * value for value in entries]
                write(frame["rhs"], rhs)
                call()
                x = read(frame["x"])
                near(x, reference(rhs))
                near([2 * x[0] + 3 * x[1], 3 * x[0] + 2 * x[1], -2 * x[2]], rhs)
                near(read(frame["dx"]), x)
                near(read(frame["db"]), reference(x))
                near(read(frame["loss"]), [0.5 * sum(v * v for v in x)])
            frames.append((frame, call))
        frames[0][1]()
        frames[0][1].close()
        frames[1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(frames[1][1])
        frames[1][1].close()
        for frame, _ in frames:
            for value in frame.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
