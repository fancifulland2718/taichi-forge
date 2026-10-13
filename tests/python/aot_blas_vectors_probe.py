"""Source-free cuBLAS vector mixed execution, validation and retirement."""

import math
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_dot_probe import near, raises, read, write


def description(operation, bindings, **parameters):
    return dict(
        kind="native", provider="cublas", operation=operation, version=1, parameters=parameters, bindings=bindings
    )


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        module = runtime.load_module(artifact)
        frames = []
        for _ in range(2):
            frame = {
                name: runtime.ndarray("f32", (37 if name in ("x", "y0", "y", "dx", "dy") else 1,))
                for name in ("x", "y0", "y", "dx", "dy", "norm", "loss", "beta", "da", "db")
            }
            frame["alpha"] = runtime.ndarray("f32", ())
            write(frame["y"], [-99] * 37)
            prepared = module.prepare_execution("vectors", frame)
            assert read(frame["y"]) == [-99] * 37
            assert prepared.info["execution_kind"] == "cuda_graph"
            assert prepared.info["resource_bytes"] == 8
            frames.append((frame, prepared))
        for iteration, (alpha, beta) in enumerate(((2, -0.75), (-0.5, 1.25), (0, 0), (0.25, -2))):
            for index, (frame, prepared) in enumerate(frames):
                x = [(i % 7 - 3 + index) / 8 for i in range(37)]
                y0 = [(i % 11 - 5 + iteration) / 16 for i in range(37)]
                for name, values in (("x", x), ("y0", y0), ("alpha", [alpha]), ("beta", [beta])):
                    write(frame[name], values)
                prepared()
                y = [-0.5 * beta * (alpha * a + 0.5 * b) for a, b in zip(x, y0)]
                norm = math.sqrt(sum(v * v for v in y))
                near(read(frame["y"]), y)
                near(read(frame["norm"]), [norm])
                near(read(frame["loss"]), [0.5 * norm * norm])
                near(read(frame["dx"]), [-0.5 * beta * alpha * v for v in y])
                near(read(frame["dy"]), [-0.25 * beta * v for v in y])
                near(read(frame["da"]), [sum(-0.5 * beta * a * v for a, v in zip(x, y))])
                near(read(frame["db"]), [sum(-0.5 * (alpha * a + 0.5 * b) * v for a, b, v in zip(x, y0, y))])

        def prepare(operation, arguments):
            return ExecutionPlan([operation]).prepare(module, arguments)

        empty = runtime.ndarray("f32", (0,))
        scalar = runtime.ndarray("f32", (1,))
        alpha = runtime.ndarray("f32", ())
        write(alpha, [-2])
        for operation, arguments in (
            (description("scal_f32", dict(x="x", alpha="a"), size=0), dict(x=empty, a=alpha)),
            (description("axpy_f32", dict(x="x", y="y", alpha="a"), size=0), dict(x=empty, y=empty, a=alpha)),
            (description("nrm2_f32", dict(x="x", output="out"), size=0), dict(x=empty, out=scalar)),
        ):
            with prepare(operation, arguments) as call:
                for _ in range(2):
                    write(scalar, [99])
                    call()
                    if operation["operation"] == "nrm2_f32":
                        near(read(scalar), [0])

        x, y = runtime.ndarray("f32", (3,)), runtime.ndarray("f32", (3,))
        write(x, [1, 2, 3])
        write(y, [9, 8, 7])
        axpy = description("axpy_f32", dict(x="x", y="y", alpha="a"), size=3)
        scal = description("scal_f32", dict(x="x", alpha="a"), size=3)
        for operation, arguments in (
            (axpy, dict(x=x, y=x, a=alpha)),
            (axpy, dict(x=x, y=y, a=x)),
            (dict(axpy, parameters=dict(size=3, alpha=2)), dict(x=x, y=y, a=alpha)),
            (dict(axpy, parameters=dict(size=3, stride=1)), dict(x=x, y=y, a=alpha)),
            (dict(axpy, parameters=dict(size=2)), dict(x=x, y=y, a=alpha)),
            (axpy, dict(x=runtime.ndarray("i32", (3,)), y=y, a=alpha)),
            (description("scal_f32", dict(x="x"), size=3, alpha=True), dict(x=x)),
            (description("scal_f32", dict(x="x", alpha="a"), size=1), dict(x=scalar, a=scalar)),
            (description("nrm2_f32", dict(x="x", output="out"), size=1), dict(x=scalar, out=scalar)),
        ):
            raises(lambda: prepare(operation, arguments))
        near(read(y), [9, 8, 7])
        # Read-only alpha may reference x; mutable destinations must not overlap.
        write(scalar, [2])
        single_y = runtime.ndarray("f32", (1,))
        write(single_y, [3])
        with prepare(
            description("axpy_f32", dict(x="x", y="y", alpha="a"), size=1), dict(x=scalar, a=scalar, y=single_y)
        ) as call:
            call()
            near(read(single_y), [7])
        with prepare(scal, dict(x=x, a=single_y)) as call:
            call()
            near(read(x), [7, 14, 21])
        parent = runtime.ndarray("f32", (8,))
        write(parent, [99, 1, 2, 3, 4, 5, 6, 77])
        views = [
            dict(kind="ndarray_view", name=name, array="parent", offset=offset, shape=[3])
            for name, offset in (("left", 1), ("right", 4))
        ]
        sliced = description("axpy_f32", dict(x="left", y="right"), size=3, alpha=-2)
        with ExecutionPlan([sliced], resources=views).prepare(module, dict(parent=parent)) as call:
            call()
            near(read(parent), [99, 1, 2, 3, 2, 1, 0, 77])
        overlap = [views[0], dict(views[1], offset=2)]
        raises(lambda: ExecutionPlan([sliced], resources=overlap).prepare(module, dict(parent=parent)))
        near(read(parent), [99, 1, 2, 3, 2, 1, 0, 77])
        frames[0][1].close()
        frames[1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(frames[1][1])
        frames[1][1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
