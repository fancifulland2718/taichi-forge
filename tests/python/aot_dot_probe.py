"""Source-free cuBLAS device-scalar gate; optional explicit JIT owner sharing."""

import array
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime


def read(value):
    result = array.array("f")
    result.frombytes(value.read())
    return result.tolist()


def write(value, entries):
    value.write(array.array("f", entries))


def near(actual, expected):
    assert len(actual) == len(expected)
    assert all(abs(a - e) <= 2e-5 * (1 + abs(e)) for a, e in zip(actual, expected)), (actual, expected)


def raises(operation):
    try:
        operation()
    except (AotError, ValueError, RuntimeError):
        return
    raise AssertionError("Invalid dot binding unexpectedly succeeded")


def description(size):
    return dict(
        kind="native",
        provider="cublas",
        operation="dot_f32",
        version=1,
        parameters={"size": size},
        bindings={"x": "x", "y": "y", "output": "dot"},
    )


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        status = runtime.probe_native_operation("dot_f32")
        assert status["available"] and status["capabilities"] == {"cuda_capture"}, status
        module = runtime.load_module(artifact)
        plan = module.execution_plan("dot_energy")

        def frame(scale):
            values = {
                name: runtime.ndarray("f32", 37 if name in ("x", "y", "dx", "dy") else 1)
                for name in ("x", "y", "dot", "loss", "dx", "dy")
            }
            x = [scale * ((i % 11) - 5) / 16 for i in range(37)]
            y = [((i % 7) - 3) / 8 for i in range(37)]
            write(values["x"], x)
            write(values["y"], y)
            write(values["dot"], [-99])
            write(values["loss"], [-77])
            run = plan.prepare(module, values)
            assert read(values["dot"]) == [-99]
            assert read(values["loss"]) == [-77]
            assert run.info["execution_kind"] == "cuda_graph"
            return values, x, y, run

        left, x, y, run = frame(1)
        right, x2, y2, run2 = frame(-0.5)
        for scale in (1, 0, -2):
            current = [scale * value for value in x]
            write(left["x"], current)
            run()
            run2()
            runtime.wait()
            for values, xv, yv in ((left, current, y), (right, x2, y2)):
                dot = sum(a * b for a, b in zip(xv, yv))
                near(read(values["dot"]), [dot])
                near(read(values["loss"]), [0.5 * dot * dot])
                near(read(values["dx"]), [dot * v for v in yv])
                near(read(values["dy"]), [dot * v for v in xv])
        assert run.info["launches"] == 3

        # Scalar output shape (), aliased read-only inputs and no DSL dispatch.
        scalar = runtime.ndarray("f32", ())
        write(scalar, [-55])
        args = dict(x=left["y"], y=left["y"], dot=scalar)
        dot_plan = ExecutionPlan([description(37)])
        square = dot_plan.prepare(module, args)
        assert read(scalar) == [-55]
        square()
        near(read(scalar), [sum(v * v for v in y)])
        square.close()
        # Empty vectors must overwrite a previous hit, including on replay.
        empty = runtime.ndarray("f32", 0)
        zero = ExecutionPlan([description(0)]).prepare(module, dict(x=empty, y=empty, dot=scalar))
        for sentinel in (-17, 42):
            write(scalar, [sentinel])
            zero()
            assert read(scalar) == [0]
        zero.close()

        # One handle switches HOST/DEVICE/HOST modes while recording, without
        # moving device results through the host or sharing across frames.
        context = dict(
            kind="native", name="blas", provider="cublas", operation="context", version=1, parameters={}, bindings={}
        )
        unused = dict(context, name="unused", provider="missing.provider")
        matrices = {name: runtime.ndarray("f32", (1, 1)) for name in ("a", "b", "c")}
        write(matrices["a"], [2])
        write(matrices["b"], [3])
        gemm = dict(
            kind="native",
            provider="cublas",
            operation="gemm_f32",
            version=1,
            parameters=dict(rows=1, columns=1, inner=1),
            bindings=dict(a="a", b="b", output="c", context="blas"),
        )
        dot = description(37)
        dot["bindings"]["context"] = "blas"
        mixed = ExecutionPlan(
            [gemm, dot, dict(gemm, parameters=dict(rows=1, columns=1, inner=1, alpha=2, beta=1))],
            resources=[context, unused],
        ).prepare(module, dict(args, **matrices))
        for _ in range(2):
            mixed()
            near(read(matrices["c"]), [18])
            near(read(scalar), [sum(v * v for v in y)])
        mixed.close()
        # Constructors cannot be executed and operations cannot be resources.
        raises(lambda: ExecutionPlan([{k: v for k, v in context.items() if k != "name"}]).prepare(module, {}))
        bad_resource = dict(gemm, name="blas")
        bad_resource["bindings"] = {key: value for key, value in gemm["bindings"].items() if key != "context"}
        raises(lambda: ExecutionPlan([dot], resources=[bad_resource]).prepare(module, dict(args, **matrices)))
        before_matrix = read(matrices["c"])
        raises(
            lambda: ExecutionPlan(
                [gemm, dot, {"kind": "kernel", "entry": "missing", "arguments": []}], resources=[context]
            ).prepare(module, dict(args, **matrices))
        )
        assert read(matrices["c"]) == before_matrix
        raises(
            lambda: ExecutionPlan([{"kind": "fill32", "destination": "blas", "value": 0}], resources=[context]).prepare(
                module, {}
            )
        )
        raises(
            lambda: ExecutionPlan(
                [{"kind": "kernel", "entry": "energy", "arguments": ["blas", "loss"]}], resources=[context]
            ).prepare(module, {"loss": left["loss"]})
        )
        wrong_type = runtime.ndarray("i32", 37)
        wrong_shape = runtime.ndarray("f32", (1, 37))
        for invalid in (dict(args, x=wrong_type), dict(args, x=wrong_shape), dict(args, dot=left["x"])):
            raises(lambda: dot_plan.prepare(module, invalid))
        raises(lambda: ExecutionPlan([description(38)]).prepare(module, args))
        for size in (-1, True, 1.5, 1 << 62):
            raises(lambda: ExecutionPlan([description(size)]).prepare(module, args))
        invalid = description(37)
        invalid["parameters"]["unknown"] = 1
        raises(lambda: ExecutionPlan([invalid]).prepare(module, args))
        # Exact output/input physical alias must fail even though sizes match.
        one = runtime.ndarray("f32", 1)
        raises(lambda: ExecutionPlan([description(1)]).prepare(module, dict(x=one, y=one, dot=one)))
        before = read(scalar)
        raises(
            lambda: ExecutionPlan(
                [
                    {"kind": "fill32", "destination": "dot", "value": 0},
                    description(37),
                    {"kind": "kernel", "entry": "missing", "arguments": []},
                ]
            ).prepare(module, args)
        )
        assert read(scalar) == before

        run()
        left["x"].close()
        raises(run)
        run.close()
        run2()
        run2.close()
        again = plan.prepare(module, right)
        again()
        if mode == "borrowed":
            ti.reset()
            raises(again)
        else:
            module.close()
            raises(again)
            assert "taichi_forge" not in sys.modules
    print("AOT_DOT_PASSED", mode, status)


if __name__ == "__main__":
    main(*sys.argv[1:])
