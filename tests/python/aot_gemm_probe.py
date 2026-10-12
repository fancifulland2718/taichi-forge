"""No-DSL GEMM consumer gate, with an optional explicit JIT-owner borrower."""

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
    assert all(abs(a - e) <= 3e-5 * (1 + abs(e)) for a, e in zip(actual, expected)), (actual[:9], expected[:9])


def raises(operation):
    try:
        operation()
    except (AotError, ValueError, RuntimeError):
        return
    raise AssertionError("Invalid GEMM binding unexpectedly succeeded")


def multiply(a, b, m, n, k, ta=False, tb=False):
    return [
        sum(a[p * m + i if ta else i * k + p] * b[j * k + p if tb else p * n + j] for p in range(k))
        for i in range(m)
        for j in range(n)
    ]


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        status = runtime.probe_native_operation("gemm_f32")
        assert status["available"] and status["capabilities"] == {"cuda_capture"}, status
        assert not runtime.probe_native_operation("unrelated.optional.provider")["available"]
        module = runtime.load_module(artifact)
        plan = module.execution_plan("matrix_energy")

        def frame(scale):
            arrays = {
                name: runtime.ndarray("f32", shape)
                for name, shape in (
                    ("a", (17, 13)),
                    ("b", (13, 9)),
                    ("seed", (17, 9)),
                    ("c", (17, 9)),
                    ("da", (17, 13)),
                    ("db", (13, 9)),
                    ("loss", 1),
                )
            }
            a = [scale * ((i % 17) - 8) / 16 for i in range(17 * 13)]
            b = [((i % 13) - 6) / 32 for i in range(13 * 9)]
            seed = [((i % 7) - 3) / 64 for i in range(17 * 9)]
            for name, values in (("a", a), ("b", b), ("seed", seed), ("c", [-99] * 153), ("loss", [-77])):
                write(arrays[name], values)
            c = [1.25 * p + 0.25 * s for p, s in zip(multiply(a, b, 17, 9, 13), seed)]
            da = [1.25 * p for p in multiply(c, b, 17, 13, 9, tb=True)]
            db = [1.25 * p for p in multiply(a, c, 13, 9, 17, ta=True)]
            prepared = plan.prepare(module, arrays)
            assert read(arrays["c"]) == [-99] * 153
            assert read(arrays["loss"]) == [-77]
            assert prepared.info["execution_kind"] == "cuda_graph"
            return arrays, (c, da, db, [0.5 * sum(x * x for x in c)]), prepared

        left, expected, run = frame(1.0)
        right, other_expected, other_run = frame(-0.5)
        for _ in range(3):
            run()
            other_run()
        runtime.wait()
        for arrays, values in ((left, expected), (right, other_expected)):
            for name, reference in zip(("c", "da", "db", "loss"), values):
                near(read(arrays[name]), reference)
        assert run.info["launches"] == 3
        run()
        left["a"].close()
        raises(run)
        run.close()
        other_run()
        near(read(right["da"]), other_expected[1])

        # All physical transpose combinations and repeated beta accumulation.
        for ta, tb in ((False, False), (True, False), (False, True), (True, True)):
            m, n, k = 3, 5, 7
            a = runtime.ndarray("f32", (k, m) if ta else (m, k))
            b = runtime.ndarray("f32", (n, k) if tb else (k, n))
            c = runtime.ndarray("f32", (m, n))
            av = [(i - 10) / 8 for i in range(m * k)]
            bv = [((i % 11) - 5) / 16 for i in range(k * n)]
            cv = [0.25] * (m * n)
            for value, values in ((a, av), (b, bv), (c, cv)):
                write(value, values)
            op = dict(
                kind="gemm_f32",
                a="a",
                b="b",
                output="c",
                rows=m,
                columns=n,
                inner=k,
                transpose_a=ta,
                transpose_b=tb,
                alpha=-0.75,
                beta=0.5,
            )
            args = dict(a=a, b=b, c=c)
            call = ExecutionPlan([op]).prepare(module, args)
            assert read(c) == cv
            for _ in range(2):
                call()
                cv = [-0.75 * x + 0.5 * y for x, y in zip(multiply(av, bv, m, n, k, ta, tb), cv)]
                near(read(c), cv)
            call.close()
            raises(lambda: ExecutionPlan([dict(op, inner=k + 1)]).prepare(module, args))
            before = read(c)
            raises(
                lambda: ExecutionPlan(
                    [
                        {"kind": "fill32", "destination": "c", "value": 0},
                        op,
                        {"kind": "kernel", "entry": "missing", "arguments": []},
                    ]
                ).prepare(module, args)
            )
            assert read(c) == before
            for value in args.values():
                value.close()

        square = runtime.ndarray("f32", (3, 3))
        alias = dict(kind="gemm_f32", a="x", b="x", output="x", rows=3, columns=3, inner=3)
        raises(lambda: ExecutionPlan([alias]).prepare(module, {"x": square}))
        wrong_type = runtime.ndarray("i32", (3, 3))
        raises(lambda: ExecutionPlan([dict(alias, output="y")]).prepare(module, {"x": square, "y": wrong_type}))
        other_run()
        module.close()
        raises(other_run)
        other_run.close()
        module = runtime.load_module(artifact)
        final = module.execution_plan("matrix_energy").prepare(module, right)
        final()
        near(read(right["c"]), other_expected[0])
        if mode == "borrowed":
            final()
            ti.reset()
            raises(final)
            runtime.close()
            ti.init(arch=ti.cuda, offline_cache=False)
            with ti.aot.borrow_runtime() as fresh:
                assert fresh.probe_native_operation("gemm_f32")["available"]
            ti.reset()
        else:
            assert "taichi_forge" not in sys.modules
    print("AOT_GEMM_PASSED", mode, status)


if __name__ == "__main__":
    main(*sys.argv[1:])
