"""Stdlib-only loaded Cholesky plans, with an independent tridiagonal oracle."""

import array
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime


def resource(dtype, rows=17, rhs_count=3, seeded=False):
    return dict(
        kind="native",
        name="solver",
        provider="cusolverdn",
        operation="cholesky_plan",
        version=1,
        parameters=dict(rows=rows, rhs_count=rhs_count, dtype=dtype, seeded=seeded),
        bindings=dict(factor="factor", status="status"),
    )


def operation(phase="factor_and_solve", rhs="rhs", solution="x"):
    return dict(
        kind="native",
        provider="cusolverdn",
        operation="cholesky",
        version=1,
        parameters=dict(phase=phase),
        bindings=dict(plan="solver", a="a", rhs=rhs, solution=solution),
    )


def write(value, entries, code="f"):
    value.write(array.array(code, entries))


def read(value, code="f"):
    result = array.array(code)
    result.frombytes(value.read())
    return result.tolist()


def solve(rhs, scale, rows=17):
    output = []
    for begin in range(0, len(rhs), rows):
        diagonal, values = [4 * scale] * rows, list(rhs[begin : begin + rows])
        for i in range(1, rows):
            multiplier = -scale / diagonal[i - 1]
            diagonal[i] += multiplier * scale
            values[i] -= multiplier * values[i - 1]
        values[-1] /= diagonal[-1]
        for i in range(rows - 2, -1, -1):
            values[i] = (values[i] + scale * values[i + 1]) / diagonal[i]
        output.extend(values)
    return output


def near(actual, expected, dtype):
    tolerance = 2e-5 if dtype == "f32" else 2e-12
    assert len(actual) == len(expected)
    assert all(abs(a - b) <= tolerance * (1 + abs(b)) for a, b in zip(actual, expected)), (actual, expected)


def raises(call):
    try:
        call()
    except (AotError, ValueError, RuntimeError):
        return
    raise AssertionError("Invalid Cholesky plan/lifetime was accepted")


def main(mode, artifact, adapter, vendor, dtype):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    code = "f" if dtype == "f32" else "d"
    rows, count = 17, 3 * 17
    matrix = [4 if i == j else -1 if i == j + 1 else 999 if j > i else 0 for i in range(rows) for j in range(rows)]
    with runtime:
        assert runtime.native_operation_capabilities("cusolverdn_cholesky") == {"cuda_ordered"}
        assert not runtime.probe_native_operation("cusolverdn_cholesky")["available"]
        runtime.configure_provider("cusolverdn", adapter_path=adapter, library_path=vendor)
        runtime.configure_provider("unused", adapter_path="absent-adapter")
        assert runtime.probe_native_operation("cusolverdn_cholesky")["available"]
        module = runtime.load_module(artifact)
        plan = module.execution_plan("energy")
        frames = []
        for sign in (1, -0.5):
            frame = {name: runtime.ndarray(dtype, (3, rows)) for name in ("rhs", "x", "db")}
            frame.update(
                a=runtime.ndarray(dtype, (rows, rows)),
                factor=runtime.ndarray(dtype, (rows, rows)),
                status=runtime.ndarray("i32", (2,)),
                loss=runtime.ndarray(dtype, (1,)),
            )
            write(frame["x"], [-99] * count, code)
            write(frame["factor"], [-77] * (rows * rows), code)
            write(frame["status"], [-1, -1], "i")
            call = plan.prepare(module, frame)
            assert call.info["execution_kind"] == "cuda_ordered"
            assert read(frame["x"], code) == [-99] * count
            assert read(frame["factor"], code) == [-77] * (rows * rows)
            assert read(frame["status"], "i") == [-1, -1]
            for scale in (1, 2):
                rhs = [sign * (i + 1) / count for i in range(count)]
                a = [scale * value for value in matrix]
                write(frame["rhs"], rhs, code)
                write(frame["a"], a, code)
                call()
                expected = solve(rhs, scale)
                near(read(frame["x"], code), expected, dtype)
                near(read(frame["db"], code), solve(expected, scale), dtype)
                near(read(frame["loss"], code), [sum(v * v for v in expected) / 2], dtype)
                assert read(frame["a"], code) == a
                assert read(frame["status"], "i") == [0, 0]
            # A seeded solve reads the actual old factor, never the changed A.
            write(frame["a"], [0] * (rows * rows), code)
            reuse = ExecutionPlan([operation("solve")], resources=[resource(dtype, seeded=True)])
            args = {key: frame[key] for key in reuse.argument_names}
            with reuse.prepare(module, args) as solve_only:
                solve_only()
                near(read(frame["x"], code), expected, dtype)
            # Exact RHS/output alias is supported by the native plan.
            args["x"] = frame["rhs"]
            with reuse.prepare(module, args) as alias:
                alias()
                near(read(frame["rhs"], code), expected, dtype)
            unseeded = ExecutionPlan([operation("solve")], resources=[resource(dtype)])
            raises(lambda: unseeded.prepare(module, args))
            invalid = dict(args, factor=frame["a"])
            raises(lambda: reuse.prepare(module, invalid))
            for field, value in (("rows", 0), ("rows", 0x7FFFFFFFFFFFFFFF), ("dtype", "i32")):
                bad_resource = resource(dtype, seeded=True)
                bad_resource["parameters"][field] = value
                malformed = ExecutionPlan([operation("solve")], resources=[bad_resource])
                raises(lambda: malformed.prepare(module, args))
            loop = ExecutionPlan(
                [
                    dict(
                        kind="while",
                        predicate="active",
                        max_iterations=1,
                        condition=[dict(kind="fill32", destination="active", value=1)],
                        body=[operation("solve")],
                    )
                ],
                resources=[resource(dtype, seeded=True), dict(kind="ndarray", name="active", dtype="i32", shape=[1])],
            )
            try:
                loop.prepare(module, args)
            except AotError as error:
                assert "GPU control cannot contain host-invoked native operations" in str(error), str(error)
            else:
                raise AssertionError("A root-only Cholesky recording entered a conditional body")
            frames.append((frame, call))
        # A failure is visible in device status; the owner can be reused later.
        frame, call = frames[0]
        write(frame["a"], [0] * (rows * rows), code)
        call()
        assert read(frame["status"], "i")[0] > 0
        write(frame["a"], matrix, code)
        call()
        assert read(frame["status"], "i") == [0, 0]
        runtime.configure_provider("cusolverdn", adapter_path=adapter + ".missing", library_path=vendor)
        raises(lambda: plan.prepare(module, frame))
        call()  # Existing owners retain their original adapter and vendor.
        call.close()  # retire queued work before dropping its capture/plan
        frames[-1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for frame, call in frames:
            raises(call)
            call.close()
            for value in frame.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
