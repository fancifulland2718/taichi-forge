"""Stdlib-only cuDSS AOT consumer and independent dense solve reference."""

import array
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime


def plan_resource(phase="solve"):
    return dict(
        kind="native", name="solver", provider="cudss", operation="plan_f32", version=1,
        parameters=dict(rows=4, nonzeros=10, matrix_type="spd", phase=phase),
        bindings=dict(row_offsets="r", column_indices="c", values="seed"),
    )


def solve_group(pairs, values=None):
    bindings = dict(plan="solver")
    for index, (rhs, output) in enumerate(pairs):
        bindings[f"rhs_{index}"] = rhs
        bindings[f"output_{index}"] = output
    if values is not None:
        bindings["values"] = values
    return dict(kind="native", provider="cudss", operation="solve_group_f32", version=1,
                parameters=dict(rhs_count=len(pairs)), bindings=bindings)


def solve(rhs, scale=1):
    matrix = [[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]]
    augmented = [[scale * float(value) for value in row] + [float(b)] for row, b in zip(matrix, rhs)]
    for column in range(4):
        pivot = augmented[column][column]
        augmented[column] = [v / pivot for v in augmented[column]]
        for row in range(4):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [v - factor * p for v, p in zip(augmented[row], augmented[column])]
    return [row[-1] for row in augmented]


def write(value, entries, dtype="f"):
    value.write(array.array(dtype, entries))


def read(value):
    result = array.array("f")
    result.frombytes(value.read())
    return result.tolist()


def near(actual, expected):
    assert len(actual) == len(expected)
    assert all(abs(a - b) <= 3e-5 * (1 + abs(b)) for a, b in zip(actual, expected)), (actual, expected)


def raises(call):
    try:
        call()
    except (AotError, ValueError, RuntimeError):
        return
    raise AssertionError("Invalid cuDSS binding or lifecycle was accepted")


def main(mode, artifact, adapter, vendor):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    values = [4, -1, -1, 4, -1, -1, 4, -1, -1, 3]
    with runtime:
        assert runtime.native_operation_capabilities("cudss_solve_group_f32") == {"cuda_capture"}
        assert not runtime.probe_native_operation("cudss_solve_group_f32")["available"]
        runtime.configure_provider("cudss", adapter_path=adapter, library_path=vendor)
        runtime.configure_provider("unused.provider", adapter_path="absent-adapter")
        assert runtime.probe_native_operation("cudss_solve_group_f32")["available"]
        module = runtime.load_module(artifact)
        plan = module.execution_plan("solve_energy")

        def frame(factor):
            arrays = {name: runtime.ndarray("f32", (4,)) for name in ("b", "b2", "x", "x2", "db")}
            arrays.update(r=runtime.ndarray("i32", (5,)), c=runtime.ndarray("i32", (10,)),
                          seed=runtime.ndarray("f32", (10,)), loss=runtime.ndarray("f32", (1,)))
            write(arrays["r"], [0, 2, 5, 8, 10], "i")
            write(arrays["c"], [0, 1, 0, 1, 2, 1, 2, 3, 2, 3], "i")
            write(arrays["seed"], values)
            rhs = [factor * v for v in (2, 4, 6, 9)]
            write(arrays["b"], rhs)
            write(arrays["b2"], [2 * v for v in rhs])
            for name in ("x", "x2", "db"):
                write(arrays[name], [-99] * 4)
            write(arrays["loss"], [-99])
            run = plan.prepare(module, arrays)
            assert run.info["execution_kind"] == "cuda_graph"
            assert read(arrays["x"]) == [-99] * 4
            assert read(arrays["loss"]) == [-99]
            x = solve(rhs)
            return arrays, run, dict(x=x, x2=[2 * v for v in x], db=solve(x), loss=[sum(v * v for v in x) / 2])

        left, run, expected = frame(1)
        right, other, expected_other = frame(-0.5)
        # A plan takes a private seed. Replay must not reread this matrix.
        write(left["seed"], [0] * 10)
        for _ in range(3):
            run()
            other()
        for arrays, reference in ((left, expected), (right, expected_other)):
            for name, value in reference.items():
                near(read(arrays[name]), value)

        # Numerical-update plans run one factorization for all RHS in a group.
        for phase in ("factor_solve", "refactor_solve"):
            arrays = right
            dynamic = runtime.ndarray("f32", (10,))
            local = ExecutionPlan([solve_group([("b", "x"), ("b2", "x2")], "values")],
                                  resources=[plan_resource(phase)])
            bindings = {name: arrays[name] for name in ("r", "c", "seed", "b", "b2", "x", "x2")}
            bindings["values"] = dynamic
            prepared = local.prepare(module, bindings)
            for scale in (2, 4, 2):
                write(dynamic, [scale * v for v in values])
                prepared()
                near(read(arrays["x"]), [v / scale for v in expected_other["x"]])
                near(read(arrays["x2"]), [v / scale for v in expected_other["x2"]])
            prepared()
            prepared.close()
            dynamic.close()

        runtime.configure_provider("cudss", adapter_path=adapter + ".missing", library_path=vendor)
        raises(lambda: plan.prepare(module, right))
        run()
        near(read(left["db"]), expected["db"])
        runtime.configure_provider("cudss", adapter_path=None)
        other()
        near(read(right["x"]), expected_other["x"])
        run()
        left["b"].close()
        raises(run)
        run.close()
        other()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(other)
        other.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
