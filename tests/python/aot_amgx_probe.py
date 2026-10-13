"""Stdlib-only AmgX consumer with an independent tridiagonal reference."""

import json
import math
import struct
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime

N = 17
NNZ = 3 * N - 2


def native(operation, bindings, **parameters):
    return dict(
        kind="native", provider="amgx", operation=operation, version=1, bindings=bindings, parameters=parameters
    )


def solver_resource(dtype, *, compute_residual=True, max_iters=60):
    config = json.dumps(
        dict(
            config_version=2,
            solver=dict(
                solver="PCG",
                preconditioner=dict(solver="NOSOLVER"),
                max_iters=max_iters,
                monitor_residual=1,
                convergence="RELATIVE_INI",
                norm="L2",
                tolerance=1e-6 if dtype == "f32" else 1e-11,
                print_solve_stats=0,
                obtain_timings=0,
            ),
        )
    )
    return dict(
        native("solver_csr", {}, rows=N, nonzeros=NNZ, dtype=dtype, config=config, compute_residual=compute_residual),
        name="solver",
    )


def setup():
    return native("setup_csr", dict(solver="solver", row_offsets="rows", column_indices="columns", values="initial"))


def solve(**parameters):
    return native(
        "solve",
        dict(
            solver="solver",
            rhs="rhs",
            solution="solution",
            status="status",
            iterations="iterations",
            residual="residual",
        ),
        **parameters,
    )


def forward(dtype, **parameters):
    return [
        dict(kind="kernel", entry=f"rhs_{dtype}", arguments=["seed", "rhs"]),
        solve(**parameters),
        dict(kind="kernel", entry=f"publish_{dtype}", arguments=["solution", "published"]),
    ]


def plans(dtype):
    resource = solver_resource(dtype)
    fixed = forward(dtype, retain_initial_guess=True)
    result = {
        f"fixed_{dtype}": ExecutionPlan(fixed, initialization=[setup()], resources=[resource]),
        f"refresh_{dtype}": ExecutionPlan(
            [
                dict(kind="kernel", entry=f"coefficients_{dtype}", arguments=["diagonal", "values"]),
                native("replace_coefficients", dict(solver="solver", values="values")),
            ]
            + fixed,
            initialization=[setup()],
            resources=[resource],
        ),
        f"rebuild_{dtype}": ExecutionPlan([setup()] + fixed, resources=[resource]),
        f"adjoint_{dtype}": ExecutionPlan(
            forward(dtype)
            + [
                native("solve", dict(solver="solver", rhs="solution", solution="adjoint", status="adj_status")),
                dict(kind="kernel", entry=f"gradient_{dtype}", arguments=["solution", "adjoint", "gradient"]),
            ],
            initialization=[setup()],
            resources=[resource],
        ),
    }
    return result


def write(array, values, dtype="f"):
    array.write(struct.pack("<" + dtype * len(values), *values))


def read(array, dtype="f"):
    data = array.read()
    return list(struct.unpack("<" + dtype * (len(data) // struct.calcsize(dtype)), data))


def near(actual, expected, tolerance):
    assert len(actual) == len(expected)
    assert all(abs(a - b) <= tolerance * (1 + abs(b)) for a, b in zip(actual, expected)), (actual, expected)


def raises(call, message=None):
    try:
        call()
    except (AotError, ValueError, RuntimeError) as error:
        if message is not None:
            assert message in str(error), str(error)
        return
    raise AssertionError("Invalid AmgX preparation or execution succeeded")


def coefficients(diagonal):
    return [diagonal if i == j else -1 for i in range(N) for j in range(max(0, i - 1), min(N, i + 2))]


def reference(diagonal, rhs):
    # Thomas elimination; no AmgX or Forge output is used as the reference.
    diagonal = [diagonal] * N
    rhs = list(rhs)
    for i in range(1, N):
        diagonal[i] -= 1 / diagonal[i - 1]
        rhs[i] += rhs[i - 1] / diagonal[i - 1]
    result = [0.0] * N
    result[-1] = rhs[-1] / diagonal[-1]
    for i in range(N - 2, -1, -1):
        result[i] = (rhs[i] + result[i + 1]) / diagonal[i]
    return result


def frame(runtime, module, name, dtype, diagonal=4):
    plan = module.execution_plan(name)
    integers = dict(rows=N + 1, columns=NNZ, status=1, iterations=1, adj_status=1)
    sizes = dict(initial=NNZ, values=NNZ, diagonal=1, residual=1, gradient=NNZ)
    arrays = {
        key: runtime.ndarray(
            "i32" if key in integers else "f64" if key == "residual" else dtype, (integers.get(key, sizes.get(key, N)),)
        )
        for key in plan.argument_names
    }
    scalar = "f" if dtype == "f32" else "d"
    write(arrays["rows"], [0] + [3 * i - 1 for i in range(1, N)] + [NNZ], "i")
    write(arrays["columns"], [j for i in range(N) for j in range(max(0, i - 1), min(N, i + 2))], "i")
    write(arrays["initial"], coefficients(diagonal), scalar)
    write(arrays["seed"], [(i % 7 - 3) / 8 for i in range(N)], scalar)
    for key in ("solution", "published"):
        write(arrays[key], [-77] * N, scalar)
    prepared = plan.prepare(module, arrays)
    assert read(arrays["published"], scalar) == [-77] * N
    assert prepared.info["execution_kind"] == "cuda_mixed"
    return arrays, prepared


def check(arrays, dtype, diagonal):
    scalar, tolerance = ("f", 2e-5) if dtype == "f32" else ("d", 2e-9)
    expected = reference(diagonal, [value * 2 + 0.125 for value in read(arrays["seed"], scalar)])
    near(read(arrays["solution"], scalar), expected, tolerance)
    near(read(arrays["published"], scalar), [value * 2 + 1 for value in expected], tolerance)
    assert read(arrays["status"], "i") == [0]
    assert 0 <= read(arrays["iterations"], "i")[0] <= 60
    assert math.isfinite(read(arrays["residual"], "d")[0])
    if "adjoint" in arrays:
        adjoint = reference(diagonal, expected)
        near(read(arrays["adjoint"], scalar), adjoint, tolerance)
        near(
            read(arrays["gradient"], scalar),
            [-adjoint[i] * expected[j] for i in range(N) for j in range(max(0, i - 1), min(N, i + 2))],
            tolerance,
        )
        assert read(arrays["adj_status"], "i") == [0]


def close_frame(arrays, prepared):
    prepared.close()
    for array in arrays.values():
        array.close()


def main(mode, artifact, adapter, vendor):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        for kind in ("amgx_setup_csr", "amgx_replace_coefficients", "amgx_solve"):
            assert runtime.native_operation_capabilities(kind) == {"cuda_synchronous"}
            assert not runtime.probe_native_operation(kind)["available"]
        runtime.configure_provider("amgx", adapter_path=adapter, library_path=vendor)
        assert runtime.probe_native_operation("amgx_solve")["available"]
        module = runtime.load_module(artifact)
        for dtype in ("f32", "f64"):
            scalar = "f" if dtype == "f32" else "d"
            left, one = frame(runtime, module, f"fixed_{dtype}", dtype)
            right, two = frame(runtime, module, f"fixed_{dtype}", dtype)
            # Preparation cannot sample numerical matrix inputs.
            write(left["initial"], coefficients(6), scalar)
            assert one.info["workspace_bytes"] == (N + 1 + NNZ) * 4
            one.initialize()
            assert one.info["workspace_bytes"] == 0  # Vendor-owned storage is opaque.
            assert read(left["published"], scalar) == [-77] * N
            write(left["initial"], coefficients(8), scalar)
            one.initialize()  # Idempotent, still A with diagonal 6.
            for key in ("rows", "columns", "initial"):
                left[key].close()
            for i in range(3):
                if i:
                    write(left["solution"], [float("nan")] * N, scalar)
                write(left["seed"], [(j % 7 - 3 + i) / 8 for j in range(N)], scalar)
                one()
                two()
                check(left, dtype, 6)
                check(right, dtype, 4)
            close_frame(left, one)
            two()
            check(right, dtype, 4)  # Retiring a sibling must not invalidate AmgX pools.
            close_frame(right, two)
            for name in ("refresh", "rebuild", "adjoint"):
                arrays, run = frame(runtime, module, f"{name}_{dtype}", dtype)
                for diagonal in (4, 5):
                    if name == "refresh":
                        write(arrays["diagonal"], [diagonal], scalar)
                    elif name == "rebuild":
                        write(arrays["initial"], coefficients(diagonal), scalar)
                    else:
                        diagonal = 4
                    run()
                    check(arrays, dtype, diagonal)
                close_frame(arrays, run)

        # Cold recording rejects uninitialized solvers and unseeded retained state.
        arrays, keep = frame(runtime, module, "fixed_f32", "f32")
        for ops, initialization, message in (
            (forward("f32"), None, "preceding setup"),
            (forward("f32", initial_guess="retained"), [setup()], "preceding solve"),
        ):
            plan = ExecutionPlan(ops, initialization=initialization, resources=[solver_resource("f32")])
            raises(lambda: plan.prepare(module, {k: arrays[k] for k in plan.argument_names}), message)
        # Initialization may seed a retained guess; zero/upload applies only to the first solve.
        for policy in ("zero", "upload"):
            plan = ExecutionPlan(
                [solve(initial_guess="retained")],
                initialization=[setup()] + forward("f32", initial_guess=policy),
                resources=[solver_resource("f32", compute_residual=False)],
            )
            write(arrays["solution"], [0] * N)
            seeded = plan.prepare(module, {k: arrays[k] for k in plan.argument_names})
            seeded.initialize()
            write(arrays["solution"], [float("nan")] * N)
            seeded()
            assert read(arrays["status"], "i") == [0]
            assert math.isnan(read(arrays["residual"], "d")[0])
            near(read(arrays["solution"]), reference(4, [v * 2 + 0.125 for v in read(arrays["seed"])]), 2e-5)
            seeded.close()
        # Diagnostics cannot overwrite RHS/solution storage, including an alias via the same scalar type.
        plan = ExecutionPlan(forward("f32"), initialization=[setup()], resources=[solver_resource("f32")])
        raises(lambda: plan.prepare(module, dict(arrays, iterations=arrays["status"])), "overlap")
        # RHS is uploaded before the result is written: identical input/output storage is valid.
        aliased = dict(arrays, solution=arrays["rhs"])
        alias_run = plan.prepare(module, aliased)
        alias_run()
        check(aliased, "f32", 4)
        alias_run.close()
        # A successful library call need not mean convergence. Preserve status and the iterate.
        limited = ExecutionPlan(
            forward("f32"), initialization=[setup()], resources=[solver_resource("f32", max_iters=1)]
        )
        unconverged = limited.prepare(module, arrays)
        unconverged()
        assert read(arrays["status"], "i")[0] != 0
        assert read(arrays["iterations"], "i") == [1]
        near(read(arrays["published"]), [2 * v + 1 for v in read(arrays["solution"])], 2e-5)
        unconverged.close()
        # Malformed CSR fails during initialize, poisons this instance and preserves another live solver.
        bad, failed = frame(runtime, module, "fixed_f32", "f32")
        write(bad["columns"], [N] * NNZ, "i")
        raises(failed.initialize)
        write(bad["columns"], [0] * NNZ, "i")
        raises(failed)
        close_frame(bad, failed)
        keep()
        check(arrays, "f32", 4)
        runtime.configure_provider("amgx", adapter_path=None)
        assert not runtime.probe_native_operation("amgx_solve")["available"]
        raises(lambda: frame(runtime, module, "fixed_f32", "f32"))
        keep()
        check(arrays, "f32", 4)
        keep()  # Owner retirement must cover the trailing asynchronous publish graph.
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(keep)
        keep.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
