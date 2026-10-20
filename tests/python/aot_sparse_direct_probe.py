"""Runtime-only sparse direct consumer with an independent dense reference."""

import array
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime


def resource_spec(name="solver", *, dtype="f32", method="LLT", ordering="AMD", format="csc", refactor=False):
    return dict(
        kind="native",
        name=name,
        provider="sparse_direct",
        operation="plan",
        version=1,
        parameters=dict(
            rows=3,
            nonzeros=7,
            dtype=dtype,
            implementation="eigen",
            method=method,
            ordering=ordering,
            format=format,
            phase="refactor_solve" if refactor else "solve",
        ),
        bindings=dict(offsets="offsets", indices="indices", values="seed"),
    )


def solve_op(pairs, values=None):
    bindings = dict(plan="solver")
    for index, (rhs, output) in enumerate(pairs):
        bindings[f"rhs_{index}"], bindings[f"output_{index}"] = rhs, output
    if values is not None:
        bindings["values"] = values
    return dict(
        kind="native",
        provider="sparse_direct",
        operation="solve",
        version=1,
        parameters=dict(rhs_count=len(pairs)),
        bindings=bindings,
    )


def dense_solve(matrix, rhs):
    rows = [[float(v) for v in row] + [float(b)] for row, b in zip(matrix, rhs)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(rows[row][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        value = rows[column][column]
        rows[column] = [item / value for item in rows[column]]
        for row in range(3):
            if row != column:
                value = rows[row][column]
                rows[row] = [a - value * b for a, b in zip(rows[row], rows[column])]
    return [row[-1] for row in rows]


def write(value, entries, code="f"):
    value.write(array.array(code, entries))


def read(value, code="f"):
    data = array.array(code)
    data.frombytes(value.read())
    return data.tolist()


def near(actual, expected, tolerance=2e-6):
    assert all(abs(a - b) <= tolerance * (1 + abs(b)) for a, b in zip(actual, expected)), (actual, expected)


def raises(call):
    try:
        call()
    except (AotError, ValueError, RuntimeError):
        return
    raise AssertionError("Invalid sparse direct binding or state was accepted")


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cpu, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cpu")
    with runtime:
        assert runtime.native_resource_capabilities("sparse_direct_plan") == {"cpu"}
        assert runtime.probe_native_operation("sparse_direct_solve")["available"]
        module = runtime.load_module(artifact)
        matrix = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
        rhs = [6, 10, 8]

        def frame(factor):
            buffers = {key: runtime.ndarray("f32", (3,)) for key in ("b", "x", "db")}
            buffers["loss"] = runtime.ndarray("f32", (1,))
            write(buffers["b"], [factor * value for value in rhs])
            write(buffers["x"], [-99] * 3)
            return buffers, module.execution_plan("fixed").prepare(module, buffers)

        left, run = frame(1)
        right, other = frame(-0.5)
        assert read(left["x"]) == [-99] * 3  # Preparation cannot execute solve.
        for _ in range(2):
            run()
            other()
        for buffers, scale in ((left, 1), (right, -0.5)):
            x = [scale, 2 * scale, 3 * scale]
            near(read(buffers["x"]), x)
            near(read(buffers["db"]), dense_solve(matrix, x))
            near(read(buffers["loss"]), [sum(v * v for v in x) / 2])
        write(left["b"], [2 * v for v in rhs])
        run()
        near(read(left["x"]), [2, 4, 6])
        near(read(right["x"]), [-0.5, -1, -1.5])

        for storage_format in ("col_major_f32", "row_major_f32"):
            exported = module.execution_plan("from_graph_" + storage_format).prepare(
                module, dict(b=left["b"], x=left["x"])
            )
            exported()
            near(read(left["x"]), [2, 4, 6])
            exported.close()
        double_rhs, double_output = runtime.ndarray("f64", (3,)), runtime.ndarray("f64", (3,))
        write(double_rhs, rhs, "d")
        exported = module.execution_plan("from_graph_row_major_f64").prepare(
            module, dict(b=double_rhs, x=double_output)
        )
        exported()
        near(read(double_output, "d"), [1, 2, 3], 1e-12)
        exported.close()

        # Unsorted CSR input becomes CSC once. Live values retain CSR order.
        general = [[4, 2, 0], [1, 3, 1], [0, 0.5, 2]]
        values = [2, 4, 1, 1, 3, 2, 0.5]
        dynamic = {key: runtime.ndarray("f64", (3,)) for key in ("b", "b2", "x", "x2")}
        dynamic["values"] = runtime.ndarray("f64", (7,))
        write(dynamic["b"], rhs, "d")
        write(dynamic["b2"], [-v for v in rhs], "d")
        active = module.execution_plan("dynamic").prepare(module, dynamic)
        for scale in (1, 2, 0.5):
            write(dynamic["values"], [scale * v for v in values], "d")
            active()
            expected = [v / scale for v in dense_solve(general, rhs)]
            near(read(dynamic["x"], "d"), expected, 1e-12)
            near(read(dynamic["x2"], "d"), [-v for v in expected], 1e-12)
        # An invalid numeric factor cannot report stale successful output.
        write(dynamic["values"], [0] * 7, "d")
        write(dynamic["x"], [-99] * 3, "d")
        raises(active)
        near(read(dynamic["x"], "d"), [-99] * 3, 0)
        write(dynamic["values"], values, "d")
        active()
        near(read(dynamic["x"], "d"), dense_solve(general, rhs), 1e-12)
        alias = dict(dynamic, x=dynamic["b"])
        raises(lambda: module.execution_plan("dynamic").prepare(module, alias))

        # Public resources release completed construction-only seed inputs.
        seed = {
            key: runtime.ndarray(dtype, (size,))
            for key, dtype, size in (("offsets", "i32", 4), ("indices", "i32", 7), ("seed", "f32", 7))
        }
        write(seed["offsets"], [0, 2, 5, 7], "i")
        write(seed["indices"], [0, 1, 0, 1, 2, 1, 2], "i")
        write(seed["seed"], [4, 1, 1, -3, 1, 1, 2])
        spec = resource_spec(method="LDLT")
        parameters = spec["parameters"]
        seed_bindings = {k: seed[v] for k, v in spec["bindings"].items()}
        raises(
            lambda: runtime.native_resource("sparse_direct", "missing", parameters=parameters, bindings=seed_bindings)
        )
        raises(
            lambda: runtime.native_resource(
                "sparse_direct", "plan", parameters=dict(parameters, method="QR"), bindings=seed_bindings
            )
        )
        raises(
            lambda: runtime.native_resource(
                "sparse_direct", "plan", parameters=dict(parameters, rows=1 << 31), bindings=seed_bindings
            )
        )
        write(seed["indices"], [0, 0, 0, 1, 2, 1, 2], "i")
        raises(lambda: runtime.native_resource("sparse_direct", "plan", parameters=parameters, bindings=seed_bindings))
        write(seed["indices"], [0, 1, 0, 1, 3, 1, 2], "i")
        raises(lambda: runtime.native_resource("sparse_direct", "plan", parameters=parameters, bindings=seed_bindings))
        write(seed["indices"], [0, 1, 0, 1, 2, 1, 2], "i")
        owner = runtime.native_resource(
            "sparse_direct", "plan", parameters=parameters, bindings={k: seed[v] for k, v in spec["bindings"].items()}
        )
        for value in seed.values():
            value.close()
        public = ExecutionPlan([solve_op([("b", "x")])]).prepare(module, dict(solver=owner, b=left["b"], x=left["x"]))
        public()
        near(read(left["x"]), dense_solve([[4, 1, 0], [1, -3, 1], [0, 1, 2]], [2 * v for v in rhs]))
        owner.close()
        raises(public)
        public.close()
        run.close()
        other()
        active.close()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(other)
        other.close()
    if mode != "borrowed":
        assert "taichi_forge" not in sys.modules
        assert "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
