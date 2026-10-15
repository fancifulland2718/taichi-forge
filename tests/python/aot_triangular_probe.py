"""Runtime-only triangular solves with independent analysis and explicit adjoints."""

import json
import sys
from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_binding_constraints_probe import failure
from tests.python.aot_dot_probe import near, read, write

INITIAL = [2, 0.2, -0.1, 1, 4, 0.3, 0.5, -1, 3]


def solve(values, data, rhs, fill, unit, transpose):
    matrix = [[values[i * 3 + j] if (i >= j if fill == "lower" else i <= j) else 0 for j in range(3)] for i in range(3)]
    if unit:
        for i in range(3):
            matrix[i][i] = 1
    if transpose:
        matrix = [list(row) for row in zip(*matrix)]
    lower = (fill == "lower") != transpose
    output = [0.0] * len(data)
    for i in (range(3) if lower else range(2, -1, -1)):
        for column in range(rhs):
            remaining = sum(matrix[i][j] * output[j * rhs + column] for j in range(3) if j != i)
            output[i * rhs + column] = (data[i * rhs + column] - remaining) / matrix[i][i]
    return output


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        for suffix in ("solve", "update"):
            capability = runtime.probe_native_operation(f"cusparse_triangular_{suffix}_f32")
            assert capability["available"], (suffix, capability)
        module = runtime.load_module(artifact)
        frames = []
        for name in ("vector", "matrix", "upper", "unit_transpose"):
            document = json.loads(module.execution_plan(name).to_json())
            params = next(r["parameters"] for r in document["resources"] if r["name"] == "forward")
            rhs = params["rhs_count"]
            shape = (3,) if rhs == 1 else (3, rhs)
            for slot in range(2):
                arrays = {
                    key: runtime.ndarray("f32", (9,) if key in ("values", "coefficients") else shape)
                    for key in ("x", "y", "dx", "values", "coefficients")
                }
                write(arrays["values"], INITIAL)
                write(arrays["y"], [99] * (3 * rhs))
                call = module.prepare_execution(name, arrays)
                assert call.info["execution_kind"] == "cuda_graph"
                assert read(arrays["y"]) == [99] * (3 * rhs), "Preparation executed a solve"
                frames.append((name, params, slot, arrays, call))
        for step in range(3):
            for _, params, slot, arrays, call in reversed(frames):
                rhs = params["rhs_count"]
                coefficients = [v * (step + 1 + slot / 4) for v in INITIAL]
                x = [(i - 2 + slot) / 3 for i in range(3 * rhs)]
                write(arrays["coefficients"], coefficients)
                write(arrays["x"], x)
                write(arrays["y"], [99] * (3 * rhs))
                call()
                y = solve(coefficients, x, rhs, params["fill_mode"], params["unit_diagonal"], params["transpose"])
                dx = solve(coefficients, y, rhs, params["fill_mode"], params["unit_diagonal"], not params["transpose"])
                near(read(arrays["y"]), y)
                near(read(arrays["dx"]), dx)

        name, _, _, arrays, _ = frames[0]
        write(arrays["y"], [99] * 3)
        failure(lambda: module.prepare_execution(name, dict(arrays, y=arrays["x"])), "overlaps")
        document = json.loads(module.execution_plan(name).to_json())
        missing = dict(
            document, operations=[op for op in document["operations"] if op.get("operation") != "triangular_update_f32"]
        )
        failure(lambda: ExecutionPlan.from_json(json.dumps(missing)).prepare(module, arrays), "preceding update")
        for patch in (
            dict(rhs_count=0),
            dict(rhs_count=True),
            dict(fill_mode="auto"),
            dict(transpose=1),
            dict(unit_diagonal=0),
            dict(mutable_values=False),
        ):
            changed = dict(
                document,
                resources=[
                    dict(r, parameters=dict(r["parameters"], **patch)) if r["name"] == "forward" else r
                    for r in document["resources"]
                ],
            )
            failure(lambda: ExecutionPlan.from_json(json.dumps(changed)).prepare(module, arrays))
        assert read(arrays["y"]) == [99] * 3
        frames[0][4].close()
        frames[1][4]()
        near(
            read(frames[1][3]["y"]),
            solve([v * 3.25 for v in INITIAL], [(i - 1) / 3 for i in range(3)], 1, "lower", False, False),
        )
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        failure(frames[1][4])
        for _, _, _, arrays, call in frames:
            call.close()
            for value in arrays.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
