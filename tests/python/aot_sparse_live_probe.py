"""Standalone live sparse consumer; only the borrowed case imports the DSL."""

import json
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_dot_probe import near, read, write, raises


def case(kind):
    n = 4 if kind == "bsr" else 3
    rhs = 2 if kind.startswith("spmm") or kind == "spsm" else 1
    dense = [[float(2 + i) if i == j else float(i + 1) / 4 if j < i else 0.0 for j in range(n)] for i in range(n)]
    transpose = [list(row) for row in zip(*dense)]

    def compress(matrix):
        if kind == "bsr":
            return [
                matrix[2 * row + i][2 * col + j]
                for row in range(2)
                for col in range(2)
                for i in range(2)
                for j in range(2)
            ]
        return [x for row in matrix for x in row]

    return n, rhs, dense, compress(dense), compress(transpose)


def reference(matrix, data, rhs, solve=False, transpose=False):
    n = len(matrix)
    a = [list(row) for row in zip(*matrix)] if transpose else matrix
    if not solve:
        return [sum(a[i][k] * data[k * rhs + j] for k in range(n)) for i in range(n) for j in range(rhs)]
    result = [0.0] * len(data)
    for i in (reversed(range(n)) if transpose else range(n)):
        others = range(i + 1, n) if transpose else range(i)
        for j in range(rhs):
            result[i * rhs + j] = (data[i * rhs + j] - sum(a[i][k] * result[k * rhs + j] for k in others)) / a[i][i]
    return result


def main(mode, artifact, kind):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    n, rhs, dense, coefficients, adjoint = case(kind)
    triangular = kind in ("spsv", "spsm")
    shape = (n,) if rhs == 1 else (n, rhs)
    with runtime:
        module = runtime.load_module(artifact)
        frames = []
        for name in ("live", "selected", "frozen"):
            arrays = {
                key: runtime.ndarray("f32", (1,) if key in ("loss", "dloss") else shape)
                for key in ("x", "y", "check", "dy", "loss", "dloss", "dx")
            }
            if name != "frozen":
                for key, data in (("coefficients", coefficients), ("adjoint_coefficients", adjoint)):
                    if key == "adjoint_coefficients" and triangular:
                        continue
                    arrays[key] = runtime.ndarray("f32", (len(data),))
                    write(arrays[key], data)
            write(arrays["y"], [99] * (n * rhs))
            prepared = module.prepare_execution(name, arrays)
            assert prepared.info["execution_kind"] == "cuda_graph"
            assert read(arrays["y"]) == [99] * (n * rhs), "Preparation executed the graph"
            frames.append((name, arrays, prepared))
        for step in range(4):
            for slot, (name, arrays, prepared) in enumerate(reversed(frames)):
                scale = 1.0 if name == "frozen" else 1 + step / 2 + slot / 4
                host = [(i - 1 + slot) / 3 for i in range(n * rhs)] if step != 2 else [0] * (n * rhs)
                for key, data in (("coefficients", coefficients), ("adjoint_coefficients", adjoint)):
                    if key in arrays:
                        write(arrays[key], [v * scale for v in data])
                write(arrays["x"], host)
                prepared()
                a = [[v * scale for v in row] for row in dense]
                y = reference(a, host, rhs, solve=triangular)
                near(read(arrays["y"]), y)
                near(read(arrays["check"]), y)
                near(read(arrays["loss"]), [0.5 * sum(v * v for v in y)])
                near(read(arrays["dx"]), reference(a, y, rhs, solve=triangular, transpose=True))
        _, arrays, _ = frames[0]
        wrong = runtime.ndarray("f32", (len(coefficients) + 1,))
        raises(lambda: module.prepare_execution("live", dict(arrays, coefficients=wrong)))
        wrong.close()
        raises(lambda: module.prepare_execution("live", {k: v for k, v in arrays.items() if k != "coefficients"}))
        document = json.loads(module.execution_plan("live").to_json())
        # A preceding device copy changes values after retained analysis. The
        # automatically exported refresh must run after it on every replay.
        updated = dict(
            document,
            operations=[dict(kind="copy", source="incoming", destination="coefficients")] + document["operations"],
        )
        incoming = runtime.ndarray("f32", (len(coefficients),))
        with ExecutionPlan.from_json(json.dumps(updated)).prepare(module, dict(arrays, incoming=incoming)) as update:
            host = [(i - 2) / 5 for i in range(n * rhs)]
            write(arrays["x"], host)
            for scale in (2.5, 0.75):
                write(incoming, [v * scale for v in coefficients])
                if not triangular:
                    write(arrays["adjoint_coefficients"], [v * scale for v in adjoint])
                update()
                a = [[v * scale for v in row] for row in dense]
                y = reference(a, host, rhs, solve=triangular)
                near(read(arrays["y"]), y)
                near(read(arrays["dx"]), reference(a, y, rhs, solve=triangular, transpose=True))
        incoming.close()
        if triangular:
            missing = dict(
                document,
                operations=[op for op in document["operations"] if op.get("operation") != "triangular_update_f32"],
            )
            raises(lambda: ExecutionPlan.from_json(json.dumps(missing)).prepare(module, arrays))
        wrong_dtype = runtime.ndarray("i32", (len(coefficients),))
        raises(lambda: module.prepare_execution("live", dict(arrays, coefficients=wrong_dtype)))
        wrong_dtype.close()
        frames[0][2].close()
        frames[1][2]()
        # Retire outstanding work while producer objects no longer exist.
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(frames[1][2])
        for _, arrays, prepared in frames:
            prepared.close()
            for value in arrays.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
