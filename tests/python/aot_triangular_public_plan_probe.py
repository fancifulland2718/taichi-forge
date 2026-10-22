"""Shared plan qualification, numeric refresh and independent retirement."""

import array
import json
import sys

from taichi_forge_runtime.aot import Runtime
from aot_triangular_probe import INITIAL, solve
from tests.python.aot_binding_constraints_probe import failure
from tests.python.aot_dot_probe import near, read, write


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        module, peer = runtime.load_module(artifact), runtime.load_module(artifact)
        rows, indices, values = runtime.ndarray("i32", 4), runtime.ndarray("i32", 9), runtime.ndarray("f32", 9)
        rows.write(array.array("i", [0, 3, 6, 9]))
        indices.write(array.array("i", [0, 1, 2] * 3))
        write(values, INITIAL)

        def matrix():
            return runtime.native_resource(
                "cusparse",
                "csr_f32",
                parameters=dict(rows=3, columns=3),
                bindings=dict(row_offsets=rows, column_indices=indices, values=values),
            )

        original, other = matrix(), matrix()
        for rhs in (1, 2):
            parameters = json.loads(module.execution_plan(str(rhs)).to_json())["operations"][0]["parameters"]
            shape = (3,) if rhs == 1 else (3, rhs)
            x, y, wrong = (runtime.ndarray("f32", shape) for _ in range(3))

            def plan(**patch):
                return runtime.native_resource(
                    "cusparse",
                    "triangular_plan_f32",
                    parameters=dict(parameters, **patch),
                    bindings=dict(matrix=original, input=x, output=y),
                )

            shared = plan()
            args = dict(matrix=original, solve=shared, x=x, y=y)
            first, second = (m.prepare_execution(str(rhs), args) for m in (module, peer))
            write(y, [91] * (3 * rhs))
            for patch, error in (
                (dict(matrix=other), "mismatch"),
                (dict(x=wrong), "binding mismatch"),
                (dict(y=wrong), "binding mismatch"),
            ):
                failure(lambda: peer.prepare_execution(str(rhs), dict(args, **patch)), error)
            for patch in (
                dict(transpose=not parameters["transpose"]),
                dict(unit_diagonal=not parameters["unit_diagonal"]),
                dict(fill_mode="upper" if parameters["fill_mode"] == "lower" else "lower"),
                dict(mutable_values=False),
            ):
                with plan(**patch) as bad:
                    failure(
                        lambda: peer.prepare_execution(str(rhs), dict(args, solve=bad)),
                        "mutable_values" if patch.get("mutable_values") is False else "contract mismatch",
                    )
            assert read(y) == [91] * (3 * rhs), "Rejected preparation wrote output"
            for scale in (1, 2, 0.5):
                coefficients = [v * scale for v in INITIAL]
                data = [float(i - 2) for i in range(3 * rhs)]
                write(values, coefficients)
                write(x, data)
                for call in (first, second):
                    call()
                    near(
                        read(y),
                        solve(
                            coefficients,
                            data,
                            rhs,
                            parameters["fill_mode"],
                            parameters["unit_diagonal"],
                            parameters["transpose"],
                        ),
                    )
            first.close()
            second()
            shared.close()
            failure(second, "retired dependency")
            second.close()
            # A separately prepared resource on the same matrix remains valid.
            with plan() as independent, peer.prepare_execution(str(rhs), dict(args, solve=independent)) as call:
                call()
                near(
                    read(y),
                    solve(
                        coefficients,
                        data,
                        rhs,
                        parameters["fill_mode"],
                        parameters["unit_diagonal"],
                        parameters["transpose"],
                    ),
                )
        original.close()
        other.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
