"""Source-free CSR descriptors, preparation rejection, updates and owner retirement."""

import array
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_dot_probe import near, raises, read, write


def native(operation, bindings, **parameters):
    return dict(
        kind="native", provider="cusparse", operation=operation, version=1, parameters=parameters, bindings=bindings
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
        for dtype in ("i32", "i64"):
            frame = {
                name: runtime.ndarray("f32", size)
                for name, size in (("x", 4), ("y0", 3), ("y", 3), ("values", 5), ("dx", 4), ("loss", 1))
            }
            write(frame["y"], [-99] * 3)
            prepared = module.prepare_execution(dtype, frame)
            assert read(frame["y"]) == [-99] * 3
            assert prepared.info["execution_kind"] == "cuda_graph"
            frames.append((frame, prepared))
        offsets, columns = [0, 2, 2, 5], [2, 0, 3, 0, 1]
        for iteration in range(3):
            for index, (frame, prepared) in enumerate(frames):
                x = [0.5 + index, -2, 1.25, -0.75]
                y0 = [-1, 2, 0.5]
                values = [value * (iteration - 1) for value in (2, -1, 0.5, 3, -2)]
                for name, data in (("x", x), ("y0", y0), ("values", values)):
                    write(frame[name], data)
                prepared()
                y = [
                    1.5 * sum(values[j] * x[columns[j]] for j in range(offsets[r], offsets[r + 1])) - 0.25 * y0[r]
                    for r in range(3)
                ]
                dx = [0] * 4
                for row in range(3):
                    for j in range(offsets[row], offsets[row + 1]):
                        dx[columns[j]] += 1.5 * values[j] * y[row]
                near(read(frame["y"]), y)
                near(read(frame["loss"]), [0.5 * sum(v * v for v in y)])
                near(read(frame["dx"]), dx)

        def indices(dtype, values):
            result = runtime.ndarray(dtype, len(values))
            result.write(array.array("i" if dtype == "i32" else "q", values))
            return result

        resource = dict(
            native(
                "csr_f32", dict(row_offsets="offsets", column_indices="columns", values="values"), rows=2, columns=2
            ),
            name="matrix",
        )
        operation = native("spmv_f32", dict(matrix="matrix", input="x", output="y"))
        storage = dict(
            offsets=indices("i32", [0, 1, 2]),
            columns=indices("i32", [1, 0]),
            values=runtime.ndarray("f32", 2),
            x=runtime.ndarray("f32", 2),
            y=runtime.ndarray("f32", 2),
        )
        write(storage["values"], [2, -3])
        write(storage["x"], [4, 5])
        write(storage["y"], [9, 8])

        def prepare(arguments=storage, res=resource, op=operation, extra=()):
            return ExecutionPlan([op], resources=[*extra, res]).prepare(module, arguments)

        for params in (dict(algorithm="csr_alg2", preprocess=False), dict(transpose=True)):
            with prepare(op=dict(operation, parameters=params)) as call:
                call()
                near(read(storage["y"]), [-15, 8] if params.get("transpose") else [10, -12])
        # cuSPARSE currently requires matching row/column index widths.
        alternate = dict(storage, offsets=indices("i64", [0, 1, 2]), columns=indices("i32", [1, 0]))
        raises(lambda: prepare(alternate))
        raises(lambda: prepare(dict(storage, columns=indices("i64", [1, 0]))))
        # Rejected preparations must never execute the output operation.
        write(storage["y"], [9, 8])
        for offsets_bad, columns_bad in (
            ([1, 1, 2], [1, 0]),
            ([0, 1, 1], [1, 0]),
            ([0, 2, 1], [1, 0]),
            ([0, 3, 2], [1, 0]),
            ([0, -1, 2], [1, 0]),
            ([0, 1, 2], [2, 0]),
            ([0, 1, 2], [-1, 0]),
            ([0, 2, 2], [1, 1]),
        ):
            bad = dict(storage, offsets=indices("i32", offsets_bad), columns=indices("i32", columns_bad))
            raises(lambda: prepare(bad))
        # Unsorted, nonadjacent duplicates also fail before creating a vendor plan.
        bad = dict(
            storage,
            offsets=indices("i32", [0, 3, 3]),
            columns=indices("i32", [1, 0, 1]),
            values=runtime.ndarray("f32", 3),
        )
        raises(lambda: prepare(bad))
        for bad in (
            dict(storage, x=storage["y"]),
            dict(storage, values=storage["y"]),
            dict(storage, x=runtime.ndarray("f32", 1)),
            dict(storage, x=runtime.ndarray("i32", 2)),
            dict(storage, offsets=runtime.ndarray("f32", 3)),
        ):
            raises(lambda: prepare(bad))
        for parameters in (dict(algorithm="auto"), dict(preprocess=1), dict(alpha=True), dict(stride=2)):
            raises(lambda: prepare(op=dict(operation, parameters=parameters)))
        near(read(storage["y"]), [9, 8])
        # Physical views must be compared, even though they have different names.
        parent = runtime.ndarray("f32", 6)
        write(parent, [99, 2, -3, 4, 5, 77])
        views = [
            dict(kind="ndarray_view", name=name, array="parent", offset=offset, shape=[2])
            for name, offset in (("values", 1), ("x", 3))
        ]
        arguments = {name: value for name, value in storage.items() if name not in ("values", "x")}
        arguments["parent"] = parent
        with prepare(arguments, extra=views) as call:
            call()
            near(read(storage["y"]), [10, -12])
        # Force output to overlap the CSR coefficients.
        alias_views = [*views, dict(kind="ndarray_view", name="y", array="parent", offset=2, shape=[2])]
        raises(lambda: prepare({k: v for k, v in arguments.items() if k != "y"}, extra=alias_views))
        near(read(parent), [99, 2, -3, 4, 5, 77])
        empty = dict(
            storage, offsets=indices("i32", [0, 0, 0]), columns=indices("i32", []), values=runtime.ndarray("f32", 0)
        )
        with prepare(empty) as call:
            for _ in range(2):
                write(storage["y"], [99, 88])
                call()
                near(read(storage["y"]), [0, 0])
        # Degenerate rectangular shapes preserve overwrite/beta semantics.
        for rows, cols in ((0, 2), (2, 0), (0, 0)):
            arguments = dict(
                offsets=indices("i32", [0] * (rows + 1)),
                columns=indices("i32", []),
                values=runtime.ndarray("f32", 0),
                x=runtime.ndarray("f32", cols),
                y=runtime.ndarray("f32", rows),
            )
            write(arguments["y"], [2] * rows)
            with prepare(
                arguments,
                res=dict(resource, parameters=dict(rows=rows, columns=cols)),
                op=dict(operation, parameters=dict(beta=0.5)),
            ) as call:
                call()
                near(read(arguments["y"]), [1] * rows)
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
