"""Source-free BSR blocks, live values, explicit block transpose and retirement."""

import array
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_dot_probe import near, raises, read, write


def native(operation, bindings, **parameters):
    return dict(
        kind="native", provider="cusparse", operation=operation, version=1, parameters=parameters, bindings=bindings
    )


def fixture(block, *, ordered=False):
    offsets, columns = [0, 2, 2, 5], [2, 0, 3, 0, 1]
    blocks = [[((k + 2 * i + 3 * j) % 9 - 4) / 8 for i in range(block) for j in range(block)] for k in range(5)]
    if ordered:
        order = [k for row in range(3) for k in sorted(range(offsets[row], offsets[row + 1]), key=columns.__getitem__)]
        columns, blocks = [columns[k] for k in order], [blocks[k] for k in order]
    dense = [[0.0] * (4 * block) for _ in range(3 * block)]
    for row in range(3):
        for k in range(offsets[row], offsets[row + 1]):
            for i in range(block):
                for j in range(block):
                    dense[row * block + i][columns[k] * block + j] = blocks[k][i * block + j]
    transpose_offsets, transpose_columns, transpose_values = [0], [], []
    for col in range(4):
        for row in range(3):
            for k in range(offsets[row], offsets[row + 1]):
                if columns[k] == col:
                    transpose_columns.append(row)
                    transpose_values.extend(blocks[k][j * block + i] for i in range(block) for j in range(block))
        transpose_offsets.append(len(transpose_columns))
    return (
        offsets,
        columns,
        [v for b in blocks for v in b],
        transpose_offsets,
        transpose_columns,
        transpose_values,
        dense,
    )


def check_frozen(runtime, module):
    block = 2
    *_, dense = fixture(block)
    frames = []
    for name, scale in (("original", 1), ("updated", 2), ("selected", 2)):
        sizes = dict(x=8, y=6, check=6, dy=6, loss=1, dloss=1, dx=8)
        arrays = {name: runtime.ndarray("f32", size) for name, size in sizes.items()}
        write(arrays["y"], [-99] * 6)
        call = module.prepare_execution(name, arrays)
        near(read(arrays["y"]), [-99] * 6)
        assert call.info["execution_kind"] == "cuda_graph"
        frames.append((arrays, call, scale))
    for iteration in (1, 0, -1):
        x = [iteration * (i - 3) / 4 for i in range(8)]
        for arrays, call, scale in frames:
            write(arrays["x"], x)
            call()
            y = [scale * sum(a * b for a, b in zip(row, x)) for row in dense]
            dx = [scale * sum(dense[row][col] * y[row] for row in range(6)) for col in range(8)]
            near(read(arrays["y"]), y)
            near(read(arrays["check"]), y)
            near(read(arrays["loss"]), [0.5 * sum(v * v for v in y)])
            near(read(arrays["dx"]), dx)
    frames[0][1].close()
    frames[1][1]()
    return [call for _, call, _ in frames]


def check_live(runtime, module):
    block = 4  # Native schema follows the vendor layout, not the producer's preset block list.
    offsets, columns, values0, toffsets, tcolumns, tvalues0, dense = fixture(block)
    frames = []
    for dtype in ("i32", "i64"):
        sizes = dict(x=16, y=12, y0=12, values=80, transpose_values=80, loss=1, dx=16)
        arrays = {name: runtime.ndarray("f32", size) for name, size in sizes.items()}
        write(arrays["y"], [-99] * 12)
        call = module.prepare_execution(dtype, arrays)
        near(read(arrays["y"]), [-99] * 12)
        frames.append((arrays, call))
    for iteration in (1, 0, -1):
        for index, (arrays, call) in enumerate(frames):
            x = [(i - 7 + index) / 8 for i in range(16)]
            y0 = [(i - 5) / 4 for i in range(12)]
            for name, host in (
                ("x", x),
                ("y0", y0),
                ("values", [iteration * v for v in values0]),
                ("transpose_values", [iteration * v for v in tvalues0]),
            ):
                write(arrays[name], host)
            call()
            y = [1.5 * iteration * sum(a * b for a, b in zip(row, x)) - 0.25 * y0[i] for i, row in enumerate(dense)]
            dx = [1.5 * iteration * sum(dense[r][c] * y[r] for r in range(12)) for c in range(16)]
            near(read(arrays["y"]), y)
            near(read(arrays["loss"]), [0.5 * sum(v * v for v in y)])
            near(read(arrays["dx"]), dx)

    def indices(dtype, values):
        result = runtime.ndarray(dtype, len(values))
        result.write(array.array("i" if dtype == "i32" else "q", values))
        return result

    resource = dict(
        native(
            "bsr_f32",
            dict(row_offsets="r", column_indices="c", values="v"),
            block_rows=2,
            block_columns=2,
            block_size=2,
        ),
        name="a",
    )
    operation = native("bsr_spmv_f32", dict(matrix="a", input="x", output="y"))
    bindings = dict(
        r=indices("i32", [0, 1, 2]),
        c=indices("i32", [1, 0]),
        v=runtime.ndarray("f32", 8),
        x=runtime.ndarray("f32", 4),
        y=runtime.ndarray("f32", 4),
    )
    write(bindings["v"], [2, 0, 0, 2, -1, 0, 0, -1])
    write(bindings["x"], [1, 2, 3, 4])
    write(bindings["y"], [9] * 4)

    def prepare(arguments=bindings, res=resource, op=operation, extra=()):
        return ExecutionPlan([op], resources=[*extra, res]).prepare(module, arguments)

    for ro, co in (
        ([1, 1, 2], [1, 0]),
        ([0, 2, 1], [1, 0]),
        ([0, 3, 2], [1, 0]),
        ([0, 1, 2], [2, 0]),
        ([0, 2, 2], [1, 1]),
    ):
        raises(lambda: prepare(dict(bindings, r=indices("i32", ro), c=indices("i32", co))))
    for bad in (
        dict(bindings, r=indices("i64", [0, 1, 2])),
        dict(bindings, v=runtime.ndarray("f32", 7)),
        dict(bindings, x=bindings["y"]),
        dict(bindings, x=runtime.ndarray("f32", 3)),
    ):
        raises(lambda: prepare(bad))
    # The output is a distinct handle over the coefficients' allocation.
    alias = dict(kind="ndarray_view", name="y", array="v", offset=2, shape=[4])
    raises(lambda: prepare({k: v for k, v in bindings.items() if k != "y"}, extra=[alias]))
    near(read(bindings["y"]), [9] * 4)
    with prepare() as call:
        call()
        near(read(bindings["y"]), [6, 8, -1, -2])
    # Zero blocks and zero scalar dimensions must preserve overwrite/beta behavior.
    for rows, cols in ((2, 2), (0, 2), (2, 0), (0, 0)):
        args = dict(
            r=indices("i32", [0] * (rows + 1)),
            c=indices("i32", []),
            v=runtime.ndarray("f32", 0),
            x=runtime.ndarray("f32", cols * 2),
            y=runtime.ndarray("f32", rows * 2),
        )
        write(args["y"], [4] * (rows * 2))
        with prepare(
            args,
            res=dict(resource, parameters=dict(block_rows=rows, block_columns=cols, block_size=2)),
            op=dict(operation, parameters=dict(beta=0.5)),
        ) as call:
            call()
            near(read(args["y"]), [2] * (rows * 2))
    return [call for _, call in frames]


def main(mode, artifact, kind):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        status = runtime.probe_native_operation("cusparse_bsr_spmv_f32")
        assert status["available"] and status["capabilities"] == {"cuda_capture"}, status
        module = runtime.load_module(artifact)
        calls = check_frozen(runtime, module) if kind == "frozen" else check_live(runtime, module)
        calls[-1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(calls[-1])
        for call in calls:
            call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
