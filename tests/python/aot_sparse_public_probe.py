"""Public CSR values can change without sharing private triangular analysis."""

import array
import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_dot_probe import near, read, write, raises


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        module = runtime.load_module(artifact)
        offsets, indices = runtime.ndarray("i32", 3), runtime.ndarray("i32", 3)
        offsets.write(array.array("i", [0, 1, 3]))
        indices.write(array.array("i", [0, 0, 1]))
        values = [runtime.ndarray("f32", 3) for _ in range(2)]
        matrices = []
        for coefficients in values:
            write(coefficients, [2, 1, 4])
            matrices.append(
                runtime.native_resource(
                    "cusparse",
                    "csr_f32",
                    parameters=dict(rows=2, columns=2),
                    bindings=dict(row_offsets=offsets, column_indices=indices, values=coefficients),
                )
            )
        frames = []
        for name in ("first", "again"):
            arrays = {name: runtime.ndarray("f32", 2) for name in ("x", "product", "recovered", "other")}
            prepared = module.prepare_execution(name, dict(arrays, left=matrices[0], right=matrices[1]))
            frames.append((arrays, prepared))
        for scale in (1, 3, 0.5):
            write(values[0], [2 * scale, scale, 4 * scale])
            write(values[1], [5, -2, 3])
            for arrays, call in frames:
                write(arrays["x"], [3, -2])
                call()
                near(read(arrays["product"]), [6 * scale, -5 * scale])
                near(read(arrays["recovered"]), [3, -2])
                near(read(arrays["other"]), [15, -12])
        frames[0][1].close()
        frames[1][1]()
        near(read(frames[1][0]["recovered"]), [3, -2])
        matrices[0].close()
        raises(frames[1][1])
        frames[1][1].close()
        matrices[1].close()
        for arrays, _ in frames:
            for value in arrays.values():
                value.close()
        for value in (*values, offsets, indices):
            value.close()
        offsets, indices = runtime.ndarray("i32", 2), runtime.ndarray("i32", 1)
        offsets.write(array.array("i", [0, 1]))
        indices.write(array.array("i", [0]))
        values, x, y = (runtime.ndarray("f32", size) for size in (4, 2, 2))
        write(values, [2, 0, 1, 4])
        block = runtime.native_resource(
            "cusparse",
            "bsr_f32",
            parameters=dict(block_rows=1, block_columns=1, block_size=2),
            bindings=dict(row_offsets=offsets, column_indices=indices, values=values),
        )
        with module.prepare_execution("bsr", dict(block=block, x=x, y=y)) as call:
            write(x, [3, -2])
            call()
            near(read(y), [6, -5])
            write(values, [4, 0, 2, 8])
            call()
            near(read(y), [12, -10])
        block.close()
        for value in (offsets, indices, values, x, y):
            value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
