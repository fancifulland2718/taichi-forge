"""Replay selected sparse programs after the producer and search owners close."""

import sys
from taichi_forge_runtime.aot import Runtime
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
        module = runtime.load_module(artifact)
        cases = []
        for name in (
            "row_streamed",
            "tree_direct",
            "pattern_preprocessed",
            "legacy_unprepared",
            "legacy_prepared",
            "legacy_updated",
            "empty",
        ):
            if name == "pattern_preprocessed":
                try:
                    module.execution_plan(name)
                except KeyError:
                    continue
            for frame_index in range(2):
                shapes = dict(input=(5, 8), product=(7, 8), output=(7, 8))
                if name == "empty":
                    del shapes["product"]
                if name.startswith("legacy"):
                    shapes.update(vector=(5,), result=(7,))
                arrays = {n: runtime.ndarray("f32", shape) for n, shape in shapes.items()}
                call = module.prepare_execution(name, arrays)
                cases.append((name, frame_index, arrays, call))
        for step in range(2):
            for name, index, arrays, call in cases:
                x = [(i - 4 + index + step) / 8 for i in range(40)]
                write(arrays["input"], x)
                if name.startswith("legacy"):
                    write(arrays["vector"], [1, -2, 3, -4, 5])
                call()
                if name == "empty":
                    near(read(arrays["output"]), [0] * 56)
                    continue
                scale = 2 if name == "legacy_updated" else 1
                a = [
                    [scale * ((r + 1) * 0.125 + c * 0.0625) if c in (r % 5, (r + 2) % 5) else 0 for c in range(5)]
                    for r in range(7)
                ]
                product = [sum(a[r][k] * x[k * 8 + j] for k in range(5)) for r in range(7) for j in range(8)]
                near(read(arrays["product"]), product)
                near(read(arrays["output"]), product if name.startswith("legacy") else [2 * v + 0.25 for v in product])
                if name.startswith("legacy"):
                    near(
                        read(arrays["result"]),
                        [sum(a[r][k] * [1, -2, 3, -4, 5][k] for k in range(5)) for r in range(7)],
                    )
        cases[0][3].close()
        cases[1][3]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        failure(cases[1][3])
        for _, _, _, call in cases[1:]:
            call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
