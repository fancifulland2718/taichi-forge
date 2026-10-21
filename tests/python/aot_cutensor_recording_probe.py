"""Two ordinary contraction instances, after producer plan and Graph close."""

import sys

from taichi_forge_runtime.aot import Runtime
from aot_cutensor_probe import near, read, write


def main(mode, artifact, adapter, vendor):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("cutensor", adapter_path=adapter, library_path=vendor)
        module = runtime.load_module(artifact)
        plan = module.execution_plan("ordinary_contraction")
        frames = []
        for factor in (1.0, -0.5):
            a, b, output = (runtime.ndarray("f32", shape) for shape in ((5, 7), (7, 3), (5, 3)))
            a_host = [factor * (i % 11 - 5) / 8 for i in range(35)]
            b_host = [(i % 7 - 3) / 4 for i in range(21)]
            product = [
                1.25 * sum(a_host[i * 7 + p] * b_host[p * 3 + j] for p in range(7)) for i in range(5) for j in range(3)
            ]
            write(a, a_host)
            write(b, b_host)
            write(output, [0.5] * 15)
            prepared = plan.prepare(module, dict(a=a, b=b, c=output, d=output))
            near(read(output), [0.5] * 15)
            frames.append((prepared, (a, b, output), product))
        expected = [[0.5] * 15 for _ in frames]
        for index in (0, 1, 0):
            prepared, arrays, product = frames[index]
            prepared()
            expected[index] = [value + 0.25 * previous for value, previous in zip(product, expected[index])]
            near(read(arrays[2]), expected[index])
        frames[0][0].close()
        prepared, arrays, product = frames[1]
        prepared()
        near(read(arrays[2]), [value + 0.25 * previous for value, previous in zip(product, expected[1])])
        prepared.close()
        module.close()
        for _, arrays, _ in frames:
            for value in arrays:
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
