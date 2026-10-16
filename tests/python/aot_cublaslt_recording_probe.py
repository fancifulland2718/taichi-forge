"""Consume a complete automatically exported matmul Graph after producer close."""

import json
import sys

from taichi_forge_runtime.aot import Runtime
from aot_cublaslt_probe import near, read, write


def main(mode, artifact, adapter, vendor):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("cublaslt", adapter_path=adapter, library_path=vendor)
        module = runtime.load_module(artifact)
        plan = module.execution_plan("matmul")
        p = json.loads(plan.to_json())["resources"][0]["parameters"]
        m, n, k, batch = (p[key] for key in ("m", "n", "k", "batch_count"))

        def shape(rows, columns):
            return (rows, columns) if batch == 1 else (batch, rows, columns)

        shapes = dict(a=shape(k, m), b=shape(n, k), seed=shape(m, n), output=shape(m, n), result=shape(m, n))
        arrays = {key: runtime.ndarray("f32", value) for key, value in shapes.items()}
        write(arrays["result"], [-99] * (batch * m * n))
        run = plan.prepare(module, arrays)
        assert read(arrays["result"]) == [-99] * (batch * m * n)
        assert run.info["execution_kind"] == "cuda_graph"
        assert run.info["workspace_bytes"] == p["algorithm"]["workspace_bytes"]
        for factor in (1, -0.5):
            a = [factor * (i % 11 - 5) / 16 for i in range(batch * k * m)]
            b = [(i % 7 - 3) / 8 for i in range(batch * n * k)]
            write(arrays["a"], a)
            write(arrays["b"], b)
            write(arrays["seed"], [0.5] * (batch * m * n))
            expected = []
            for z in range(batch):
                az, bz = a[z * k * m : (z + 1) * k * m], b[z * n * k : (z + 1) * n * k]
                product = [
                    0.75 * sum(az[p * m + i] * bz[j * k + p] for p in range(k)) for i in range(m) for j in range(n)
                ]
                result = [1.0] * (m * n)
                for _ in range(2):
                    result = [value + 0.25 * old for value, old in zip(product, result)]
                    if p["epilogue"] == "relu":
                        result = [max(value, 0) for value in result]
                expected.extend(3 * value for value in result)
            run()
            near(read(arrays["result"]), expected)
        run()
        run.close()
        with plan.prepare(module, arrays) as replacement:
            replacement()
            near(read(arrays["result"]), expected)
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
