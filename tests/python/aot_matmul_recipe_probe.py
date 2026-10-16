"""Check complete selected matmul recipes using a scalar no-DSL reference."""

import json
import math
import sys

from taichi_forge_runtime.aot import Runtime
from aot_cublaslt_probe import near, raises, read, write


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
        frames = []
        for name in ("baseline", "packed_fused", "packed_separate", "ordinary"):
            plan = module.execution_plan(name)
            document = json.loads(plan.to_json())
            p = next(r for r in document["resources"] if r["kind"] == "native")["parameters"]
            m, n, k, batch = (p[key] for key in ("m", "n", "k", "batch_count"))

            def shape(rows, columns):
                return (rows, columns) if batch == 1 else (batch, rows, columns)

            shapes = dict(
                a=shape(k, m),
                b=shape(n, k),
                seed=shape(m, n),
                output=shape(m, n),
                doutput=shape(m, n),
                loss=(1,),
                dloss=(1,),
            )
            arrays = {key: runtime.ndarray("f32", value) for key, value in shapes.items()}
            write(arrays["output"], [-99] * (batch * m * n))
            run = plan.prepare(module, arrays)
            assert read(arrays["output"]) == [-99] * (batch * m * n)
            assert run.info["execution_kind"] == "cuda_graph"
            scratch_bytes = sum(math.prod(r["shape"]) * 4 for r in document["resources"] if r["kind"] == "ndarray")
            assert run.info["resource_bytes"] == scratch_bytes
            assert run.info["workspace_bytes"] == p["algorithm"]["workspace_bytes"]
            # These must fail even when the vendor sees only private packed inputs.
            for bad in (runtime.ndarray("i32", shapes["a"]), runtime.ndarray("f32", (*shapes["a"][:-2], 1, k * m))):
                raises(lambda: plan.prepare(module, dict(arrays, a=bad)))
                bad.close()
            if m == n == k:
                raises(lambda: plan.prepare(module, dict(arrays, output=arrays["a"])))
                raises(lambda: plan.prepare(module, dict(arrays, output=arrays["b"])))
            frames.append((arrays, run))

        for factor in (1, -0.5):
            a = [factor * (i % 11 - 5) / 16 for i in range(batch * k * m)]
            b = [(i % 7 - 3) / 8 for i in range(batch * n * k)]
            expected = []
            for z in range(batch):
                az, bz = a[z * k * m : (z + 1) * k * m], b[z * n * k : (z + 1) * n * k]
                expected.extend(
                    max(0.75 * sum(az[p * m + i] * bz[j * k + p] for p in range(k)) + 0.125, 0)
                    for i in range(m)
                    for j in range(n)
                )
            for arrays, run in frames:
                write(arrays["a"], a)
                write(arrays["b"], b)
                write(arrays["seed"], [0.5] * (batch * m * n))
                run()
                near(read(arrays["output"]), expected)
                near(read(arrays["doutput"]), expected)
                near(read(arrays["loss"]), [0.5 * sum(value * value for value in expected)])
        frames[0][1].close()
        frames[1][1]()
        near(read(frames[1][0]["doutput"]), expected)
        frames[1][1].close()
        frames[3][1]()
        near(read(frames[3][0]["output"]), expected)
        frames[3][1].close()
        frames[2][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(frames[2][1])
        frames[2][1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
