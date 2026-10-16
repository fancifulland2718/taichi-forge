"""Independent selected contraction consumer with scalar forward/adjoint checks."""

import copy
import json
import math
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from aot_cutensor_probe import near, raises, read, write


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
        adjoint = module.execution_plan("adjoint")
        frames = []
        for name in ("baseline", "packed_vendor", "packed_separate", "ordinary"):
            plan = module.execution_plan(name)
            document = json.loads(plan.to_json())
            public = {
                c["array"]: tuple(c["exact_shape"]) for c in document["constraints"] if c["kind"] == "array_layout"
            }
            a_shape, b_shape, out_shape = (public[key] for key in ("a", "b", "output"))
            k, m = a_shape[-2:]
            n = b_shape[-1]
            batch = a_shape[0] if len(a_shape) == 3 else 1
            shapes = dict(
                a=a_shape,
                b=b_shape,
                c=out_shape,
                seed=out_shape,
                output=out_shape,
                doutput=out_shape,
                da=a_shape,
                db=b_shape,
                loss=(1,),
                dloss=(1,),
            )
            arrays = {key: runtime.ndarray("f32", shape) for key, shape in shapes.items()}
            bindings = {key: value for key, value in arrays.items() if key not in ("da", "db")}
            adjoint_bindings = {key: arrays[key] for key in ("a", "b", "doutput", "da", "db")}
            write(arrays["output"], [-99] * math.prod(out_shape))
            run = plan.prepare(module, bindings)
            assert read(arrays["output"]) == [-99] * math.prod(out_shape)
            assert run.info["execution_kind"] == "cuda_graph"
            resource = next(r for r in document["resources"] if r["kind"] == "native")
            assert run.info["workspace_bytes"] == resource["parameters"]["expected_workspace_bytes"]
            assert run.info["resource_bytes"] == sum(
                math.prod(r["shape"]) * 4 for r in document["resources"] if r["kind"] == "ndarray"
            )
            backward = adjoint.prepare(module, adjoint_bindings)
            for bad in (runtime.ndarray("i32", a_shape), runtime.ndarray("f32", (*a_shape[:-2], 1, k * m))):
                raises(lambda: plan.prepare(module, dict(bindings, a=bad)))
                bad.close()
            if m == n == k:
                raises(lambda: plan.prepare(module, dict(bindings, output=arrays["a"])))
                raises(lambda: plan.prepare(module, dict(bindings, output=arrays["b"])))
            wrong = copy.deepcopy(document)
            next(r for r in wrong["resources"] if r["kind"] == "native")["parameters"][
                "expected_workspace_bytes"
            ] += 256
            raises(lambda: ExecutionPlan.from_json(json.dumps(wrong)).prepare(module, bindings))
            frames.append((arrays, run, backward))
            if name == "packed_separate":
                # C/D aliasing is legal even with a private intermediate product.
                shared = dict(arrays, c=arrays["output"])
                frames.append(
                    (
                        shared,
                        plan.prepare(module, dict(bindings, c=arrays["output"])),
                        adjoint.prepare(module, adjoint_bindings),
                    )
                )

        for factor in (1, -0.5):
            a = [factor * (i % 11 - 5) / 16 for i in range(batch * k * m)]
            b = [(i % 7 - 3) / 8 for i in range(batch * k * n)]
            output, da, db = [], [], []
            for z in range(batch):
                az, bz = a[z * k * m : (z + 1) * k * m], b[z * k * n : (z + 1) * k * n]
                dz = [
                    max(0.75 * sum(az[p * m + i] * bz[p * n + j] for p in range(k)) + 0.125, 0)
                    for i in range(m)
                    for j in range(n)
                ]
                output.extend(dz)
                da.extend(
                    0.75 * sum(dz[i * n + j] * bz[p * n + j] for j in range(n)) for p in range(k) for i in range(m)
                )
                db.extend(
                    0.75 * sum(az[p * m + i] * dz[i * n + j] for i in range(m)) for p in range(k) for j in range(n)
                )
            for arrays, run, backward in frames:
                write(arrays["a"], a)
                write(arrays["b"], b)
                write(arrays["seed"], [0.5] * (batch * m * n))
                run()
                backward()
                for key, expected in (
                    ("output", output),
                    ("doutput", output),
                    ("da", da),
                    ("db", db),
                    ("loss", [0.5 * sum(value * value for value in output)]),
                ):
                    near(read(arrays[key]), expected)
        for _, run, backward in frames[:-1]:
            run()
            backward()
            run.close()
            backward.close()
        arrays, run, backward = frames[-1]
        run()
        backward()
        near(read(arrays["db"]), db)
        run()
        backward()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(run)
        raises(backward)
        run.close()
        backward.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
