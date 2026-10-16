"""Stdlib consumer for selected sparse matmuls and downstream real gradients."""

import copy
import json
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_cusparselt_probe import half, read, write
from tests.python.aot_cutensor_probe import near, raises


def inputs(m, n, k, products, factor, shift):
    host = dict(a=[factor * (1 + (i + p) % 3) / 8 if (p + shift) % 4 < 2 else 0 for i in range(m) for p in range(k)])
    for index in range(products):
        host[f"b{index}"] = [((j * 3 + p + index) % 5 - 2) / 8 for j in range(n) for p in range(k)]
        host[f"c{index}"] = [(i % 7 - 3 + index) / 16 for i in range(m * n)]
    return host


def check(arrays, host, m, n, k, products):
    a = host["a"]
    loss = 0
    for index in range(products):
        b, c = host[f"b{index}"], host[f"c{index}"]
        d = [
            half(max(0, 0.75 * sum(a[i * k + p] * b[j * k + p] for p in range(k)) + 0.25 * c[i * n + j]))
            for i in range(m)
            for j in range(n)
        ]
        da = [
            0.75 * sum(d[i * n + j] * b[j * k + p] for j in range(n)) if a[i * k + p] else 0
            for i in range(m)
            for p in range(k)
        ]
        db = [0.75 * sum(d[i * n + j] * a[i * k + p] for i in range(m)) for j in range(n) for p in range(k)]
        near(read(arrays[f"d{index}"]), d)
        for name, values in (("output", d), ("doutput", d), ("da", da), ("db", db), ("dc", [0.25 * x for x in d])):
            near(read(arrays[f"{name}{index}"], "f"), values)
        loss += 0.5 * sum(value * value for value in d)
    near(read(arrays["loss"], "f"), [loss])


def main(mode, artifact, adapter, vendor, products):
    products = int(products)
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("cusparselt", adapter_path=adapter, library_path=vendor)
        module = runtime.load_module(artifact)
        frames = []
        for name in ("ordinary", "baseline", "vendor", "separate"):
            plan = module.execution_plan(name)
            document = json.loads(plan.to_json())
            parameters = document["resources"][0]["parameters"]
            m, n, k = (parameters[key] for key in ("m", "n", "k"))
            shapes = dict(a=(m, k), loss=(1,), dloss=(1,))
            for index in range(products):
                for key, shape in (
                    ("b", (n, k)),
                    ("c", (m, n)),
                    ("d", (m, n)),
                    ("output", (m, n)),
                    ("doutput", (m, n)),
                    ("da", (m, k)),
                    ("db", (n, k)),
                    ("dc", (m, n)),
                ):
                    shapes[f"{key}{index}"] = shape
            arrays = {
                key: runtime.ndarray("f16" if key == "a" or key[0] in "bcd" and key[1:].isdigit() else "f32", shape)
                for key, shape in shapes.items()
            }
            write(arrays["d0"], [-99] * (m * n))
            run = plan.prepare(module, arrays)
            assert read(arrays["d0"]) == [-99] * (m * n)
            assert run.info["execution_kind"] == "cuda_graph"
            assert run.info["workspace_bytes"] == sum(parameters["expected_resources"])
            assert run.info["resource_bytes"] == 0
            frames.append((arrays, run))
            for key in ("a", "c1", "d1"):
                raises(lambda: plan.prepare(module, dict(arrays, d0=arrays[key])))
            for bad in (runtime.ndarray("i16", (m, k)), runtime.ndarray("f16", (1, m * k))):
                raises(lambda: plan.prepare(module, dict(arrays, a=bad)))
                bad.close()
            # Resource costs belong to the selected plan, not a consumer search.
            for index in range(3):
                wrong = copy.deepcopy(document)
                wrong["resources"][0]["parameters"]["expected_resources"][index] += 256
                raises(lambda: ExecutionPlan.from_json(json.dumps(wrong)).prepare(module, arrays))
            wrong = copy.deepcopy(document)
            # The vendor resolves -1 to a real algorithm. Exact restoration must reject that drift.
            wrong["resources"][0]["parameters"]["algorithm_id"] = -1
            raises(lambda: ExecutionPlan.from_json(json.dumps(wrong)).prepare(module, arrays))
            if name == "separate":
                shared = dict(arrays, c0=arrays["d0"], b1=arrays["b0"])
                frames.append((shared, plan.prepare(module, shared)))

        for factor, shift in ((1, 0), (-0.5, 2)):
            original = inputs(m, n, k, products, factor, shift)
            for arrays, run in frames:
                host = dict(original)
                if arrays["b1"] is arrays["b0"]:
                    host["b1"] = host["b0"]
                for name, values in host.items():
                    write(arrays[name], values)
                run()
                check(arrays, host, m, n, k, products)
        for _, run in frames[:-1]:
            run()
            run.close()
        _, run = frames[-1]
        run()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(run)
        run.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
