"""Standalone/borrowed selected sparse solves with scalar forward and adjoint references."""

import json
import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cudss_probe import near, raises, read, write


def solve(rhs, scale):
    n = len(rhs)
    diagonal = [scale * (4 + 2 * i / (n - 1)) for i in range(n)]
    result = list(rhs)
    for i in range(1, n):
        ratio = -scale / diagonal[i - 1]
        diagonal[i] -= ratio * -scale
        result[i] -= ratio * result[i - 1]
    result[-1] /= diagonal[-1]
    for i in range(n - 2, -1, -1):
        result[i] = (result[i] + scale * result[i + 1]) / diagonal[i]
    return result


def main(mode, artifact, adapter, vendor):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("cudss", adapter_path=adapter, library_path=vendor)
        module = runtime.load_module(artifact)
        frames = []
        for name in ("ordinary", "baseline", "selected"):
            plan = module.execution_plan(name)
            p = next(r for r in json.loads(plan.to_json())["resources"] if r["kind"] == "native")["parameters"]
            n, fixed = p["rows"], p["phase"] == "solve"
            arrays = {
                key: runtime.ndarray("f32", 1 if key in ("loss", "dloss") else n)
                for key in plan.argument_names
                if key != "values"
            }
            arrays.update({f"db{i}": runtime.ndarray("f32", n) for i in range(3)})
            if not fixed:
                arrays["values"] = runtime.ndarray("f32", 3 * n - 2)
            for key in ("x0", "x1", "x2", "loss"):
                write(arrays[key], [-99] * (1 if key == "loss" else n))
            args = {key: arrays[key] for key in plan.argument_names}
            run = plan.prepare(module, args)
            assert read(arrays["x0"]) == [-99] * n and read(arrays["loss"]) == [-99]
            assert run.info["execution_kind"] == "cuda_graph"
            adjoint_plan = module.execution_plan(name + "_adjoint")
            backward = adjoint_plan.prepare(module, {key: arrays[key] for key in adjoint_plan.argument_names})
            assert run.info["workspace_bytes"] > 0  # The resource owns real cuDSS vendor storage.
            raises(lambda: plan.prepare(module, dict(args, b1=arrays["b0"])))
            raises(lambda: plan.prepare(module, dict(args, x1=arrays["x0"])))
            raises(lambda: plan.prepare(module, dict(args, x0=arrays["b0"])))
            for bad in (runtime.ndarray("i32", n), runtime.ndarray("f32", (1, n))):
                raises(lambda: plan.prepare(module, dict(args, b0=bad)))
                bad.close()
            frames.append((arrays, run, backward))
        for step in range(2):
            scale = 1 if fixed else 0.75 + step
            rhs = [[(i + 1 + step) * (j + 1) / 4 for i in range(n)] for j in range(3)]
            expected = [solve(b, scale) for b in rhs]
            adjoints = [solve(x, scale) for x in expected]
            for arrays, run, backward in frames:
                for i in range(3):
                    write(arrays[f"b{i}"], rhs[i])
                if not fixed:
                    values = []
                    for i in range(n):
                        if i:
                            values.append(-scale)
                        values.append(scale * (4 + 2 * i / (n - 1)))
                        if i + 1 < n:
                            values.append(-scale)
                    write(arrays["values"], values)
                run()
                backward()
                for i in range(3):
                    near(read(arrays[f"x{i}"]), expected[i])
                    near(read(arrays[f"dx{i}"]), expected[i])
                    near(read(arrays[f"db{i}"]), adjoints[i])
                near(read(arrays["loss"]), [0.5 * sum(x * x for column in expected for x in column)])
        frames[0][1]()
        frames[0][1].close()
        frames[0][2].close()
        for _, run, backward in frames[1:]:
            run()
            backward()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for _, run, backward in frames[1:]:
            raises(run)
            raises(backward)
            run.close()
            backward.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
