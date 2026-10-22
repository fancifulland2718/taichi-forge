"""Independent sessions reconstructed from an automatically exported AmgX Graph."""

import sys
import json
from copy import deepcopy
from pathlib import Path

from taichi_forge_runtime.aot import Runtime
from aot_amgx_probe import N, NNZ, coefficients, reference, read, write, near, raises


def pcg_step(rhs, initial):
    def multiply(vector):
        return [4 * x - (vector[i - 1] if i else 0) - (vector[i + 1] if i + 1 < N else 0) for i, x in enumerate(vector)]

    residual = [b - ax for b, ax in zip(rhs, multiply(initial))]
    alpha = sum(r * r for r in residual) / sum(r * ar for r, ar in zip(residual, multiply(residual)))
    return [x + alpha * r for x, r in zip(initial, residual)]


def check_binding_states(runtime, module, artifact):
    arrays = {
        name: runtime.ndarray("f64", NNZ if name == "initial" else N)
        for name in ("initial", "a_rhs", "b_rhs", "a_solution", "b_solution")
    }
    rhs_a, rhs_b = [1 + i % 3 for i in range(N)], [2 - i % 2 for i in range(N)]
    write(arrays["initial"], coefficients(4), "d")
    write(arrays["a_rhs"], rhs_a, "d")
    write(arrays["b_rhs"], rhs_b, "d")
    write(arrays["a_solution"], [0.25] * N, "d")
    write(arrays["b_solution"], [-0.5] * N, "d")
    call = module.prepare_execution("binding_states", arrays)
    # Reusing a binding identity with different physical ports fails in prepare.
    from taichi_forge_runtime.aot import ExecutionPlan

    doc = json.loads(module.execution_plan("binding_states").to_json())
    doc["operations"][1]["parameters"]["binding_id"] = doc["operations"][0]["parameters"]["binding_id"]
    raises(lambda: ExecutionPlan.from_json(json.dumps(doc)).prepare(module, arrays), "inconsistent ports")
    a = [0.25] * N
    for step in range(2):
        first = pcg_step(rhs_a, a)
        b = pcg_step(rhs_b, [-0.5] * N if not step else first)
        a = pcg_step(rhs_a, b)
        call()
        near(read(arrays["a_solution"], "d"), a, 1e-9)
        near(read(arrays["b_solution"], "d"), b, 1e-9)
        if not step:
            jit_a, jit_b = json.loads(Path(artifact).with_suffix(".json").read_text(encoding="utf8"))
            near(a, jit_a, 1e-9)
            near(b, jit_b, 1e-9)
        write(arrays["a_solution"], [float("nan")] * N, "d")
        write(arrays["b_solution"], [float("nan")] * N, "d")
    call.close()


def main(mode, artifact, adapter, vendor, dtype):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    code = "f" if dtype == "f32" else "d"
    tolerance = 3e-5 if dtype == "f32" else 1e-9
    with runtime:
        runtime.configure_provider("amgx", adapter_path=adapter, library_path=vendor)
        modules = (runtime.load_module(artifact), runtime.load_module(artifact))
        frames = []
        for module in modules:
            arrays = {
                name: runtime.ndarray(dtype, size)
                for name, size in (
                    ("initial", NNZ),
                    ("values", NNZ),
                    ("diagonal", 1),
                    ("rhs", N),
                    ("seed", N),
                    ("solution", N),
                    ("output", N),
                )
            }
            write(arrays["initial"], coefficients(4), code)
            write(arrays["solution"], [0] * N, code)
            write(arrays["output"], [91] * N, code)
            call = module.prepare_execution("step", arrays)
            assert read(arrays["output"], code) == [91] * N
            assert call.info["execution_kind"] == "cuda_mixed"
            frames.append((arrays, call))
        for step, diagonal in enumerate((4, 6, 3)):
            for slot, (arrays, call) in enumerate(frames):
                seed = [float(step + slot + i % 3) / 4 for i in range(N)]
                write(arrays["seed"], seed, code)
                write(arrays["diagonal"], [diagonal], code)
                if step:
                    write(arrays["solution"], [float("nan")] * N, code)
                    write(arrays["initial"], [float("nan")] * NNZ, code)  # Setup runs once.
                call()
                near(
                    read(arrays["output"], code),
                    [2 * x + 1 for x in reference(diagonal, [2 * x + 0.125 for x in seed])],
                    tolerance,
                )
        modules[0].close()
        raises(frames[0][1], "retired")
        frames[1][1]()
        near(
            read(frames[1][0]["output"], code),
            [2 * x + 1 for x in reference(diagonal, [2 * x + 0.125 for x in seed])],
            tolerance,
        )
        if dtype == "f64":
            check_binding_states(runtime, modules[1], artifact)
        else:
            arrays = frames[1][0]
            arrays["choice"] = runtime.ndarray("i32", 1)
            write(arrays["initial"], coefficients(4), code)
            conditional = modules[1].prepare_execution("conditional", arrays)
            # A solve in only one branch cannot establish a retained guess at
            # the join. Both branches can; setup after the join invalidates it.
            from taichi_forge_runtime.aot import ExecutionPlan

            document = json.loads(modules[1].execution_plan("conditional").to_json())
            captured = deepcopy(document)
            captured["operations"][1]["lowering"]["kind"] = "cuda_conditional"
            raises(
                lambda: ExecutionPlan.from_json(json.dumps(captured)).prepare(modules[1], arrays),
                "GPU control cannot contain host-invoked",
            )
            retained = deepcopy(document["operations"][1]["bodies"][0][-1])
            retained["parameters"].update(initial_guess="retained", binding_id="retained_suffix")
            document["operations"].insert(2, retained)
            raises(
                lambda: ExecutionPlan.from_json(json.dumps(document)).prepare(modules[1], arrays),
                "preceding solve",
            )
            branch = document["operations"][1]
            branch["has_default"] = True
            branch["bodies"].append(deepcopy(branch["bodies"][0]))
            joined = ExecutionPlan.from_json(json.dumps(document)).prepare(modules[1], arrays)
            for selected in (0, 1):
                write(arrays["choice"], [selected], "i")
                write(arrays["seed"], [1] * N, code)
                write(arrays["solution"], [0] * N, code)
                write(arrays["diagonal"], [4], code)
                joined()
                near(read(arrays["output"], code), [2 * x + 1 for x in reference(4, [2.125] * N)], tolerance)
            joined.close()
            document["operations"].insert(2, deepcopy(document["initialization"][0]))
            raises(
                lambda: ExecutionPlan.from_json(json.dumps(document)).prepare(modules[1], arrays),
                "preceding solve",
            )
            for selected, diagonal in ((0, 4), (1, 4), (0, 6), (1, 6)):
                write(arrays["choice"], [selected], "i")
                write(arrays["seed"], [1] * N, code)
                write(arrays["solution"], [13] * N, code)
                write(arrays["diagonal"], [diagonal], code)
                conditional()
                expected = [2 * x + 1 for x in reference(diagonal, [2.125] * N)] if selected else [27] * N
                near(read(arrays["output"], code), expected, tolerance)
            conditional.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
