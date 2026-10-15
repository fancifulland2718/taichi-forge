"""Source-free conditional cuDSS reference and prepared-owner checks."""

import json
import struct
import sys

from taichi_forge_runtime.aot import Runtime

if __package__:
    from .aot_cudss_probe import near, read, raises, solve, write
else:
    from aot_cudss_probe import near, read, raises, solve, write


def shapes(phase):
    floats = {
        name: 4
        for name in ("source", "source2", "rhs", "rhs2", "solution", "solution2", "adjoint_rhs", "adjoint", "gradient")
    }
    floats["loss"] = 1
    if phase != "solve":
        floats.update(base_values=10, values=10)
    ints = {
        name: 1
        for name in ("counter", "checks", "active", "target", "reverse_count", "reverse_checks", "reverse_active")
    }
    ints["terminal"] = 6
    return floats, ints


def reference(source, source2, limit, phase, budget=3):
    rounds = min(limit, budget)
    x, x2 = source, source2
    for iteration in range(rounds):
        scale = 1 if phase == "solve" else iteration + 2
        x, x2 = solve(x, scale), solve(x2, scale)
    gradient = x
    for iteration in reversed(range(rounds)):
        gradient = solve(gradient, 1 if phase == "solve" else iteration + 2)
    return dict(solution=x, solution2=x2, gradient=gradient, loss=[sum(v * v for v in x) / 2]), [
        rounds,
        rounds + 1,
        int(rounds < limit),
        0,
        rounds + 1,
        0,
    ]


def main(mode, artifact, adapter, vendor, phase):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("cudss", adapter_path=adapter, library_path=vendor)
        module = runtime.load_module(artifact)
        floats, ints = shapes(phase)
        prepared = []
        for name in ("step", "step", "zero", "nested"):
            arrays = {n: runtime.ndarray("f32", count) for n, count in floats.items()}
            arrays.update({n: runtime.ndarray("i32", count) for n, count in ints.items()})
            if phase != "solve":
                write(arrays["base_values"], [4, -1, -1, 4, -1, -1, 4, -1, -1, 3])
            write(arrays["solution"], [-99] * 4)
            call = module.prepare_execution(name, arrays)
            assert call.info["execution_kind"] == "cuda_graph"
            assert read(arrays["solution"]) == [-99] * 4
            prepared.append((name, arrays, call))
        for iteration, limit in enumerate((0, 2, 9, 1, 0)):
            for index, (name, arrays, call) in enumerate(prepared):
                source = [(i - 1.5) * (index + 1) / 4 + iteration / 8 for i in range(4)]
                source2 = [-2 * v + 0.125 for v in source]
                write(arrays["source"], source)
                write(arrays["source2"], source2)
                write(arrays["target"], [limit], "i")
                call()
                expected, terminal = reference(source, source2, limit, phase, 0 if name == "zero" else 3)
                for output, values in expected.items():
                    near(read(arrays[output]), values)
                assert list(struct.unpack("<6i", arrays["terminal"].read())) == terminal
        # The artifact's one seed resource serves both forward and reverse
        # groups; each prepared frame owns independent mutable vendor state.
        description = json.loads(module.execution_plan("step").to_json())
        assert len([r for r in description["resources"] if r.get("operation") == "plan_f32"]) == 1
        prepared[0][-1]()
        prepared[0][-1].close()
        raises(prepared[0][-1])
        prepared[1][-1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for _, _, call in prepared[1:]:
            raises(call)
            call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
