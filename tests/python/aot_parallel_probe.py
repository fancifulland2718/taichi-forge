"""Run CUDA fork/join artifacts with stdlib-only or a borrowed JIT owner."""

import sys

from taichi_forge_runtime.aot import AotError, Runtime, Scalar
from tests.python.aot_cutensor_probe import near, raises
from tests.python.aot_graph_recording_probe import read, write


def main(mode, artifact, case):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        module = runtime.load_module(artifact)
        calls, frames = [], []
        if case == "selected":
            count, names = 17, ("a0", "b0", "output")
        elif case == "gradient":
            count, names = 34, ("source", "output", "gradient")
        else:
            branches, count = int(case), 257
            names = ("source", "ready", "result") + tuple(f"lane_{i}_{j}" for i in range(branches) for j in range(2))
        for frame in range(2):
            arrays = {name: runtime.ndarray("f32", count) for name in names}
            bindings = dict(arrays)
            if case == "gradient":
                bindings.update(branch_0=Scalar("i32", 0), branch_1=Scalar("i32", 1))
            elif case != "selected":
                bindings["scale"] = Scalar("f32", 1.0005 if frame == 0 else 0.999)
            call = module.prepare_execution("step", bindings)
            assert call.info["execution_kind"] == "cuda_graph", call.info
            assert call.info["dispatch_count"] == (
                9 if case == "selected" else 6 if case == "gradient" else 1 + 3 * branches
            )
            calls.append(call)
            frames.append(arrays)
        if case == "selected":
            try:
                module.prepare_execution("step", dict(frames[0], b0=frames[0]["a0"]))
            except (AotError, ValueError):
                pass
            else:
                raise AssertionError("Cross-branch aliases were accepted")
            baseline_arrays = {name: runtime.ndarray("f32", count) for name in names}
            baseline = module.prepare_execution("baseline", baseline_arrays)
        for iteration in range(3):
            for frame, (call, arrays) in enumerate(zip(calls, frames)):
                values = [(i - 8) / 32 + iteration + frame for i in range(count)]
                if case == "selected":
                    for current in (arrays, baseline_arrays):
                        write(current["a0"], values)
                        write(current["b0"], [2 * x for x in values])
                    baseline()
                else:
                    write(arrays["source"], values)
                call()
                if case == "selected":
                    near(read(arrays["output"]), [3 * x - 4 for x in values])
                    near(read(arrays["output"]), read(baseline_arrays["output"]))
                elif case == "gradient":
                    near(read(arrays["output"]), [x * x for x in values])
                    near(read(arrays["gradient"]), [2 * x * (1 + i // 17) for i, x in enumerate(values)])
                else:
                    scale = 1.0005 if frame == 0 else 0.999
                    expected = [x + 1 for x in values]
                    for _ in range(32):
                        expected = [x * scale + 0.001 for x in expected]
                    actual = read(arrays["result"])
                    assert all(abs(a - x * branches) <= 1e-4 * max(1, abs(a)) for a, x in zip(actual, expected))
        calls[0]()
        calls[0].close()
        calls[1]()
        if case == "selected":
            baseline.close()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(calls[1])
        calls[1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
