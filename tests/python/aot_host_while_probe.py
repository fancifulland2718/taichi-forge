"""Source-free exact loop replay, gradients, and independent mutable instances."""

import json
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from aot_amgx_probe import read, write, near, raises


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        modules = [runtime.load_module(artifact) for _ in range(2)]
        frames = []
        for slot, module in enumerate(modules):
            arrays = {name: runtime.ndarray("f32", 5) for name in ("x", "y", "dx", "dy")}
            arrays.update(limit=runtime.ndarray("i32", 1), state=runtime.ndarray("i32", 2))
            x = [0.25 + slot + i / 2 for i in range(5)]
            write(arrays["x"], x)
            plans = [module.execution_plan(f"loop{budget}") for budget in (0, 3)]
            calls = [plan.prepare(module, {name: arrays[name] for name in plan.argument_names}) for plan in plans]
            frames.append((arrays, x, calls))
            for budget, call in zip((0, 3), calls):
                for limit in (0, 1, 7):
                    write(arrays["limit"], [limit], "i")
                    call()
                    count = min(limit, budget)
                    assert read(arrays["state"], "i") == [count, count + 1]
                    near(read(arrays["y"]), [a * a for a in x] if count else [-91] * 5, 1e-6)
                    near(read(arrays["dx"]), [count * 2 * a for a in x], 1e-6)
            # The inverted polarity is part of the wire contract. A huge
            # representable budget with a false condition must not be unrolled.
            doc = json.loads(module.execution_plan("loop3").to_json())
            doc["operations"][-1].update(continue_while_nonzero=False, max_iterations=0x7FFFFFFFFFFFFFFF)
            call = ExecutionPlan.from_json(json.dumps(doc)).prepare(module, arrays)
            write(arrays["limit"], [1], "i")
            call()
            assert read(arrays["state"], "i") == [0, 1]
            call.close()
        modules[0].close()
        raises(frames[0][2][1], "retired")
        arrays, x, calls = frames[1]
        write(arrays["limit"], [2], "i")
        calls[1]()
        near(read(arrays["dx"]), [4 * a for a in x], 1e-6)
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
