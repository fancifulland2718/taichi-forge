"""Tiny counter oracle; large budgets terminate early without large work."""

import json
import statistics
import struct
import sys
import time


def cases(budget):
    return [(0, 2), (2, 3), (66, 0), (1, 66), (0, 1)] if budget == 65 else [(0, 2), (2, 3), (1, 2), (0, 1)]


def expected(values, limits, budget, regions):
    outer, inner = (min(value, budget) for value in limits)
    total = outer * inner * regions
    return (
        [x * x * (total + 1) for x in values],
        [2 * x * (total + 1) for x in values],
        [outer, total],
    )


def main(mode, artifact, budget, regions, arch="cuda", comparison="gradient"):
    from taichi_forge_runtime.aot import Runtime

    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False, enable_fallback=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        begin = time.perf_counter()
        module = runtime.load_module(artifact)
        costs = dict(load_ms=(time.perf_counter() - begin) * 1000, prepare_ms=[], complete_ms=[])
        frames = []
        for scale, name in ((1, "step"), (2, "again")):
            arrays = {n: runtime.ndarray("f32", 4) for n in ("x", "y", "dx", "dy")}
            arrays.update({n: runtime.ndarray("i32", 2) for n in ("limits", "terminal")})
            values = [scale * x for x in (0.5, 1, 2, 3)]
            arrays["x"].write(struct.pack("<4f", *values))
            arrays["y"].write(struct.pack("<4f", *([-99] * 4)))
            begin = time.perf_counter()
            call = module.prepare_execution(name, arrays)
            costs["prepare_ms"].append((time.perf_counter() - begin) * 1000)
            if arch == "cuda":
                assert call.info["execution_kind"] == "cuda_graph", call.info
            assert struct.unpack("<4f", arrays["y"].read()) == (-99,) * 4
            frames.append((arrays, call, values))
        for limits in cases(budget):
            for arrays, call, values in frames:
                arrays["limits"].write(struct.pack("<2i", *limits))
                begin = time.perf_counter()
                call()
                runtime.wait()
                costs["complete_ms"].append((time.perf_counter() - begin) * 1000)
                y, dx, terminal = expected(values, limits, budget, regions)
                assert list(struct.unpack("<4f", arrays["y"].read())) == y
                assert list(struct.unpack("<4f", arrays["dx"].read())) == dx
                assert list(struct.unpack("<2i", arrays["terminal"].read())) == terminal
        frames[0][1].close()
        arrays, call, values = frames[1]
        arrays["limits"].write(struct.pack("<2i", 2, 1))
        samples = []
        for _ in range(16):
            begin = time.perf_counter()
            call()
            runtime.wait()
            samples.append((time.perf_counter() - begin) * 1e6)
        costs["gradient_warm_complete_us"] = statistics.median(samples)
        if comparison == "primal":
            begin = time.perf_counter()
            primal = module.prepare_execution("primal", arrays)
            costs["primal_prepare_ms"] = (time.perf_counter() - begin) * 1000
            primal()
            runtime.wait()
            samples = []
            for _ in range(16):
                begin = time.perf_counter()
                primal()
                runtime.wait()
                samples.append((time.perf_counter() - begin) * 1e6)
            costs["primal_warm_complete_us"] = statistics.median(samples)
            y, _, terminal = expected(values, [2, 1], budget, regions)
            assert list(struct.unpack("<4f", arrays["y"].read())) == y
            assert list(struct.unpack("<2i", arrays["terminal"].read())) == terminal
            primal.close()
        call()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        try:
            frames[1][1]()
        except RuntimeError:
            pass
        else:
            raise AssertionError("retired nested plan was reusable")
        frames[1][1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    print(json.dumps(costs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
