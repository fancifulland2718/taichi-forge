"""No-source consumption of selected device-update plans and their gradients."""

import json
import statistics
import struct
import sys
import time

from aot_nested_capacity_probe import cases, expected


def main(mode, artifact, budget):
    from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime

    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False, enable_fallback=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
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
                y, dx, terminal = expected(values, limits, budget, 2)
                assert list(struct.unpack("<4f", arrays["y"].read())) == y
                assert list(struct.unpack("<4f", arrays["dx"].read())) == dx
                assert list(struct.unpack("<2i", arrays["terminal"].read())) == terminal

        arrays, _, _ = frames[0]
        description = json.loads(module.execution_plan("step").to_json())
        description["resources"] = [r for r in description["resources"] if r["name"] != "ip0"]
        public_predicate = ExecutionPlan.from_json(json.dumps(description))
        for dtype, count in (("f32", 1), ("i32", 2)):
            invalid = runtime.ndarray(dtype, count)
            try:
                module.prepare_execution(public_predicate, dict(arrays, ip0=invalid))
            except AotError as error:
                assert "one aligned scalar i32" in str(error), str(error)
            else:
                raise AssertionError("Invalid inner predicate accepted")
            invalid.close()
        predicate = runtime.ndarray("i32", 1)
        bound = module.prepare_execution(public_predicate, dict(arrays, ip0=predicate))
        bound()
        predicate.close()  # Retires the submitted graph before its control storage.
        try:
            bound()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Retired predicate was reusable")
        bound.close()
        frames[0][1].close()
        arrays, call, _ = frames[1]
        arrays["limits"].write(struct.pack("<2i", 2, 1))
        samples = []
        for _ in range(16):
            begin = time.perf_counter()
            call()
            runtime.wait()
            samples.append((time.perf_counter() - begin) * 1e6)
        costs["warm_complete_us"] = statistics.median(samples)
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        try:
            call()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Retired runtime/module was reusable")
        call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    print(json.dumps(costs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
