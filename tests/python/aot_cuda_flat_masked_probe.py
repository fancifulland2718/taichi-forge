"""Consume a flat masked artifact without importing its producer source."""

import json
import statistics
import struct
import sys
import time


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
        costs = dict(load_ms=(time.perf_counter() - begin) * 1000, prepare_ms=[])
        frames = []
        for scale, name in ((1, "step"), (2, "again")):
            arrays = {n: runtime.ndarray("f32", 4) for n in ("x", "y", "dx", "dy")}
            arrays.update({n: runtime.ndarray("i32", 1) for n in ("limits", "predicate")})
            arrays["terminal"] = runtime.ndarray("i32", 2)
            values = [scale * x for x in (0.5, 1, 2, 3)]
            arrays["x"].write(struct.pack("<4f", *values))
            arrays["y"].write(struct.pack("<4f", *([-7] * 4)))
            begin = time.perf_counter()
            call = module.prepare_execution(name, arrays)
            costs["prepare_ms"].append((time.perf_counter() - begin) * 1000)
            assert call.info["execution_kind"] == "cuda_graph", call.info
            assert struct.unpack("<4f", arrays["y"].read()) == (-7,) * 4
            frames.append((arrays, call, values))

        def check(frame, limit):
            arrays, _, values = frame
            steps = min(budget, max(0, limit))
            assert struct.unpack("<2i", arrays["terminal"].read()) == (steps, steps + 1)
            assert list(struct.unpack("<4f", arrays["y"].read())) == [x * x if steps else -99 for x in values]
            assert list(struct.unpack("<4f", arrays["dx"].read())) == [2 * steps * x for x in values]
            assert struct.unpack("<4f", arrays["dy"].read()) == (0 if steps else 1,) * 4

        for limit in (0, 2, 7, -1, 0):
            for frame in frames:
                arrays, call, _ = frame
                arrays["limits"].write(struct.pack("<i", limit))
                call()
                runtime.wait()
                check(frame, limit)

        arrays = frames[0][0]
        document = json.loads(module.execution_plan("step").to_json())
        # Wrong launch ABIs must fail even when no body iteration will execute.
        loop_index = next(i for i, op in enumerate(document["operations"]) if op["kind"] == "while")
        for region in ("condition", "body"):
            invalid = json.loads(json.dumps(document))
            loop = invalid["operations"][loop_index]
            if region == "condition":
                loop[region] = [loop["body"][-1]]
            else:
                loop[region][0]["entry"] = "advance"
            try:
                module.prepare_execution(ExecutionPlan.from_json(json.dumps(invalid)), arrays)
            except AotError as error:
                assert "Unsupported fixed CUDA Graph dispatch" in str(error), str(error)
            else:
                raise AssertionError("Invalid CUDA launch ABI accepted")
        for dtype, count in (("f32", 1), ("i32", 2)):
            invalid = runtime.ndarray(dtype, count)
            try:
                module.prepare_execution("step", dict(arrays, predicate=invalid))
            except AotError as error:
                # An external initial condition can encounter the ordinary
                # kernel ABI check before the loop's scalar control check.
                assert any(
                    message in str(error)
                    for message in ("one aligned scalar i32", "Ndarray dtype or shape does not match")
                ), str(error)
            else:
                raise AssertionError("Invalid predicate accepted")
            invalid.close()
        arrays["limits"].write(struct.pack("<i", 2))
        frames[0][1]()
        arrays["predicate"].close()
        check(frames[0], 2)
        try:
            frames[0][1]()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Retired predicate was reusable")
        frames[0][1].close()
        arrays, call, _ = frames[1]
        arrays["limits"].write(struct.pack("<i", 2))
        samples = []
        for _ in range(16):
            begin = time.perf_counter()
            call()
            runtime.wait()
            samples.append((time.perf_counter() - begin) * 1e6)
        check(frames[1], 2)
        costs["warm_complete_us"] = statistics.median(samples)
        call()  # Retire in-flight work through the owner/module path.
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        try:
            call()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Retired owner was reusable")
        call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    print(json.dumps(costs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
