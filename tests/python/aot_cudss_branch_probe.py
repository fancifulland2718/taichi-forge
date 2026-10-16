"""No producer code: conditional solve, kernel.grad and explicit SPD adjoint."""

import json
import struct
import sys
import time

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cudss_probe import near, read, raises, solve, write


def expect(arrays, kind, choice, scale, value_scale=None):
    selected = (0 if choice else 1) if kind == "if" else choice
    if selected in (0, 1):
        matrix_scale = scale * (selected + 1)
        if selected == 0 and value_scale is not None:
            matrix_scale = value_scale
        x = solve([(i + 1) * (scale + 1) for i in range(4)], matrix_scale)
        near(read(arrays["x"]), x)
        near(read(arrays["loss"]), [v * v for v in x])
        near(read(arrays["gradient"]), solve([2 * v for v in x], matrix_scale))
    else:
        assert read(arrays["x"]) == [-71] * 4
        assert read(arrays["loss"]) == [-91] * 4
        assert read(arrays["gradient"]) == [0] * 4
    assert struct.unpack("<2i", arrays["state"].read()) == (-77, 1)


def check(runtime, artifact):
    start = time.perf_counter()
    module = runtime.load_module(artifact)
    stats = dict(load_ms=(time.perf_counter() - start) * 1000)
    values = [4, -1, -1, 4, -1, -1, 4, -1, -1, 3]
    for kind in ("if", "switch"):
        frames = []
        costs = []
        for scale in (1, 3):
            start = time.perf_counter()
            arrays = {
                name: runtime.ndarray("f32", 4) for name in ("rhs", "x", "dx", "loss", "dloss", "gradient")
            }
            arrays.update({name: runtime.ndarray("f32", 10) for name in ("values0", "values1")})
            arrays["choice"] = runtime.ndarray("i32", 1)
            arrays["state"] = runtime.ndarray("i32", 2)
            write(arrays["values0"], [v * scale for v in values])
            write(arrays["values1"], [v * scale * 2 for v in values])
            write(arrays["rhs"], [(i + 1) * (scale + 1) for i in range(4)])
            write(arrays["x"], [71] * 4)
            write(arrays["choice"], [1], "i")
            bound = time.perf_counter()
            call = module.prepare_execution(kind, arrays)
            prepared = time.perf_counter()
            assert read(arrays["x"]) == [71] * 4
            assert call.info["execution_kind"] == "cuda_graph"
            raises(lambda: module.prepare_execution(kind, dict(arrays, rhs=arrays["x"])))
            first = time.perf_counter()
            call()
            runtime.wait()
            done = time.perf_counter()
            expect(arrays, kind, 1, scale)
            costs.append(
                dict(
                    binding_ms=(bound - start) * 1000,
                    prepare_ms=(prepared - bound) * 1000,
                    first_complete_ms=(done - first) * 1000,
                )
            )
            frames.append((arrays, call, scale))
        for choice in (-2147483648, -1, 0, 1, 2, 2147483647, 0):
            for arrays, call, scale in frames:
                write(arrays["choice"], [choice], "i")
                call()
                expect(arrays, kind, choice, scale)
        arrays, call, scale = frames[0]
        write(arrays["values0"], [v * 4 for v in values])
        selected = 1 if kind == "if" else 0
        write(arrays["choice"], [selected], "i")
        call()
        expect(arrays, kind, selected, scale, value_scale=4)
        start = time.perf_counter()
        for _ in range(16):
            call()
        submitted = time.perf_counter()
        runtime.wait()
        done = time.perf_counter()
        expect(arrays, kind, selected, scale, value_scale=4)
        stats[kind] = dict(
            frames=costs,
            host_us=(submitted - start) * 1e6 / 16,
            complete_us=(done - start) * 1e6 / 16,
            workspace_bytes=call.info["workspace_bytes"],
        )
        arrays["values0"].close()
        raises(call)
        call.close()
        other, second, scale = frames[1]
        second()
        expect(other, kind, 0, scale)
        second.close()
        for arrays, _, _ in frames:
            for value in arrays.values():
                value.close()
    module.close()
    return stats


if __name__ == "__main__":
    mode, artifact, adapter, vendor = sys.argv[1:]
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("cudss", adapter_path=adapter, library_path=vendor)
        stats = check(runtime, artifact)
        if mode == "borrowed":
            ti.reset()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    print(json.dumps(stats))
