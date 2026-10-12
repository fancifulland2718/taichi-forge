"""Standalone provider gate: no DSL or numpy dependency in the consumer."""

import array
import cmath
import itertools
import math
import os
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime


def read(value):
    result = array.array("f")
    result.frombytes(value.read())
    return result.tolist()


def write(value, components):
    value.write(array.array("f", components))


def close_enough(actual, expected, atol=0.004, rtol=3e-5):
    assert len(actual) == len(expected)
    assert all(abs(a - e) <= atol + rtol * abs(e) for a, e in zip(actual, expected)), (actual[:8], expected[:8])


def raises(operation):
    try:
        operation()
    except (AotError, ValueError, RuntimeError):
        return
    raise AssertionError("Invalid FFT binding unexpectedly succeeded")


def dft(values, dimensions):
    indices = list(itertools.product(*(range(n) for n in dimensions)))
    result = []
    for k in indices:
        z = sum(
            values[j] * cmath.exp(-2j * math.pi * sum(a * b / n for a, b, n in zip(k, coord, dimensions)))
            for j, coord in enumerate(indices)
        )
        result.extend((z.real, z.imag))
    return result


def main(mode, artifact):
    path = os.environ.get("TI_TEST_CUFFT_DIRECTORY")
    directory = os.add_dll_directory(path) if path and os.name == "nt" else None
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        status = runtime.probe_native_operation("fft_c2c_f32")
        assert status["available"] and status["library_version"] > 0, status
        assert status["capabilities"] == {"cuda_capture"}
        assert not runtime.probe_native_operation("unrelated.optional.provider")["available"]
        assert runtime.probe_native_operation("sort")["available"]
        module = runtime.load_module(artifact)
        plan = module.execution_plan("spectral_energy")

        def frame(multiplier):
            arrays = {name: runtime.ndarray("f32", 514, element_shape=(2,)) for name in ("x", "frequency", "dx")}
            arrays["loss"] = runtime.ndarray("f32", 1)
            values = [multiplier * ((i % 23) - 11) / 64 for i in range(1028)]
            write(arrays["x"], values)
            write(arrays["frequency"], [-99] * 1028)
            write(arrays["dx"], [-88] * 1028)
            write(arrays["loss"], [-77])
            prepared = plan.prepare(module, arrays)
            assert read(arrays["dx"]) == [-88] * 1028
            assert read(arrays["loss"]) == [-77]
            assert prepared.info["execution_kind"] == "cuda_graph"
            return arrays, values, prepared

        first, values, run = frame(1.0)
        second, other_values, other_run = frame(-0.25)
        expected = []
        for start in (0, 514):
            expected += dft([2 * complex(values[i], values[i + 1]) for i in range(start, start + 514, 2)], (257,))
        for _ in range(3):
            run()
            other_run()
        runtime.wait()
        close_enough(read(first["frequency"]), expected)
        close_enough(read(first["dx"]), [4 * 257 * x for x in values])
        close_enough(read(second["dx"]), [4 * 257 * x for x in other_values])
        close_enough(read(first["loss"]), [2 * 257 * sum(x * x for x in values)], atol=0.02)
        info = run.info
        assert info["launches"] == 3

        # Retiring one plan and its in-flight input leaves the other frame live.
        run()
        first["x"].close()
        raises(run)
        run.close()
        other_run()
        close_enough(read(second["dx"]), [4 * 257 * x for x in other_values])

        # Different rank, exact in-place replay, and explicit unnormalized inverse.
        data = runtime.ndarray("f32", 30, element_shape=(2,))
        components = [((i % 7) - 3) / 8 for i in range(60)]
        write(data, components)
        fft = {
            "kind": "fft_c2c_f32",
            "input": "data",
            "output": "data",
            "dimensions": [3, 5],
            "batch_count": 2,
            "direction": "forward",
        }
        forward = ExecutionPlan([fft]).prepare(module, {"data": data})
        inverse = ExecutionPlan([dict(fft, direction="inverse")]).prepare(module, {"data": data})
        assert read(data) == components
        forward()
        expected = []
        for start in (0, 30):
            expected += dft([complex(components[i], components[i + 1]) for i in range(start, start + 30, 2)], (3, 5))
        close_enough(read(data), expected)
        inverse()
        close_enough(read(data), [15 * x for x in components])
        raises(lambda: ExecutionPlan([dict(fft, dimensions=[2, 5])]).prepare(module, {"data": data}))
        raises(lambda: ExecutionPlan([dict(fft, dimensions=[1, 1, 3, 5])]).prepare(module, {"data": data}))
        before = read(data)
        raises(
            lambda: ExecutionPlan(
                [
                    {"kind": "fill32", "destination": "data", "value": 0},
                    fft,
                    {"kind": "kernel", "entry": "missing", "arguments": []},
                ]
            ).prepare(module, {"data": data})
        )
        assert read(data) == before  # Even a late failure performs no FFT or fill.

        write(data, components)
        volume = ExecutionPlan([dict(fft, dimensions=[2, 3, 5], batch_count=1)]).prepare(module, {"data": data})
        volume()
        close_enough(read(data), dft([complex(components[i], components[i + 1]) for i in range(0, 60, 2)], (2, 3, 5)))
        volume.close()

        # cuFFT 12.2 keeps these small transforms in local/shared storage. A
        # 64K transform exercises a real vendor workspace with only 512 KiB
        # per array and no additional user-kernel compilation or DFT reference.
        extent = 65536
        large_input, large_first, large_second = (runtime.ndarray("f32", extent, element_shape=(2,)) for _ in range(3))
        write(large_input, [0.25, -0.5] * extent)
        large_plan = ExecutionPlan(
            [
                {
                    "kind": "fft_c2c_f32",
                    "input": "x",
                    "output": "y",
                    "dimensions": [extent],
                    "direction": "forward",
                }
            ]
        )
        left = large_plan.prepare(module, {"x": large_input, "y": large_first})
        right = large_plan.prepare(module, {"x": large_input, "y": large_second})
        assert left.info["workspace_bytes"] == right.info["workspace_bytes"]
        workspace_bytes = left.info["workspace_bytes"]
        left()
        right()
        left.close()
        right()
        expected_large = [0.25 * extent, -0.5 * extent] + [0.0] * (2 * extent - 2)
        close_enough(read(large_first), expected_large)
        close_enough(read(large_second), expected_large)
        right.close()

        other_run()
        module.close()
        raises(other_run)
        other_run.close()
        forward.close()
        inverse.close()
        again = runtime.load_module(artifact)
        last = ExecutionPlan([fft]).prepare(again, {"data": data})
        last()
        if mode == "borrowed":
            ti.reset()  # Native provider plans retire before the shared executor.
            raises(last)
            last.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
        assert "numpy" not in sys.modules
    if directory is not None:
        directory.close()
    print("FFT provider OK", mode, status, info, "workspace_fixture_bytes", workspace_bytes)


if __name__ == "__main__":
    main(*sys.argv[1:])
