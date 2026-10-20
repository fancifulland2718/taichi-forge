"""Small independent/borrowed consumers with source imports forbidden."""

import array
import json
import os
from pathlib import Path
import statistics
import sys
import time

from aot_consumer_probe import NoFrontend, raises


def write(value, values):
    value.write(array.array("f", values))


def read(value):
    values = array.array("f")
    values.frombytes(value.read())
    return list(values)


def main(mode, artifact, result):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime import aot
        runtime = aot.Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0)))
    else:
        import taichi_forge as ti
        from taichi_forge_runtime import aot
        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    measurements = []
    with runtime:
        assert runtime.native_operation_capabilities("vulkan_buffer_commands") == {"vulkan_inline"}
        start = time.perf_counter()
        modules = [runtime.load_module(artifact) for _ in range(2)]
        load_ms = (time.perf_counter() - start) * 1000
        frames = [dict(input=runtime.ndarray("f32", 8), output=runtime.ndarray("f32", 8),
                       doutput=runtime.ndarray("f32", 8), loss=runtime.ndarray("f32", 1),
                       dloss=runtime.ndarray("f32", 1)) for _ in modules]
        field_outputs = [runtime.ndarray("f32", 23) for _ in modules]
        initial = list(range(16)) + [77] * 7
        for module, output in zip(modules, field_outputs):
            with module.prepare_execution("field_read", dict(out=output)) as call:
                call()
            assert read(output) == initial
        with modules[0].prepare_execution("field", dict(out=field_outputs[0])) as call:
            call()
            call()
        expected = list(initial)
        expected[9:13] = [1, 2, 3, 4]
        assert read(field_outputs[0]) == expected
        with modules[1].prepare_execution("field_read", dict(out=field_outputs[1])) as call:
            call()
        assert read(field_outputs[1]) == initial
        for module, frame in zip(modules, frames):
            for name in ("copy", "parent"):
                plan = module.execution_plan(name)
                selected = {key: frame[key] for key in plan.argument_names}
                write(frame["output"], [-99] * 8)
                start = time.perf_counter()
                call = plan.prepare(module, selected)
                prepare_ms = (time.perf_counter() - start) * 1000
                assert read(frame["output"]) == [-99] * 8
                for scale in (1, 0, -2):
                    write(frame["input"], [scale * i for i in range(8)])
                    call()
                    expected = [-99, -99, scale, 2 * scale, 3 * scale, 4 * scale, -99, -99]
                    assert read(frame["output"]) == expected
                    if name == "parent":
                        assert read(frame["doutput"]) == [0, 0, *expected[2:6], 0, 0]
                        assert read(frame["loss"]) == [sum(x * x / 2 for x in expected[2:6])]
                submit, complete = [], []
                for _ in range(5):
                    start = time.perf_counter_ns()
                    call()
                    submitted = time.perf_counter_ns()
                    runtime.wait()
                    submit.append((submitted - start) / 1000)
                    complete.append((time.perf_counter_ns() - start) / 1000)
                measurements.append(dict(name=name, prepare_ms=prepare_ms,
                    submit_us=statistics.median(submit), complete_us=statistics.median(complete), info=call.info))
                call.close()
            with module.prepare_execution("barrier", {}) as barrier:
                barrier()
        frame = frames[0]
        plan = modules[0].execution_plan("copy")
        saved = read(frame["output"])
        short = runtime.ndarray("f32", 4)
        for bindings in (dict(input=short, output=frame["output"]),
                         dict(input=frame["output"], output=frame["output"])):
            raises(aot.AotError, lambda: plan.prepare(modules[0], bindings))
            assert read(frame["output"]) == saved
        short.close()
        call = plan.prepare(modules[0], dict(input=frame["input"], output=frame["output"]))
        ticket = call.submit()
        modules[0].close()
        ticket.wait()
        ticket.close()
        raises(aot.AotError, call)
        call.close()
        # Retiring one module does not invalidate the other physical instance.
        frame = frames[1]
        with modules[1].prepare_execution("copy", dict(input=frame["input"], output=frame["output"])) as sibling:
            sibling()
            frame["input"].close()
            raises(aot.AotError, sibling)
        modules[1].close()
        for frame in frames:
            for value in frame.values():
                value.close()
        for value in field_outputs:
            value.close()
    Path(result).write_text(json.dumps(dict(load_two_modules_ms=load_ms, cases=measurements), indent=2))
    print("AOT buffer commands, parent gradient, isolation and retirement passed", mode)


if __name__ == "__main__":
    main(*sys.argv[1:])
