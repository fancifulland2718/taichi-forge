"""Independent whole-tensor consumers, prefix checks, actual AD and retirement."""

import array
import json
import os
from pathlib import Path
import sys

from aot_struct_consumer_probe import NoFrontend, reject


def main(mode, arch, artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime import aot

        runtime = aot.Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0)) if arch == "vulkan" else 0
        )
    else:
        import taichi_forge as ti
        from taichi_forge_runtime import aot

        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
        runtime = ti.aot.borrow_runtime()
    description = json.loads(Path(artifact).with_suffix(".json").read_text())
    dtype, expected = description["dtype"], description["expected"]
    code = dict(i32="i", f32="f", f64="d")[dtype]
    initial = [-9] * (3 * description["lanes"]) + description["values"]

    def read(value):
        result = array.array(code)
        result.frombytes(value.read())
        return list(result)

    with runtime:
        modules = [runtime.load_module(artifact) for _ in range(2)]
        frames = [dict(out=runtime.ndarray(dtype, len(expected))) for _ in modules]
        for module, frame in zip(modules, frames):
            with module.prepare_execution("read", frame) as call:
                call()
            assert read(frame["out"]) == initial
        for name in ("bad_shape", "bad_extent", "bad_alias", "bad_strategy"):
            reject((RuntimeError, aot.AotError), lambda: modules[0].prepare_execution(name, frames[0]))
        with modules[0].prepare_execution("read", frames[0]) as call:
            call()
        assert read(frames[0]["out"]) == initial
        for name in ("cold", "warm", "parent"):
            with modules[0].prepare_execution(name, frames[0]) as call:
                call()
                call()
            assert read(frames[0]["out"]) == expected
        with modules[1].prepare_execution("prefix", frames[1]) as call:
            call()
        assert read(frames[1]["out"]) == description["prefix_expected"]
        if dtype != "i32":
            suffix = dict(frames[0], **{name: runtime.ndarray(dtype, len(expected)) for name in ("energy", "dx", "dy")})
            suffix["dy"].write(array.array(code, [1] * len(expected)).tobytes())
            with modules[0].prepare_execution("suffix", suffix) as call:
                call()
            assert read(suffix["dx"]) == expected
            assert read(suffix["energy"]) == [0.5 * x * x for x in expected]
        call = modules[0].prepare_execution("cold", frames[0])
        ticket = call.submit()
        modules[0].close()
        ticket.wait()
        ticket.close()
        reject((RuntimeError, aot.AotError), call)
        call.close()
        with modules[1].prepare_execution("read", frames[1]) as call:
            call()
        assert read(frames[1]["out"]) == description["prefix_expected"]
        modules[1].close()
    print(
        "Tensor reductions, prefix, parent, actual suffix AD and retirement passed",
        mode,
        arch,
        dtype,
        description["shape"],
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
