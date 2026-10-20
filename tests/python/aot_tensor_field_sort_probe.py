"""Runtime-only sort records preserve payload bits, gradients and retirement."""

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
    info = json.loads(Path(artifact).with_suffix(".json").read_text())
    n = info["n"]
    bits = "Q" if info["bits_dtype"] == "u64" else "I"

    def read(value, code):
        result = array.array(code)
        result.frombytes(value.read())
        return list(result)

    with runtime:
        modules = [runtime.load_module(artifact) for _ in range(2)]
        frames = [
            dict(
                out_keys=runtime.ndarray("u32", n),
                out_values=runtime.ndarray(info["bits_dtype"], n * info["lanes"]),
                ids=runtime.ndarray("i32", n),
            )
            for _ in modules
        ]
        input_keys = runtime.ndarray("f32", n)
        input_values = runtime.ndarray(info["dtype"], n, element_shape=info["shape"])
        input_values.write(array.array(bits, info["values"]).tobytes())
        reset_frame = dict(input_keys=input_keys, input_values=input_values)

        def reset(which, case):
            input_keys.write(array.array("I", case["keys"]).tobytes())
            with modules[which].prepare_execution("reset", reset_frame) as call:
                call()

        def observe(which):
            with modules[which].prepare_execution("read", frames[which]) as call:
                call()
            return read(frames[which]["out_keys"], "I"), read(frames[which]["out_values"], bits)

        def check(which, case):
            assert read(frames[which]["out_keys"], "I") == case["sorted_keys"]
            assert read(frames[which]["out_values"], bits) == case["sorted_values"]
            assert read(frames[which]["ids"], "i") == case["ids"]

        original = (info["cases"][0]["keys"], info["values"])
        for which in range(2):
            assert observe(which) == original
        for name in ("cold", "warm", "parent"):
            with modules[0].prepare_execution(name, frames[0]) as call:
                for case in info["cases"]:
                    reset(0, case)
                    call()
                    call()
                    check(0, case)
                    assert observe(1) == original
        suffix = dict(frames[0], **{name: runtime.ndarray("f32", n) for name in ("source", "energy", "dx", "dy")})
        suffix["source"].write(array.array("f", info["source"]).tobytes())
        with modules[0].prepare_execution("suffix", suffix) as call:
            for case in info["cases"]:
                reset(0, case)
                # Reverse consumes output adjoints; each independent VJP needs
                # its own seed even when replaying the same prepared plan.
                suffix["dy"].write(array.array("f", [1] * n).tobytes())
                call()
                check(0, case)
                actual = read(suffix["dx"], "f")
                assert actual == case["gradient"], (case["ids"], actual, case["gradient"], read(suffix["dy"], "f"))
                assert read(suffix["energy"], "f") == case["energy"]
        reset(0, info["cases"][0])
        call = modules[0].prepare_execution("cold", frames[0])
        ticket = call.submit()
        modules[0].close()
        ticket.wait()
        ticket.close()
        reject((RuntimeError, aot.AotError), call)
        call.close()
        with modules[1].prepare_execution("warm", frames[1]) as other:
            other()
        check(1, info["cases"][0])
        modules[1].close()
    print(
        "Tensor Field native sort bits, mutable inputs, parent, suffix AD and retirement passed",
        mode,
        arch,
        info["dtype"],
        info["shape"],
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
