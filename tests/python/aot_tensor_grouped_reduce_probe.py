"""Runtime-only packed grouped reductions preserve clear/replay dependencies."""

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
    dtype = info["dtype"]
    code = dict(f32="f", f64="d", i32="i")[dtype]
    count = (info["groups"] + info["n"]) * info["lanes"]
    expected = info["expected"] + info["source"]
    cleared = [0] * len(info["expected"]) + info["source"]
    original = info["initial"] + info["source"]

    def read(value):
        data = array.array(code)
        data.frombytes(value.read())
        return list(data)

    with runtime:
        modules = [runtime.load_module(artifact) for _ in range(2)]
        outputs = [runtime.ndarray(dtype, count) for _ in modules]
        keys = [runtime.ndarray("i32", info["n"]) for _ in modules]
        initial = runtime.ndarray(dtype, info["groups"], element_shape=info["shape"])
        initial.write(array.array(code, info["initial"]).tobytes())
        frames = [
            dict(out=out, **({"input_keys": key} if info["public_keys"] else {})) for out, key in zip(outputs, keys)
        ]

        def set_keys(which, ids):
            keys[which].write(array.array("i", ids).tobytes())
            if not info["public_keys"]:
                with modules[which].prepare_execution("set_keys", dict(key_updates=keys[which])) as call:
                    call()

        def reset(which):
            with modules[which].prepare_execution("reset", dict(initial_values=initial)) as call:
                call()

        def observe(which):
            with modules[which].prepare_execution("read", dict(out=outputs[which])) as call:
                call()
            return read(outputs[which])

        for which in range(2):
            set_keys(which, info["keys"])
            assert observe(which) == original
        for name in ("bad_shape", "bad_extent"):
            reject((RuntimeError, aot.AotError), lambda: modules[0].prepare_execution(name, frames[0]))
        assert observe(0) == original
        for name in ("cold", "warm", "parent"):
            reset(0)
            with modules[0].prepare_execution(name, frames[0]) as call:
                call()
                call()
                assert read(outputs[0]) == expected
                set_keys(0, [-1] * info["n"])
                call()
                assert read(outputs[0]) == cleared
                set_keys(0, info["keys"])
                call()
                assert read(outputs[0]) == expected
            assert observe(1) == original
        if dtype != "i32":
            suffix = dict(frames[0], **{name: runtime.ndarray(dtype, count) for name in ("energy", "dx", "dy")})
            suffix["dy"].write(array.array(code, [1] * count).tobytes())
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
        with modules[1].prepare_execution("warm", frames[1]) as other:
            other()
        assert read(outputs[1]) == expected
        modules[1].close()
    print(
        "Tensor grouped clear/replay, mutable keys, parent, suffix AD and retirement passed",
        mode,
        arch,
        dtype,
        info["shape"],
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
