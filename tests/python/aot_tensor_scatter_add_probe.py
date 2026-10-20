"""Standalone packed accumulation with mutable indices and isolated roots."""

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
    count = (info["bound"] + info["n"]) * info["lanes"]

    def expected(delta=None, times=1):
        delta = info["delta"] if delta is None else delta
        return [a + times * b for a, b in zip(info["initial"], delta)] + info["source"]

    def read(value):
        data = array.array(code)
        data.frombytes(value.read())
        return list(data)

    with runtime:
        modules = [runtime.load_module(artifact) for _ in range(2)]
        outputs = [runtime.ndarray(dtype, count) for _ in modules]
        indices = [runtime.ndarray("i32", info["n"]) for _ in modules]
        initial = runtime.ndarray(dtype, info["bound"], element_shape=info["shape"])
        initial.write(array.array(code, info["initial"]).tobytes())
        frames = [
            dict(out=out, **({"input_indices": index} if info["public_indices"] else {}))
            for out, index in zip(outputs, indices)
        ]

        def set_indices(which, keys):
            indices[which].write(array.array("i", keys).tobytes())
            if not info["public_indices"]:
                with modules[which].prepare_execution("set_indices", dict(index_updates=indices[which])) as call:
                    call()

        def reset(which):
            with modules[which].prepare_execution("reset", dict(initial_values=initial)) as call:
                call()

        def observe(which):
            with modules[which].prepare_execution("read", dict(out=outputs[which])) as call:
                call()
            return read(outputs[which])

        for which in range(2):
            set_indices(which, info["indices"])
            assert observe(which) == info["initial"] + info["source"]
        for name in ("bad_shape", "bad_extent", "bad_strategy", "bad_alias"):
            reject((RuntimeError, aot.AotError), lambda: modules[0].prepare_execution(name, frames[0]))
        assert observe(0) == info["initial"] + info["source"]
        for name in ("cold", "warm", "parent"):
            reset(0)
            with modules[0].prepare_execution(name, frames[0]) as call:
                call()
                call()
                assert read(outputs[0]) == expected(times=2)
                set_indices(0, [-1] * info["n"])
                call()
                assert read(outputs[0]) == expected(times=2)
                set_indices(0, info["indices"])
        with modules[1].prepare_execution("prefix", frames[1]) as call:
            call()
        assert read(outputs[1]) == expected(info["prefix_delta"])
        if dtype != "i32":
            reset(0)
            suffix = dict(frames[0], **{name: runtime.ndarray(dtype, count) for name in ("energy", "dx", "dy")})
            suffix["dy"].write(array.array(code, [1] * count).tobytes())
            with modules[0].prepare_execution("suffix", suffix) as call:
                call()
            assert read(suffix["dx"]) == expected()
            assert read(suffix["energy"]) == [0.5 * x * x for x in expected()]
        reset(0)
        call = modules[0].prepare_execution("cold", frames[0])
        ticket = call.submit()
        modules[0].close()
        ticket.wait()
        ticket.close()
        reject((RuntimeError, aot.AotError), call)
        call.close()
        assert observe(1) == expected(info["prefix_delta"])
        modules[1].close()
    print(
        "Tensor scatter-add, mutable indices, prefix, parent, suffix AD and retirement passed",
        mode,
        arch,
        dtype,
        info["shape"],
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
