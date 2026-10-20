"""Runtime-only affine roots, parent Graphs, isolation and actual suffix AD."""

import json
import os
from pathlib import Path
import sys

from aot_struct_consumer_probe import NoFrontend, read, reject


def main(mode, arch, artifact, invalid_artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime import aot
        runtime = aot.Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0)) if arch == "vulkan" else 0)
    else:
        import taichi_forge as ti
        from taichi_forge_runtime import aot
        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
        runtime = ti.aot.borrow_runtime()
    n = json.loads(Path(artifact).with_suffix(".json").read_text())["size"]
    initial = [i * .5 for i in range(n)]
    expected = [6 * value + 13 for value in initial]
    with runtime:
        modules = [runtime.load_module(artifact) for _ in range(2)]
        frames = [{name: runtime.ndarray(dtype, n) for name, dtype in
                   (("out", "f32"), ("source", "f32"), ("tags", "i32"))} for _ in modules]

        def check(frame, values):
            assert read(frame["out"]) == values
            assert read(frame["source"]) == initial
            import array
            assert frame["tags"].read() == array.array("i", [37] * n).tobytes()

        for module, frame in zip(modules, frames):
            with module.prepare_execution("read", frame) as call:
                call()
            check(frame, [-9] * n)
        for name in ("bad_copy", "bad_view"):
            reject((RuntimeError, aot.AotError),
                   lambda: modules[0].prepare_execution(name, dict(out=frames[0]["out"])))
        invalid = runtime.load_module(invalid_artifact)
        reject((RuntimeError, aot.AotError),
               lambda: invalid.prepare_execution("cold", frames[0]))
        invalid.close()
        for name in ("cold", "warm", "parent"):
            with modules[0].prepare_execution(name, frames[0]) as call:
                call()
            check(frames[0], expected)
        with modules[1].prepare_execution("prefix", frames[1]) as call:
            call()
        check(frames[1], expected[:11] + [-9] * (n - 11))
        with modules[0].prepare_execution("read", frames[0]) as call:
            call()
        check(frames[0], expected)
        full_frame = dict(frames[0], **{name: runtime.ndarray("f32", n) for name in ("energy", "dx", "dy")})
        full = modules[0].prepare_execution("suffix", full_frame)
        full()
        assert read(full_frame["dx"]) == expected
        assert read(full_frame["energy"]) == [.5 * value * value for value in expected]
        ticket = full.submit()
        modules[0].close()
        ticket.wait()
        ticket.close()
        reject((RuntimeError, aot.AotError), full)
        full.close()
        with modules[1].prepare_execution("read", frames[1]) as call:
            call()
        check(frames[1], expected[:11] + [-9] * (n - 11))
        modules[1].close()
    print("Affine root, prefix, parent, state isolation, suffix AD and retirement passed", mode, arch, n)


if __name__ == "__main__":
    main(*sys.argv[1:])
