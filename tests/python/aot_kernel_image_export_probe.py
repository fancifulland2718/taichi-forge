"""No producer imports: aliased images, real numeric AD and binding families."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime
from aot_optix_probe import raises, read, write


def check(frame, values, arch):
    if arch == "vulkan":
        values = [values[j * 3 + i] for i in range(3) for j in range(5)]
    expected = [2 * v for v in values]
    assert read(frame["output"]) == expected
    assert read(frame["dout"]) == expected
    assert read(frame["loss"]) == [sum(v * v * 0.5 for v in expected)]


def main(mode, artifact, recipe, result, arch="cuda"):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    costs, retained = [], []
    with runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000

        def arrays():
            frame = {
                name: runtime.ndarray("f32", (1,) if name == "loss" else (3, 5))
                for name in ("output", "dout", "loss")
            }
            write(frame["output"], [-99] * 15)
            return frame

        for offset in (0, -7):
            pixels = runtime.ndarray("f32", (3, 5))
            values = list(range(offset, offset + 15))
            write(pixels, values)
            frame = arrays()
            start = time.perf_counter()
            call = module.prepare_execution("step", dict(frame, pixels=pixels))
            prepare_ms = (time.perf_counter() - start) * 1000
            assert read(frame["output"]) == [-99] * 15
            call.initialize()
            # Seed changes after preparation must not silently alter the image.
            write(pixels, [77] * 15)
            call()
            check(frame, values, arch)
            if recipe == "selected" and arch == "cuda":
                other = arrays()
                sibling = call.prepare_binding(other)
                assert sibling.binding_family_info["executables"] == 1
                assert (
                    sibling.binding_family_info["identity"]
                    == call.binding_family_info["identity"]
                )
                for binding, target in ((sibling, other), (call, frame)):
                    binding()
                    check(target, values, arch)
                call.close()
                sibling()
                check(other, values, arch)
                call, frame = sibling, other
            submitted, completed = [], []
            for _ in range(8):
                start = time.perf_counter_ns()
                call()
                submitted.append((time.perf_counter_ns() - start) / 1000)
                runtime.wait()
                completed.append((time.perf_counter_ns() - start) / 1000)
            check(frame, values, arch)
            costs.append(
                dict(
                    prepare_ms=prepare_ms,
                    submit_us=statistics.median(submitted),
                    complete_us=statistics.median(completed),
                    info=call.info,
                )
            )
            # Native resource inputs follow the existing prepared dependency
            # contract, even when their content is sampled only during prepare.
            retained.append((call, frame, values, pixels))
        for call, frame, values, _ in retained:
            call()
            check(frame, values, arch)
        # Retire in flight through the actual parent owner on each route.
        retained[-1][0]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for call, _, _, pixels in retained:
            raises(call)
            call.close()
            pixels.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(result).write_text(json.dumps(dict(load_ms=load_ms, cases=costs), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
