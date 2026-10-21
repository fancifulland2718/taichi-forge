"""Rebuild line expansion without the producer raster frontend or shader tools."""

import json
import os
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_optix_probe import raises, read, write


def geometry(variant):
    shift = (-0.15, 0.2, 4)[variant]
    points = [
        (-4, -4, 0),
        (4, 4, 0),
        (-0.7, -0.3, 0),
        (0.7, -0.3, 0),
        (-0.6, 0.3, -0.2),
        (0.6, 0.3, -0.2),
        (3, 3, 0),
        (4, 4, 0),
    ]
    colors = [(1, 1, 1), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 1, 1)]
    if variant == 1:
        colors[2:6] = reversed(colors[2:6])
    return (
        [value for x, y, z in points for value in (x + shift, y, z)],
        [v for c in colors for v in c],
        [0, 0, *([0, 2, 1, 3] if variant == 1 else [0, 1, 2, 3]), 0],
    )


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        # Pixel-exact producer comparisons require the same physical device.
        runtime = Runtime(
            "vulkan",
            device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", os.environ.get("TI_VISIBLE_DEVICE", "0"))),
        )
    folder = Path(artifact).parent
    references = json.loads((folder / "references.json").read_text())
    with runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        costs = dict(load_ms=(time.perf_counter() - start) * 1000, cases=[])
        frame = {name: runtime.ndarray("f32", 8, element_shape=(3,)) for name in ("vertices", "colors")}
        frame.update(output=runtime.ndarray("f32", 32 * 24 * 5), indices=runtime.ndarray("i32", 7))
        calls = []
        for name in ("plain", "indexed", "empty"):
            plan = module.execution_plan(name)
            start = time.perf_counter()
            call = plan.prepare(module, {key: frame[key] for key in plan.argument_names})
            prepare_ms = (time.perf_counter() - start) * 1000
            for variant in range(3):
                vertices, colors, indices = geometry(variant)
                write(frame["vertices"], vertices)
                write(frame["colors"], colors)
                write(frame["indices"], indices, "i")
                call()
                assert read(frame["output"]) == references[name][variant], name
            # Time visible geometry after the hit-to-empty correctness case.
            vertices, colors, indices = geometry(0)
            write(frame["vertices"], vertices)
            write(frame["colors"], colors)
            write(frame["indices"], indices, "i")
            call()
            assert read(frame["output"]) == references[name][0]
            submits, completes = [], []
            for _ in range(4):
                start = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                submits.append((submitted - start) / 1000)
                completes.append((time.perf_counter_ns() - start) / 1000)
            costs["cases"].append(
                dict(
                    name=name,
                    prepare_ms=prepare_ms,
                    submit_us=statistics.median(submits),
                    complete_us=statistics.median(completes),
                    info=call.info,
                )
            )
            calls.append(call)
        calls[1]()
        calls[1].close()
        frame["vertices"].close()
        raises(calls[0])
        calls[2]()  # Empty plan never retained the geometry.
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(calls[2])
        for call in calls:
            call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    (folder / f"{mode}.json").write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
