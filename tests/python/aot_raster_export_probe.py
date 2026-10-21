"""Run frozen RasterPass plans with no producer, Camera, Window or shader compiler."""

import json
import os
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime
from aot_optix_probe import raises, read, write


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        # Pixel-exact producer comparisons require the same physical device.
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", os.environ.get("TI_VISIBLE_DEVICE", "0"))))
    folder = Path(artifact).parent
    references = json.loads((folder / "references.json").read_text())
    costs = []
    with runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        frame = {
            name: runtime.ndarray("f32", size) for name, size in dict(output=3840, dout=3840, loss=1, dloss=1).items()
        }
        frame["centers"] = runtime.ndarray("f32", 2, element_shape=(3,))
        frame["transforms"] = runtime.ndarray("f32", 2, element_shape=(4, 4))
        for name, fmt in (("color", "rgba8"), ("depth", "depth32f")):
            frame[name] = runtime.native_resource(
                "image", "sampled", parameters=dict(shape=[32, 24], format=fmt, attachment=True)
            )
        calls = []
        for name in ("private", "public", "empty"):
            plan = module.execution_plan(name)
            bindings = {key: frame[key] for key in plan.argument_names}
            start = time.perf_counter()
            call = plan.prepare(module, bindings)
            prepare_ms = (time.perf_counter() - start) * 1000
            assert call.info["execution_kind"] == "vulkan_commands"
            for offset, expected in zip((-0.2, 0.2, 5), references):
                matrix = [float(row == col) for row in range(4) for col in range(4)]
                matrices = matrix * 2
                matrices[16 + 3] = 0.8 + offset
                write(frame["transforms"], matrices)
                write(frame["centers"], [9, 9, 0, offset, 0.45, 0])
                call()
                if name == "empty":
                    expected = [0, 0, 0, 1, 0] * (32 * 24)
                assert read(frame["output"]) == expected
                assert read(frame["dout"]) == expected
                loss = sum(x * x / 2 for x in expected)
                assert abs(read(frame["loss"])[0] - loss) < max(1e-4, loss * 3e-5)
            submits, completes = [], []
            for _ in range(4):
                start = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                submits.append((submitted - start) / 1000)
                completes.append((time.perf_counter_ns() - start) / 1000)
            costs.append(
                dict(
                    name=name,
                    prepare_ms=prepare_ms,
                    submit_us=statistics.median(submits),
                    complete_us=statistics.median(completes),
                    info=call.info,
                )
            )
            if name == "public":
                with runtime.native_resource(
                    "image", "sampled", parameters=dict(shape=[33, 24], format="rgba8", attachment=True)
                ) as wrong:
                    raises(lambda: plan.prepare(module, dict(bindings, color=wrong)), "layout contract")
                call()
            calls.append(call)
        calls[0]()
        calls[0].close()  # Pending work retires before private geometry/attachments.
        frame["color"].close()
        raises(calls[1])
        calls[2]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(calls[2])
        for call in calls:
            call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    (folder / f"{mode}.json").write_text(json.dumps(dict(load_ms=load_ms, costs=costs), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
