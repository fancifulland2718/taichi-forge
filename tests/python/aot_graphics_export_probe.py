"""Use exported graphics recordings with no producer imports in standalone mode."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from aot_optix_probe import raises, read, write


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan")
    folder, costs = Path(artifact).parent, []
    with runtime:
        module = runtime.load_module(artifact)
        frame = {}
        owners, calls = [], []
        for name, dtype, usage, values in (
            ("vertices", "f32", "vertex", [v for xy in [(-1, -1), (3, -1), (-1, 3)] for v in [*xy, 1, 0, 0, 0.5, 0.5]]),
            ("positions", "f32", "vertex", [-1, -1, 3, -1, -1, 3]),
            ("colors", "f32", "vertex", [9, 9, 9, 0, 0.25, 0.75]),
            ("indices", "u32", "index", [0, 1, 2]),
            ("uniform", "f32", "uniform", [1, 0, 0, 0]),
        ):
            frame[name] = runtime.ndarray(dtype, len(values), usage=(usage,))
            write(frame[name], values, "I" if dtype == "u32" else "f")
        for name, count in (("features", 256), ("dfeatures", 256), ("loss", 1)):
            frame[name] = runtime.ndarray("f32", count)
        owners += list(frame.values())
        for route in ("step", "seeded", "public", "single"):
            inputs = dict(frame)
            seed = runtime.ndarray("f32", (1, 1), element_shape=(4,))
            write(seed, [0.5, 1, 0.25, 1])
            if route == "public":
                for i in range(2):
                    blobs = {
                        f"shader_{s}": runtime.host_blob((folder / f"shader_{i}_{s}.spv").read_bytes())
                        for s in range(2)
                    }
                    inputs[f"pipeline_{i}"] = runtime.native_resource(
                        "vulkan_graphics",
                        "pipeline",
                        parameters=json.loads((folder / f"pipeline_{i}.json").read_text()),
                        bindings=blobs,
                    )
                    for blob in blobs.values():
                        blob.close()
                for name, fmt, shape in (
                    ("color", "rgba32f", [8, 8]),
                    ("depth", "depth32f", [8, 8]),
                    ("source", "rgba32f", [1, 1]),
                ):
                    inputs[name] = runtime.native_resource(
                        "image",
                        "sampled",
                        parameters=dict(format=fmt, shape=shape, attachment=name != "source"),
                        bindings=dict(pixels=seed) if name == "source" else {},
                    )
                owners += [owner for name, owner in inputs.items() if name not in frame]
            else:
                inputs["pixels"] = seed
            target_seed = None
            if route == "seeded":
                target_seed = runtime.ndarray("f32", (8, 8), element_shape=(4,))
                write(target_seed, [7] * 256)
                inputs["target_pixels"] = target_seed
            start = time.perf_counter()
            required = module.execution_plan(route).argument_names
            call = module.prepare_execution(route, {name: inputs[name] for name in required})
            prepare_ms = (time.perf_counter() - start) * 1000
            if route != "public":
                assert call.info["native_resource_bytes"] == (1024 if route == "single" else 1296)
            start = time.perf_counter()
            call.initialize()
            initialize_ms = (time.perf_counter() - start) * 1000
            seed.close()
            if target_seed is not None:
                target_seed.close()
            host, complete = [], []
            for color in ([0, 0.25, 0.75], [0.25, 0.5, 0], [0.75, 0, 0.25]):
                write(frame["colors"], [9, 9, 9, *color])
                start = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                host.append((submitted - start) / 1000)
                complete.append((time.perf_counter_ns() - start) / 1000)
                expected = [
                    v for i in range(64) for v in ([0.5, 0, 0, 1] if i % 8 < 4 and route != "single" else [*color, 1])
                ]
                assert read(frame["features"]) == expected
                if route != "single":
                    assert read(frame["dfeatures"]) == expected
                    assert read(frame["loss"]) == [sum(x * x * 0.5 for x in expected)]
            calls.append(call)
            costs.append(
                dict(
                    route=route,
                    prepare_ms=prepare_ms,
                    initialize_ms=initialize_ms,
                    host_us=statistics.median(host),
                    complete_us=statistics.median(complete),
                )
            )
        mrt_frame = dict(
            mrt_vertices=runtime.ndarray("f32", 15, usage=("vertex",)),
            ids=runtime.ndarray("u32", 32),
            primitives=runtime.ndarray("i32", 32),
        )
        write(mrt_frame["mrt_vertices"], [v for xy in [(-1, -1), (3, -1), (-1, 3)] for v in [*xy, 0.25, 0.5, 0.75]])
        mrt = module.prepare_execution("mrt", mrt_frame)
        assert mrt.info["native_resource_bytes"] == 768
        for _ in range(2):
            mrt()
            assert read(mrt_frame["ids"], "I") == [0xF1234567 if i % 8 < 4 else 0xFEDCBA98 for i in range(32)]
            assert read(mrt_frame["primitives"], "i") == [-123456789 if i % 8 < 4 else -987654321 for i in range(32)]
        calls.append(mrt)
        owners += list(mrt_frame.values())
        # An exported clear is not a blanket initialization claim. Moving it
        # inside a conditional must fail if the rest consumes the attachment.
        document = json.loads(module.execution_plan("step").to_json())
        first, *tail = document["operations"]
        document["operations"] = [
            dict(
                kind="branch",
                selector="selector",
                selection="nonzero",
                has_default=False,
                lowering=dict(kind="vulkan_host", version=1),
                condition=[dict(kind="fill32", destination="selector", value=0)],
                bodies=[[first]],
            ),
            *tail,
        ]
        plan = ExecutionPlan(
            document["operations"],
            resources=document["resources"],
            gradients=document["gradients"],
            initialization=document["initialization"],
        )
        selector = runtime.ndarray("i32", 1)
        pixels = runtime.ndarray("f32", (1, 1), element_shape=(4,))
        write(pixels, [0.5, 1, 0.25, 1])
        raises(lambda: plan.prepare(module, dict(frame, selector=selector, pixels=pixels)), "unconditional upload")
        selector.close()
        pixels.close()
        calls[0]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for call in calls:
            raises(call)
            call.close()
        for owner in owners:
            owner.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    (folder / f"{mode}.json").write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
