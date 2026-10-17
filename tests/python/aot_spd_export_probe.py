"""Consume exported SPD Graphs without producer Python or compiler dependencies."""

import json
from pathlib import Path
import statistics
import sys
import time
from taichi_forge_runtime.aot import Runtime
from aot_optix_probe import raises, read, write
from aot_spd_probe import pyramid


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan")
    folder = Path(artifact).parent
    spec = json.loads((folder / "spec.json").read_text())
    width, height = spec["shape"]
    total = sum((width >> (i + 1)) * (height >> (i + 1)) for i in range(spec["levels"]))
    costs = []
    with runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        frame = {
            name: runtime.ndarray("f32", shape)
            for name, shape in dict(
                seed=(width, height), scale=(1,), result=(total,), dresult=(total,), loss=(1,), dloss=(1,)
            ).items()
        }
        write(frame["seed"], [0] * (width * height))
        images = [
            runtime.native_resource(
                "image",
                "sampled",
                parameters=dict(format="r32f", storage=True, shape=shape, mip_levels=levels),
                bindings=dict(pixels=frame["seed"]) if levels == 1 else {},
            )
            for shape, levels in (([width, height], 1), ([width // 2, height // 2], spec["levels"]))
        ]
        blob = runtime.host_blob((folder / "shader.spv").read_bytes())
        spd = runtime.native_resource(
            "spd",
            "plan",
            parameters=dict(reduction=spec["reduction"], expected_plan=spec["expected_plan"]),
            bindings=dict(source=images[0], output=images[1], shader=blob),
        )
        blob.close()
        frame.update(public_spd=spd, public_source=images[0], public_pyramid=images[1])
        calls = []
        for recipe in ("ordinary", "selected"):
            for policy in ("private", "images", "public"):
                name = recipe + "_" + policy
                plan = module.execution_plan(name)
                start = time.perf_counter()
                call = plan.prepare(module, {n: frame[n] for n in plan.argument_names})
                prepare_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                call.initialize()
                initialize_ms = (time.perf_counter() - start) * 1000
                assert call.info["execution_kind"] == "vulkan_commands"
                assert call.info["workspace_bytes"] == (0 if policy == "public" else 4)
                for scale in (1, 0, -2):
                    write(frame["scale"], [scale])
                    call()
                    pixels = [scale * ((x * 17 + y * 31) % 251) / 256 for y in range(height) for x in range(width)]
                    expected = pyramid(pixels, width, height, spec["levels"], spec["reduction"])
                    assert read(frame["result"]) == read(frame["dresult"]) == expected
                    loss = sum(x * x / 2 for x in expected)
                    assert abs(read(frame["loss"])[0] - loss) < max(3e-6, loss * 3e-5)
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
                        initialize_ms=initialize_ms,
                        submit_us=statistics.median(submits),
                        complete_us=statistics.median(completes),
                        info=call.info,
                    )
                )
                calls.append(call)
        frame["seed"].close()
        for call in calls:
            call()
        # An identically shaped public image cannot replace the plan's fixed one.
        replacement = runtime.native_resource(
            "image", "sampled", parameters=dict(format="r32f", storage=True, shape=[width, height])
        )
        plan = module.execution_plan("selected_public")
        bad = {n: frame[n] for n in plan.argument_names}
        bad["public_source"] = replacement
        raises(lambda: plan.prepare(module, bad))
        replacement.close()
        calls[0]()
        calls[0].close()  # pending work retires before its private counter/images
        spd.close()
        raises(calls[-1])
        for call in calls[1:]:
            call.close()
        module.close()
        for value in frame.values():
            value.close()
    if mode == "borrowed":
        ti.reset()
    else:
        assert "taichi_forge" not in sys.modules
    (folder / f"{mode}.json").write_text(json.dumps(dict(load_ms=load_ms, costs=costs), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
