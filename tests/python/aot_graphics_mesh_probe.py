"""Consume mesh/task exports without importing the producer or compiling code."""

import copy
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
        assert runtime.native_resource_capabilities("vulkan_graphics_pipeline", version=2)
        assert runtime.native_operation_capabilities("vulkan_graphics_pass_direct", version=4)
        frame = {
            name: runtime.ndarray("f32", count)
            for name, count in (("choice", 4), ("color", 4), ("output", 4), ("dout", 4), ("loss", 1))
        }
        spec = json.loads((folder / "pipeline.json").read_text())
        blobs = {
            f"shader_{i}": runtime.host_blob((folder / f"shader_{i}.spv").read_bytes())
            for i in range(spec["shader_count"])
        }
        raises(
            lambda: runtime.native_resource("vulkan_graphics", "pipeline", version=1, parameters=spec, bindings=blobs),
            "vertex/fragment",
        )
        frame["pipeline"] = runtime.native_resource(
            "vulkan_graphics", "pipeline", version=2, parameters=spec, bindings=blobs
        )
        for blob in blobs.values():
            blob.close()
        frame["target"] = runtime.native_resource(
            "image", "sampled", parameters=dict(format="rgba32f", shape=[8, 8], attachment=True)
        )
        document = json.loads(module.execution_plan("private").to_json())
        for mutation, message in (
            ("downgrade", "version 4"),
            ("zero", "group counts exceed"),
            ("oversized", "group counts exceed"),
            ("both", "mesh"),
            ("kind", "kinds must agree"),
        ):
            altered = copy.deepcopy(document)
            operation = next(op for op in altered["operations"] if op.get("provider") == "vulkan_graphics")
            draw = operation["parameters"]["draws"][0]
            if mutation == "downgrade":
                operation["version"] = 3
            elif mutation in ("zero", "oversized"):
                draw["mesh"] = [0 if mutation == "zero" else 0xFFFFFFFF, 2, 2]
            elif mutation == "both":
                draw["element_count"] = 3
            else:
                del draw["mesh"]
                draw["element_count"] = 3
            plan = ExecutionPlan.from_json(json.dumps(altered))
            raises(lambda: plan.prepare(module, {name: frame[name] for name in plan.argument_names}), message)
        calls = []
        for route in ("private", "public"):
            plan = module.execution_plan(route)
            started = time.perf_counter()
            call = module.prepare_execution(route, {name: frame[name] for name in plan.argument_names})
            prepare_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            call.initialize()
            initialize_ms = (time.perf_counter() - started) * 1000
            assert call.info["execution_kind"] == "vulkan_commands"
            assert call.info["resource_bytes"] == 4
            assert call.info["native_resource_bytes"] == (1024 if route == "private" else 0)
            host, complete = [], []
            for color in ([1, 0, 0, 1], [0, 1, 0, 1], [0.25, 0.5, 0.75, 1], [0, 0, 0, 1]):
                write(frame["choice"], color)
                started = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                host.append((submitted - started) / 1000)
                complete.append((time.perf_counter_ns() - started) / 1000)
                assert read(frame["output"]) == color
                assert read(frame["dout"]) == color
                assert read(frame["loss"]) == [sum(v * v * 0.5 for v in color)]
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
        replacement = runtime.ndarray("f32", 4)
        write(frame["choice"], [0.5, 0, 0.25, 1])
        plan = module.execution_plan("public")
        inputs = dict(frame, color=replacement)
        rebound = module.prepare_execution("public", {name: inputs[name] for name in plan.argument_names})
        rebound()
        runtime.wait()
        assert read(frame["dout"]) == [0.5, 0, 0.25, 1]
        frame["pipeline"].close()
        raises(rebound)
        raises(calls[1])
        calls[0]()
        module.close()
        if mode == "borrowed":
            ti.reset()
        rebound.close()
        for call in calls:
            call.close()
        replacement.close()
        for value in frame.values():
            value.close()
    folder.joinpath(mode + ".json").write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
