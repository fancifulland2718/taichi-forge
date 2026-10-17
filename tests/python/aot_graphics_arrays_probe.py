"""Consume exported raster arrays without producer imports or shader compilers."""

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
    colors = ([1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1])
    bank_values = [-91] * 256
    for i, color in enumerate(colors):
        bank_values[64 * (i + 1) : 64 * (i + 1) + 4] = color
    costs, calls = [], []
    with runtime:
        module = runtime.load_module(artifact)
        assert runtime.native_operation_capabilities("vulkan_graphics_pass_direct", version=3)
        frame = {}
        for name, dtype, count, usage in (
            ("vertices", "f32", 15, ("vertex",)),
            ("selector", "u32", 4, ("storage", "uniform")),
            ("choice", "u32", 1, ("storage",)),
            ("bank", "f32", 256, ("storage",)),
            ("output", "f32", 4, ("storage",)),
            ("dout", "f32", 4, ("storage",)),
            ("loss", "f32", 1, ("storage",)),
            *((f"c{i}", "f32", 4, ("storage",)) for i in range(3)),
        ):
            frame[name] = runtime.ndarray(dtype, count, usage=usage)
        write(frame["vertices"], [v for xy in ((-1, -1), (3, -1), (-1, 3)) for v in (*xy, 0, 0, 0)])
        write(frame["selector"], [0] * 4, "I")
        write(frame["bank"], bank_values)
        for i, color in enumerate(colors):
            write(frame[f"c{i}"], color)
        document = json.loads(module.execution_plan("public_views").to_json())
        for mutation, message in (
            ("downgrade", "version 3"),
            ("both", "exactly one"),
            ("short", "at least two"),
            ("uniform", "storage buffers"),
            ("mismatch", "match"),
            ("outside", "exceeds parent shape"),
        ):
            altered = copy.deepcopy(document)
            raster = next(op for op in altered["operations"] if op.get("provider") == "vulkan_graphics")
            table = raster["parameters"]["draws"][0]["buffers"][0]
            if mutation == "downgrade":
                raster["version"] = 2
            elif mutation == "both":
                table["array"] = table["arrays"][0]
            elif mutation == "short":
                table["arrays"] = table["arrays"][:1]
            elif mutation == "uniform":
                table["storage"] = False
            elif mutation == "mismatch":
                table["arrays"] = table["arrays"][:2]
            else:
                next(r for r in altered["resources"] if r["name"] == "c0")["offset"] = 256
            plan = ExecutionPlan.from_json(json.dumps(altered))
            raises(lambda: plan.prepare(module, {name: frame[name] for name in plan.argument_names}), message)
        for route in ("export", "public_views", "private_views"):
            plan = module.execution_plan(route)
            started = time.perf_counter()
            call = module.prepare_execution(route, {name: frame[name] for name in plan.argument_names})
            prepare_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            call.initialize()
            initialize_ms = (time.perf_counter() - started) * 1000
            assert call.info["resource_bytes"] == (1028 if route == "private_views" else 4)
            assert call.info["execution_kind"] == "vulkan_commands"
            host, complete = [], []
            for selected in (0, 2, 1, 3):
                write(frame["choice"], [selected], "I")
                started = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                host.append((submitted - started) / 1000)
                complete.append((time.perf_counter_ns() - started) / 1000)
                color = colors[selected % 3]
                expected = [color[0], color[1], color[2] + 0.5, 1]
                assert read(frame["output"]) == expected
                assert read(frame["dout"]) == expected
                assert read(frame["loss"]) == [sum(v * v * 0.5 for v in expected)]
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
        # OpArrayLength observes the view's finite range, not its parent allocation.
        shortened = copy.deepcopy(document)
        next(r for r in shortened["resources"] if r["name"] == "c0")["shape"] = [3]
        short_plan = ExecutionPlan.from_json(json.dumps(shortened))
        with short_plan.prepare(module, {name: frame[name] for name in short_plan.argument_names}) as call:
            write(frame["choice"], [0], "I")
            call()
            runtime.wait()
            assert read(frame["output"]) == [1, 0, 0.375, 1]
            assert read(frame["dout"]) == [1, 0, 0.375, 1]
        # A fresh prepared binding must read the replacement parent, not a stale set.
        replacement = runtime.ndarray("f32", 256)
        replaced_values = bank_values.copy()
        replaced_values[64:68] = [0.25, 0.5, 0.75, 1]
        write(replacement, replaced_values)
        plan = module.execution_plan("public_views")
        inputs = dict(frame, bank=replacement)
        rebound = module.prepare_execution("public_views", {name: inputs[name] for name in plan.argument_names})
        write(frame["choice"], [0], "I")
        rebound()
        runtime.wait()
        assert read(frame["output"]) == [0.25, 0.5, 1.25, 1]
        assert read(frame["dout"]) == [0.25, 0.5, 1.25, 1]
        rebound.close()
        frame["bank"].close()
        raises(calls[1])
        calls[2]()
        runtime.wait()
        assert read(frame["output"]) == [1, 0, 0.5, 1]
        calls[2]()
        module.close()
        if mode == "borrowed":
            ti.reset()
        for call in calls:
            call.close()
        replacement.close()
        for value in frame.values():
            value.close()
    Path(artifact).with_name(mode + ".json").write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
