"""Source-free indirect command/count consumption with exact view ranges."""

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
    costs = []
    with runtime:
        module = runtime.load_module(artifact)
        assert runtime.native_operation_capabilities("vulkan_graphics_pass_indirect")
        document = json.loads(module.execution_plan("public_views").to_json())
        operation = next(op for op in document["operations"] if op.get("provider") == "vulkan_graphics")
        indexed = "index" in operation["parameters"]["draws"][1]
        frame = {}
        for name, dtype, count, usage in (
            ("vertices", "f32", 15, ("vertex",)),
            ("background", "f32", 15, ("vertex",)),
            ("indices", "u32", 3, ("index",)),
            ("active", "u32", 1, ("storage",)),
            ("commands", "u32", 6 if indexed else 5, ("storage", "indirect")),
            ("count", "u32", 2, ("storage", "indirect")),
            ("bank", "u32", 32, ("storage", "indirect")),
            ("output", "f32", 4, ("storage",)),
            ("dout", "f32", 4, ("storage",)),
            ("loss", "f32", 1, ("storage",)),
        ):
            frame[name] = runtime.ndarray(dtype, count, usage=usage)
        for name, color in (("vertices", [1, 0, 0]), ("background", [0, 0, 1])):
            write(frame[name], [v for xy in ((-1, -1), (3, -1), (-1, 3)) for v in (*xy, *color)])
        write(frame["indices"], [0, 1, 2], "I")
        write(frame["bank"], [0] * 32, "I")
        for mutation, message in (
            ("operation", "pass_indirect"),
            ("short", "too small"),
            ("wide", "too small"),
            ("stride", "four-byte"),
            ("index", "index limit"),
        ):
            altered = copy.deepcopy(document)
            op = next(op for op in altered["operations"] if op.get("provider") == "vulkan_graphics")
            indirect = op["parameters"]["draws"][1]["indirect"]
            if mutation == "operation":
                op["operation"], op["version"] = "pass_direct", 4
            elif mutation == "short":
                next(r for r in altered["resources"] if r["name"] == "commands")["shape"][0] -= 1
            elif mutation == "wide":
                indirect["command_offset"] = (1 << 32) + 4
            elif mutation == "stride":
                indirect["stride"] = 12
            else:
                indirect["index_element_limit"] = 0 if indexed else 1
            plan = ExecutionPlan.from_json(json.dumps(altered))
            raises(lambda: plan.prepare(module, {name: frame[name] for name in plan.argument_names}), message)
        public_plan = module.execution_plan("public_views")
        wrong_usage = runtime.ndarray("u32", 32, usage=("vertex",))
        wrong_type = runtime.ndarray("f32", 32, usage=("storage", "indirect"))
        for bank in (wrong_usage, wrong_type):
            inputs = dict(frame, bank=bank)
            raises(lambda: public_plan.prepare(module, {name: inputs[name] for name in public_plan.argument_names}))
            bank.close()
        calls = []
        for route in ("export", "public_views", "private_views"):
            plan = module.execution_plan(route)
            started = time.perf_counter()
            call = module.prepare_execution(route, {name: frame[name] for name in plan.argument_names})
            prepare_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            call.initialize()
            initialize_ms = (time.perf_counter() - started) * 1000
            assert call.info["execution_kind"] == "vulkan_commands"
            assert call.info["resource_bytes"] == (132 if route == "private_views" else 4)
            host, complete = [], []
            for active in (0, 1, 0, 1):
                write(frame["active"], [active], "I")
                started = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                host.append((submitted - started) / 1000)
                complete.append((time.perf_counter_ns() - started) / 1000)
                expected = [1, 0, 0, 1] if active else [0, 0, 1, 1]
                assert read(frame["output"]) == expected
                assert read(frame["dout"]) == expected
                assert read(frame["loss"]) == [1]
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
        replacement = runtime.ndarray("u32", 32, usage=("storage", "indirect"))
        write(replacement, [0] * 32, "I")
        write(frame["bank"], [0] * 32, "I")
        inputs = dict(frame, bank=replacement)
        rebound = module.prepare_execution("public_views", {name: inputs[name] for name in public_plan.argument_names})
        rebound()
        runtime.wait()
        assert read(frame["dout"]) == [1, 0, 0, 1]
        rebound.close()
        frame["bank"].close()
        raises(calls[1])
        calls[2]()
        runtime.wait()
        assert read(frame["output"]) == [1, 0, 0, 1]
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
