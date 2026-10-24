"""Independent runtime consumer of a versioned shared-AS raster plan."""

import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from aot_optix_probe import raises, read, write

VERTICES = [-1, -1, 0.5, 1, -1, 0.5, 0, 1, 0.5]
POSE = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    costs, calls, owners = [], [], []
    with runtime:
        assert runtime.native_operation_capabilities("vulkan_graphics_pass_direct", version=2) == {"vulkan_inline"}
        assert runtime.native_operation_capabilities("vulkan_graphics_pass_direct") == {"vulkan_inline"}
        module = runtime.load_module(artifact)
        frame = {
            name: runtime.ndarray(dtype, shape)
            for name, dtype, shape in (
                ("seed_vertices", "f32", (3, 3)),
                ("seed_indices", "i32", (1, 3)),
                ("poses", "f32", (1, 12)),
                ("shift", "f32", 1),
                ("output", "f32", 8),
                ("dout", "f32", 8),
                ("loss", "f32", 1),
            )
        }
        frame["vertices"] = runtime.ndarray("f32", 6, usage=("vertex",))
        write(frame["seed_vertices"], VERTICES)
        write(frame["seed_indices"], [0, 1, 2], "i")
        write(frame["poses"], POSE)
        write(frame["vertices"], [-1, -1, 3, -1, -1, 3])
        owners += list(frame.values())
        blas = runtime.native_resource(
            "vulkan_ray",
            "triangle_blas",
            parameters=dict(vertex_count=3, triangle_count=1),
            bindings=dict(vertices=frame["seed_vertices"], indices=frame["seed_indices"]),
        )
        scene = runtime.native_resource(
            "vulkan_ray",
            "instance_tlas",
            parameters=dict(instance_count=1, custom_indices=[17]),
            bindings=dict(blas_0=blas, transforms=frame["poses"]),
        )
        owners += [scene, blas]
        frame["scene"] = scene
        # Schema v1 must not silently accept v2 AS declarations. Reflection also
        # rejects missing or duplicate descriptor assignments before recording.
        document = json.loads(module.execution_plan("private").to_json())
        for mutation, message in (
            ("downgrade", "Unknown native schema binding"),
            ("missing", "missing"),
            ("duplicate", "duplicated"),
        ):
            altered = copy.deepcopy(document)
            raster = next(op for op in altered["operations"] if op.get("provider") == "vulkan_graphics")
            if mutation == "downgrade":
                raster["version"] = 1
            elif mutation == "missing":
                raster["parameters"]["draws"][0]["scenes"] = []
            else:
                raster["parameters"]["draws"][0]["scenes"] *= 2
            plan = ExecutionPlan.from_json(json.dumps(altered))
            raises(lambda: plan.prepare(module, {name: frame[name] for name in plan.argument_names}), message)
        for route in ("private", "public"):
            plan = module.execution_plan(route)
            start = time.perf_counter()
            call = module.prepare_execution(route, {name: frame[name] for name in plan.argument_names})
            assert call.info["execution_kind"] == "vulkan_commands"
            prepare_ms = (time.perf_counter() - start) * 1000
            host, complete = [], []
            for shift in (0, 4, 0):
                write(frame["shift"], [shift])
                start = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                host.append((submitted - start) / 1000)
                complete.append((time.perf_counter_ns() - start) / 1000)
                expected = [1, 0, 17, 17, 0, -1, -1, 17] if shift == 0 else [0, -1, -1, -1] * 2
                assert read(frame["output"]) == expected
                assert read(frame["dout"]) == expected
                assert read(frame["loss"]) == [sum(0.5 * x * x for x in expected)]
            costs.append(
                dict(
                    route=route,
                    prepare_ms=prepare_ms,
                    host_us=statistics.median(host),
                    complete_us=statistics.median(complete),
                    info=call.info,
                )
            )
            calls.append(call)
        scene.close()
        raises(calls[1])
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
    Path(artifact).with_name(mode + ".json").write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
