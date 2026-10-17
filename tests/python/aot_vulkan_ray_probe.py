"""Runtime-only Vulkan AS reconstruction, replay, retirement and query oracle."""

import json
import os
import statistics
import sys
import time
from pathlib import Path

from taichi_forge_runtime.aot import ExecutionPlan, Runtime

if __package__:
    from .aot_optix_probe import RAYS, read, write, raises
else:
    from aot_optix_probe import RAYS, read, write, raises

VERTICES = [-1, -1, 1, 1, -1, 1, 0, 1, 1]


def native(operation, bindings, **parameters):
    return dict(
        kind="native", provider="vulkan_ray", operation=operation, version=1, bindings=bindings, parameters=parameters
    )


def plan_parts(private):
    resources = [
        dict(kind="ndarray", name="vertices", dtype="f32", shape=[3, 3]),
        dict(kind="ndarray", name="dloss", dtype="f32", shape=[1]),
    ]
    if private:
        resources.insert(
            0,
            dict(
                native("triangle_scene", dict(vertices="seed", indices="indices"), vertex_count=3, triangle_count=1),
                name="scene",
            ),
        )
    operations = [
        dict(kind="kernel", entry="move", arguments=["seed", "shift", "vertices"]),
        native("refit", dict(scene="scene", vertices="vertices")),
        native("query", dict(scene="scene", rays="rays", hits="hits", hit_indices="ids"), ray_count=4),
        native("query", dict(scene="scene", rays="rays", hits="legacy"), ray_count=4),
        dict(kind="fill32", destination="dhits", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["hits", "loss"]),
        dict(kind="kernel", entry="energy_grad", arguments=["hits", "loss"]),
    ]
    return operations, resources


def check(frame, shift):
    hits, ids, legacy, gradients = [], [], [], []
    loss = 0.0
    for ray in RAYS:
        t = (1 + shift - ray[2]) / ray[6]
        if ray[0] == 0 and ray[3] <= t <= ray[7]:
            hits += [t, 0.25, 0.5, 0]
            ids += [0, 0, 0, 1]
            legacy += [t, 0, 0, 1]
            gradients += [t, 0, 0, 0]
            loss += 0.5 * t * t
        else:
            hits += [-1, 0, 0, 0]
            ids += [-1, -1, -1, 0]
            legacy += [-1, -1, -1, 0]
            gradients += [0, 0, 0, 0]
    for key, expected in (("hits", hits), ("legacy", legacy), ("dhits", gradients), ("loss", [loss])):
        actual = read(frame[key])
        assert len(actual) == len(expected)
        assert all(abs(a - b) < 1e-6 for a, b in zip(actual, expected)), (key, actual, expected)
    assert read(frame["ids"], "i") == ids


def create_scene(runtime):
    # Packed vector input and scalar Nx3 input have the same geometry contract.
    seed = runtime.ndarray("f32", 3, element_shape=(3,))
    indices = runtime.ndarray("i32", (1, 3))
    write(seed, VERTICES)
    write(indices, [0, 1, 2], "i")
    scene = runtime.native_resource(
        "vulkan_ray",
        "triangle_scene",
        parameters=dict(vertex_count=3, triangle_count=1),
        bindings=dict(vertices=seed, indices=indices),
    )
    # Successful construction completed the seed copy/build, not just enqueue.
    seed.close()
    indices.close()
    return scene


def frame_arrays(runtime):
    shapes = dict(seed=(3, 3), rays=(4, 8), shift=(1,), hits=(4, 4), ids=(4, 4), legacy=(4, 4), dhits=(4, 4), loss=(1,))
    frame = {name: runtime.ndarray("i32" if name == "ids" else "f32", shape) for name, shape in shapes.items()}
    write(frame["seed"], VERTICES)
    write(frame["rays"], sum(RAYS, []))
    write(frame["shift"], [0])
    return frame


def subrange_checks(runtime, module, scene, frame):
    parent = runtime.ndarray("f32", 49)
    ids = runtime.ndarray("u32", 4, element_shape=(4,))
    values = [123] + sum(RAYS, []) + [99] * 16
    views = [
        dict(kind="ndarray_view", name="rays", array="parent", offset=1, shape=[4, 8]),
        dict(kind="ndarray_view", name="hits", array="parent", offset=33, shape=[4, 4]),
    ]
    for typed in (True, False):
        write(parent, values)
        original = read(parent)
        bindings = dict(scene="scene", rays="rays", hits="hits")
        arguments = dict(scene=scene, parent=parent)
        if typed:
            bindings["hit_indices"] = "ids"
            arguments["ids"] = ids
        plan = ExecutionPlan([native("query", bindings, ray_count=4)], resources=views)
        call = plan.prepare(module, arguments)
        call()
        actual = read(parent)
        assert actual[:33] == original[:33]
        assert actual[33:] == read(frame["hits" if typed else "legacy"])
        if typed:
            assert read(ids, "I") == [x & 0xFFFFFFFF for x in read(frame["ids"], "i")]
        call.close()
    parent.close()
    ids.close()


def main(mode, artifact, output):
    start = time.perf_counter()
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    result = dict(runtime_ms=(time.perf_counter() - start) * 1000)
    with runtime:
        assert runtime.device_capabilities().get("spirv_has_ray_query") == 1
        assert runtime.native_operation_capabilities("vulkan_ray_query") == {"vulkan_inline"}
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        result["load_ms"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        scene = create_scene(runtime)
        result["resource_ms"] = (time.perf_counter() - start) * 1000
        frame = frame_arrays(runtime)
        write(frame["hits"], [99] * 16)
        start = time.perf_counter()
        call = module.prepare_execution("public", dict(frame, scene=scene))
        result["prepare_ms"] = (time.perf_counter() - start) * 1000
        assert read(frame["hits"]) == [99] * 16  # Preparation does not run the step.
        result["info"] = call.info
        result["private_bytes"] = call.info["resource_bytes"]
        for shift in (0, 0.75, 4, 0):
            write(frame["shift"], [shift])
            call()
            check(frame, shift)
        # Same native scene, another module and immutable output frame.
        other_module = runtime.load_module(artifact)
        other_frame = frame_arrays(runtime)
        other = other_module.prepare_execution("public", dict(other_frame, scene=scene))
        other()
        check(other_frame, 0)

        # Reject aliases by physical ranges, not just argument names or dtype.
        from taichi_forge_runtime.aot._runtime import NdArray

        alias = NdArray(frame["rays"]._memory, "f32", 1, (4, 4), ())
        raises(lambda: module.prepare_execution("public", dict(frame, scene=scene, hits=alias)), "overlap")
        flat = runtime.ndarray("f32", 32)
        raises(lambda: module.prepare_execution("public", dict(frame, scene=scene, rays=flat)), "shape")
        flat.close()
        # Invalid resource preparation must leave an already-prepared scene usable.
        bad = runtime.ndarray("u32", (1, 3))
        raises(
            lambda: runtime.native_resource(
                "vulkan_ray",
                "triangle_scene",
                parameters=dict(vertex_count=3, triangle_count=1),
                bindings=dict(vertices=frame["seed"], indices=bad),
            ),
            "dtype",
        )
        bad.close()
        call()
        check(frame, 0)
        subrange_checks(runtime, module, scene, frame)
        if mode == "borrowed":
            # Distinct imported registrations can refer to one physical buffer.
            storage = ti.ndarray(ti.f32, shape=(4, 8))
            first, second = runtime.import_ndarray(storage), runtime.import_ndarray(storage)
            imported_alias = NdArray(second._memory, "f32", 1, (4, 4), ())
            raises(
                lambda: module.prepare_execution("public", dict(frame, scene=scene, rays=first, hits=imported_alias)),
                "overlap",
            )
            first.close()
            second.close()
        submits, totals = [], []
        for _ in range(8):
            start = time.perf_counter_ns()
            call()
            submitted = time.perf_counter_ns()
            runtime.wait()
            submits.append((submitted - start) / 1000)
            totals.append((time.perf_counter_ns() - start) / 1000)
        result.update(submit_us=statistics.median(submits), complete_us=statistics.median(totals))

        call()  # Closing a shared scene retires every dependent recording.
        scene.close()
        raises(call)
        raises(other)
        scene.close()
        replacement = create_scene(runtime)
        rebound = module.prepare_execution("public", dict(frame, scene=replacement))
        rebound()
        check(frame, 0)
        # Persisted private reconstruction uses the same factory and can coexist.
        indices = runtime.ndarray("i32", (1, 3))
        write(indices, [0, 1, 2], "i")
        private = module.prepare_execution("private", dict(frame, indices=indices))
        result["private_info"] = private.info
        assert private.info["workspace_bytes"] > result["info"]["workspace_bytes"]
        for shift in (0.75, 4, 0):
            write(frame["shift"], [shift])
            private()
            check(frame, shift)
        if mode == "borrowed":
            private()
            ti.reset()
            raises(private, "closed")
        else:
            assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
