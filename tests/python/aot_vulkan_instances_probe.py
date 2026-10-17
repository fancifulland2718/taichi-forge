"""Runtime-only shared Vulkan BLAS/TLAS, GPU refit and owner retirement."""

import json
import os
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime

if __package__:
    from .aot_vulkan_ray_probe import VERTICES, native, read, write, raises
else:
    from aot_vulkan_ray_probe import VERTICES, native, read, write, raises

TRANSFORMS = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 1]
RAYS = [
    [0, 0, 0, 0.01, 0, 0, 1, 4],
    [3, 0, 0, 0.01, 0, 0, 1, 4],
    [3, 0, 0, 0.01, 0, 0, 1, 1.5],
    [0, 0, 3, 0.01, 0, 0, -1, 4],
    [8, 0, 0, 0.01, 0, 0, 1, 4],
]


def plan_parts(private):
    resources = [
        dict(kind="ndarray", name="vertices", dtype="f32", shape=[3, 3]),
        dict(kind="ndarray", name="transforms", dtype="f32", shape=[2, 12]),
        dict(kind="ndarray", name="dloss", dtype="f32", shape=[1]),
    ]
    if private:
        resources += [
            dict(
                native("triangle_blas", dict(vertices="seed", indices="indices"), vertex_count=3, triangle_count=1),
                name="blas",
            ),
            dict(
                native(
                    "instance_tlas",
                    dict(blas_0="blas", blas_1="blas", transforms="poses"),
                    instance_count=2,
                    custom_indices=[17, 21],
                ),
                name="scene",
            ),
        ]
    return [
        dict(kind="kernel", entry="move", arguments=["seed", "poses", "control", "vertices", "transforms"]),
        native("refit", dict(scene="blas", vertices="vertices")),
        native("instance_refit", dict(scene="scene", transforms="transforms")),
        native("query", dict(scene="scene", rays="rays", hits="hits", hit_indices="ids"), ray_count=len(RAYS)),
        native("query", dict(scene="scene", rays="rays", hits="legacy"), ray_count=len(RAYS)),
        dict(kind="fill32", destination="dhits", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["hits", "loss"]),
        dict(kind="kernel", entry="energy_grad", arguments=["hits", "loss"]),
    ], resources


def check(frame, dz=0, dx=0, custom=(17, 21), masks=(255, 255), gradients=True):
    expected = dict(hits=[], ids=[], legacy=[], dhits=[], loss=[0.0])
    for ray in RAYS:
        hit = None
        for ordinal, (x, z) in enumerate(((dx, 1 + dz), (3 + dx, 2 + dz))):
            t = (z - ray[2]) / ray[6]
            if masks[ordinal] and abs(ray[0] - x) < 1e-6 and ray[3] <= t <= ray[7]:
                if hit is None or t < hit[0]:
                    hit = t, ordinal
        if hit:
            t, ordinal = hit
            expected["hits"] += [t, 0.25, 0.5, 0]
            expected["ids"] += [0, ordinal, custom[ordinal], 1]
            expected["legacy"] += [t, 0, custom[ordinal], 1]
            expected["dhits"] += [t, 0, 0, 0]
            expected["loss"][0] += 0.5 * t * t
        else:
            expected["hits"] += [-1, 0, 0, 0]
            expected["ids"] += [-1, -1, -1, 0]
            expected["legacy"] += [-1, -1, -1, 0]
            expected["dhits"] += [0, 0, 0, 0]
    for key in ("hits", "ids", "legacy", "dhits", "loss") if gradients else ("hits", "ids"):
        actual = read(frame[key], "i" if key == "ids" else "f")
        assert len(actual) == len(expected[key])
        assert all(abs(a - b) < 1e-6 for a, b in zip(actual, expected[key])), (key, actual, expected[key])


def create_blas(runtime, opaque=True):
    seed = runtime.ndarray("f32", 3, element_shape=(3,))
    indices = runtime.ndarray("i32", (1, 3))
    write(seed, VERTICES)
    write(indices, [0, 1, 2], "i")
    blas = runtime.native_resource(
        "vulkan_ray",
        "triangle_blas",
        parameters=dict(vertex_count=3, triangle_count=1, opaque=opaque),
        bindings=dict(vertices=seed, indices=indices),
    )
    seed.close()
    indices.close()
    return blas


def create_tlas(runtime, blas, custom=(17, 21), masks=(255, 255), sbt_record_offsets=None):
    seed = runtime.ndarray("f32", 2, element_shape=(3, 4))
    write(seed, TRANSFORMS)
    parameters = dict(instance_count=2, custom_indices=list(custom), visibility_masks=list(masks))
    if sbt_record_offsets is not None:
        parameters["sbt_record_offsets"] = list(sbt_record_offsets)
    tlas = runtime.native_resource(
        "vulkan_ray",
        "instance_tlas",
        parameters=parameters,
        bindings=dict(blas_0=blas, blas_1=blas, transforms=seed),
    )
    seed.close()
    return tlas


def frame_arrays(runtime):
    shapes = dict(
        seed=(3, 3),
        poses=(2, 12),
        control=(2,),
        rays=(len(RAYS), 8),
        hits=(len(RAYS), 4),
        ids=(len(RAYS), 4),
        legacy=(len(RAYS), 4),
        dhits=(len(RAYS), 4),
        loss=(1,),
    )
    result = {key: runtime.ndarray("i32" if key == "ids" else "f32", shape) for key, shape in shapes.items()}
    write(result["seed"], VERTICES)
    write(result["poses"], TRANSFORMS)
    write(result["control"], [0, 0])
    write(result["rays"], sum(RAYS, []))
    return result


def rejection_checks(runtime, module, blas, scene, frame):
    args = dict(frame, blas=blas, scene=scene)
    raises(lambda: module.prepare_execution("public", dict(args, scene=blas)), "type mismatch")
    raises(lambda: module.prepare_execution("public", dict(args, blas=scene)), "type mismatch")
    for parameters, bindings, message in (
        (dict(instance_count=2, custom_indices=[1]), dict(blas_0=blas, blas_1=blas), "metadata"),
        (dict(instance_count=2, visibility_masks=[255, 256]), dict(blas_0=blas, blas_1=blas), "bit width"),
        (dict(instance_count=2, custom_indices=[0, 1 << 24]), dict(blas_0=blas, blas_1=blas), "bit width"),
        (dict(instance_count=2, sbt_record_offsets=[1]), dict(blas_0=blas, blas_1=blas), "metadata"),
        (dict(instance_count=2, sbt_record_offsets=[0, 1 << 24]), dict(blas_0=blas, blas_1=blas), "bit width"),
        (dict(instance_count=2, sbt_record_offsets=[-1, 0]), dict(blas_0=blas, blas_1=blas), "integer"),
        (dict(instance_count=2, sbt_record_offsets=[True, 0]), dict(blas_0=blas, blas_1=blas), "integer"),
        (dict(instance_count=2), dict(blas_0=scene, blas_1=blas), "type mismatch"),
        (dict(instance_count=2), dict(blas_0=blas), "bindings"),
    ):
        raises(
            lambda: runtime.native_resource(
                "vulkan_ray", "instance_tlas", parameters=parameters, bindings=dict(bindings, transforms=frame["poses"])
            ),
            message,
        )
    bad = runtime.ndarray("i32", (2, 12))
    raises(
        lambda: runtime.native_resource(
            "vulkan_ray",
            "instance_tlas",
            parameters=dict(instance_count=2),
            bindings=dict(blas_0=blas, blas_1=blas, transforms=bad),
        ),
        "dtype",
    )
    bad.close()


def main(mode, artifact, output):
    start = time.perf_counter()
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    costs = dict(runtime_ms=(time.perf_counter() - start) * 1000)
    with runtime:
        assert runtime.native_operation_capabilities("vulkan_ray_instance_refit") == {"vulkan_inline"}
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        costs["load_ms"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        blas = create_blas(runtime, opaque=False)
        scene = create_tlas(runtime, blas)
        costs["resources_ms"] = (time.perf_counter() - start) * 1000
        frame = frame_arrays(runtime)
        initial = ExecutionPlan(
            [
                native("query", dict(scene="scene", rays="rays", hits="hits", hit_indices="ids"), ray_count=len(RAYS)),
            ]
        ).prepare(module, dict(scene=scene, **{name: frame[name] for name in ("rays", "hits", "ids")}))
        initial()
        check(frame, gradients=False)  # Cold build must already use the supplied device matrices.
        initial.close()
        write(frame["hits"], [99] * (len(RAYS) * 4))
        start = time.perf_counter()
        call = module.prepare_execution("public", dict(frame, blas=blas, scene=scene))
        costs["prepare_ms"] = (time.perf_counter() - start) * 1000
        costs["public_info"] = call.info
        assert read(frame["hits"]) == [99] * (len(RAYS) * 4)
        for dz, dx in ((0, 0), (0.75, 0), (4, 0), (0, 8), (0, 0)):
            write(frame["control"], [dz, dx])
            call()
            check(frame, dz, dx)
        rejection_checks(runtime, module, blas, scene, frame)

        # A second TLAS shares one BLAS but has independent masks/IDs and outputs.
        other_module = runtime.load_module(artifact)
        other_scene = create_tlas(runtime, blas, (23, 29), (255, 0))
        other_frame = frame_arrays(runtime)
        # Subrange device matrix packing: scalar offset1, then exact [2,3,4].
        packed = runtime.ndarray("f32", 25)
        write(packed, [123] + TRANSFORMS)
        other_plan = ExecutionPlan(
            [
                native("instance_refit", dict(scene="scene", transforms="matrices")),
                native("query", dict(scene="scene", rays="rays", hits="hits", hit_indices="ids"), ray_count=len(RAYS)),
            ],
            resources=[dict(kind="ndarray_view", name="matrices", array="packed", offset=1, shape=[2, 3, 4])],
        )
        other = other_plan.prepare(
            other_module,
            dict(scene=other_scene, packed=packed, **{name: other_frame[name] for name in ("rays", "hits", "ids")}),
        )
        for dz in (0.5, 0):
            write(frame["control"], [dz, 0])
            call()
            other()
            check(frame, dz)
            check(other_frame, dz, custom=(23, 29), masks=(255, 0), gradients=False)
            assert read(packed) == [123] + TRANSFORMS
        # Closing one TLAS does not retire recordings using its sibling.
        scene.close()
        raises(call)
        other()
        check(other_frame, custom=(23, 29), masks=(255, 0), gradients=False)
        replacement = create_tlas(runtime, blas)
        call = module.prepare_execution("public", dict(frame, blas=blas, scene=replacement))
        call()
        check(frame)
        submits, totals = [], []
        for _ in range(8):
            start = time.perf_counter_ns()
            call()
            submitted = time.perf_counter_ns()
            runtime.wait()
            submits.append((submitted - start) / 1000)
            totals.append((time.perf_counter_ns() - start) / 1000)
        costs.update(submit_us=statistics.median(submits), complete_us=statistics.median(totals))
        call()
        other()
        blas.close()  # Transitive dependency retirement across both TLAS owners.
        raises(call)
        raises(other)
        blas.close()

        indices = runtime.ndarray("i32", (1, 3))
        write(indices, [0, 1, 2], "i")
        start = time.perf_counter()
        private = module.prepare_execution("private", dict(frame, indices=indices))
        costs["private_prepare_ms"] = (time.perf_counter() - start) * 1000
        costs["private_info"] = private.info
        assert private.info["workspace_bytes"] > costs["public_info"]["workspace_bytes"]
        for dz, dx in ((0, 0), (0, 8), (4, 0), (0, 0)):
            write(frame["control"], [dz, dx])
            private()
            check(frame, dz, dx)
        if mode == "borrowed":
            private()
            ti.reset()
            raises(private, "closed")
        else:
            assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
