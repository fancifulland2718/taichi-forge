"""Stdlib-only shared-GAS consumer with an independent transformed-triangle oracle."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_optix_probe import VERTICES, native, raises, read, write

SECOND = [-1, -1, 1, 0, 1, 1, 1, -1, 1]
BASE_TRANSFORMS = [[1, 0, 0, x, 0, 1, 0, 0, 0, 0, 1, 0] for x in (0, 4, 8)]
RAYS = [
    [0, 0, 0, 0.01, 0, 0, 1, 3],
    [0, 0, 0, 0.01, 0, 0, 1, 1.5],
    [4, 0, 0, 0.01, 0, 0, 1, 3],
    [4, 0, 0, 0.01, 0, 0, 1, 0.5],
    [8, 0, 0, 0.01, 0, 0, 1, 3],
    [0, 0, 3, 0.01, 0, 0, -1, 4],
    [4, 0, 3, 0.01, 0, 0, -1, 4],
    [8, 0, 3, 0.01, 0, 0, -1, 4],
    [2, 0, 0, 0.01, 0, 0, 1, 3],
    [6, 0, 0, 0.01, 0, 0, 1, 3],
]


def plan_parts(policy):
    resources = [dict(native("context", {}), name="context")]
    for name, vertices, indices, count in (("gas0", "seed", "indices", 6), ("gas1", "seed1", "indices1", 3)):
        resources.append(
            dict(
                native(
                    "triangle_gas",
                    dict(context="context", vertices=vertices, indices=indices),
                    vertex_count=count,
                    triangle_count=count // 3,
                    allow_update=name == "gas0",
                ),
                name=name,
            )
        )
    resources.append(
        dict(
            native(
                "instance_scene",
                dict(context="context", transforms="initial_transforms", gas_0="gas0", gas_1="gas0", gas_2="gas1"),
                instance_count=3,
                custom_indices=[17, 29, 37],
                visibility_masks=[255, 7, 255],
                opaque=[True, False, True],
                allow_update=True,
            ),
            name="scene",
        )
    )
    resources += [
        dict(kind="ndarray", name="vertices", dtype="f32", shape=[6, 3]),
        dict(kind="ndarray", name="transforms", dtype="f32", shape=[3, 12]),
        dict(kind="ndarray", name="dloss", dtype="f32", shape=[1]),
    ]
    common = dict(scene="scene", rays="rays")
    if policy == "table":
        common["faces"] = "faces"
    parameters = dict(ray_count=len(RAYS), front_only=policy == "front_only")
    operations = [
        dict(kind="kernel", entry="animate", arguments=["seed", "shift", "vertices", "requested", "transforms"]),
        native("instance_refit", dict(scene="scene", transforms="transforms")),
        native("refit", dict(scene="gas0", vertices="vertices")),
        # This second IAS update is required after GAS bounds change. It must
        # preserve the just-produced instance transforms without host readback.
        native("instance_refit", dict(scene="scene")),
        native("query", dict(common, hits="hits", hit_indices="ids"), **parameters),
        native("query", dict(common, hits="occluded"), occlusion=True, **parameters),
        dict(kind="fill32", destination="dhits", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["hits", "loss"]),
        dict(kind="kernel", entry="energy_grad", arguments=["hits", "loss"]),
    ]
    return operations, resources


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def oracle(shift, transforms, rules, hidden=False):
    meshes = [VERTICES, VERTICES, SECOND]
    hits, ids, flags, gradients, loss = [], [], [], [], 0.0
    for ray in RAYS:
        nearest = None
        base = 0
        for instance, (source, transform) in enumerate(zip(meshes, transforms)):
            count = len(source) // 9
            for primitive in range(count):
                if hidden and instance == 2:
                    continue
                vertices = []
                for i in range(3):
                    vertex = source[9 * primitive + 3 * i : 9 * primitive + 3 * i + 3]
                    if instance < 2:
                        vertex = [x + y for x, y in zip(vertex, shift)]
                    vertices.append(
                        [dot(transform[4 * j : 4 * j + 3], vertex) + transform[4 * j + 3] for j in range(3)]
                    )
                a, b, c = vertices
                e1, e2 = sub(b, a), sub(c, a)
                direction = ray[4:7]
                if rules[base + primitive] and dot(direction, cross(e1, e2)) >= 0:
                    continue
                p = cross(direction, e2)
                determinant = dot(e1, p)
                if abs(determinant) < 1e-12:
                    continue
                offset = sub(ray[:3], a)
                u = dot(offset, p) / determinant
                q = cross(offset, e1)
                v = dot(direction, q) / determinant
                t = dot(e2, q) / determinant
                if u >= 0 and v >= 0 and u + v <= 1 and ray[3] <= t <= ray[7]:
                    if nearest is None or t < nearest[0]:
                        nearest = (t, u, v, primitive, instance)
            base += count
        if nearest is None:
            hits += [-1, 0, 0, 0]
            ids += [-1, -1, -1, 0]
            flags.append(0)
            gradients += [0, 0, 0, 0]
        else:
            t, u, v, primitive, instance = nearest
            hits += [t, u, v, 0]
            ids += [primitive, instance, (17, 29, 37)[instance], 1]
            flags.append(1)
            gradients += [t, 0, 0, 0]
            loss += 0.5 * t * t
    return dict(hits=hits, ids=ids, occluded=flags, dhits=gradients, loss=[loss])


def check(frame, shift, transforms, rules, hidden=False):
    for name, expected in oracle(shift, transforms, rules, hidden).items():
        actual = read(frame[name], "i" if name in ("ids", "occluded") else "f")
        assert len(actual) == len(expected)
        assert all(abs(a - b) < 2e-5 for a, b in zip(actual, expected)), (name, actual, expected)


def main(mode, artifact, adapter, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("optix", adapter_path=adapter)
        assert runtime.native_operation_capabilities("optix_instance_refit") == {"cuda_ordered"}
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        retained, records = [], []
        for policy in ("two_sided", "front_only", "table"):
            plan = module.execution_plan(policy)
            for binding in range(2):
                shapes = dict(
                    seed=(6, 3),
                    seed1=(3, 3),
                    indices=(2, 3),
                    indices1=(1, 3),
                    initial_transforms=(3, 12),
                    requested=(3, 12),
                    shift=(3,),
                    rays=(len(RAYS), 8),
                    hits=(len(RAYS), 4),
                    ids=(len(RAYS), 4),
                    occluded=(len(RAYS),),
                    loss=(1,),
                    dhits=(len(RAYS), 4),
                    faces=(5,),
                )
                frame = {
                    name: runtime.ndarray(
                        "i32" if name in ("indices", "indices1", "ids", "occluded", "faces") else "f32", shapes[name]
                    )
                    for name in plan.argument_names
                }
                write(frame["seed"], VERTICES)
                write(frame["seed1"], SECOND)
                write(frame["indices"], range(6), "i")
                write(frame["indices1"], range(3), "i")
                write(frame["initial_transforms"], sum(BASE_TRANSFORMS, []))
                write(frame["rays"], sum(RAYS, []))
                write(frame["hits"], [-99] * (len(RAYS) * 4))
                rules = [int(policy == "front_only")] * 5
                if policy == "table":
                    rules = [1, 1, 0, 1, 1] if binding == 0 else [0, 1, 1, 1, 0]
                    write(frame["faces"], rules, "i")
                if policy == "table" and binding == 0:
                    write(frame["initial_transforms"], [0] * 36)
                    raises(lambda: plan.prepare(module, frame), "finite invertible")
                    write(frame["initial_transforms"], sum(BASE_TRANSFORMS, []))
                    document = json.loads(plan.to_json())
                    document["resources"].insert(1, dict(native("context", {}), name="other_context"))
                    next(item for item in document["resources"] if item["name"] == "gas0")["bindings"][
                        "context"
                    ] = "other_context"
                    invalid = ExecutionPlan.from_json(json.dumps(document))
                    raises(lambda: invalid.prepare(module, frame), "same-context")
                    with_bad_size = runtime.ndarray("i32", 4)
                    raises(lambda: plan.prepare(module, dict(frame, faces=with_bad_size)), "compact")
                    with_bad_size.close()
                    document = json.loads(plan.to_json())
                    next(item for item in document["operations"] if item.get("operation") == "query")["bindings"][
                        "scene"
                    ] = "gas0"
                    invalid = ExecutionPlan.from_json(json.dumps(document))
                    raises(lambda: invalid.prepare(module, frame), "incompatible ports")
                start = time.perf_counter()
                call = plan.prepare(module, frame)
                prepare_ms = (time.perf_counter() - start) * 1000
                assert read(frame["hits"]) == [-99] * (len(RAYS) * 4)
                write(frame["indices"], [99] * 6, "i")
                if policy == "table":
                    write(frame["faces"], [1 - value for value in rules], "i")
                mirrored = [row[:] for row in BASE_TRANSFORMS]
                mirrored[0][0], mirrored[0][10] = -1, 1.5
                mirrored[1][0], mirrored[1][5], mirrored[1][10] = 2, 0.5, 1.5
                for shift, transforms in (
                    ([0, 0, 0], BASE_TRANSFORMS),
                    ([2, 0, 0], BASE_TRANSFORMS),
                    ([0, 0, 0.25], mirrored),
                    ([0, 0, 4], BASE_TRANSFORMS),
                    ([0, 0, 0], BASE_TRANSFORMS),
                ):
                    write(frame["shift"], shift)
                    write(frame["requested"], sum(transforms, []))
                    call()
                    check(frame, shift, transforms, rules)
                submits, completes = [], []
                for _ in range(8):
                    start = time.perf_counter_ns()
                    call()
                    submitted = time.perf_counter_ns()
                    runtime.wait()
                    submits.append((submitted - start) / 1000)
                    completes.append((time.perf_counter_ns() - start) / 1000)
                records.append(
                    dict(
                        policy=policy,
                        binding=binding,
                        prepare_ms=prepare_ms,
                        warm_submit_us=statistics.median(submits),
                        warm_complete_us=statistics.median(completes),
                        info=call.info,
                    )
                )
                retained.append((frame, call))
                if policy == "two_sided" and binding == 0:
                    document = json.loads(plan.to_json())
                    next(item for item in document["resources"] if item["name"] == "scene")["parameters"][
                        "visibility_masks"
                    ][2] = 0
                    write(frame["indices"], range(6), "i")
                    hidden = ExecutionPlan.from_json(json.dumps(document)).prepare(module, frame)
                    try:
                        hidden()
                        check(frame, [0, 0, 0], BASE_TRANSFORMS, rules, hidden=True)
                    finally:
                        hidden.close()
        retained[0][1]()
        retained[0][1].close()
        retained[-1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for frame, call in retained[1:]:
            raises(call)
            call.close()
        for frame, _ in retained:
            for value in frame.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(dict(mode=mode, load_ms=load_ms, records=records), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
