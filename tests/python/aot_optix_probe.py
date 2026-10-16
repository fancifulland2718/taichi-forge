"""Source-independent OptiX scene/refit/query consumer with a geometry oracle."""

import array
import json
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime

VERTICES = [-1, -1, 1, 1, -1, 1, 0, 1, 1, -1, -1, 2, 0, 1, 2, 1, -1, 2]
RAYS = [
    [0, 0, 0, 0.01, 0, 0, 1, 3],
    [0, 0, 0, 0.01, 0, 0, 1, 1.5],
    [0, 0, 3, 0.01, 0, 0, -1, 4],
    [4, 0, 0, 0.01, 0, 0, 1, 3],
]


def native(operation, bindings, **parameters):
    return dict(
        kind="native", provider="optix", operation=operation, version=1, parameters=parameters, bindings=bindings
    )


def plan_parts(policy):
    context = dict(native("context", {}), name="context")
    scene = dict(
        native(
            "triangle_scene",
            dict(context="context", vertices="seed", indices="indices"),
            vertex_count=6,
            triangle_count=2,
            allow_update=True,
        ),
        name="scene",
    )
    resources = [
        context,
        scene,
        dict(kind="ndarray", name="vertices", dtype="f32", shape=[6, 3]),
        dict(kind="ndarray", name="dloss", dtype="f32", shape=[1]),
    ]
    common = dict(scene="scene", rays="rays")
    parameters = dict(ray_count=4, front_only=policy == "front_only")
    if policy == "table":
        common["faces"] = "faces"
    operations = [
        dict(kind="kernel", entry="move", arguments=["seed", "shift", "vertices"]),
        native("refit", dict(scene="scene", vertices="vertices")),
        native("query", dict(common, hits="hits", hit_indices="ids"), **parameters),
        native("query", dict(common, hits="occluded"), occlusion=True, **parameters),
        dict(kind="fill32", destination="dhits", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["hits", "loss"]),
        dict(kind="kernel", entry="energy_grad", arguments=["hits", "loss"]),
    ]
    return operations, resources


def write(value, entries, dtype="f"):
    value.write(array.array(dtype, entries))


def read(value, dtype="f"):
    values = array.array(dtype)
    values.frombytes(value.read())
    return list(values)


def raises(fn, text=None):
    try:
        fn()
    except (RuntimeError, ValueError) as error:
        if text:
            assert text in str(error), str(error)
    else:
        raise AssertionError("invalid call succeeded")


def check(frame, shift, rules):
    hits, ids, gradients, occluded = [], [], [], []
    loss = 0.0
    for ray in RAYS:
        accepted = []
        if ray[0] == 0:
            for primitive, z in enumerate((1 + shift, 2 + shift)):
                front = ray[6] * (1 if primitive == 0 else -1) < 0
                t = (z - ray[2]) / ray[6]
                if (not rules[primitive] or front) and ray[3] <= t <= ray[7]:
                    accepted.append((t, primitive))
        if accepted:
            t, primitive = min(accepted)
            # Both independent triangles have the same center barycentrics.
            hits.extend([t, 0.25, 0.5, 0] if primitive == 0 else [t, 0.5, 0.25, 0])
            ids.extend([primitive, 0, 0, 1])
            gradients.extend([t, 0, 0, 0])
            occluded.append(1)
            loss += 0.5 * t * t
        else:
            hits.extend([-1, 0, 0, 0])
            ids.extend([-1, -1, -1, 0])
            gradients.extend([0, 0, 0, 0])
            occluded.append(0)
    for key, expected in (("hits", hits), ("dhits", gradients), ("loss", [loss])):
        actual = read(frame[key])
        assert len(actual) == len(expected)
        assert all(abs(a - b) < 1e-5 for a, b in zip(actual, expected)), (key, actual, expected)
    assert read(frame["ids"], "i") == ids
    assert read(frame["occluded"], "i") == occluded


def main(mode, artifact, adapter, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    records = []
    with runtime:
        assert runtime.native_operation_capabilities("optix_query") == {"cuda_ordered"}
        runtime.configure_provider("optix", adapter_path=adapter)
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        retained = []
        for policy in ("two_sided", "front_only", "table"):
            plan = module.execution_plan(policy)
            for binding in range(2):
                shapes = dict(
                    seed=(6, 3),
                    indices=(2, 3),
                    rays=(4, 8),
                    shift=(1,),
                    hits=(4, 4),
                    ids=(4, 4),
                    occluded=(4,),
                    loss=(1,),
                    dhits=(4, 4),
                    faces=(2,),
                )
                frame = {
                    name: runtime.ndarray(
                        "i32" if name in ("indices", "ids", "occluded", "faces") else "f32", shapes[name]
                    )
                    for name in plan.argument_names
                }
                write(frame["seed"], VERTICES)
                write(frame["indices"], range(6), "i")
                write(frame["rays"], sum(RAYS, []))
                write(frame["hits"], [-99] * 16)
                rules = [int(policy == "front_only")] * 2
                if policy == "table":
                    rules = [binding, 1]
                    write(frame["faces"], rules, "i")
                if binding == 0:
                    write(frame["indices"], [0, 1, 2, 3, 4, 6], "i")
                    raises(lambda: plan.prepare(module, frame), "index")
                    write(frame["indices"], range(6), "i")
                    if policy == "table":
                        write(frame["faces"], [2, 1], "i")
                        raises(lambda: plan.prepare(module, frame), "face rules")
                        write(frame["faces"], rules, "i")
                    if policy == "two_sided":
                        # Same byte count does not establish a valid ray layout.
                        flat_rays = runtime.ndarray("f32", 32)
                        try:
                            raises(lambda: plan.prepare(module, dict(frame, rays=flat_rays)), "shape")
                        finally:
                            flat_rays.close()
                        float_ids = runtime.ndarray("f32", (4, 4))
                        try:
                            raises(lambda: plan.prepare(module, dict(frame, ids=float_ids)), "compact")
                        finally:
                            float_ids.close()
                        document = json.loads(plan.to_json())
                        next(item for item in document["resources"] if item.get("name") == "scene")["parameters"][
                            "allow_update"
                        ] = False
                        frozen = ExecutionPlan.from_json(json.dumps(document))
                        raises(lambda: frozen.prepare(module, frame), "updateable")
                start = time.perf_counter()
                call = plan.prepare(module, frame)
                prepare_ms = (time.perf_counter() - start) * 1000
                assert call.info["execution_kind"] == "cuda_ordered"
                assert read(frame["hits"]) == [-99] * 16
                # Topology/material rules are frozen at the cold boundary.
                write(frame["indices"], [99] * 6, "i")
                if policy == "table":
                    write(frame["faces"], [1 - value for value in rules], "i")
                for shift in (0.0, 0.25, 4.0, 0.0):
                    write(frame["shift"], [shift])
                    call()
                    check(frame, shift, rules)
                samples, submits = [], []
                for _ in range(8):
                    start = time.perf_counter_ns()
                    call()
                    submitted = time.perf_counter_ns()
                    runtime.wait()
                    samples.append((time.perf_counter_ns() - start) / 1000)
                    submits.append((submitted - start) / 1000)
                records.append(
                    dict(
                        policy=policy,
                        binding=binding,
                        prepare_ms=prepare_ms,
                        warm_complete_us=statistics.median(samples),
                        warm_submit_us=statistics.median(submits),
                        info=call.info,
                    )
                )
                retained.append((frame, call))
                if policy == "front_only" and binding == 0:
                    # any_hit uses the extended ABI; only B is a valid forward
                    # candidate, so it has the same independent oracle here.
                    document = json.loads(plan.to_json())
                    for operation in document["operations"]:
                        if operation.get("operation") == "query":
                            operation["parameters"]["any_hit"] = True
                    write(frame["indices"], range(6), "i")
                    any_hit = ExecutionPlan.from_json(json.dumps(document)).prepare(module, frame)
                    try:
                        any_hit()
                        check(frame, 0, rules)
                    finally:
                        any_hit.close()
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
    from pathlib import Path

    Path(output).write_text(json.dumps(dict(mode=mode, load_ms=load_ms, records=records), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
