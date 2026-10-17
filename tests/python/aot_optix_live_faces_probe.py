"""Source-independent OptiX queries whose per-instance tables change on device."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_optix_instances_probe import BASE_TRANSFORMS, RAYS, SECOND, oracle, plan_parts as instance_parts
from tests.python.aot_optix_probe import VERTICES, native, raises, read, write

POLICIES = {
    "split": ["table", "table", "front_only"],
    "shared": ["table", "table", "two_sided"],
    "uniform": ["two_sided", "front_only", "two_sided"],
    "two_sided": ["two_sided"] * 3,
    "single": ["table"],
}


def plan_parts(policy):
    operations, resources = instance_parts("two_sided")
    operations = operations[4:]
    resources = [item for item in resources if item["name"] not in ("vertices", "transforms")]
    if policy == "single":
        resources = [
            resources[0],
            dict(
                native(
                    "triangle_scene",
                    dict(context="context", vertices="seed", indices="indices"),
                    vertex_count=6,
                    triangle_count=2,
                ),
                name="scene",
            ),
            resources[-1],
        ]
    else:
        operations.insert(0, native("instance_refit", dict(scene="scene", transforms="requested")))
    rules = POLICIES[policy]
    if "table" in rules:
        operations.insert(0, dict(kind="kernel", entry="update_rules", arguments=["epoch", "face0", "face1"]))
    for operation in operations:
        if operation.get("operation") != "query":
            continue
        operation["version"] = 2
        operation["parameters"].pop("front_only")
        operation["parameters"].update(face_rules=rules, face_table_count=rules.count("table"))
        for i in range(rules.count("table")):
            operation["bindings"][f"faces_{i}"] = "face0" if i == 0 or policy == "shared" else "face1"
    return operations, resources


def expected(policy, epoch, transforms):
    values = [epoch, 1]
    other = values if policy == "shared" else [1 - epoch, 1]
    rules = dict(
        split=values + other + [1],
        shared=values + other + [0],
        uniform=[0, 0, 1, 1, 0],
        two_sided=[0] * 5,
        single=values + [0, 0, 0],
    )[policy]
    if policy == "single":
        transforms = [row[:] for row in BASE_TRANSFORMS]
        transforms[1][3] = transforms[2][3] = 1000
    result = oracle([0, 0, 0], transforms, rules)
    if policy == "single":
        for i in range(len(RAYS)):
            if result["ids"][4 * i + 3]:
                result["ids"][4 * i + 2] = 0
    return result


def check(frame, policy, epoch, transforms):
    for name, target in expected(policy, epoch, transforms).items():
        actual = read(frame[name], "i" if name in ("ids", "occluded") else "f")
        assert len(actual) == len(target)
        assert all(abs(a - b) < 2e-5 for a, b in zip(actual, target)), (policy, epoch, name, actual, target)


def main(mode, artifact, adapter, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("optix", adapter_path=adapter)
        assert runtime.native_operation_capabilities("optix_query", version=2) == {"cuda_ordered"}
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        retained, records = [], []
        for policy in POLICIES:
            plan = module.execution_plan(policy)
            for binding in range(2 if policy == "split" else 1):
                shapes = dict(
                    seed=(6, 3),
                    seed1=(3, 3),
                    indices=(2, 3),
                    indices1=(1, 3),
                    initial_transforms=(3, 12),
                    requested=(3, 12),
                    epoch=(1,),
                    face0=(2,),
                    face1=(2,),
                    rays=(len(RAYS), 8),
                    hits=(len(RAYS), 4),
                    ids=(len(RAYS), 4),
                    occluded=(len(RAYS),),
                    loss=(1,),
                    dhits=(len(RAYS), 4),
                )
                types = dict(
                    indices="i32", indices1="i32", ids="i32", occluded="i32", epoch="i32", face0="u32", face1="u32"
                )
                frame = {name: runtime.ndarray(types.get(name, "f32"), shapes[name]) for name in plan.argument_names}
                write(frame["seed"], VERTICES)
                write(frame["indices"], range(6), "i")
                write(frame["rays"], sum(RAYS, []))
                write(frame["hits"], [-99] * (4 * len(RAYS)))
                if policy != "single":
                    write(frame["seed1"], SECOND)
                    write(frame["indices1"], range(3), "i")
                    write(frame["initial_transforms"], sum(BASE_TRANSFORMS, []))
                if "table" in POLICIES[policy]:
                    # Values are not ready until update_rules executes. A v1-style
                    # cold validation or snapshot would reject/cache these sentinels.
                    write(frame["face0"], [0xFFFFFFFF] * 2, "I")
                    write(frame["face1"], [0xFFFFFFFF] * 2, "I")
                if policy == "split" and binding == 0:
                    bad_size = runtime.ndarray("u32", 3)
                    try:
                        raises(lambda: plan.prepare(module, dict(frame, face0=bad_size)), "compact")
                    finally:
                        bad_size.close()
                    document = json.loads(plan.to_json())
                    for item in document["operations"]:
                        if item.get("operation") == "query":
                            item["parameters"]["face_rules"] = ["table", "table"]
                    wrong_count = ExecutionPlan.from_json(json.dumps(document))
                    raises(lambda: wrong_count.prepare(module, frame), "instance count")
                    document = json.loads(plan.to_json())
                    document["resources"].append(
                        dict(kind="ndarray_view", name="alias", array="ids", offset=0, shape=[2])
                    )
                    for item in document["operations"]:
                        if item.get("operation") == "query":
                            item["bindings"]["faces_0"] = "alias"
                    alias = ExecutionPlan.from_json(json.dumps(document))
                    raises(lambda: alias.prepare(module, frame), "overlap")
                start = time.perf_counter()
                call = plan.prepare(module, frame)
                prepare_ms = (time.perf_counter() - start) * 1000
                assert read(frame["hits"]) == [-99] * (4 * len(RAYS))
                mirrored = [row[:] for row in BASE_TRANSFORMS]
                # Keep the transformed plane away from ray 1's exact t_max;
                # interval endpoint behavior is checked separately from faces.
                mirrored[0][0], mirrored[0][10] = -1, 1.25
                mirrored[1][0], mirrored[1][5] = 2, 0.5
                for step, transforms in enumerate((BASE_TRANSFORMS, BASE_TRANSFORMS, mirrored, BASE_TRANSFORMS)):
                    epoch = (step + binding) % 2
                    if "epoch" in frame:
                        write(frame["epoch"], [epoch], "i")
                    if "requested" in frame:
                        write(frame["requested"], sum(transforms, []))
                    call()
                    check(frame, policy, epoch, transforms)
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
                retained.append((frame, call, policy, epoch))
        # Independent bindings still work after all subsequent preparations.
        for frame, call, policy, epoch in retained[:2]:
            call()
            check(frame, policy, epoch, BASE_TRANSFORMS)
        retained[0][1]()
        retained[0][1].close()
        retained[-1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for frame, call, _, _ in retained[1:]:
            raises(call)
            call.close()
        for frame, *_ in retained:
            for value in frame.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(dict(mode=mode, load_ms=load_ms, records=records), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
