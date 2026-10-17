"""Rebuild exported shared geometry and IAS recordings in a fresh consumer."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_optix_instances_probe import BASE_TRANSFORMS, RAYS, SECOND, check
from tests.python.aot_optix_probe import VERTICES, raises, read, write


def main(mode, artifact, adapter, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("optix", adapter_path=adapter)
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        costs = dict(load_ms=(time.perf_counter() - start) * 1000, records=[])
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
            faces0=(2,),
            faces1=(1,),
        )
        retained = []
        costs["public_resources"] = check_public_resources(runtime, module, artifact, shapes)
        for policy in ("two_sided", "table", "query_only"):
            for binding in range(2):
                plan = module.execution_plan(policy)
                frame = {
                    name: runtime.ndarray(
                        (
                            "i32"
                            if name
                            in (
                                "indices",
                                "indices1",
                                "ids",
                                "occluded",
                                "faces0",
                                "faces1",
                            )
                            else "f32"
                        ),
                        shapes[name],
                    )
                    for name in plan.argument_names
                }
                write(frame["seed"], VERTICES)
                write(frame["seed1"], SECOND)
                write(frame["indices"], range(6), "i")
                write(frame["indices1"], range(3), "i")
                initial = [row[:] for row in BASE_TRANSFORMS]
                for row in initial:
                    row[11] += 0.25 * binding
                write(frame["initial_transforms"], sum(initial, []))
                write(frame["rays"], sum(RAYS, []))
                write(frame["hits"], [-99] * (4 * len(RAYS)))
                start = time.perf_counter()
                call = plan.prepare(module, frame)
                prepare_ms = (time.perf_counter() - start) * 1000
                assert read(frame["hits"]) == [-99] * (4 * len(RAYS))
                for name in ("indices", "indices1", "initial_transforms"):
                    write(
                        frame[name],
                        [99] * (6 if name == "indices" else 3 if name == "indices1" else 36),
                        "i" if name != "initial_transforms" else "f",
                    )
                if policy == "query_only":
                    write(frame["seed"], [99] * 18)
                    write(frame["seed1"], [99] * 9)
                mirrored = [row[:] for row in BASE_TRANSFORMS]
                mirrored[0][0], mirrored[0][10] = -1, 1.25
                mirrored[1][0], mirrored[1][5], mirrored[1][10] = 2, 0.5, 1.25
                cases = (
                    ([0, 0, 0], BASE_TRANSFORMS),
                    ([2, 0, 0], BASE_TRANSFORMS),
                    ([0, 0, 0.25], mirrored),
                    ([0, 0, 4], BASE_TRANSFORMS),
                    ([0, 0, 0], BASE_TRANSFORMS),
                )
                for index, (shift, transforms) in enumerate(cases):
                    rules = [0] * 5 if policy == "two_sided" else [index % 2, 1, 1, 1, (index + binding) % 2]
                    if "faces0" in frame:
                        write(frame["faces0"], rules[:2], "i")
                        write(frame["faces1"], rules[4:], "i")
                    if "requested" in frame:
                        write(frame["shift"], shift)
                        write(frame["requested"], sum(transforms, []))
                    call()
                    check(
                        frame,
                        [0, 0, 0] if policy == "query_only" else shift,
                        initial if policy == "query_only" else transforms,
                        rules,
                    )
                samples = []
                for _ in range(8):
                    start = time.perf_counter_ns()
                    call()
                    runtime.wait()
                    samples.append((time.perf_counter_ns() - start) / 1000)
                costs["records"].append(
                    dict(
                        policy=policy,
                        binding=binding,
                        prepare_ms=prepare_ms,
                        warm_complete_us=statistics.median(samples),
                        info=call.info,
                    )
                )
                retained.append(
                    (
                        frame,
                        call,
                        initial if policy == "query_only" else BASE_TRANSFORMS,
                        rules,
                    )
                )
        for frame, call, transforms, rules in retained[:-1]:
            call()
            check(frame, [0, 0, 0], transforms, rules)
            call.close()
            for array in frame.values():
                array.close()
        retained[-1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(retained[-1][1])
        retained[-1][1].close()
        for array in retained[-1][0].values():
            array.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


def check_public_resources(runtime, module, artifact, shapes):
    owners = []

    def array(dtype, shape, values):
        value = runtime.ndarray(dtype, shape)
        write(value, values, "i" if dtype == "i32" else "f")
        owners.append(value)
        return value

    context = runtime.native_resource("optix", "context")
    seed = array("f32", (6, 3), VERTICES)
    topology = array("i32", (2, 3), range(6))
    gas = runtime.native_resource(
        "optix",
        "triangle_gas",
        parameters=dict(vertex_count=6, triangle_count=2, allow_update=True),
        bindings=dict(context=context, vertices=seed, indices=topology),
    )
    gas1 = runtime.native_resource(
        "optix",
        "triangle_gas",
        parameters=dict(vertex_count=3, triangle_count=1),
        bindings=dict(
            context=context,
            vertices=array("f32", (3, 3), SECOND),
            indices=array("i32", (1, 3), range(3)),
        ),
    )
    scene = runtime.native_resource(
        "optix",
        "instance_scene",
        parameters=dict(
            instance_count=3,
            allow_update=True,
            custom_indices=[17, 29, 37],
            visibility_masks=[255, 7, 255],
            opaque=[True, False, True],
        ),
        bindings=dict(
            context=context,
            transforms=array("f32", (3, 12), sum(BASE_TRANSFORMS, [])),
            gas_0=gas,
            gas_1=gas,
            gas_2=gas1,
        ),
    )
    resources = dict(shared_context=context, shared_gas=gas, other_gas=gas1, shared_scene=scene)
    records, calls = [], []

    def frame_for(name):
        frame = {}
        for key in module.execution_plan(name).argument_names:
            if key in resources:
                frame[key] = resources[key]
                continue
            integer = key in (
                "indices",
                "indices1",
                "ids",
                "occluded",
                "faces0",
                "faces1",
            )
            frame[key] = runtime.ndarray("i32" if integer else "f32", shapes[key])
            owners.append(frame[key])
        for key, values, dtype in (
            ("seed", VERTICES, "f"),
            ("seed1", SECOND, "f"),
            ("indices", range(6), "i"),
            ("indices1", range(3), "i"),
            ("initial_transforms", sum(BASE_TRANSFORMS, []), "f"),
            ("rays", sum(RAYS, []), "f"),
            ("faces0", [1, 1], "i"),
            ("faces1", [1], "i"),
        ):
            if key in frame:
                write(frame[key], values, dtype)
        return frame

    peer = runtime.load_module(artifact)
    query_frame = frame_for("public_query")
    query = peer.prepare_execution("public_query", query_frame)
    wrong = runtime.native_resource(
        "optix",
        "triangle_gas",
        parameters=dict(vertex_count=6, triangle_count=2, allow_update=True),
        bindings=dict(context=context, vertices=seed, indices=topology),
    )
    public_frame = frame_for("public")
    raises(
        lambda: module.prepare_execution("public", dict(public_frame, shared_gas=wrong)),
        "alias mismatch",
    )
    wrong_scene = runtime.native_resource(
        "optix",
        "instance_scene",
        parameters=dict(
            instance_count=3,
            allow_update=True,
            custom_indices=[17, 29, 37],
            visibility_masks=[255, 7, 255],
            opaque=[True, False, True],
        ),
        bindings=dict(
            context=context,
            transforms=array("f32", (3, 12), sum(BASE_TRANSFORMS, [])),
            gas_0=gas,
            gas_1=wrong,
            gas_2=gas1,
        ),
    )
    # Even a query-only plan must preserve sharing inside one public IAS. It
    # does not have a second public GAS root to catch this mismatch indirectly.
    raises(
        lambda: peer.prepare_execution("public_query", dict(query_frame, shared_scene=wrong_scene)),
        "alias mismatch",
    )
    wrong_scene.close()
    wrong.close()
    wrong_context = runtime.native_resource("optix", "context")
    mixed_frame = frame_for("mixed")
    raises(
        lambda: module.prepare_execution("mixed", dict(mixed_frame, shared_context=wrong_context)),
        "alias mismatch",
    )
    wrong_context.close()
    for name in ("public", "mixed", "public_context"):
        frame = public_frame if name == "public" else mixed_frame if name == "mixed" else frame_for(name)
        start = time.perf_counter()
        call = module.prepare_execution(name, frame)
        prepare_ms = (time.perf_counter() - start) * 1000
        for shift in ([0, 0, 0], [0, 0, 4], [0, 0, 0]):
            write(frame["shift"], shift)
            write(frame["requested"], sum(BASE_TRANSFORMS, []))
            call()
            check(frame, shift, BASE_TRANSFORMS, [1] * 5)
            if name == "public":
                # An independent module observes exactly the updated GAS/IAS.
                query()
                check(query_frame, shift, BASE_TRANSFORMS, [1] * 5)
        records.append(
            dict(
                policy=name,
                prepare_ms=prepare_ms,
                resource_bytes=call.info["resource_bytes"],
            )
        )
        calls.append(call)
    # Both public and partially reconstructed resource chains retire together.
    calls[0]()
    context.close()
    for call in (*calls, query):
        raises(call)
        call.close()
    peer.close()
    for owner in (scene, gas, gas1, *owners):
        owner.close()
    return records


if __name__ == "__main__":
    main(*sys.argv[1:])
