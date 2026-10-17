"""Packed vector records and matrix/scalar-3D transforms in AOT OptiX."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_optix_instances_probe import BASE_TRANSFORMS, RAYS, SECOND, check, plan_parts as scalar_parts
from tests.python.aot_optix_probe import VERTICES, raises, read, write


def plan_parts(policy):
    operations, resources = scalar_parts("table")
    for item in resources:
        if item["name"] == "vertices":
            item.update(shape=[6], element_shape=[3])
        elif item["name"] == "transforms":
            item.update(shape=[3], element_shape=[3, 4])
    if policy == "scalar_transforms":
        for item in operations:
            if item.get("operation") == "instance_refit" and "transforms" in item["bindings"]:
                item["bindings"]["transforms"] = "requested_scalar"
    return operations, resources


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
        load_ms = (time.perf_counter() - start) * 1000
        retained, records = [], []
        for policy in ("tensor", "scalar_transforms"):
            plan = module.execution_plan(policy)
            layouts = dict(
                seed=((6,), (3,)),
                seed1=((3,), (3,)),
                indices=((2,), (3,)),
                indices1=((1,), (3,)),
                initial_transforms=((3,), (3, 4)),
                requested=((3,), (3, 4)),
                shift=((3,), ()),
                rays=((len(RAYS),), (8,)),
                hits=((len(RAYS),), (4,)),
                ids=((len(RAYS),), (4,)),
                occluded=((len(RAYS),), (1,)),
                loss=((1,), ()),
                dhits=((len(RAYS),), (4,)),
                faces=((5,), (1,)),
            )
            if policy == "scalar_transforms":
                layouts.update(
                    initial_transforms=((3, 3, 4), ()),
                    requested_scalar=((3, 3, 4), ()),
                    faces=((5, 1), ()),
                    occluded=((len(RAYS), 1), ()),
                )
            integer_names = {"indices", "indices1", "ids", "occluded", "faces"}
            frame = {
                name: runtime.ndarray(
                    "i32" if name in integer_names else "f32", layouts[name][0], element_shape=layouts[name][1]
                )
                for name in plan.argument_names
            }
            write(frame["seed"], VERTICES)
            write(frame["seed1"], SECOND)
            write(frame["indices"], range(6), "i")
            write(frame["indices1"], range(3), "i")
            write(frame["initial_transforms"], sum(BASE_TRANSFORMS, []))
            write(frame["rays"], sum(RAYS, []))
            write(frame["hits"], [-99] * (4 * len(RAYS)))
            rules = [1, 1, 0, 1, 1] if policy == "tensor" else [0, 1, 1, 1, 0]
            write(frame["faces"], rules, "i")
            if policy == "tensor":
                wrong_ids = runtime.ndarray("i32", (len(RAYS),), element_shape=(2, 2))
                wrong_transforms = runtime.ndarray("f32", (3, 4, 3))
                try:
                    raises(lambda: plan.prepare(module, dict(frame, ids=wrong_ids)), "layout")
                    raises(lambda: plan.prepare(module, dict(frame, initial_transforms=wrong_transforms)), "shape")
                finally:
                    wrong_ids.close()
                    wrong_transforms.close()
            start = time.perf_counter()
            call = plan.prepare(module, frame)
            prepare_ms = (time.perf_counter() - start) * 1000
            assert read(frame["hits"]) == [-99] * (4 * len(RAYS))
            write(frame["indices"], [99] * 6, "i")
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
                if "requested_scalar" in frame:
                    write(frame["requested_scalar"], sum(transforms, []))
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
                    prepare_ms=prepare_ms,
                    warm_submit_us=statistics.median(submits),
                    warm_complete_us=statistics.median(completes),
                    info=call.info,
                )
            )
            retained.append((frame, call, rules))
        retained[0][1]()
        check(retained[0][0], [0, 0, 0], BASE_TRANSFORMS, retained[0][2])
        retained[0][1].close()
        retained[1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(retained[1][1])
        retained[1][1].close()
        for frame, *_ in retained:
            for value in frame.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(dict(mode=mode, load_ms=load_ms, records=records), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
