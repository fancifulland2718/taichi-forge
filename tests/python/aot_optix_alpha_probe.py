"""Managed alpha images in an independent consumer; no producer ray objects."""

import json
import statistics
import sys
import time
from pathlib import Path

from taichi_forge_runtime.aot import ExecutionPlan, Runtime

if __package__:
    from .aot_optix_probe import RAYS, raises, read, write
else:
    from aot_optix_probe import RAYS, raises, read, write


def native(provider, operation, bindings, version=1, **parameters):
    return dict(
        kind="native",
        provider=provider,
        operation=operation,
        version=version,
        parameters=parameters,
        bindings=bindings,
    )


def plan_parts(channels):
    resources = [
        dict(native("optix", "context", {}), name="context"),
        dict(
            native(
                "optix",
                "triangle_gas",
                dict(context="context", vertices="vertices", indices="indices"),
                vertex_count=3,
                triangle_count=1,
            ),
            name="gas",
        ),
        dict(
            native(
                "optix",
                "instance_scene",
                dict(
                    context="context", gas_0="gas", gas_1="gas", transforms="transforms"
                ),
                instance_count=2,
                opaque=[False, False],
                custom_indices=[11, 22],
                allow_update=True,
            ),
            name="scene",
        ),
        dict(
            native(
                "image",
                "sampled",
                {},
                shape=[2, 4],
                format="r32f" if channels == 1 else "rgba32f",
                filter="nearest",
                address_modes=["clamp_to_edge", "clamp_to_edge"],
            ),
            name="texture",
        ),
        dict(kind="ndarray", name="dloss", dtype="f32", shape=[1]),
    ]
    common = dict(
        scene="scene",
        rays="rays",
        faces_0="faces",
        uvs_0="uva",
        uvs_1="uvb",
        image_0="texture",
        image_1="texture",
    )
    policy = dict(
        ray_count=4,
        face_rules=["table", "front_only"],
        face_table_count=1,
        alpha_masks=[dict(channel=channels - 1), dict(channel=channels - 1)],
        mask_count=2,
    )
    operations = [
        dict(
            kind="kernel",
            entry=f"paint_{channels}",
            arguments=["config", "pixels", "uva", "uvb", "faces"],
        ),
        native("image", "upload", dict(image="texture", pixels="pixels")),
        native("optix", "instance_refit", dict(scene="scene", transforms="transforms")),
        native(
            "optix",
            "query",
            dict(common, hits="hits", hit_indices="ids"),
            version=3,
            **policy,
        ),
        native(
            "optix",
            "query",
            dict(common, hits="occluded"),
            version=3,
            occlusion=True,
            **policy,
        ),
        dict(kind="fill32", destination="dhits", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["hits", "loss"]),
        dict(kind="kernel", entry="energy_grad", arguments=["hits", "loss"]),
    ]
    return operations, resources


def check(frame, config, rays=RAYS):
    alpha_a, alpha_b, face_a, uv_swap = config
    if uv_swap:
        alpha_a, alpha_b = alpha_b, alpha_a
    expected_hits, expected_ids, expected_grad, expected_occ = [], [], [], []
    loss = 0.0
    for ray in rays:
        candidates = []
        if ray[0] == 0:
            for instance, z, alpha, cull in (
                (0, 1, alpha_a, face_a),
                (1, 2, alpha_b, 1),
            ):
                front = ray[6] * (1 if instance == 0 else -1) < 0
                t = (z - ray[2]) / ray[6]
                if alpha >= 0.5 and (not cull or front) and ray[3] <= t <= ray[7]:
                    candidates.append((t, instance))
        if candidates:
            t, instance = min(candidates)
            expected_hits.extend([t, 0.25, 0.5, 0])
            expected_ids.extend([0, instance, (11, 22)[instance], 1])
            expected_grad.extend([t, 0, 0, 0])
            expected_occ.append(1)
            loss += t * t * 0.5
        else:
            expected_hits.extend([-1, 0, 0, 0])
            expected_ids.extend([-1, -1, -1, 0])
            expected_grad.extend([0] * 4)
            expected_occ.append(0)
    for key, expected in (
        ("hits", expected_hits),
        ("dhits", expected_grad),
        ("loss", [loss]),
    ):
        actual = read(frame[key])
        assert all(abs(a - b) < 1e-5 for a, b in zip(actual, expected)), (
            key,
            config,
            actual,
            expected,
        )
    assert read(frame["ids"], "i") == expected_ids
    assert read(frame["occluded"], "i") == expected_occ


def main(mode, artifact, adapter, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    retained, timings = [], []
    with runtime:
        runtime.configure_provider("optix", adapter_path=adapter)
        assert runtime.native_operation_capabilities("optix_query", version=3) == {
            "cuda_ordered"
        }
        assert runtime.native_operation_capabilities("image_upload") == {"cuda_capture"}
        module = runtime.load_module(artifact)
        for channels in (1, 4):
            plan = module.execution_plan(f"alpha_{channels}")
            for binding in range(2):
                shapes = dict(
                    vertices=(3, 3),
                    indices=(1, 3),
                    transforms=(2, 3, 4),
                    rays=(4, 8),
                    pixels=(2, 4) if channels == 1 else (2, 4, 4),
                    config=(4,),
                    uva=(3, 2),
                    uvb=(3, 2),
                    faces=(1,),
                    hits=(4, 4),
                    ids=(4, 4),
                    occluded=(4,),
                    dhits=(4, 4),
                    loss=(1,),
                )
                frame = {
                    name: runtime.ndarray(
                        (
                            "i32"
                            if name in ("indices", "faces", "ids", "occluded")
                            else "f32"
                        ),
                        shapes[name],
                    )
                    for name in plan.argument_names
                }
                write(frame["vertices"], [-1, -1, 1, 1, -1, 1, 0, 1, 1])
                write(frame["indices"], [0, 1, 2], "i")
                # A identity; B rotates 180 degrees about X, then translates Z.
                write(
                    frame["transforms"],
                    [
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        -1,
                        0,
                        0,
                        0,
                        0,
                        -1,
                        3,
                    ],
                )
                write(frame["rays"], sum(RAYS, []))
                write(frame["hits"], [-99] * 16)
                if binding == 0:
                    missing_upload = json.loads(plan.to_json())
                    missing_upload["operations"] = [
                        op
                        for op in missing_upload["operations"]
                        if op.get("provider") != "image"
                    ]
                    raises(
                        lambda: ExecutionPlan.from_json(
                            json.dumps(missing_upload)
                        ).prepare(module, frame),
                        "unconditional upload",
                    )
                    conditional = json.loads(plan.to_json())
                    conditional["version"] = 11
                    upload = conditional["operations"][1]
                    conditional["operations"][1] = dict(
                        kind="branch",
                        selector="faces",
                        selection="nonzero",
                        has_default=False,
                        lowering=dict(kind="cuda_conditional", version=1),
                        condition=[dict(kind="fill32", destination="faces", value=1)],
                        bodies=[[upload]],
                    )
                    raises(
                        lambda: ExecutionPlan.from_json(
                            json.dumps(conditional)
                        ).prepare(module, frame),
                        "unconditional upload",
                    )
                    opaque = json.loads(plan.to_json())
                    next(r for r in opaque["resources"] if r["name"] == "scene")[
                        "parameters"
                    ]["opaque"][0] = True
                    raises(
                        lambda: ExecutionPlan.from_json(json.dumps(opaque)).prepare(
                            module, frame
                        ),
                        "opaque instance",
                    )
                    wrong_pixels = runtime.ndarray(
                        "f32", (4, 2) if channels == 1 else (4, 2, 4)
                    )
                    try:
                        raises(
                            lambda: plan.prepare(
                                module, dict(frame, pixels=wrong_pixels)
                            ),
                            "shape",
                        )
                    finally:
                        wrong_pixels.close()
                start = time.perf_counter()
                call = plan.prepare(module, frame)
                prepare_ms = (time.perf_counter() - start) * 1000
                assert call.info["execution_kind"] == "cuda_ordered"
                assert read(frame["hits"]) == [-99] * 16
                # Mask indices must use the GAS snapshot as well as traversal.
                write(frame["indices"], [99] * 3, "i")
                for config in (
                    (0, 1, 0, 0),
                    (1, 1, 0, 0),
                    (1, 1, 1, 0),
                    (1, 0, 1, 0),
                    (0, 0, 0, 0),
                    (1, 0, 0, 1),
                    (1, 0, 0, 0),
                ):
                    write(frame["config"], config)
                    call()
                    check(frame, config)
                transforms = read(frame["transforms"])
                translated = transforms[:]
                translated[3] = translated[15] = 8
                write(frame["transforms"], translated)
                call()
                check(frame, (0, 0, 0, 0))
                write(frame["transforms"], transforms)
                call()
                check(frame, config)
                samples, submits = [], []
                for _ in range(12):
                    start = time.perf_counter_ns()
                    call()
                    submitted = time.perf_counter_ns()
                    runtime.wait()
                    samples.append((time.perf_counter_ns() - start) / 1000)
                    submits.append((submitted - start) / 1000)
                timings.append(
                    dict(
                        channels=channels,
                        binding=binding,
                        prepare_ms=prepare_ms,
                        warm_complete_us=statistics.median(samples),
                        warm_submit_us=statistics.median(submits),
                        info=call.info,
                    )
                )
                if binding == 0:
                    # First-hit semantics have one candidate inside this interval.
                    first = json.loads(plan.to_json())
                    for op in first["operations"]:
                        if op.get("operation") == "query":
                            op["parameters"]["any_hit"] = True
                    write(frame["indices"], [0, 1, 2], "i")
                    first_call = ExecutionPlan.from_json(json.dumps(first)).prepare(
                        module, frame
                    )
                    short_rays = [ray[:-1] + [1.5] for ray in RAYS]
                    write(frame["rays"], sum(short_rays, []))
                    for config in ((1, 1, 0, 0), (0, 1, 0, 0)):
                        write(frame["config"], config)
                        first_call()
                        check(frame, config, short_rays)
                    first_call.close()
                    write(frame["rays"], sum(RAYS, []))
                    # An explicit initial seed is copied once at preparation.
                    # Subsequent writes to its source are not implicit uploads.
                    seeded = json.loads(plan.to_json())
                    seeded["operations"] = [
                        op
                        for op in seeded["operations"]
                        if op.get("provider") != "image"
                    ]
                    next(r for r in seeded["resources"] if r["name"] == "texture")[
                        "bindings"
                    ] = {"pixels": "pixels"}
                    pixels = [0.0] * (8 * channels)
                    pixels[-1] = 1.0
                    write(frame["pixels"], pixels)
                    seeded_call = ExecutionPlan.from_json(json.dumps(seeded)).prepare(
                        module, frame
                    )
                    write(frame["config"], (1, 0, 0, 0))
                    seeded_call()
                    check(frame, (0, 1, 0, 0))
                    seeded_call.close()
                retained.append((frame, call))
        retained[0][1]()
        retained[0][1].close()
        retained[-1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for frame, call in retained:
            raises(call)
            call.close()
            for value in frame.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
