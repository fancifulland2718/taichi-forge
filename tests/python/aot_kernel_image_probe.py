"""Independent consumption of qualified image kernel bindings and capture leases."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime

if __package__:
    from .aot_optix_probe import raises, read, write
else:
    from aot_optix_probe import raises, read, write


def native(provider, operation, bindings, **parameters):
    return dict(
        kind="native",
        provider=provider,
        operation=operation,
        version=1,
        bindings=bindings,
        parameters=parameters,
    )


def plan_parts(policy):
    resources, initialization = [], []
    for name, shape, fmt, seed, filter in (
        ("one", [4], "r8", "pixels1", "linear"),
        ("linear", [3, 5], "r32f", "pixels2", "linear"),
        ("nearest", [3, 5], "r32f", "pixels2", "nearest"),
        ("three", [2, 3, 4], "r32f", "pixels3", "linear"),
    ):
        bindings = {} if seed == "pixels2" and policy != "seeded" else dict(pixels=seed)
        if policy == "initialized":
            bindings = {}
            initialization.append(
                native("image", "upload", dict(image=name, pixels=seed))
            )
        resources.append(
            dict(
                native(
                    "image", "sampled", bindings, shape=shape, format=fmt, filter=filter
                ),
                name=name,
            )
        )
    operations = []
    if policy not in ("seeded", "initialized"):
        operations += [
            dict(kind="kernel", entry="paint", arguments=["scale", "pixels2"]),
            native("image", "upload", dict(image="linear", pixels="pixels2")),
            native("image", "upload", dict(image="nearest", pixels="pixels2")),
        ]
    operations.append(
        dict(
            kind="kernel",
            entry="sample",
            arguments=["one", "linear", "nearest", "three", "output"],
        )
    )
    if policy == "ordered":
        resources += [
            dict(native("optix", "context", {}), name="context"),
            dict(
                native(
                    "optix",
                    "triangle_scene",
                    dict(context="context", vertices="vertices", indices="indices"),
                    vertex_count=3,
                    triangle_count=1,
                ),
                name="scene",
            ),
        ]
        operations.append(
            native(
                "optix",
                "query",
                dict(scene="scene", rays="rays", hits="hits", hit_indices="ids"),
                ray_count=1,
            )
        )
    operations += [
        dict(kind="fill32", destination="doutput", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["output", "loss"]),
        dict(kind="kernel", entry="backward", arguments=["output", "loss"]),
    ]
    return dict(
        operations=operations,
        resources=resources,
        initialization=initialization or None,
        gradients=dict(output="doutput", loss="dloss"),
        binding_strategy=(
            "cuda_immutable_argument_frames" if policy == "family" else None
        ),
    )


def check(frame, scale, arch="cuda"):
    expected = [
        128 / 255,
        1,
        4 * scale,
        11 * scale,
        5.75 * scale,
        6 * scale,
        11,
        14,
        11.5,
    ]
    if arch == "vulkan":
        expected = [
            128 / 255,
            1,
            12 * scale,
            5 * scale,
            3.25 * scale,
            4 * scale,
            22,
            13,
            11.5,
        ]
    # CUDA linear interpolation uses fractional fixed precision.
    for name in ("output", "doutput"):
        actual = read(frame[name])
        assert all(abs(a - b) < 3e-3 for a, b in zip(actual, expected)), (
            name,
            actual,
            expected,
        )
    assert abs(read(frame["loss"])[0] - sum(v * v * 0.5 for v in expected)) < 0.04
    if "hits" in frame:
        assert read(frame["hits"]) == [1, 0.25, 0.5, 0]
        assert read(frame["ids"], "i") == [0, 0, 0, 1]


def main(mode, artifact, adapter, output, arch="cuda"):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    retained, costs = [], []
    with runtime:
        if adapter:
            runtime.configure_provider("optix", adapter_path=adapter)
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        policies = ["seeded", "updated"] + (
            ["family", "initialized"] if arch == "cuda" else []
        )
        if arch == "vulkan":
            policies += ["depth", "depth_seeded", "depth_initialized"]
        policies += ["ordered"] if adapter else []
        shapes = dict(
            pixels1=(4,),
            pixels2=(3, 5),
            pixels3=(2, 3, 4),
            output=(9,),
            doutput=(9,),
            loss=(1,),
            dloss=(1,),
            scale=(1,),
            vertices=(3, 3),
            indices=(1, 3),
            rays=(1, 8),
            hits=(1, 4),
            ids=(1, 4),
        )
        for policy in policies:
            plan = module.execution_plan(policy)
            for binding in range(2):
                frame = {
                    name: runtime.ndarray(
                        (
                            "u8"
                            if name == "pixels1"
                            else "i32" if name in ("indices", "ids") else "f32"
                        ),
                        shapes[name],
                    )
                    for name in plan.argument_names
                }
                write(frame["pixels1"], [0, 64, 128, 255], "B")
                write(
                    frame["pixels2"],
                    (
                        [i / 16 for i in range(15)]
                        if policy in ("depth_seeded", "depth_initialized")
                        else list(range(15))
                    ),
                )
                write(frame["pixels3"], list(range(24)))
                write(frame["output"], [-99] * 9)
                if policy == "ordered":
                    write(frame["vertices"], [-1, -1, 1, 1, -1, 1, 0, 1, 1])
                    write(frame["indices"], [0, 1, 2], "i")
                    write(frame["rays"], [0, 0, 0, 0.01, 0, 0, 1, 3])
                if policy == "updated" and binding == 0:
                    missing = json.loads(plan.to_json())
                    missing["operations"] = [
                        op
                        for op in missing["operations"]
                        if op.get("provider") != "image"
                    ]
                    raises(
                        lambda: ExecutionPlan.from_json(json.dumps(missing)).prepare(
                            module, frame
                        ),
                        "unconditional upload",
                    )
                    wrong = json.loads(plan.to_json())
                    next(
                        op for op in wrong["operations"] if op.get("entry") == "sample"
                    )["arguments"][0] = "three"
                    invalid = ExecutionPlan.from_json(json.dumps(wrong))
                    raises(
                        lambda: invalid.prepare(
                            module,
                            {
                                k: v
                                for k, v in frame.items()
                                if k in invalid.argument_names
                            },
                        ),
                        "dimension/format",
                    )
                    wrong = json.loads(plan.to_json())
                    next(
                        op for op in wrong["operations"] if op.get("entry") == "sample"
                    )["arguments"][-1] = "linear"
                    invalid = ExecutionPlan.from_json(json.dumps(wrong))
                    raises(
                        lambda: invalid.prepare(
                            module,
                            {
                                k: v
                                for k, v in frame.items()
                                if k in invalid.argument_names
                            },
                        ),
                        "sampled Texture",
                    )
                    conditional = json.loads(plan.to_json())
                    conditional["version"] = 11 if arch == "cuda" else 12
                    uploads = conditional["operations"][1:3]
                    conditional["operations"][1:3] = [
                        dict(
                            kind="branch",
                            selector="selector",
                            selection="nonzero",
                            has_default=False,
                            lowering=dict(
                                kind=(
                                    "cuda_conditional"
                                    if arch == "cuda"
                                    else "vulkan_host"
                                ),
                                version=1,
                            ),
                            condition=[
                                dict(kind="fill32", destination="selector", value=1)
                            ],
                            bodies=[uploads],
                        )
                    ]
                    selector = runtime.ndarray("i32", (1,))
                    raises(
                        lambda: ExecutionPlan.from_json(
                            json.dumps(conditional)
                        ).prepare(module, dict(frame, selector=selector)),
                        "unconditional upload",
                    )
                    selector.close()
                    assert read(frame["output"]) == [-99] * 9
                start = time.perf_counter()
                call = plan.prepare(module, frame)
                prepare_ms = (time.perf_counter() - start) * 1000
                assert call.info["execution_kind"] == (
                    "vulkan_commands"
                    if arch == "vulkan"
                    else "cuda_ordered" if policy == "ordered" else "cuda_graph"
                )
                assert read(frame["output"]) == [-99] * 9
                initialize_ms = None
                if policy.endswith("initialized"):
                    start = time.perf_counter()
                    call.initialize()
                    initialize_ms = (time.perf_counter() - start) * 1000
                    for name in ("pixels1", "pixels2", "pixels3"):
                        frame[name].close()
                if policy.endswith("seeded"):
                    write(frame["pixels2"], [99] * 15)
                for scale in (
                    (1,)
                    if policy in ("seeded", "initialized")
                    else (
                        (1 / 16,)
                        if policy in ("depth_seeded", "depth_initialized")
                        else (1 / 16, 0, 1 / 32) if policy == "depth" else (1, -2, 0, 3)
                    )
                ):
                    if "scale" in frame:
                        write(frame["scale"], [scale])
                    call()
                    check(frame, scale, arch)
                if policy == "family":
                    changed_pixels = runtime.ndarray("f32", shapes["pixels2"])
                    raises(
                        lambda: call.prepare_binding(
                            dict(frame, pixels2=changed_pixels)
                        ),
                        "non-updatable native operands",
                    )
                    changed_pixels.close()
                    # A rejected update is cold and leaves the original usable.
                    call()
                    check(frame, 3)
                    sibling_frame = dict(frame)
                    for name in (
                        "output",
                        "doutput",
                        "loss",
                        "dloss",
                        "scale",
                    ):
                        sibling_frame[name] = runtime.ndarray("f32", shapes[name])
                    sibling = call.prepare_binding(sibling_frame)
                    write(sibling_frame["scale"], [-1])
                    sibling()
                    check(sibling_frame, -1)
                    call()
                    check(frame, 3)
                    assert (
                        call.binding_family_info["identity"]
                        == sibling.binding_family_info["identity"]
                    )
                    call.close()
                    sibling()
                    check(sibling_frame, -1)
                    retained.append((sibling_frame, sibling))
                    call = sibling
                samples, submits = [], []
                for _ in range(8):
                    start = time.perf_counter_ns()
                    call()
                    submitted = time.perf_counter_ns()
                    runtime.wait()
                    samples.append((time.perf_counter_ns() - start) / 1000)
                    submits.append((submitted - start) / 1000)
                costs.append(
                    dict(
                        policy=policy,
                        binding=binding,
                        prepare_ms=prepare_ms,
                        initialize_ms=initialize_ms,
                        complete_us=statistics.median(samples),
                        submit_us=statistics.median(submits),
                        info=call.info,
                    )
                )
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
    Path(output).write_text(json.dumps(dict(load_ms=load_ms, cases=costs), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
