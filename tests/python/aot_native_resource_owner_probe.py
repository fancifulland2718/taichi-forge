"""Source-independent public resource ownership and failure boundaries."""

import ctypes as C
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import weakref

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from taichi_forge_runtime.aot import _c_api as abi
from aot_optix_probe import raises, read, write


def native(provider, operation, bindings, **parameters):
    return dict(
        kind="native",
        provider=provider,
        operation=operation,
        version=1,
        parameters=parameters,
        bindings=bindings,
    )


def create_image(runtime, values):
    pixels = runtime.ndarray("f32", (3, 5))
    write(pixels, values)
    image = runtime.native_resource(
        "image",
        "sampled",
        parameters=dict(shape=[3, 5], format="r32f", filter="nearest"),
        bindings=dict(pixels=pixels),
    )
    pixels.close()  # Completed constructor inputs have no lingering edge.
    return image


def arrays(runtime):
    return {
        name: runtime.ndarray("f32", (1,) if name in ("loss", "dloss") else (4,))
        for name in ("output", "doutput", "loss", "dloss")
    }


def check(frame, values, arch):
    indices = [4, 11, 7, 7] if arch == "cuda" else [12, 5, 7, 7]
    expected = [float(values[i]) for i in indices]
    assert read(frame["output"]) == expected
    assert read(frame["doutput"]) == expected
    assert read(frame["loss"]) == [sum(0.5 * x * x for x in expected)]


def native_wrong_handle(runtime, handle):
    request = json.dumps(
        native("image", "sampled", dict(pixels="input"), shape=[3, 5], format="r32f")
    ).encode()
    arg = abi.Argument()
    arg.type, arg.value.native_resource = 6, handle
    named = abi.NamedArgument(b"input", arg)
    raises(
        lambda: runtime._api.call(
            runtime._api.ti_create_native_resource_ext,
            runtime._handle,
            len(request),
            request,
            1,
            C.byref(named),
            None,
        ),
        "foreign",
    )


def image_checks(runtime, module, artifact, arch):
    start = time.perf_counter()
    image = create_image(runtime, list(range(15)))
    create_ms = (time.perf_counter() - start) * 1000
    other_module = runtime.load_module(artifact)
    first, second = arrays(runtime), arrays(runtime)
    start = time.perf_counter()
    call = module.prepare_execution("step", dict(first, image=image, alias=image))
    other = other_module.prepare_execution(
        "step", dict(second, image=image, alias=image)
    )
    prepare_ms = (time.perf_counter() - start) * 1000
    alias_document = json.loads(other_module.execution_plan("step").to_json())
    alias_document["version"] = 17
    if arch == "cuda":
        alias_document["binding_strategy"] = "cuda_immutable_argument_frames"
    alias_document["constraints"] = [
        dict(
            kind="resource_alias",
            left="image",
            right="alias",
            left_path=[],
            right_path=[],
        )
    ]
    alias_plan = ExecutionPlan.from_json(json.dumps(alias_document))
    accepted = alias_plan.prepare(other_module, dict(second, image=image, alias=image))
    accepted()
    check(second, list(range(15)), arch)
    if arch == "cuda":
        sibling = accepted.prepare_binding(first)
        accepted.close()
        sibling()
        check(first, list(range(15)), arch)
        sibling.close()
    accepted.close()
    different = create_image(runtime, list(range(15)))
    raises(
        lambda: alias_plan.prepare(
            other_module, dict(second, image=image, alias=different)
        ),
        "alias mismatch",
    )
    different.close()
    alias_document["constraints"][0]["left_path"] = ["missing"]
    raises(
        lambda: ExecutionPlan.from_json(json.dumps(alias_document)).prepare(
            other_module, dict(second, image=image, alias=image)
        ),
        "retained input port",
    )
    for prepared, frame in ((call, first), (other, second)):
        prepared()
        check(frame, list(range(15)), arch)
    direct = other_module.kernel("sample").bind(image, image, second["output"])
    direct()
    fixed = direct.prepare()
    fixed()
    assert read(second["output"]) == (
        [4.0, 11.0, 7.0, 7.0] if arch == "cuda" else [12.0, 5.0, 7.0, 7.0]
    )
    module.close()
    raises(call)
    call.close()
    other()
    check(second, list(range(15)), arch)

    # Same owner queue orders cross-plan uploads and sampling without a host wait.
    pixels = runtime.ndarray("f32", (3, 5))
    values = list(range(15, 30))
    write(pixels, values)
    upload_plan = ExecutionPlan(
        [native("image", "upload", dict(image="image", pixels="pixels"))]
    )
    upload = upload_plan.prepare(other_module, dict(image=image, pixels=pixels))
    upload()
    other()
    check(second, values, arch)
    pixels.close()
    raises(upload)
    upload.close()
    other()  # Closing the uploader's source must not close the image.
    check(second, values, arch)

    if arch == "cuda":
        document = json.loads(other_module.execution_plan("step").to_json())
        document.update(version=10, binding_strategy="cuda_immutable_argument_frames")
        family_plan = ExecutionPlan.from_json(json.dumps(document))
        family = family_plan.prepare(
            other_module, dict(second, image=image, alias=image)
        )
        sibling = family.prepare_binding(first)
        replacement = create_image(runtime, [-1] * 15)
        raises(
            lambda: family.prepare_binding(
                dict(first, image=replacement, alias=replacement)
            ),
            "resource changed",
        )
        replacement.close()
        family.close()
        sibling()
        check(first, values, arch)
    else:
        sibling = None

    # The C API itself rejects foreign and wrong-kind identities, not only Python.
    memory = runtime.ndarray("f32", (1,))
    native_wrong_handle(runtime, memory._memory._handle)
    memory.close()
    with Runtime(arch) as foreign:
        native_wrong_handle(foreign, image._handle)
    submitted, completed = [], []
    for _ in range(8):
        start = time.perf_counter_ns()
        other()
        submitted.append((time.perf_counter_ns() - start) / 1000)
        runtime.wait()
        completed.append((time.perf_counter_ns() - start) / 1000)
    old = image._handle
    other()
    image.close()
    for prepared in (other, fixed, sibling):
        if prepared is not None:
            raises(prepared)
            prepared.close()
    native_wrong_handle(runtime, old)
    replacement = create_image(runtime, [2] * 15)
    assert replacement._handle != old
    fresh = other_module.prepare_execution(
        "step", dict(first, image=replacement, alias=replacement)
    )
    fresh()
    check(first, [2] * 15, arch)
    fresh.close()
    replacement.close()

    # Dropping a Python image variable cannot retire its prepared user.
    image = create_image(runtime, [3] * 15)
    ref = weakref.ref(image)
    call = other_module.prepare_execution(
        "step", dict(second, image=image, alias=image)
    )
    del image
    gc.collect()
    assert ref() is not None
    call()
    check(second, [3] * 15, arch)
    call.close()
    gc.collect()
    assert ref() is None

    # Uninitialized images do not gain sampling facts from a merely recorded upload.
    empty = runtime.native_resource(
        "image", "sampled", parameters=dict(shape=[3, 5], format="r32f")
    )
    raises(
        lambda: other_module.prepare_execution(
            "step", dict(first, image=empty, alias=empty)
        ),
        "unconditional upload",
    )
    empty.close()

    # Initialization-only public resources release both Python and native edges.
    document = json.loads(other_module.execution_plan("step").to_json())
    document["initialization"] = [document["operations"].pop(0)]
    document["version"] = max(document["version"], 2)
    if arch == "cuda":
        document.update(version=10, binding_strategy="cuda_immutable_argument_frames")
    initial_plan = ExecutionPlan.from_json(json.dumps(document))
    image = create_image(runtime, [5] * 15)
    initial = initial_plan.prepare(other_module, dict(first, image=image, alias=image))
    initial_sibling = initial.prepare_binding(second) if arch == "cuda" else None
    initial.initialize()
    image.close()
    initial()
    check(first, [5] * 15, arch)
    if initial_sibling is not None:
        # Initialization belongs to the family and wrote the first frame only.
        write(second["output"], [6] * 4)
        initial_sibling()
        assert read(second["doutput"]) == [6] * 4
        initial_sibling.close()
    initial.close()
    for frame in (first, second):
        for value in frame.values():
            value.close()
    return other_module, dict(
        create_ms=create_ms,
        prepare_ms=prepare_ms,
        submit_us=statistics.median(submitted),
        complete_us=statistics.median(completed),
    )


def optix_checks(runtime, module, adapter):
    if not adapter:
        return
    runtime.configure_provider("optix", adapter_path=adapter)
    # Exercise both a retained memory root and an opaque parent root.
    for close_root in ("vertices", "context"):
        context = runtime.native_resource("optix", "context")
        vertices, indices = runtime.ndarray("f32", (3, 3)), runtime.ndarray(
            "i32", (1, 3)
        )
        write(vertices, [-1, -1, 1, 1, -1, 1, 0, 1, 1])
        write(indices, [0, 1, 2], "i")
        gas = runtime.native_resource(
            "optix",
            "triangle_gas",
            parameters=dict(vertex_count=3, triangle_count=1, allow_update=True),
            bindings=dict(context=context, vertices=vertices, indices=indices),
        )
        transforms = runtime.ndarray("f32", (1, 12))
        write(transforms, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
        scene = runtime.native_resource(
            "optix",
            "instance_scene",
            parameters=dict(instance_count=1),
            bindings=dict(context=context, transforms=transforms, gas_0=gas),
        )
        rays, hits, ids = (
            runtime.ndarray("f32", (1, 8)),
            runtime.ndarray("f32", (1, 4)),
            runtime.ndarray("i32", (1, 4)),
        )
        write(rays, [0, 0, 0, 0.01, 0, 0, 1, 3])
        plan = ExecutionPlan(
            [
                native("optix", "refit", dict(scene="gas", vertices="vertices")),
                native(
                    "optix",
                    "query",
                    dict(scene="scene", rays="rays", hits="hits", hit_indices="ids"),
                    ray_count=1,
                ),
            ],
            constraints=[
                dict(
                    kind="resource_alias",
                    left="scene",
                    right="gas",
                    left_path=["gas_0"],
                    right_path=[],
                ),
                dict(
                    kind="resource_alias",
                    left="scene",
                    right="gas",
                    left_path=["gas_0", "context"],
                    right_path=["context"],
                ),
            ],
        )
        arguments = dict(
            scene=scene, gas=gas, vertices=vertices, rays=rays, hits=hits, ids=ids
        )
        wrong = runtime.native_resource(
            "optix",
            "triangle_gas",
            parameters=dict(vertex_count=3, triangle_count=1, allow_update=True),
            bindings=dict(context=context, vertices=vertices, indices=indices),
        )
        raises(
            lambda: plan.prepare(module, dict(arguments, gas=wrong)), "alias mismatch"
        )
        wrong.close()
        call = plan.prepare(module, arguments)
        call()
        assert read(hits)[0] == 1
        assert read(ids, "i")[0] == 0
        call()
        (vertices if close_root == "vertices" else context).close()
        raises(call)
        replacement_vertices = runtime.ndarray("f32", (3, 3))
        raises(
            lambda: plan.prepare(module, dict(arguments, vertices=replacement_vertices)),
            "retired dependency",
        )
        replacement_vertices.close()
        call.close()
        for owner in (
            scene,
            gas,
            context,
            transforms,
            vertices,
            indices,
            rays,
            hits,
            ids,
        ):
            owner.close()


def main(mode, arch, artifact, adapter, result):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(artifact)
        if arch == "cpu":
            raises(
                lambda: runtime.native_resource(
                    "image", "sampled", parameters=dict(shape=[3, 5], format="r32f")
                ),
                "unavailable",
            )
            frame = arrays(runtime)
            write(frame["output"], [1, 2, 3, 4])
            call = module.prepare_execution("step", frame)
            document = json.loads(module.execution_plan("step").to_json())
            document.update(
                version=17,
                constraints=[
                    dict(
                        kind="resource_alias",
                        left="output",
                        right="loss",
                        left_path=[],
                        right_path=[],
                    )
                ],
            )
            raises(
                lambda: ExecutionPlan.from_json(json.dumps(document)).prepare(
                    module, frame
                ),
                "public native resources",
            )
            call()
            assert read(frame["doutput"]) == [1, 2, 3, 4]
            costs = {}
        else:
            module, costs = image_checks(runtime, module, artifact, arch)
            if arch == "cuda":
                optix_checks(runtime, module, adapter)
            image = create_image(runtime, [4] * 15)
            frame = arrays(runtime)
            call = module.prepare_execution(
                "step", dict(frame, image=image, alias=image)
            )
            call()
            check(frame, [4] * 15, arch)
        # Exercise actual owner teardown while the final call is in flight.
        call()
        if mode == "borrowed":
            ti.reset()
        else:
            runtime.close()
        raises(call)
        call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(result).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
