"""Source-independent programmable Vulkan RT and managed image consumer."""

import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_optix_probe import raises, read, write

VERTICES = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1]
POSES = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0]


def native(provider, operation, bindings, **parameters):
    return dict(
        kind="native", provider=provider, operation=operation, version=1, bindings=bindings, parameters=parameters
    )


def pipeline_spec(images=False):
    return dict(
        shader_count=1 if images else 4,
        shaders=[dict(stage=s) for s in (["raygen"] if images else ["raygen", "miss", "closest_hit", "any_hit"])],
        groups=(
            [dict(kind="raygen", general=0)]
            if images
            else [
                dict(kind="raygen", general=0),
                dict(kind="miss", general=1),
                dict(kind="triangles", closest_hit=2),
                dict(kind="triangles", closest_hit=2, any_hit=3),
            ]
        ),
    )


def launch_spec(images=False):
    buffers = [dict(set=0, binding=2 if images else 1, kind="storage_buffer", access="read" if images else "write")]
    result = dict(
        dimensions=[3, 2, 1] if images else [5, 1, 1],
        raygen=dict(group=0),
        miss=[],
        hit=[],
        buffer_count=1,
        buffers=buffers,
        image_count=2 if images else 0,
        images=[],
        scene_count=0 if images else 1,
        scenes=[] if images else [dict(set=0, binding=0)],
    )
    if images:
        result.update(
            push_constants=dict(offset=0, size=16),
            images=[
                dict(set=0, binding=0, kind="sampled_image", access="read"),
                dict(set=0, binding=1, kind="storage_image", access="write"),
            ],
        )
        # SPIR-V reflection rounds this uniform block to a 16-byte extent.
        payload = struct.pack("<4f", 0.5, 0, 0, 0)
    else:
        result["miss"] = [dict(group=1, offset=4 * i, size=4) for i in range(2)]
        result["hit"] = [dict(group=2 if i in (2, 5) else 3, offset=8 + 4 * i, size=4) for i in range(6)]
        payload = struct.pack("<8I", 200, 201, *(10 * i + 3 for i in range(6)))
    return result, payload


def operations(images=False):
    params, _ = launch_spec(images)
    if images:
        result = [native("image", "upload", dict(image="source", pixels="pixels"))]
        bindings = dict(pipeline="pipeline", payload="payload", buffer_0="gain", image_0="source", image_1="target")
        decode = dict(kind="kernel", entry="sample", arguments=["target", "features"])
    else:
        result = [
            native("vulkan_ray", "refit", dict(scene="blas", vertices="vertices")),
            native("vulkan_ray", "instance_refit", dict(scene="scene", transforms="poses")),
        ]
        bindings = dict(pipeline="pipeline", payload="payload", scene_0="scene", buffer_0="output")
        decode = dict(kind="kernel", entry="decode", arguments=["output", "features"])
    result += [
        native("vulkan_ray_program", "launch", bindings, **params),
        decode,
        dict(kind="fill32", destination="dfeatures", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["features", "loss"]),
        dict(kind="kernel", entry="backward", arguments=["features", "loss"]),
    ]
    return result


def private_resources(images=False):
    if images:
        return [
            dict(native("image", "sampled", dict(pixels="pixels"), shape=[3, 2], format="rgba32f"), name="source"),
            dict(
                native("image", "sampled", dict(pixels="pixels"), shape=[3, 2], format="rgba32f", storage=True),
                name="target",
            ),
        ]
    return [
        dict(
            native(
                "vulkan_ray",
                "triangle_blas",
                dict(vertices="vertices", indices="indices"),
                vertex_count=6,
                triangle_count=2,
                opaque=False,
            ),
            name="blas",
        ),
        dict(
            native(
                "vulkan_ray",
                "instance_tlas",
                dict(blas_0="blas", blas_1="blas", transforms="poses"),
                instance_count=2,
                custom_indices=[7, 11],
                sbt_record_offsets=[1, 4],
            ),
            name="scene",
        ),
    ]


def public_pipeline(runtime, directory, images, variant=None):
    spec = pipeline_spec(images)
    code = [(directory / f"{'image' if images else 'ray'}_{i}.spv").read_bytes() for i in range(spec["shader_count"])]
    if variant == "code":
        # A valid changed generator word leaves behavior unchanged but changes code identity.
        code[0] = code[0][:8] + bytes([code[0][8] ^ 1]) + code[0][9:]
    elif variant == "groups":
        spec["groups"][2], spec["groups"][3] = spec["groups"][3], spec["groups"][2]
    shaders = [runtime.host_blob(value) for value in code]
    try:
        if variant is None:
            raises(
                lambda: runtime.native_resource(
                    "vulkan_ray_program",
                    "pipeline",
                    parameters=dict(spec, expected_pipeline="0" * 64),
                    bindings={f"shader_{i}": value for i, value in enumerate(shaders)},
                ),
                "contract mismatch",
            )
        if not images and variant is None:
            with runtime.host_blob(b"bad") as bad:
                bindings = {f"shader_{i}": v for i, v in enumerate(shaders)}
                raises(
                    lambda: runtime.native_resource(
                        "vulkan_ray_program", "pipeline", parameters=spec, bindings=dict(bindings, shader_0=bad)
                    ),
                    "SPIR-V",
                )
        return runtime.native_resource(
            "vulkan_ray_program",
            "pipeline",
            parameters=spec,
            bindings={f"shader_{i}": v for i, v in enumerate(shaders)},
        )
    finally:
        for blob in shaders:
            blob.close()


def check_gradients(frame, expected):
    assert read(frame["features"]) == expected, (read(frame["features"]), expected)
    gradient = [max(v, 0) for v in expected]
    assert read(frame["dfeatures"]) == gradient
    assert abs(read(frame["loss"])[0] - sum(0.5 * v * v for v in gradient)) < 1e-4


def main(mode, artifact, output_file, recipe="explicit"):
    start = time.perf_counter()
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    costs = dict(runtime_ms=(time.perf_counter() - start) * 1000, cases={})
    with runtime:
        assert runtime.device_capabilities().get("vulkan_has_ray_tracing_pipeline", 0) == 1
        assert "vulkan_inline" in runtime.native_operation_capabilities("vulkan_ray_program_launch")
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        costs["load_ms"] = (time.perf_counter() - start) * 1000
        directory = Path(artifact).parent
        for images in (False, True):
            count = 24 if images else 5
            frame = {
                "features": runtime.ndarray("f32", count),
                "dfeatures": runtime.ndarray("f32", count),
                "loss": runtime.ndarray("f32", 1),
            }
            if images:
                frame.update(pixels=runtime.ndarray("f32", (3, 2, 4)), gain=runtime.ndarray("f32", 1))
                pixels = [i * 0.25 for i in range(24)]
                write(frame["pixels"], pixels)
            else:
                frame.update(
                    vertices=runtime.ndarray("f32", (6, 3)),
                    indices=runtime.ndarray("i32", (2, 3)),
                    poses=runtime.ndarray("f32", (2, 12)),
                    output=runtime.ndarray("u32", (5, 8)),
                )
                write(frame["vertices"], VERTICES)
                write(frame["indices"], list(range(6)), "i")
                write(frame["poses"], POSES)
            start = time.perf_counter()
            pipeline = public_pipeline(runtime, directory, images)
            creation = (time.perf_counter() - start) * 1000
            owners = {"pipeline": pipeline}
            for spec in private_resources(images):
                available = dict(frame, **owners)
                owners[spec["name"]] = runtime.native_resource(
                    spec["provider"],
                    spec["operation"],
                    parameters=spec["parameters"],
                    bindings={k: available[v] for k, v in spec["bindings"].items()},
                )
            prefix = "images" if images else "rays"
            calls = {}
            for private in (False, True):
                name = prefix + ("_private" if private else "_public")
                available = dict(frame, **owners)
                supplied = {k: available[k] for k in module.execution_plan(name).argument_names}
                start = time.perf_counter()
                call = module.prepare_execution(name, supplied)
                costs["cases"][name] = dict(prepare_ms=(time.perf_counter() - start) * 1000, info=call.info)
                calls[name] = call
                if not private and recipe != "explicit":
                    for variant in (("code",) if images else ("code", "groups")):
                        with public_pipeline(runtime, directory, images, variant) as wrong:
                            raises(
                                lambda: module.prepare_execution(name, dict(supplied, pipeline=wrong)),
                                "contract mismatch",
                            )
                for state in (0, 1, 2, 0):
                    if images:
                        write(frame["gain"], [state + 1])
                        pixels = [(i + (state if recipe == "explicit" else 0)) * 0.25 for i in range(24)]
                        write(frame["pixels"], pixels)
                        expected = [v * (state + 1) + 0.5 for v in pixels]
                    else:
                        vertices = [v + (0.25 if state == 1 and i % 3 == 2 else 0) for i, v in enumerate(VERTICES)]
                        poses = POSES[:]
                        poses[3] += 30 if state == 2 else 0
                        poses[15] += 30 if state == 2 else 0
                        write(frame["vertices"], vertices)
                        write(frame["poses"], poses)
                        expected = [-1] * 5 if state == 2 else [2, 1, 2, 1, -1]
                        if state == 1:
                            expected = [v + 0.25 if v > 0 else v for v in expected]
                    call()
                    check_gradients(frame, expected)
                    if not images:
                        values = read(frame["output"], "I")
                        expected_ids = [[1, 0, 7, 13], [0, 0, 7, 23], [1, 1, 11, 43], [0, 1, 11, 53]]
                        for i in range(5):
                            ids = [0xFFFFFFFF] * 3 + [200 + i % 2] if state == 2 or i == 4 else expected_ids[i]
                            assert values[8 * i : 8 * i + 4] == ids
                        if state != 2:
                            geometry = struct.unpack("<40f", struct.pack("<40I", *values))
                            for i in range(4):
                                assert geometry[8 * i + 5 : 8 * i + 8] == (0.25, 0.25, 1.0)
                complete = []
                for _ in range(8):
                    start = time.perf_counter_ns()
                    call()
                    runtime.wait()
                    complete.append((time.perf_counter_ns() - start) / 1000)
                costs["cases"][name]["complete_us"] = statistics.median(complete)
                if not private:
                    sibling_module = runtime.load_module(artifact)
                    sibling = sibling_module.prepare_execution(name, supplied)
                    sibling()
                    check_gradients(frame, expected)
                    sibling.close()
                    sibling_module.close()
                    call()
                    check_gradients(frame, expected)
            costs["cases"][prefix + "_public"]["pipeline_ms"] = creation
            # Invalid preparation must not retire an existing valid executable.
            if images:
                params, payload = launch_spec(True)
                with runtime.host_blob(payload) as blob:
                    bindings = dict(
                        pipeline=pipeline,
                        payload=blob,
                        buffer_0=frame["gain"],
                        image_0=owners["source"],
                        image_1=owners["target"],
                    )
                    spec = native("vulkan_ray_program", "launch", {k: k for k in bindings}, **params)
                    for patch, message in (
                        (dict(image_1=owners["source"]), "storage usage"),
                        (dict(image_0=owners["target"]), "sample and store"),
                    ):
                        raises(lambda: ExecutionPlan([spec]).prepare(module, dict(bindings, **patch)), message)
            if not images:
                params, payload = launch_spec()
                with runtime.host_blob(payload) as blob:
                    bindings = dict(pipeline=pipeline, payload=blob, scene_0=owners["scene"], buffer_0=frame["output"])
                    for patch, message in (
                        (dict(dimensions=[0, 1, 1]), "dispatch"),
                        (dict(expected_pipeline="0" * 64), "contract mismatch"),
                        (dict(hit=params["hit"][:4]), "SBT offsets"),
                        (dict(raygen=dict(group=0, offset=len(payload), size=4)), "payload"),
                        (dict(buffers=[dict(params["buffers"][0], binding=0)]), "Duplicate"),
                    ):
                        spec = native("vulkan_ray_program", "launch", {k: k for k in bindings}, **dict(params, **patch))
                        raises(lambda: ExecutionPlan([spec]).prepare(module, bindings), message)
                calls["rays_public"]()
                check_gradients(frame, [2, 1, 2, 1, -1])
            calls[prefix + "_public"]()
            (owners["source"] if images else pipeline).close()
            raises(calls[prefix + "_public"])
            write(frame["features"], [-99] * count)
            calls[prefix + "_private"]()
            check_gradients(frame, expected)
            if images and mode == "borrowed":
                ti.reset()
                raises(calls[prefix + "_private"])
            else:
                for call in calls.values():
                    call.close()
            for owner in owners.values():
                owner.close()
        if mode == "standalone":
            assert "taichi_forge" not in sys.modules
    Path(output_file).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
