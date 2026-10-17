"""Small independent geometry oracle; no producer import in standalone mode."""

import json
from pathlib import Path
import statistics
import struct
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_optix_probe import raises, read, write
from tests.python.aot_vulkan_micromap_probe import asset, geometry, POSES


def native(operation, bindings, version=1, **parameters):
    return dict(
        kind="native", provider="optix", operation=operation, version=version, bindings=bindings, parameters=parameters
    )


def check(frame, coords, states, alpha, limit, shift, dz):
    expected_t, gradient, total = [], [], 0
    for prefix, gate in (("", alpha), ("plain_", True)):
        expected_hits, expected_ids, expected_occ = [], [], []
        for i, state in enumerate(states):
            front = state == 1 or (state >= 2 and gate)
            t = 2 - dz if front else 3
            hit = t <= limit and not shift
            x, y = coords[2 * i] - 3 * (i // 4), coords[2 * i + 1]
            expected_hits += [t, x, y, 0] if hit else [-1, 0, 0, 0]
            expected_ids += [i // 4, 0 if front else 1, 11 if front else 22, 1] if hit else [-1, -1, -1, 0]
            expected_occ.append(int(hit))
            if not prefix:
                expected_t.append(t if hit else -1)
                gradient += [t if hit else 0, 0, 0, 0]
                total += 0.5 * t * t if hit else 0
        actual = read(frame[prefix + "hits"])
        assert len(actual) == len(expected_hits) and all(abs(a - b) < 1e-5 for a, b in zip(actual, expected_hits)), (
            prefix,
            actual,
            expected_hits,
        )
        assert read(frame[prefix + "ids"], "i") == expected_ids
        assert read(frame[prefix + "occluded"], "i") == expected_occ
    # First accepted may be the rear opaque surface; restricted intervals have
    # exactly one possible accepted layer and must agree with closest-hit.
    first, ids = read(frame["first_hits"]), read(frame["first_ids"], "i")
    for i, t in enumerate(expected_t):
        assert (ids[4 * i + 3] == 1) == (t >= 0)
        if t < 0 or limit < 3:
            assert abs(first[4 * i] - t) < 1e-5
        else:
            assert abs(first[4 * i] - t) < 1e-5 or abs(first[4 * i] - 3) < 1e-5
    assert all(abs(a - b) < 1e-5 for a, b in zip(read(frame["dhits"]), gradient))
    assert abs(read(frame["loss"])[0] - total) < 1e-4


def main(mode, artifact, adapter):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    records, retained = [], []
    with runtime:
        runtime.configure_provider("optix", adapter_path=adapter)
        assert runtime.probe_native_resource("optix_triangle_gas", version=2)["available"]
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        for case in ("two", "indexed", "predefined"):
            vertices, coords, states = geometry(case)
            triangles, count = len(vertices) // 9, len(states)
            shapes = [
                ("vertices", "f32", (triangles * 3, 3)),
                ("indices", "i32", (triangles, 3)),
                ("poses", "f32", (2, 12)),
                ("rays", "f32", (count, 8)),
                ("uvs", "f32", (triangles * 3, 2)),
                ("dhits", "f32", (count, 4)),
                ("loss", "f32", (1,)),
                ("dloss", "f32", (1,)),
            ]
            for prefix in ("", "plain_", "first_"):
                shapes += [(prefix + "hits", "f32", (count, 4)), (prefix + "ids", "i32", (count, 4))]
                if prefix != "first_":
                    shapes += [(prefix + "occluded", "i32", (count,))]
            frame = {name: runtime.ndarray(dtype, shape) for name, dtype, shape in shapes}
            write(frame["vertices"], vertices)
            write(frame["indices"], range(triangles * 3), "i")
            write(frame["poses"], POSES)
            write(frame["uvs"], [0.25] * (triangles * 6))
            pixels = runtime.ndarray("f32", (2, 2))
            write(pixels, [0] * 4)
            image = runtime.native_resource(
                "image", "sampled", parameters=dict(shape=[2, 2], format="r32f"), bindings=dict(pixels=pixels)
            )
            frame["alpha_image"] = image
            upload = ExecutionPlan(
                [
                    dict(
                        kind="native",
                        provider="image",
                        operation="upload",
                        version=1,
                        bindings=dict(image="image", pixels="pixels"),
                        parameters={},
                    )
                ]
            ).prepare(module, dict(image=image, pixels=pixels))
            write(pixels, [0] * 4)
            upload()
            data, descriptors, mapping, _ = asset(case)
            blobs = dict(data=runtime.host_blob(data), descriptors=runtime.host_blob(descriptors))
            if mapping is not None:
                blobs["indices"] = runtime.host_blob(mapping)
            omm = runtime.native_resource(
                "optix", "opacity_micromap", parameters=dict(triangle_count=triangles), bindings=blobs
            )
            if case == "two":
                negative_contracts(runtime, module, frame, omm, blobs)
            if case == "indexed":
                frame["public_omm"] = omm
            plan = module.execution_plan(case)
            start = time.perf_counter()
            call = plan.prepare(module, {name: frame[name] for name in plan.argument_names})
            prepare_ms = (time.perf_counter() - start) * 1000
            call.initialize()
            if case != "indexed":
                omm.close()
            for name, blob in blobs.items():
                if case != "indexed" or name != "data":
                    blob.close()
            # An explicit public plan argument retains the ordinary binding
            # lifetime, even though the constructed GAS consumes its host data.
            for alpha, limit, shift, dz in (
                (0, 4, 0, 0),
                (1, 2.5, 0, 0),
                (0, 2.5, 0, 0),
                (1, 2.75, 0, -0.5),
                (1, 4, 100, 0),
                (0, 4, 0, 0),
            ):
                write(pixels, [alpha] * 4)
                upload()
                pose = POSES.copy()
                pose[3] = pose[15] = shift
                write(frame["poses"], pose)
                moved = vertices.copy()
                moved[2::3] = [dz] * (triangles * 3)
                write(frame["vertices"], moved)
                rays = [v for i in range(count) for v in (*coords[2 * i : 2 * i + 2], 3, 0.01, 0, 0, -1, limit)]
                write(frame["rays"], rays)
                call()
                check(frame, coords, states, alpha, limit, shift, dz)
            submits, completes = [], []
            for _ in range(5):
                start = time.perf_counter_ns()
                call()
                submits.append((time.perf_counter_ns() - start) / 1000)
                runtime.wait()
                completes.append((time.perf_counter_ns() - start) / 1000)
            records.append(
                dict(
                    case=case,
                    prepare_ms=prepare_ms,
                    submit_us=statistics.median(submits),
                    complete_us=statistics.median(completes),
                    info=call.info,
                )
            )
            retained.append((call, frame, upload, image, pixels))
            if case == "indexed":
                call()
                omm.close()
                raises(call)
                blobs["data"].close()
        retained[0][0]()
        retained[0][3].close()
        raises(retained[0][0])
        retained[-1][0]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for call, frame, upload, image, pixels in retained:
            raises(call)
            call.close()
            upload.close()
            image.close()
            pixels.close()
            for value in frame.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(artifact).with_name(mode + ".json").write_text(json.dumps(dict(load_ms=load_ms, cases=records), indent=2))


def negative_contracts(runtime, module, frame, omm, blobs):
    with runtime.host_blob(b"\0") as malformed:
        raises(
            lambda: runtime.native_resource(
                "optix",
                "opacity_micromap",
                parameters=dict(triangle_count=1),
                bindings=dict(blobs, descriptors=malformed),
            ),
            "packed input",
        )
    context = runtime.native_resource("optix", "context")
    bindings = dict(context=context, vertices=frame["vertices"], indices=frame["indices"], opacity_micromap=omm)
    parameters = dict(vertex_count=3, triangle_count=1, allow_update=True)
    raises(
        lambda: runtime.native_resource(
            "optix",
            "triangle_gas",
            version=2,
            parameters=dict(parameters, expected_micromap="0" * 64),
            bindings=bindings,
        ),
        "fingerprint",
    )
    with runtime.host_blob(struct.pack("<IHH", 0, 1, 3)) as malformed:
        invalid = runtime.native_resource(
            "optix", "opacity_micromap", parameters=dict(triangle_count=1), bindings=dict(blobs, descriptors=malformed)
        )
        raises(
            lambda: runtime.native_resource(
                "optix",
                "triangle_gas",
                version=2,
                parameters=parameters,
                bindings=dict(bindings, opacity_micromap=invalid),
            ),
            "level or format",
        )
        invalid.close()
    gas = runtime.native_resource("optix", "triangle_gas", version=2, parameters=parameters, bindings=bindings)
    omm.close()  # Direct GAS construction consumed the immutable host input.
    poses = runtime.ndarray("f32", (1, 12))
    write(poses, POSES[:12])
    scene_bindings = dict(context=context, gas_0=gas, transforms=poses)
    raises(
        lambda: runtime.native_resource(
            "optix", "instance_scene", parameters=dict(instance_count=1, opaque=[True]), bindings=scene_bindings
        ),
        "cannot be declared opaque",
    )
    scene = runtime.native_resource(
        "optix", "instance_scene", parameters=dict(instance_count=1), bindings=scene_bindings
    )
    operation = native(
        "query", dict(scene="scene", rays="rays", hits="hits", hit_indices="ids"), ray_count=4, front_only=True
    )
    raises(
        lambda: ExecutionPlan([operation]).prepare(
            module, dict(scene=scene, rays=frame["rays"], hits=frame["hits"], ids=frame["ids"])
        ),
        "opacity micromaps",
    )
    write(frame["rays"], [100, 0, 3, 0.01, 0, 0, -1, 4] * 4)
    operation["parameters"]["front_only"] = False
    call = ExecutionPlan([operation]).prepare(
        module, dict(scene=scene, rays=frame["rays"], hits=frame["hits"], ids=frame["ids"])
    )
    call()
    assert read(frame["hits"])[::4] == [-1] * 4
    call.close()
    scene.close()
    gas.close()
    poses.close()
    context.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
