"""Baked opacity classification survives AOT reconstruction and retirement."""

import json
import os
from pathlib import Path
import struct
import sys
import statistics
import time

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_vulkan_instances_probe import native, raises, read, write


def asset(case):
    if case == "predefined":
        return b"", b"", struct.pack("<4i", -1, -2, -3, -4), [0, 1, 2, 3]
    if case == "indexed":
        return bytes([0xE4]), struct.pack("<IHH", 0, 1, 2), struct.pack("<6i", 0, 0, -1, -2, -3, -4), [0, 1, 2, 3]
    return bytes([0xA]), struct.pack("<IHH", 0, 1, 1), None, [0, 1, 0, 1]


def geometry(case):
    _, _, mapping, states = asset(case)
    indices = list(struct.unpack(f"<{len(mapping) // 4}i", mapping)) if mapping is not None else [0]
    vertices, coords, expected_states = [], [], []
    for i, index in enumerate(indices):
        vertices += [3 * i, 0, 0, 3 * i + 1, 0, 0, 3 * i, 1, 0]
        coords += [
            c for x, y in ((1 / 6, 1 / 6), (1 / 3, 1 / 3), (2 / 3, 1 / 6), (1 / 6, 2 / 3)) for c in (3 * i + x, y)
        ]
        expected_states += states if index >= 0 else [-index - 1] * 4
    return vertices, coords, expected_states


POSES = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]


def resources(case, blobs):
    data, descriptors, indices, _ = asset(case)
    count = len(indices) // 4 if indices is not None else len(descriptors) // 8
    bindings = dict(data=blobs[0], descriptors=blobs[1])
    if indices is not None:
        bindings["indices"] = blobs[2]
    return [
        dict(native("opacity_micromap", bindings, triangle_count=count), name="omm"),
        dict(
            native(
                "triangle_blas",
                dict(vertices="vertices", indices="indices", opacity_micromap="omm"),
                vertex_count=count * 3,
                triangle_count=count,
            ),
            name="front",
        ),
        dict(
            native(
                "triangle_blas",
                dict(vertices="vertices", indices="indices"),
                vertex_count=count * 3,
                triangle_count=count,
            ),
            name="back",
        ),
        dict(
            native(
                "instance_tlas",
                dict(blas_0="front", blas_1="back", transforms="poses"),
                instance_count=2,
                custom_indices=[11, 22],
            ),
            name="scene",
        ),
    ]


def plan_operations():
    return [
        native("refit", dict(scene="front", vertices="vertices")),
        native("instance_refit", dict(scene="scene", transforms="poses")),
        dict(kind="kernel", entry="query", arguments=["scene", "coords", "result", "gate", "limit"]),
        dict(kind="fill32", destination="dresult", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["result", "loss"]),
        dict(kind="kernel", entry="backward", arguments=["result", "loss"]),
    ]


def check(frame, states, gate, limit, dz):
    expected, gradient, total = [], [], 0
    for state in states:
        front = state == 1 or (state >= 2 and gate)
        t = (2 - dz) if front else 3
        accepted = t <= limit
        expected += [
            float(11 if front else 22) if accepted else -1,
            t if accepted else -1,
            int(accepted),
            int(accepted),
        ]
        gradient += [0, t if accepted else 0, 0, 0]
        total += 0.5 * t * t if accepted else 0
    actual = read(frame["result"])
    assert all(abs(a - b) < 1e-5 for a, b in zip(actual, expected)), (actual, expected)
    assert all(abs(a - b) < 1e-5 for a, b in zip(read(frame["dresult"]), gradient))
    assert abs(read(frame["loss"])[0] - total) < 1e-4


def main(mode, artifact, output, recipe="explicit"):
    start = time.perf_counter()
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    with runtime:
        costs = dict(runtime_ms=(time.perf_counter() - start) * 1000, cases={})
        assert runtime.device_capabilities().get("vulkan_has_opacity_micromap", 0) == 1
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        costs["module_load_ms"] = (time.perf_counter() - start) * 1000
        for case in ("two", "indexed", "predefined"):
            vertices, coords, states = geometry(case)
            count = len(vertices) // 9
            frame = {
                name: runtime.ndarray(dtype, shape)
                for name, dtype, shape in (
                    ("vertices", "f32", (count * 3, 3)),
                    ("indices", "i32", (count, 3)),
                    ("poses", "f32", (2, 12)),
                    ("coords", "f32", (len(states), 2)),
                    ("result", "f32", (len(states), 4)),
                    ("dresult", "f32", (len(states), 4)),
                    ("loss", "f32", (1,)),
                    ("gate", "i32", (1,)),
                    ("limit", "f32", (1,)),
                )
            }
            write(frame["vertices"], vertices)
            write(frame["indices"], list(range(count * 3)), "i")
            write(frame["poses"], POSES)
            write(frame["coords"], coords)
            data, descriptors, indices, _ = asset(case)
            source = bytearray(data)
            seeds = dict(data=runtime.host_blob(source), descriptors=runtime.host_blob(descriptors))
            source[:] = bytes(len(source))  # The public blob owns an immutable copy.
            if indices is not None:
                seeds["indices"] = runtime.host_blob(indices)
            start = time.perf_counter()
            omm = runtime.native_resource(
                "vulkan_ray", "opacity_micromap", parameters=dict(triangle_count=count), bindings=seeds
            )
            case_costs = dict(micromap_create_ms=(time.perf_counter() - start) * 1000)
            if case == "two":
                with runtime.host_blob(b"\0") as malformed:
                    raises(
                        lambda: runtime.native_resource(
                            "vulkan_ray",
                            "opacity_micromap",
                            parameters=dict(triangle_count=1),
                            bindings=dict(seeds, descriptors=malformed),
                        ),
                        "descriptor",
                    )
                with runtime.host_blob(struct.pack("<i", -5)) as malformed:
                    raises(
                        lambda: runtime.native_resource(
                            "vulkan_ray",
                            "opacity_micromap",
                            parameters=dict(triangle_count=1),
                            bindings=dict(seeds, indices=malformed),
                        ),
                        "index",
                    )
            for blob in seeds.values():
                blob.close()  # Construction inputs are complete and released.
            blas_params = dict(vertex_count=count * 3, triangle_count=count)
            inputs = dict(vertices=frame["vertices"], indices=frame["indices"])
            raises(
                lambda: runtime.native_resource(
                    "vulkan_ray",
                    "triangle_blas",
                    parameters=dict(blas_params, opaque=True),
                    bindings=dict(inputs, opacity_micromap=omm),
                ),
                "opaque",
            )
            front = runtime.native_resource(
                "vulkan_ray", "triangle_blas", parameters=blas_params, bindings=dict(inputs, opacity_micromap=omm)
            )
            sibling = runtime.native_resource(
                "vulkan_ray", "triangle_blas", parameters=blas_params, bindings=dict(inputs, opacity_micromap=omm)
            )
            sibling.close()  # Closing one BLAS leaves its shared OMM and sibling usable.
            back = runtime.native_resource("vulkan_ray", "triangle_blas", parameters=blas_params, bindings=inputs)
            scene = runtime.native_resource(
                "vulkan_ray",
                "instance_tlas",
                parameters=dict(instance_count=2, custom_indices=[11, 22]),
                bindings=dict(blas_0=front, blas_1=back, transforms=frame["poses"]),
            )
            supplied = dict(frame, front=front, scene=scene)
            public = module.prepare_execution(
                "public", {key: supplied[key] for key in module.execution_plan("public").argument_names}
            )
            start = time.perf_counter()
            private = module.prepare_execution(case, frame)
            case_costs["private_prepare_ms"] = (time.perf_counter() - start) * 1000
            case_costs["private_info"] = private.info
            shared = module.prepare_execution(case + "_shared_omm", dict(frame, omm=omm)) if recipe == "graph" else None
            for call in (public, private, *([shared] if shared else [])):
                for gate, limit, dz in ((0, 4, 0), (1, 4, 0.25), (0, 2.5, 0), (1, 0.5, 0), (1, 4, 0)):
                    write(frame["vertices"], [v + (dz if i % 3 == 2 else 0) for i, v in enumerate(vertices)])
                    write(frame["gate"], [gate], "i")
                    write(frame["limit"], [limit])
                    call()
                    check(frame, states, gate, limit, dz)
            samples = []
            for _ in range(8):
                start = time.perf_counter_ns()
                private()
                runtime.wait()
                samples.append((time.perf_counter_ns() - start) / 1000)
            case_costs["warm_complete_us"] = statistics.median(samples)
            costs["cases"][case] = case_costs
            public()
            omm.close()  # Retires the transitive public BLAS/TLAS recording.
            raises(public)
            if shared:
                raises(shared)
                shared.close()
            private()
            check(frame, states, 1, 4, 0)
            public.close()
            if not (mode == "borrowed" and case == "predefined"):
                private.close()
            scene.close()
            front.close()
            back.close()
        if mode == "borrowed":
            private()
            ti.reset()
            raises(private)
            private.close()
        else:
            assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
