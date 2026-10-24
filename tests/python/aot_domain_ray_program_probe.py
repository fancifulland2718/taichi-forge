"""Independent programmable RT: cold SBTs, private AS/images and real gradients."""

from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import struct
import sys
import time

from taichi_forge_runtime import runtime_contract_manifest
from taichi_forge_runtime.aot import ExecutionPlan, Runtime, Scalar
from tests.python.aot_vulkan_ray_program_probe import POSES, VERTICES, check_gradients
from tests.python.aot_optix_probe import raises, read, write


def frame_arrays(runtime, scope, images, lane):
    shapes = dict(features=(24 if images else 5,), dfeatures=(24 if images else 5,), loss=(1,))
    shapes.update(
        dict(pixels=(3, 2, 4), gain=(1,))
        if images
        else dict(vertices=(6, 3), indices=(2, 3), poses=(2, 12), output=(5, 8))
    )
    frame = {
        key: scope.enter_context(
            closing(runtime.ndarray("i32" if key == "indices" else "u32" if key == "output" else "f32", shape))
        )
        for key, shape in shapes.items()
    }
    if images:
        write(frame["pixels"], [(i + lane) * 0.25 for i in range(24)])
        write(frame["gain"], [1])
    else:
        write(frame["vertices"], VERTICES)
        write(frame["indices"], list(range(6)), "i")
        write(frame["poses"], POSES)
    write(frame["features"], [99] * shapes["features"][0])
    frame["phase"] = Scalar("u64", 1 << (48 + lane))
    return frame


def check(frame, images, state, lane):
    expected = (
        [(i + lane) * 0.25 * (state + 1) + 0.5 for i in range(24)]
        if images
        else [-1] * 5 if state == 2 else [2.25, 1.25, 2.25, 1.25, -1] if state == 1 else [2, 1, 2, 1, -1]
    )
    check_gradients(frame, expected)
    if not images:
        values = read(frame["output"], "I")
        ids = [[1, 0, 7, 13], [0, 0, 7, 23], [1, 1, 11, 43], [0, 1, 11, 53]]
        geometry = struct.unpack("<40f", struct.pack("<40I", *values))
        for i in range(5):
            expected_ids = [0xFFFFFFFF] * 3 + [200 + i % 2] if state == 2 or i == 4 else ids[i]
            assert values[8 * i : 8 * i + 4] == expected_ids
            if state != 2 and i < 4:
                assert geometry[8 * i + 5 : 8 * i + 8] == (0.25, 0.25, 1)


def main(mode, artifact, output_file, recipe="domain"):
    assert recipe == "domain"
    assert runtime_contract_manifest()["features"]["vulkan_execution_domain_ray_programs"]
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        owner = ti.aot.borrow_runtime()
    else:
        owner = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    evidence, retained = [], []
    with ExitStack() as scope:
        runtime = scope.enter_context(owner)
        module = scope.enter_context(runtime.load_module(artifact))
        domains = [
            scope.enter_context(runtime.execution_domain(capacity=1, instance_capacity=1, gpu_timing=True))
            for _ in range(2)
        ]
        for images in (False, True):
            plan = module.execution_plan("images_private" if images else "rays_private")
            with ExitStack() as group:
                frames = [frame_arrays(runtime, group, images, lane) for lane in range(2)]
                # Fail after the explicit initializer's launch was bound. All
                # deferred owners must roll back without a partial execution.
                bad = json.loads(plan.to_json())
                launch = next(op for op in bad["operations"] if op.get("provider") == "vulkan_ray_program")
                launch["parameters"]["buffers"][0]["binding"] = 0
                raises(
                    lambda: ExecutionPlan.from_json(json.dumps(bad)).prepare(
                        module, frames[0], execution_domain=domains[0]
                    ),
                    "Duplicate",
                )
                assert domains[0].info["accepting"] and domains[0].info["pending"] == 0
                calls = []
                for lane, (domain, frame) in enumerate(zip(domains, frames)):
                    started = time.perf_counter()
                    call = group.enter_context(
                        plan.prepare(module, frame, execution_domain=domain, mutable_parameters=("phase",))
                    )
                    prepare_ms = (time.perf_counter() - started) * 1000
                    assert read(frame["features"]) == [99] * (24 if images else 5)
                    raises(lambda: call.submit(values=dict(phase=0)))
                    started = time.perf_counter()
                    call.initialize()
                    initialize_ms = (time.perf_counter() - started) * 1000
                    check(frame, images, 0, lane)  # Explicit initialization consumed its own SBT.
                    evidence.append(
                        dict(
                            images=images, lane=lane, prepare_ms=prepare_ms, initialize_ms=initialize_ms, info=call.info
                        )
                    )
                    calls.append(call)
                for step in (0, 1, 2, 0):
                    tickets = []
                    for lane, call in enumerate(calls):
                        values = dict(phase=(1 << (48 + lane)) | ((step + lane) % 3))
                        ticket = call.submit(values=values, sample_gpu=step == 1)
                        assert ticket is not None
                        values["phase"] = 7
                        tickets.append(ticket)
                    for lane, ticket in enumerate(tickets):
                        ticket.wait()
                        check(frames[lane], images, (step + lane) % 3, lane)
                        timing = ticket.gpu_timing
                        assert timing["status"] == ("ready" if step == 1 else "not_sampled")
                        if step == 1:
                            assert timing["duration_ns"] > 0 and timing["scope"] == "execution_envelope"
                            assert timing["sequence"] == ticket.info["sequence"]
                            retained.append((ticket, timing))
                        else:
                            ticket.close()
                pending = calls[0].submit(values=dict(phase=(1 << 48) | 2))
                calls[0].close()
                pending.wait()
                check(frames[0], images, 2, 0)
                pending.close()
                with calls[1].submit(values=dict(phase=(1 << 49) | 1)) as ticket:
                    ticket.wait()
                    check(frames[1], images, 1, 1)
                calls[1].close()

                if not images:
                    # Exercise the launch's own closure check, without an AS
                    # build/refit operation masking a shared-owner admission.
                    blas = group.enter_context(
                        closing(
                            runtime.native_resource(
                                "vulkan_ray",
                                "triangle_blas",
                                parameters=dict(vertex_count=6, triangle_count=2, opaque=False),
                                bindings=dict(vertices=frames[0]["vertices"], indices=frames[0]["indices"]),
                            )
                        )
                    )
                    scene = group.enter_context(
                        closing(
                            runtime.native_resource(
                                "vulkan_ray",
                                "instance_tlas",
                                parameters=dict(instance_count=2, custom_indices=[7, 11], sbt_record_offsets=[1, 4]),
                                bindings=dict(blas_0=blas, blas_1=blas, transforms=frames[0]["poses"]),
                            )
                        )
                    )
                    description = json.loads(plan.to_json())
                    description["resources"] = [
                        r for r in description["resources"] if r["name"] not in ("blas", "scene")
                    ]
                    description.pop("initialization", None)
                    description["operations"] = [
                        op
                        for op in description["operations"]
                        if op.get("provider") != "vulkan_ray" and op.get("entry") != "ray_state"
                    ]
                    shared = ExecutionPlan(
                        description["operations"],
                        resources=description["resources"],
                        gradients=description["gradients"],
                    )
                    available = dict(frames[0], scene=scene)
                    arguments = {name: available[name] for name in shared.argument_names}
                    with shared.prepare(module, arguments) as ordinary:
                        ordinary.initialize()
                        ordinary.run()
                        check(frames[0], False, 2, 0)
                    with shared.prepare(module, arguments, execution_domain=domains[0]) as reader:
                        reader.initialize()
                        with reader.submit() as ticket:
                            ticket.wait()
                            check(frames[0], False, 2, 0)

                with plan.prepare(
                    module, frames[0], execution_domain=domains[0], mutable_parameters=("phase",)
                ) as rebound:
                    rebound.initialize()
                    with rebound.submit(values=dict(phase=(1 << 48) | 1), sample_gpu=True) as ticket:
                        ticket.wait()
                        check(frames[0], images, 1, 0)
        if mode == "borrowed":
            frame = frame_arrays(runtime, scope, False, 0)
            call = scope.enter_context(
                module.execution_plan("rays_private").prepare(
                    module, frame, execution_domain=domains[0], mutable_parameters=("phase",)
                )
            )
            call.initialize()
            ticket = call.submit(values=dict(phase=1 << 48), sample_gpu=True)
            ti.reset()
            ticket.wait()
            assert ticket.done()
            retained.append((ticket, ticket.gpu_timing))
            raises(lambda: call.submit(values=dict(phase=0)))
    for ticket, timing in retained:
        assert timing["status"] == "ready" and ticket.gpu_timing == timing
        ticket.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(output_file).write_text(json.dumps(dict(instances=evidence, timing=[t for _, t in retained]), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
