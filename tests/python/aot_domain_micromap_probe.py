"""Selected-domain OMM construction, shared owners, refit and explicit AD."""

from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import sys
import time

from taichi_forge_runtime.aot import Runtime, ExecutionPlan
from taichi_forge_runtime import runtime_contract_manifest
from tests.python.aot_vulkan_micromap_probe import POSES, asset, geometry, check
from tests.python.aot_vulkan_instances_probe import raises, write


def main(artifact, mode, output):
    assert runtime_contract_manifest()["features"]["vulkan_execution_domain_opacity_micromaps"]
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        owner = ti.aot.borrow_runtime()
    else:
        owner = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    evidence = []
    with ExitStack() as scope:
        runtime = scope.enter_context(owner)
        module = scope.enter_context(runtime.load_module(artifact))
        for case in ("two", "indexed", "predefined"):
            vertices, coords, states = geometry(case)
            count = len(vertices) // 9
            with ExitStack() as group:
                domains = [group.enter_context(runtime.execution_domain(capacity=1, instance_capacity=1)) for _ in range(2)]
                frames = []
                for _ in domains:
                    frame = {
                        name: group.enter_context(closing(runtime.ndarray(dtype, shape)))
                        for name, dtype, shape in (
                            ("vertices", "f32", (count * 3, 3)), ("indices", "i32", (count, 3)),
                            ("poses", "f32", (2, 12)), ("coords", "f32", (len(states), 2)),
                            ("result", "f32", (len(states), 4)), ("dresult", "f32", (len(states), 4)),
                            ("loss", "f32", (1,)), ("gate", "i32", (1,)), ("limit", "f32", (1,)),
                        )
                    }
                    write(frame["vertices"], vertices)
                    write(frame["indices"], list(range(count * 3)), "i")
                    write(frame["poses"], POSES)
                    write(frame["coords"], coords)
                    frames.append(frame)
                plan = module.execution_plan(case)
                # Fail after the OMM factory has constructed its device object.
                # Capacity and resources must remain reusable for a valid retry.
                bad = json.loads(plan.to_json())
                blas = next(r for r in bad["resources"] if r.get("operation") == "triangle_blas"
                            and "opacity_micromap" in r.get("bindings", {}))
                blas["parameters"]["opaque"] = True
                raises(lambda: ExecutionPlan.from_json(json.dumps(bad)).prepare(
                    module, frames[0], execution_domain=domains[0]), "opaque")
                assert domains[0].info["pending"] == 0 and domains[0].info["accepting"]
                calls = []
                for domain, frame in zip(domains, frames):
                    start = time.perf_counter()
                    call = group.enter_context(plan.prepare(module, frame, execution_domain=domain))
                    prepare_ms = (time.perf_counter() - start) * 1000
                    raises(call.submit)
                    call.initialize()
                    evidence.append(dict(case=case, prepare_ms=prepare_ms, info=call.info))
                    calls.append(call)
                steps = ((0, 4, 0), (1, 4, .25), (0, 2.5, 0), (1, .5, 0), (1, 4, 0))
                for step in range(len(steps)):
                    tickets, expected = [], []
                    for lane, (call, frame) in enumerate(zip(calls, frames)):
                        gate, limit, dz = steps[(step + lane) % len(steps)]
                        write(frame["vertices"], [v + (dz if i % 3 == 2 else 0) for i, v in enumerate(vertices)])
                        write(frame["gate"], [gate], "i")
                        write(frame["limit"], [limit])
                        tickets.append(call.submit())
                        expected.append((gate, limit, dz))
                    for ticket, frame, values in zip(tickets, frames, expected):
                        assert ticket is not None
                        ticket.wait()
                        check(frame, states, *values)
                        ticket.close()
                pending = calls[0].submit()
                calls[0].close()
                pending.wait()
                check(frames[0], states, *expected[0])
                pending.close()
                with calls[1].submit() as ticket:
                    ticket.wait()
                    check(frames[1], states, *expected[1])
                calls[1].close()

                # Two private BLAS/TLAS instances retain one public immutable OMM.
                data, desc, indices, _ = asset(case)
                blobs = {name: runtime.host_blob(value) for name, value in
                         (("data", data), ("descriptors", desc), ("indices", indices)) if value is not None}
                omm = group.enter_context(closing(runtime.native_resource(
                    "vulkan_ray", "opacity_micromap", parameters=dict(triangle_count=count), bindings=blobs)))
                for blob in blobs.values():
                    blob.close()
                shared = module.execution_plan(case + "_shared_omm")
                shared_calls = [group.enter_context(shared.prepare(module, dict(frame, omm=omm), execution_domain=domain))
                                for frame, domain in zip(frames, domains)]
                for call in shared_calls:
                    call.initialize()
                tickets = [call.submit() for call in shared_calls]
                omm.close()  # Transitively retires both BLAS/TLAS readers.
                for call, ticket, frame, values in zip(shared_calls, tickets, frames, expected):
                    ticket.wait()
                    check(frame, states, *values)
                    raises(call.submit)
                    ticket.close()
                    call.close()
        if mode == "borrowed":
            ti.reset()
        else:
            assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(evidence, indent=2), encoding="utf8")
    print("micromap_domain_ok")


if __name__ == "__main__":
    main(*sys.argv[1:])
