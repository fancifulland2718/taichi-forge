"""Live native AS loans: canonical BLAS closure, old recordings and retirement."""

from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from taichi_forge_runtime import runtime_contract_manifest
from aot_domain_ray_probe import TRANSFORMS, check, frame_arrays
from aot_optix_probe import raises, write
from aot_vulkan_ray_probe import native


def main(folder, mode):
    assert runtime_contract_manifest()["features"]["vulkan_execution_domain_native_ray_loans"]
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        owner = ti.aot.borrow_runtime()
    else:
        owner = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    evidence = []
    with ExitStack() as scope:
        runtime = scope.enter_context(owner)
        module = scope.enter_context(runtime.load_module(folder))
        domains = [scope.enter_context(runtime.execution_domain(capacity=1, instance_capacity=1)) for _ in range(3)]
        frames = [frame_arrays(runtime, scope, True) for _ in domains]
        blas = scope.enter_context(closing(runtime.native_resource(
            "vulkan_ray", "triangle_blas", parameters=dict(vertex_count=6, triangle_count=2),
            bindings=dict(vertices=frames[0]["seed"], indices=frames[0]["indices"]))))
        scenes = [scope.enter_context(closing(runtime.native_resource(
            "vulkan_ray", "instance_tlas", parameters=dict(instance_count=2, custom_indices=[19, 23]),
            bindings=dict(blas_0=blas, blas_1=blas, transforms=frame["initial_transforms"]))))
                  for frame in frames[:2]]
        document = json.loads(module.execution_plan("instances").to_json())
        query_index = next(i for i, op in enumerate(document["operations"]) if op.get("entry") == "query_kernel")
        tail = document["operations"][query_index + 1:]
        resources = [r for r in document["resources"] if r["name"] == "dloss"]

        def query_plan(*, writes=False, kernel=True):
            query = (document["operations"][query_index] if kernel else
                     native("query", dict(scene="scene", rays="rays", hits="hits", hit_indices="ids"), ray_count=6))
            operations = ([native("instance_refit", dict(scene="scene", transforms="initial_transforms"))]
                          if writes else []) + [query] + tail
            return ExecutionPlan(operations, resources=resources, gradients=document["gradients"])

        def prepare(plan, frame, scene, domain=None):
            available = dict(frame, scene=scene)
            values = {name: available[name] for name in plan.argument_names}
            started = time.perf_counter()
            call = scope.enter_context(plan.prepare(module, values, execution_domain=domain))
            call.initialize()
            evidence.append(dict(prepare_ms=(time.perf_counter() - started) * 1000, info=call.info))
            return call

        def run(call, frame, shift, *, kernel=True):
            with call.submit() as ticket:
                assert ticket is not None
                ticket.wait()
                check(frame, shift, True, kernel)

        reader_plan, writer_plan = query_plan(), query_plan(writes=True, kernel=False)
        # These recordings predate every loan; warm submit must see conflicts.
        old_reader = prepare(reader_plan, frames[0], scenes[0])
        old_writer = prepare(writer_plan, frames[0], scenes[0])
        build_plan = ExecutionPlan([native("blas_build", dict(scene="scene", vertices="seed", indices="indices"))])
        old_blas_writer = prepare(build_plan, frames[0], blas)

        readers = [prepare(reader_plan, frames[lane], scenes[0], domains[lane]) for lane in (0, 1)]
        raises(old_writer.run, "AS loan")
        raises(old_blas_writer.run, "AS loan")
        raises(lambda: prepare(writer_plan, frames[2], scenes[0], domains[2]), "AS loan")
        # Same BLAS, different TLAS: this write is compatible with both readers.
        other_writer = prepare(writer_plan, frames[2], scenes[1], domains[2])
        moved = list(TRANSFORMS)
        moved[11] += 2
        moved[23] += 2
        write(frames[2]["initial_transforms"], moved)
        tickets = [call.submit() for call in readers] + [other_writer.submit()]
        for lane, ticket in enumerate(tickets):
            assert ticket is not None
            ticket.wait()
            check(frames[lane], 2 if lane == 2 else 0, True, lane != 2)
            ticket.close()
        old_reader.run()  # Ordinary read remains compatible with shared readers.
        check(frames[0], 0, True, True)
        readers[0].close()
        run(readers[1], frames[1], 0)
        raises(old_writer.run, "AS loan")  # The second reader still owns a loan.
        readers[1].close()
        old_writer.run()
        check(frames[0], 0, True)
        # Two distinct TLAS writers may coexist while sharing the BLAS read.
        writer = prepare(writer_plan, frames[0], scenes[0], domains[0])
        raises(old_reader.run, "AS loan")
        raises(old_blas_writer.run, "AS loan")
        raises(lambda: prepare(reader_plan, frames[1], scenes[0], domains[1]), "AS loan")
        write(frames[0]["initial_transforms"], moved)
        pending = writer.submit()
        writer.close()  # Return access only after this exact invocation retires.
        pending.wait()
        check(frames[0], 2, True)
        pending.close()
        old_reader.run()  # Device visibility must be restored to ordinary work.
        check(frames[0], 2, True, True)
        run(other_writer, frames[2], 2, kernel=False)
        other_writer.close()
        old_blas_writer.run()  # All transitive readers have now retired.

        # A failed borrow did not consume instance capacity or leave a partial
        # loan. The same domains and owners can be prepared and reused again.
        resumed = prepare(reader_plan, frames[1], scenes[0], domains[1])
        run(resumed, frames[1], 2)
        ticket = resumed.submit()
        if mode == "borrowed":
            ti.reset()
            raises(resumed.submit)
        else:
            domains[1].close()
            raises(resumed.submit)
        ticket.wait()
        assert ticket.done()
        ticket.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(folder).with_name(f"shared-{mode}.json").write_text(json.dumps(evidence, indent=2), encoding="utf8")
    print("shared_native_ray_ok")


if __name__ == "__main__":
    main(*sys.argv[1:])
