"""Runtime-only private image ownership, queue initialization and gradients."""

from contextlib import ExitStack
import json
import os
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from aot_optix_probe import read, write


def main(artifact):
    with ExitStack() as scope:
        runtime = scope.enter_context(Runtime(
            "vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))))
        module = runtime.load_module(artifact)
        queues = [scope.enter_context(runtime.execution_domain(capacity=2, instance_capacity=1, gpu_timing=True))
                  for _ in range(2)]
        for policy in ("seeded", "uploaded"):
            calls, frames = [], []
            plan = module.execution_plan(policy)
            for lane, queue in enumerate(queues):
                frame = {name: runtime.ndarray("f32", shape)
                         for name, shape in (("seed", (2, 2)), ("output", 4), ("doutput", 4),
                                             ("loss", 1), ("dloss", 1), ("delta", 1))}
                for value in frame.values():
                    scope.callback(value.close)
                write(frame["seed"], [lane * 10 + i for i in range(4)])
                if policy == "seeded":
                    # Externally owned images do not acquire a private lease.
                    external = scope.enter_context(runtime.native_resource(
                        "image", "sampled", parameters=dict(shape=[2, 2], format="r32f", storage=True),
                        bindings=dict(pixels=frame["seed"])))
                    description = json.loads(plan.to_json())
                    description["resources"] = []
                    borrowed = ExecutionPlan.from_json(json.dumps(description))
                    supplied = dict(frame, image=external)
                    try:
                        borrowed.prepare(module, {key: supplied[key] for key in borrowed.argument_names},
                                         execution_domain=queue)
                    except RuntimeError as error:
                        assert "Shared Vulkan image requires a sampled shader-read use" in str(error), str(error)
                    else:
                        raise AssertionError("Borrowed image accepted without a layout contract")
                call = plan.prepare(module, {key: frame[key] for key in plan.argument_names},
                                    execution_domain=queue)
                assert call.info["native_resource_bytes"] == 16
                call.initialize()
                calls.append(call)
                frames.append(frame)
            total = 0
            for increment in (1, 3, -2):
                total += increment
                for frame in frames:
                    write(frame["delta"], [float(increment)])
                tickets = [call.submit(sample_gpu=True) for call in calls]
                assert all(ticket is not None for ticket in tickets)
                for lane, (ticket, frame) in enumerate(zip(tickets, frames)):
                    ticket.wait()
                    expected = [lane * 10 + i + total for i in range(4)]
                    assert read(frame["output"]) == expected
                    assert read(frame["doutput"]) == expected
                    assert read(frame["loss"]) == [sum(v * v * 0.5 for v in expected)]
                    assert ticket.gpu_timing["status"] == "ready"
                    ticket.close()
            write(frames[0]["delta"], [9.0])
            pending = calls[0].submit()
            calls[0].close()  # Retire an accepted submission and its private image.
            pending.wait()
            pending.close()
            write(frames[1]["delta"], [1.0])
            ticket = calls[1].submit()
            ticket.wait()
            assert read(frames[1]["output"]) == [13 + i for i in range(4)]
            ticket.close()
            calls[1].close()
        module.close()
    assert "taichi_forge" not in sys.modules
    print("private_image_instances=4")


if __name__ == "__main__":
    main(*sys.argv[1:])
