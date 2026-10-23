"""Runtime-only validation of private image control, failure and retirement."""

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
        queues = [scope.enter_context(runtime.execution_domain(capacity=2, instance_capacity=1))
                  for _ in range(2)]
        for name in ("alias", "zero", "chunks", "conditional", "coarse_conditional", "nested"):
            with ExitStack() as case:
                calls, frames = [], []
                plan = module.execution_plan(name)
                for lane, queue in enumerate(queues):
                    frame = {key: runtime.ndarray("f32", shape)
                             for key, shape in (("seed", (2, 2)), ("output", 4), ("doutput", 4),
                                                ("loss", 1), ("dloss", 1), ("delta", 1))}
                    frame.update({key: runtime.ndarray("i32", 1)
                                  for key in ("target", "outer_target")})
                    for array in frame.values():
                        case.callback(array.close)
                    write(frame["seed"], [lane * 10 + i for i in range(4)])
                    write(frame["delta"], [float(lane + 1)])
                    write(frame["outer_target"], [lane + 1], "i")
                    if name == "chunks":
                        external = runtime.native_resource(
                            "image", "sampled", parameters=dict(shape=[2, 2], format="r32f", storage=True),
                            bindings=dict(pixels=frame["seed"]))
                        case.callback(external.close)
                        description = json.loads(plan.to_json())
                        description["resources"] = [r for r in description["resources"] if r["name"] != "image"]
                        borrowed = ExecutionPlan.from_json(json.dumps(description))
                        supplied = dict(frame, image=external)
                        try:
                            borrowed.prepare(module, {key: supplied[key] for key in borrowed.argument_names},
                                             execution_domain=queue)
                        except RuntimeError as error:
                            assert "private managed image" in str(error), str(error)
                        else:
                            raise AssertionError("Borrowed image accepted without a layout contract")
                    try:
                        call = plan.prepare(module, {key: frame[key] for key in plan.argument_names},
                                            execution_domain=queue)
                    except RuntimeError as error:
                        assert name == "alias" and "both sampled and storage" in str(error), str(error)
                        continue
                    assert name != "alias", "Aliased sampled/storage task accepted"
                    case.callback(call.close)
                    call.initialize()
                    calls.append(call)
                    frames.append(frame)
                if name == "alias":
                    # Following plans reuse both domains after failed prepare.
                    continue
                totals = [0, 0]
                for target in (0, 1, 5, 0):
                    for frame in frames:
                        write(frame["target"], [target], "i")
                    tickets = [call.submit() for call in calls]
                    for lane, (ticket, frame) in enumerate(zip(tickets, frames)):
                        assert ticket is not None
                        ticket.wait()
                        rounds = 0 if name == "zero" else min(3, target)
                        rounds *= lane + 1 if name == "nested" else 1
                        totals[lane] += 2 * (lane + 1) * rounds
                        expected = [lane * 10 + i + totals[lane] for i in range(4)]
                        assert read(frame["output"]) == expected, (name, lane, target, read(frame["output"]), expected)
                        assert read(frame["doutput"]) == expected
                        assert read(frame["loss"]) == [sum(v * v * 0.5 for v in expected)]
                        ticket.close()
                # Retire accepted work; the second instance must remain valid.
                write(frames[0]["target"], [1], "i")
                pending = calls[0].submit()
                calls[0].close()
                pending.wait()
                pending.close()
                ticket = calls[1].submit()
                ticket.wait()
                assert read(frames[1]["output"]) == [10 + i + totals[1] for i in range(4)]
                ticket.close()
        module.close()
    assert "taichi_forge" not in sys.modules
    print("private_image_control_instances=10")


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])
    except RuntimeError as error:
        if "No independent Vulkan compute queue is available" in str(error) or "timeline support" in str(error):
            print("Selected Vulkan device cannot qualify independent image control:", error)
            sys.exit(77)
        raise
