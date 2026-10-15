"""Bounded mixed execution in a consumer with no producer DSL dependency."""

import array
from contextlib import closing
import json
import os
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime


def segments(runtime, module):
    frame = {name: runtime.ndarray("i32", size) for name, size in (("offsets", 4), ("extent", 2), ("output", 65))}
    plan = module.execution_plan("step")
    with plan.prepare(module, frame) as call:
        for offsets in ([0, 17, 17, 65], [0, 1, 33, 65], [0, 0, 0, 65]):
            frame["offsets"].write(array.array("i", offsets))
            for count in (65, 0, 1, 16, 17, 18, 32, 33):
                frame["extent"].write(array.array("i", [count, 0]))
                call()
                actual = array.array("i")
                actual.frombytes(frame["output"].read())
                expected = [0] * 65
                for index in range(3):
                    begin, end = min(offsets[index], count), min(offsets[index + 1], count)
                    for i in range(begin, end):
                        expected[i] = index + 1 + (expected[begin - 1] if begin else 0)
                assert actual.tolist() == expected, (offsets, count, actual.tolist())
        with closing(runtime.ndarray("i32", 3)) as bad:
            try:
                plan.prepare(module, dict(frame, offsets=bad))
            except (AotError, ValueError):
                pass
            else:
                raise AssertionError("Invalid offsets ABI accepted")
    for value in frame.values():
        value.close()


def main(mode, arch, artifact, case="bounded"):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0
        )

    def write(a, values, dtype="f"):
        a.write(array.array(dtype, values))

    def read(a, dtype="f"):
        values = array.array(dtype)
        values.frombytes(a.read())
        return values.tolist()

    with runtime:
        module = runtime.load_module(artifact)
        if case == "segments":
            segments(runtime, module)
            module.close()
            if mode == "standalone":
                assert "taichi_forge" not in sys.modules
            return
        plan = module.execution_plan("step")
        indirect = any("launch" in op for op in json.loads(plan.to_json())["operations"])
        calls = []
        for factor in (1, -0.5):
            frame = {
                name: runtime.ndarray(
                    "i32" if name in ("extent", "raw", "visited") else "f32",
                    2 if name == "extent" else 1 if name in ("raw", "loss", "dloss", "visited") else 65,
                )
                for name in plan.argument_names
            }
            xs = [factor * i / 16 for i in range(65)]
            write(frame["x"], xs)
            write(frame["y"], [-99] * 65)
            call = plan.prepare(module, frame)
            assert read(frame["y"]) == [-99] * 65
            for raw in (33, 0, 1, 32, 65, 70, -1, 2):
                count = max(0, min(raw, 65))
                write(frame["raw"], [raw], "i")
                call()
                assert read(frame["extent"], "i") == [count, int(count != raw)]
                assert read(frame["visited"], "i") == [min(65, ((count + 31) // 32) * 32) if indirect else 65]
                y = [v * v if i < count else 0 for i, v in enumerate(xs)]
                dx = [2 * v**3 if i < count else 0 for i, v in enumerate(xs)]
                for key, expected in (("y", y), ("dy", y), ("dx", dx), ("loss", [sum(v * v for v in y) / 2])):
                    assert all(abs(a - b) <= 3e-5 * (1 + abs(b)) for a, b in zip(read(frame[key]), expected)), key
            calls.append((frame, call))
        if indirect:
            document = json.loads(plan.to_json())
            payload = next(op for op in document["operations"] if "launch" in op)
            unsupported = ExecutionPlan(
                [
                    dict(
                        kind="while",
                        predicate="raw",
                        max_iterations=1,
                        condition=[],
                        body=[payload],
                        lowering=dict(
                            kind="vulkan_structured",
                            version=1,
                            counter="counter",
                            chunks=[dict(iterations=1, strategy="compact")],
                        ),
                    )
                ],
                resources=[*document["resources"], dict(kind="ndarray", name="counter", dtype="i32", shape=[1])],
            )
            unsupported_frame = {name: calls[0][0][name] for name in unsupported.argument_names}
            try:
                unsupported.prepare(module, unsupported_frame)
            except (AotError, ValueError) as error:
                assert "flat Vulkan" in str(error), error
            else:
                raise AssertionError("Indirect dispatch inside control was silently flattened")
            external = module.execution_plan("external_packet")
            packet_name = (set(external.argument_names) - set(plan.argument_names)).pop()
            for dtype, shape in (("u32", 2), ("i32", 3), ("u32", (1, 3))):
                with closing(runtime.ndarray(dtype, shape)) as bad:
                    try:
                        external.prepare(module, dict(calls[0][0], **{packet_name: bad}))
                    except (AotError, ValueError):
                        pass
                    else:
                        raise AssertionError("Invalid indirect packet accepted")
            packet = runtime.ndarray("u32", 3)
            packet_call = external.prepare(module, dict(calls[0][0], **{packet_name: packet}))
            packet_call()
            packet.close()
            try:
                packet_call()
            except (AotError, ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("Closed indirect packet was not retired")
            packet_call.close()
        for shape, dtype in ((1, "i32"), (3, "i32"), (2, "f32")):
            bad = runtime.ndarray(dtype, shape)
            try:
                plan.prepare(module, dict(calls[0][0], extent=bad))
            except (AotError, ValueError):
                pass
            else:
                raise AssertionError("Invalid extent ABI accepted")
            bad.close()
        calls[0][1]()
        calls[0][1].close()
        calls[1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        try:
            calls[1][1]()
        except (AotError, ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("Retired bounded execution accepted")
        calls[1][1].close()
        for frame, _call in calls:
            for value in frame.values():
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
