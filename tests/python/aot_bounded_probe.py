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

        ti.init(
            arch=getattr(ti, arch), offline_cache=False, **({"saturating_grid_dim": 2} if case == "adaptive" else {})
        )
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
        document = json.loads(plan.to_json())
        launch = next((op["launch"] for op in document["operations"] if "launch" in op), {})
        indirect = launch.get("kind") == "vulkan_indirect"
        adaptive = launch.get("kind") == "cuda_adaptive"
        if adaptive:
            os.environ["TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY"] = (
                "per_node" if launch["update_policy"] == "grouped_stateful" else "grouped_stateful"
            )
        exact = adaptive or launch.get("kind") in ("cpu_exact", "cuda_logical_exact")
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
            if adaptive:
                assert call.info["execution_kind"] == "cuda_graph"
                assert call.info["workspace_bytes"] == (96 if launch["update_policy"] == "grouped_stateful" else 64)
            assert read(frame["y"]) == [-99] * 65
            for raw in (33, 33, 0, 0, 1, 32, 65, 70, -1, 2):
                count = max(0, min(raw, 65))
                write(frame["raw"], [raw], "i")
                call()
                assert read(frame["extent"], "i") == [count, int(count != raw)]
                assert read(frame["visited"], "i") == [
                    (
                        2 * count
                        if adaptive
                        else count if exact else min(65, ((count + 31) // 32) * 32) if indirect else 65
                    )
                ]
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
        if exact:
            payload = next(op for op in document["operations"] if "launch" in op)
            try:
                module.kernel(payload["entry"])(*(calls[0][0][name] for name in payload["arguments"]))
            except (AotError, ValueError, RuntimeError) as error:
                assert "bounded launch recipe" in str(error), error
            else:
                raise AssertionError("Direct exact kernel invocation without recipe accepted")
            controlled = ExecutionPlan(
                [dict(kind="while", predicate="raw", max_iterations=1, condition=[], body=[payload])]
            )
            try:
                controlled.prepare(module, {name: calls[0][0][name] for name in controlled.argument_names})
            except (AotError, ValueError, RuntimeError) as error:
                assert "flat CPU/CUDA" in str(error), error
            else:
                raise AssertionError("Unqualified exact control combination was accepted")
            ordinary = dict(payload)
            del ordinary["launch"]
            for operation in (
                ordinary,
                dict(
                    payload,
                    launch=dict(
                        kind="cpu_exact" if arch == "cuda" else "cuda_logical_exact",
                        extent=launch["extent"],
                        capacity=launch["capacity"],
                    ),
                ),
            ):
                invalid = ExecutionPlan([operation])
                try:
                    invalid.prepare(module, {name: calls[0][0][name] for name in invalid.argument_names})
                except (AotError, ValueError, RuntimeError) as error:
                    assert "bounded launch recipe" in str(error) or "backend mismatch" in str(error), error
                else:
                    raise AssertionError("Incompatible exact kernel launch accepted")
            if adaptive:
                invalid = ExecutionPlan([dict(payload, launch=dict(launch, block_dim=17))])
                try:
                    invalid.prepare(module, {name: calls[0][0][name] for name in invalid.argument_names})
                except (AotError, ValueError, RuntimeError) as error:
                    assert "matching block dimension" in str(error), error
                else:
                    raise AssertionError("Mismatched compiled block dimension accepted")
                other = dict(
                    payload,
                    arguments=[
                        "other_extent" if name == "extent" else "other_visited" if name == "visited" else name
                        for name in payload["arguments"]
                    ],
                    launch=dict(launch, extent="other_extent"),
                )
                distinct = ExecutionPlan([payload, other])
                with (
                    closing(runtime.ndarray("i32", 2)) as extra_extent,
                    closing(runtime.ndarray("i32", 1)) as extra_visited,
                ):
                    for alias in (True, False):
                        frame = {
                            name: calls[0][0][name]
                            for name in distinct.argument_names
                            if name not in ("other_extent", "other_visited")
                        }
                        frame.update(
                            other_extent=frame["extent"] if alias else extra_extent, other_visited=extra_visited
                        )
                        write(frame["extent"], [17, 0], "i")
                        write(extra_extent, [33, 0], "i")
                        write(frame["visited"], [0], "i")
                        write(extra_visited, [0], "i")
                        with distinct.prepare(module, frame) as separate:
                            # Logical names stay independent even if initially aliased.
                            assert separate.info["workspace_bytes"] == 64
                            separate()
                            assert read(frame["visited"], "i") == [17]
                            assert read(extra_visited, "i") == [17 if alias else 33]
            # The launch extent is a retained resource even if not a kernel argument.
            detached = dict(payload, launch=dict(launch, extent="launch_extent"))
            retained = ExecutionPlan([detached])
            private_extent = runtime.ndarray("i32", 2)
            write(private_extent, [2, 0], "i")
            retained_frame = {name: calls[0][0][name] for name in retained.argument_names if name != "launch_extent"}
            retained_call = retained.prepare(module, dict(retained_frame, launch_extent=private_extent))
            retained_call()
            private_extent.close()
            try:
                retained_call()
            except (AotError, ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("Closed launch-only extent was not retired")
            retained_call.close()
        for shape, dtype in ((1, "i32"), (3, "i32"), (2, "f32"), ((1, 2), "i32")):
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
