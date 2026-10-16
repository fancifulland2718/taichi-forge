"""No-DSL CUB consumer: explicit provider, fixed topology, actual kernel AD."""

import array
import json
import os
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_binding_constraints_probe import failure, imported
from tests.python.aot_cutensor_probe import near
from tests.python.aot_segmented_scan_probe import expected, inputs, read, write


def main(mode, artifact, dtype, n, inclusive):
    n, inclusive = int(n), bool(int(inclusive))
    capacity = n + 7
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider(
            "cub_source",
            adapter_path=os.environ["TI_CUB_SOURCE_TEST_ADAPTER"],
            library_path=os.environ["TI_CUB_SOURCE_TEST_LIBRARY"],
        )
        assert runtime.probe_native_operation("cub_source_execute")["available"]
        module = runtime.load_module(artifact)
        frames = [
            {
                name: runtime.ndarray(t, shape)
                for name, t, shape in (
                    ("input", dtype, capacity),
                    ("result", dtype, capacity),
                    ("w", "f32", capacity),
                    ("dw", "f32", capacity),
                    ("loss", "f32", 1),
                )
            }
            for _ in range(2)
        ]
        for frame in frames:
            write(frame["result"], [73] * capacity, dtype)
            frame["w"].write(array.array("f", [0.5] * capacity))
        calls = [module.prepare_execution("step", frame) for frame in frames]
        assert all(read(frame["result"], dtype) == [73] * capacity for frame in frames)
        offsets = [0, 0, 1, 32, 32, n, n]

        def check(call, frame, iteration):
            data = inputs(capacity, dtype, iteration)
            write(frame["input"], data, dtype)
            call()
            reference = expected(offsets, data, dtype, inclusive)
            assert read(frame["result"], dtype) == reference

            def floats(name):
                values = array.array("f")
                values.frombytes(frame[name].read())
                return list(values)

            near(floats("dw"), [(v & 7) * 0.5 for v in reference[:n]] + [0] * 7)
            near(floats("loss"), [sum(v & 7 for v in reference[:n]) * 0.125])

        for iteration in (1, 0, -1):
            for index, (call, frame) in enumerate(zip(calls, frames)):
                assert call.info["execution_kind"] == "cuda_graph"
                check(call, frame, iteration * (index + 1))
        document = json.loads(module.execution_plan("step").to_json())
        for key, value, message in (
            ("binary_sha256", "0" * 64, "identity"),
            ("cub_version", 1, "identity"),
            ("workspace_bytes", 0, "workspace"),
            ("target_code", ["sm_999"], "target code"),
            ("minimum_driver_api_version", 999999, "driver"),
        ):
            bad = json.loads(json.dumps(document))
            resource = next(r for r in bad["resources"] if r["kind"] == "native")
            resource["parameters"][key] = value
            failure(
                lambda: module.prepare_execution(ExecutionPlan.from_json(json.dumps(bad)), frames[0]), message
            )
        failure(lambda: module.prepare_execution("step", dict(frames[0], result=frames[0]["input"])), "overlap")
        alias = imported(runtime, frames[0]["input"], "cuda")
        alias._dtype, alias._code = dtype, frames[0]["input"]._code
        failure(lambda: module.prepare_execution("step", dict(frames[0], result=alias)), "overlap")
        alias.close()
        document.update(version=10, binding_strategy="cuda_immutable_argument_frames")
        family = module.prepare_execution(ExecutionPlan.from_json(json.dumps(document)), frames[0])
        sibling = family.prepare_binding(frames[1])
        assert family.binding_family_info["executables"] == 1
        check(family, frames[0], 1)
        check(sibling, frames[1], -1)
        family.close()
        check(sibling, frames[1], 0)
        target, iterations = (runtime.ndarray("i32", 1) for _ in range(2))
        loop_args = dict(frames[0], target=target, iterations=iterations)
        loop = module.prepare_execution("loop", loop_args)
        for budget in (0, 1, 3, 5):
            target.write(array.array("i", [budget]))
            if budget:
                check(loop, frames[0], budget)
            else:
                write(frames[0]["result"], [73] * capacity, dtype)
                frames[0]["dw"].write(array.array("f", [91] * capacity))
                loop()
                assert read(frames[0]["result"], dtype) == [73] * capacity
                untouched = array.array("f")
                untouched.frombytes(frames[0]["dw"].read())
                assert list(untouched) == [91] * capacity
            count = array.array("i")
            count.frombytes(iterations.read())
            assert list(count) == [min(budget, 3)]
        loop.close()
        calls[0].close()
        check(calls[1], frames[1], 2)
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        failure(calls[1])
        failure(sibling)
        calls[1].close()
        sibling.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
