"""Source-free native compute consumer; gradients cover the downstream DSL kernel."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import statistics
import struct
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_optix_probe import raises, read, write


def dispatch(contract, source="input", destination="output"):
    return dict(
        kind="native",
        provider="vulkan_compute",
        operation="dispatch",
        version=1,
        bindings=dict(pipeline="pipeline", array_0=source, array_1=destination, array_2="scale"),
        parameters=dict(
            expected_shader=contract,
            groups=[2, 1, 1],
            array_count=3,
            buffers=[
                dict(set=0, binding=0, array=0, storage=True),
                dict(set=0, binding=1, array=1, storage=True),
                dict(set=1, binding=3, array=2, storage=False),
            ],
        ),
    )


def operations(contract):
    return [
        dispatch(contract, "input", "temp"),
        dispatch(contract, "temp", "output"),
        dict(kind="fill32", destination="doutput", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["output", "loss"]),
        dict(kind="kernel", entry="backward", arguments=["output", "loss"]),
    ]


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan")
    assets = Path(__file__).with_name("assets")
    code = (assets / "aot_compute.comp.spv").read_bytes()
    contract = hashlib.sha256(code).hexdigest()
    with runtime:
        assert "vulkan_inline" in runtime.native_operation_capabilities("vulkan_compute_dispatch")
        assert "vulkan_inline" in runtime.native_resource_capabilities("vulkan_compute_pipeline")
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        costs = dict(load_ms=(time.perf_counter() - start) * 1000, cases=[])

        def pipeline(data, expected=None):
            with runtime.host_blob(data) as blob:
                return runtime.native_resource(
                    "vulkan_compute",
                    "pipeline",
                    parameters=dict(expected_shader=expected or hashlib.sha256(data).hexdigest()),
                    bindings=dict(shader=blob),
                )

        raises(lambda: pipeline(code, "0" * 64), "contract mismatch")
        raises(lambda: pipeline(code[:-1]), "complete SPIR-V")
        raises(lambda: pipeline(bytes(20)), "SPIR-V magic")
        raises(lambda: pipeline((assets / "aot_compute_push.comp.spv").read_bytes()), "push constants")
        raises(lambda: pipeline((assets / "graphics_sync.vert.spv").read_bytes()), "compute main")
        # Change only OpExecutionMode LocalSize, keeping a structurally valid module.
        words = list(struct.unpack(f"<{len(code)//4}I", code))
        offset = 5
        while offset < len(words):
            size, opcode = words[offset] >> 16, words[offset] & 0xFFFF
            if opcode == 16 and words[offset + 2] == 17:
                words[offset + 3] = 0xFFFFFFFF
                break
            offset += size
        else:
            raise AssertionError("fixture has no literal LocalSize")
        raises(lambda: pipeline(struct.pack(f"<{len(words)}I", *words)), "physical device limits")
        owner = pipeline(code)  # Source host_blob is closed; pipeline consumed it.
        frame = {
            name: runtime.ndarray("f32", count)
            for name, count in dict(input=16, output=16, doutput=16, loss=1, dloss=1).items()
        }
        frame["scale"] = runtime.ndarray("f32", 4, usage=("uniform", "storage"))
        frame["pipeline"] = owner
        write(frame["scale"], [2, 0, 0, 0])
        calls = []
        for name in ("private", "public"):
            plan = module.execution_plan(name)
            start = time.perf_counter()
            call = plan.prepare(module, {key: frame[key] for key in plan.argument_names})
            prepare_ms = (time.perf_counter() - start) * 1000
            for base in (0, 3, -5):
                write(frame["input"], [float(j + base) for j in range(16)])
                call()
                expected = [float((j + base) * 4) for j in range(16)]
                assert read(frame["output"]) == expected
                assert read(frame["doutput"]) == expected
                assert read(frame["loss"]) == [sum(x * x / 2 for x in expected)]
            submits, completes = [], []
            for _ in range(5):
                start = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                submits.append((submitted - start) / 1000)
                completes.append((time.perf_counter_ns() - start) / 1000)
            costs["cases"].append(
                dict(
                    name=name,
                    prepare_ms=prepare_ms,
                    submit_us=statistics.median(submits),
                    complete_us=statistics.median(completes),
                    info=call.info,
                )
            )
            calls.append(call)

        def prepare(op, overrides=None):
            plan = ExecutionPlan([op])
            bindings = dict(frame, **(overrides or {}))
            return plan.prepare(module, {key: bindings[key] for key in plan.argument_names})

        # Cold validation failures leave existing recordings executable.
        for mutate, message in [
            (lambda p: p.update(expected_shader="0" * 64), "contract mismatch"),
            (lambda p: p.update(groups=[0xFFFFFFFF, 1, 1]), "physical device limits"),
            (lambda p: p["buffers"].append(p["buffers"][0]), "duplicated"),
            (lambda p: p["buffers"].pop(), "descriptor sets"),
            (lambda p: p["buffers"][2].update(storage=True), "type/count/coverage mismatch"),
            (lambda p: p["buffers"][2].update(binding=4), "absent from shader"),
        ]:
            op = dispatch(contract)
            mutate(op["parameters"])
            raises(lambda: prepare(op), message)
        with closing(runtime.ndarray("f32", 1, usage=("uniform",))) as small:
            raises(lambda: prepare(dispatch(contract), dict(scale=small)), "range/alignment")
        with closing(runtime.ndarray("f32", 4, usage=("storage",))) as wrong:
            raises(lambda: prepare(dispatch(contract), dict(scale=wrong)), "required usage")
        op = dispatch(contract)
        op["parameters"]["groups"] = [0, 1, 1]
        with prepare(op) as empty:
            write(frame["output"], [9] * 16)
            empty()
            assert read(frame["output"]) == [9] * 16
        # New array bindings prepare a new immutable execution; old one survives.
        with closing(runtime.ndarray("f32", 16)) as replacement:
            write(replacement, [7] * 16)
            with prepare(dispatch(contract), dict(input=replacement)) as rebound:
                rebound()
                assert read(frame["output"]) == [14] * 16
        calls[0]()
        owner.close()  # Retire dependent public execution, leave private pipeline live.
        raises(calls[1])
        calls[0]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(calls[0])
        for call in calls:
            call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(artifact).with_name(f"{mode}.json").write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
