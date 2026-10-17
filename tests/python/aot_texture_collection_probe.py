"""Source-independent image tables, GPU updates and downstream gradients."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime

if __package__:
    from .aot_kernel_image_probe import native
    from .aot_optix_probe import raises, read, write
else:
    from aot_kernel_image_probe import native
    from aot_optix_probe import raises, read, write


def plan_parts(private=True):
    resources = []
    if private:
        resources = [
            dict(native("image", "sampled", dict(pixels="seed"), shape=[2, 2], format="r32f"), name="first"),
            dict(native("image", "sampled", {}, shape=[3, 2], format="r32f"), name="second"),
            dict(
                native(
                    "image",
                    "collection",
                    dict(image_0="second", image_1="first", image_2="second"),
                    dimensions=2,
                    image_count=3,
                ),
                name="table",
            ),
        ]
    return dict(
        resources=resources,
        operations=[
            dict(kind="kernel", entry="paint", arguments=["scale", "pixels", "selectors"]),
            native("image", "upload", dict(image="second", pixels="pixels")),
            dict(kind="kernel", entry="sample", arguments=["table", "selectors", "output"]),
            dict(kind="fill32", destination="doutput", value=0),
            dict(kind="fill32", destination="loss", value=0),
            dict(kind="fill32", destination="dloss", value=1065353216),
            dict(kind="kernel", entry="energy", arguments=["output", "loss"]),
            dict(kind="kernel", entry="backward", arguments=["output", "loss"]),
        ],
        gradients=dict(output="doutput", loss="dloss"),
    )


def main(mode, artifact, result):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan")
    costs = []
    with runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        frame = {
            name: runtime.ndarray("i32" if name == "selectors" else "f32", shape)
            for name, shape in dict(
                seed=(2, 2), pixels=(3, 2), selectors=(6,), scale=(1,), output=(6,), doutput=(6,), loss=(1,), dloss=(1,)
            ).items()
        }
        write(frame["seed"], [2] * 4)
        write(frame["output"], [-99] * 6)
        private = module.execution_plan("private")
        # Uploads inside a branch/zero-trip loop cannot establish unconditional
        # initialization. The shared recording contract tests those merges;
        # this route specifically checks every unique collection member.
        missing = json.loads(private.to_json())
        missing["operations"] = [op for op in missing["operations"] if op.get("provider") != "image"]
        invalid = ExecutionPlan.from_json(json.dumps(missing))
        raises(lambda: invalid.prepare(module, {n: frame[n] for n in invalid.argument_names}), "unconditional upload")
        assert read(frame["output"]) == [-99] * 6

        retained = []
        for policy in ("private", "public"):
            first = second = table = None
            bindings = dict(frame)
            if policy == "public":
                first = runtime.native_resource(
                    "image",
                    "sampled",
                    parameters=dict(shape=[2, 2], format="r32f"),
                    bindings=dict(pixels=frame["seed"]),
                )
                second = runtime.native_resource("image", "sampled", parameters=dict(shape=[3, 2], format="r32f"))

                def collection(members):
                    return runtime.native_resource(
                        "image",
                        "collection",
                        parameters=dict(dimensions=2, image_count=len(members)),
                        bindings={f"image_{i}": image for i, image in enumerate(members)},
                    )

                table = collection([second, first, second])
                bindings.update(table=table, second=second)
            plan = module.execution_plan(policy)
            bindings = {n: bindings[n] for n in plan.argument_names}
            start = time.perf_counter()
            call = plan.prepare(module, bindings)
            prepare_ms = (time.perf_counter() - start) * 1000
            assert call.info["execution_kind"] == "vulkan_commands"
            if policy == "private":
                assert call.info["native_resource_bytes"] == 40

            def check(scale, alternate=False):
                expected = ([4, 4.5 * scale, 4] if alternate else [4.5 * scale, 4, 4.5 * scale]) * 2
                for name in ("output", "doutput"):
                    assert all(abs(a - b) < 1e-5 for a, b in zip(read(frame[name]), expected)), (
                        name,
                        read(frame[name]),
                        expected,
                    )
                assert abs(read(frame["loss"])[0] - sum(v * v * 0.5 for v in expected)) < 1e-4

            for scale in (1, 0, -2, 3):
                write(frame["scale"], [scale])
                call()
                check(scale)
            samples, submits = [], []
            for _ in range(4):
                start = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                samples.append((time.perf_counter_ns() - start) / 1000)
                submits.append((submitted - start) / 1000)
            costs.append(
                dict(
                    policy=policy,
                    prepare_ms=prepare_ms,
                    complete_us=statistics.median(samples),
                    submit_us=statistics.median(submits),
                    info=call.info,
                )
            )
            if policy == "public":
                wrong = collection([first, second])
                raises(lambda: plan.prepare(module, dict(bindings, table=wrong)), "dimension/capacity")
                wrong.close()
                replacement = collection([first, second, first])
                sibling = plan.prepare(module, dict(bindings, table=replacement))
                sibling()
                check(3, True)
                call()
                check(3)
                # Explicit close waits for work and invalidates both tables.
                sibling()
                first.close()
                raises(call)
                raises(sibling)
                for value in (call, sibling, replacement, table, second):
                    value.close()
            else:
                retained.append(call)
        retained[0]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(retained[0])
        retained[0].close()
        for value in frame.values():
            value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(result).write_text(json.dumps(dict(load_ms=load_ms, cases=costs), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
