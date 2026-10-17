"""Standalone SPD pyramid oracle and downstream AD; no shader compiler imports."""

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


def plan_parts(spec, key, byte_size, private=True):
    width, height = spec["shape"]
    params = dict(reduction=spec["reduction"], expected_plan=spec["expected_plan"])
    resources = []
    if private:
        resources = [
            dict(kind="host_blob", name="shader", data=key, byte_size=byte_size),
            dict(native("image", "sampled", {}, shape=[width, height], format="r32f", storage=True), name="source"),
            dict(
                native(
                    "image",
                    "sampled",
                    {},
                    shape=[width // 2, height // 2],
                    format="r32f",
                    storage=True,
                    mip_levels=spec["levels"],
                ),
                name="pyramid",
            ),
            dict(
                native("spd", "plan", dict(source="source", output="pyramid", shader="shader"), **params), name="plan"
            ),
        ]
    return dict(
        resources=resources,
        gradients=dict(output="doutput", loss="dloss"),
        operations=[
            native("image", "upload", dict(image="source", pixels="pixels")),
            native(
                "spd", "run", dict(plan="plan", source="source", output="pyramid"), expected_plan=spec["expected_plan"]
            ),
            dict(kind="kernel", entry="sample", arguments=["pyramid", "output"]),
            dict(kind="fill32", destination="doutput", value=0),
            dict(kind="fill32", destination="loss", value=0),
            dict(kind="fill32", destination="dloss", value=1065353216),
            dict(kind="kernel", entry="energy", arguments=["output", "loss"]),
            dict(kind="kernel", entry="backward", arguments=["output", "loss"]),
        ],
    )


def pyramid(values, width, height, levels, reduction):
    result = []
    for _ in range(levels):
        following = []
        for y in range(height // 2):
            for x in range(width // 2):
                v = [values[(2 * y + dy) * width + 2 * x + dx] for dy in (0, 1) for dx in (0, 1)]
                following.append(min(v) if reduction == "min" else max(v) if reduction == "max" else sum(v) / 4)
        result += following
        values, width, height = following, width // 2, height // 2
    return result


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan")
    folder, costs = Path(artifact).parent, []
    spec = json.loads((folder / "spec.json").read_text())
    width, height = spec["shape"]
    total = sum((width >> (i + 1)) * (height >> (i + 1)) for i in range(spec["levels"]))
    with runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        assert runtime.native_resource_capabilities("spd_plan")
        assert runtime.native_operation_capabilities("spd_run")
        frame = {
            name: runtime.ndarray("f32", shape)
            for name, shape in dict(
                pixels=(width, height), output=(total,), doutput=(total,), loss=(1,), dloss=(1,)
            ).items()
        }
        code = runtime.host_blob((folder / "shader.spv").read_bytes())
        images = [
            runtime.native_resource(
                "image", "sampled", parameters=dict(format="r32f", storage=True, shape=shape, mip_levels=levels)
            )
            for shape, levels in (([width, height], 1), ([width // 2, height // 2], spec["levels"]))
        ]
        inputs = dict(source=images[0], output=images[1], shader=code)
        params = dict(reduction=spec["reduction"], expected_plan=spec["expected_plan"])
        start = time.perf_counter()
        resource = runtime.native_resource("spd", "plan", parameters=params, bindings=inputs)
        resource_ms = (time.perf_counter() - start) * 1000
        raises(
            lambda: runtime.native_resource(
                "spd", "plan", parameters=dict(params, expected_plan="0" * 64), bindings=inputs
            ),
            "contract mismatch",
        )
        malformed = runtime.host_blob(b"bad")
        raises(
            lambda: runtime.native_resource("spd", "plan", parameters=params, bindings=dict(inputs, shader=malformed)),
            "complete SPIR-V",
        )
        malformed.close()
        code.close()  # all host code consumed synchronously by the factory
        frame.update(plan=resource, source=images[0], pyramid=images[1])
        retained = []
        for policy in ("private", "public"):
            plan = module.execution_plan(policy)
            bindings = {n: frame[n] for n in plan.argument_names}
            start = time.perf_counter()
            call = plan.prepare(module, bindings)
            prepare_ms = (time.perf_counter() - start) * 1000
            assert call.info["execution_kind"] == "vulkan_commands"
            if policy == "private":
                assert call.info["native_resource_bytes"] == (width * height + total) * 4
                assert call.info["workspace_bytes"] == 4
            for scale in (1, 0, -2):
                pixels = [scale * ((x * 17 + y * 31) % 251) / 256 for y in range(height) for x in range(width)]
                write(frame["pixels"], pixels)
                call()
                expected = pyramid(pixels, width, height, spec["levels"], spec["reduction"])
                for name in ("output", "doutput"):
                    assert all(abs(a - b) < 3e-6 for a, b in zip(read(frame[name]), expected)), name
                expected_loss = sum(x * x / 2 for x in expected)
                assert abs(read(frame["loss"])[0] - expected_loss) < max(3e-6, expected_loss * 3e-5)
            submits, completes = [], []
            for _ in range(4):
                start = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                submits.append((submitted - start) / 1000)
                completes.append((time.perf_counter_ns() - start) / 1000)
            costs.append(
                dict(
                    policy=policy,
                    prepare_ms=prepare_ms,
                    submit_us=statistics.median(submits),
                    complete_us=statistics.median(completes),
                    info=call.info,
                )
            )

            def reject(change, message):
                doc = json.loads(plan.to_json())
                change(doc)
                bad = ExecutionPlan.from_json(json.dumps(doc))
                raises(lambda: bad.prepare(module, {n: frame[n] for n in bad.argument_names}), message)

            reject(lambda d: d["operations"].pop(0), "unconditional upload")
            reject(lambda d: d["operations"][1]["bindings"].update(output="source"), "fixed source and output")
            reject(lambda d: d["operations"][1]["parameters"].update(expected_plan="0" * 64), "contract mismatch")
            retained.append(call)
        # Distinct private preparations own distinct counters and image resources.
        sibling = module.prepare_execution(
            "private", {n: frame[n] for n in module.execution_plan("private").argument_names}
        )
        sibling()
        retained[0]()
        resource.close()
        raises(retained[1])
        retained[1].close()
        module.close()
        raises(retained[0])
        retained[0].close()
        sibling.close()
        for image in images:
            image.close()
        for value in frame.values():
            value.close()
    if mode == "borrowed":
        ti.reset()
    else:
        assert "taichi_forge" not in sys.modules
    (folder / f"{mode}.json").write_text(
        json.dumps(dict(load_ms=load_ms, resource_ms=resource_ms, costs=costs), indent=2)
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
