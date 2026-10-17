"""Rebuild raster resources without a producer or a Python graphics frontend."""

import json
from contextlib import closing
import os
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_optix_probe import raises, read, write
from tests.python.aot_vulkan_ray_program_probe import native


def pipeline_spec(instanced=False):
    return dict(
        shader_count=2,
        shaders=[dict(stage="vertex"), dict(stage="fragment")],
        vertices=(
            [dict(binding=0, stride=8), dict(binding=1, stride=12, instance=True)]
            if instanced
            else [dict(binding=0, stride=28)]
        ),
        attributes=(
            [
                dict(location=0, binding=0, format="rg32f", offset=0),
                dict(location=1, binding=1, format="rgb32f", offset=0),
            ]
            if instanced
            else [
                dict(location=0, binding=0, format="rg32f", offset=0),
                dict(location=1, binding=0, format="rgb32f", offset=8),
                dict(location=2, binding=0, format="rg32f", offset=20),
            ]
        ),
        color_targets=[dict(blending=False)],
    )


def pass_spec(instanced=False):
    draw = dict(pipeline=0, element_count=3, vertices=[dict(binding=0, array=0)], buffers=[], images=[])
    bindings = dict(pipeline_0="instance_pipeline" if instanced else "pipeline", color_0="target", depth="depth")
    if instanced:
        draw.update(index=2, index_min=0, index_max=2, first_instance=1)
        draw["vertices"].append(dict(binding=1, array=1))
        bindings.update(array_0="positions", array_1="colors", array_2="indices")
    else:
        draw.update(buffers=[dict(set=0, binding=3, array=1)], images=[dict(set=0, binding=5, image=0)])
        bindings.update(array_0="vertices", array_1="uniform", image_0="source")
    return native(
        "vulkan_graphics",
        "pass_direct",
        bindings,
        pipeline_count=1,
        array_count=3 if instanced else 2,
        image_count=0 if instanced else 1,
        color_count=1,
        colors=[dict(clear=not instanced, clear_bits=[0, 0, 1065353216, 1065353216])],
        depth_clear=not instanced,
        viewport=[4 if instanced else 0, 0, 4, 8],
        draws=[draw],
    )


def image_resources():
    return [
        dict(native("image", "sampled", {}, shape=[8, 8], format="rgba32f", attachment=True), name="target"),
        dict(native("image", "sampled", {}, shape=[8, 8], format="depth32f", attachment=True), name="depth"),
        dict(native("image", "sampled", dict(pixels="pixels"), shape=[1, 1], format="rgba32f"), name="source"),
    ]


def operations():
    return [
        pass_spec(),
        pass_spec(True),
        dict(kind="kernel", entry="sample", arguments=["target", "features"]),
        dict(kind="fill32", destination="dfeatures", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["features", "loss"]),
        dict(kind="kernel", entry="backward", arguments=["features", "loss"]),
    ]


def check(frame, color):
    expected = [v for i in range(64) for v in ([0.5, 0, 0, 1] if i % 8 < 4 else [*color, 1])]
    assert read(frame["features"]) == expected, read(frame["features"])
    assert read(frame["dfeatures"]) == expected
    assert abs(read(frame["loss"])[0] - sum(v * v * 0.5 for v in expected)) < 1e-4


def main(mode, artifact, output):
    start = time.perf_counter()
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    costs = dict(runtime_ms=(time.perf_counter() - start) * 1000, cases={})
    with runtime:
        assert "vulkan_inline" in runtime.native_operation_capabilities("vulkan_graphics_pass_direct")
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        costs["load_ms"] = (time.perf_counter() - start) * 1000
        frame = {
            name: runtime.ndarray("f32", size)
            for name, size in [("features", 256), ("dfeatures", 256), ("loss", 1), ("pixels", (1, 1, 4))]
        }
        frame.update(
            vertices=runtime.ndarray("f32", 21, usage=("storage", "vertex")),
            positions=runtime.ndarray("f32", 6, usage=("vertex",)),
            colors=runtime.ndarray("f32", 6, usage=("vertex",)),
            indices=runtime.ndarray("u32", 3, usage=("index",)),
            uniform=runtime.ndarray("f32", 4, usage=("uniform",)),
        )
        write(frame["vertices"], [v for xy in [(-1, -1), (3, -1), (-1, 3)] for v in [*xy, 1, 0, 0, 0.5, 0.5]])
        write(frame["positions"], [-1, -1, 3, -1, -1, 3])
        write(frame["indices"], [0, 1, 2], "I")
        write(frame["uniform"], [1, 0, 0, 0])
        write(frame["pixels"], [0.5] * 4)
        write(frame["colors"], [1, 0, 0, 0, 1, 0])
        owners = {}
        for instanced in (False, True):
            blobs = [
                runtime.host_blob((Path(artifact).parent / f"shader_{int(instanced)}_{i}.spv").read_bytes())
                for i in range(2)
            ]
            spec = pipeline_spec(instanced)
            bindings = {f"shader_{i}": blob for i, blob in enumerate(blobs)}
            if instanced:
                with runtime.host_blob((Path(artifact).parent / "push_constant.spv").read_bytes()) as push:
                    raises(
                        lambda: runtime.native_resource(
                            "vulkan_graphics", "pipeline", parameters=spec, bindings=dict(bindings, shader_0=push)
                        ),
                        "push constants",
                    )
            raises(
                lambda: runtime.native_resource(
                    "vulkan_graphics", "pipeline", parameters=dict(spec, expected_pipeline="0" * 64), bindings=bindings
                ),
                "contract mismatch",
            )
            owners["instance_pipeline" if instanced else "pipeline"] = runtime.native_resource(
                "vulkan_graphics", "pipeline", parameters=spec, bindings=bindings
            )
            for blob in blobs:
                blob.close()
        for spec in image_resources():
            owners[spec["name"]] = runtime.native_resource(
                spec["provider"],
                spec["operation"],
                parameters=spec["parameters"],
                bindings={k: frame[v] for k, v in spec["bindings"].items()},
            )
        available = dict(frame, **owners)

        def prepare(ops, extra=None):
            plan = ExecutionPlan(ops)
            supplied = dict(available, **(extra or {}))
            return plan.prepare(module, {k: supplied[k] for k in plan.argument_names})

        # No seed exists for either attachment. LOAD and conditional-only clear
        # cannot initialize the sequence; a failed preparation leaves owners live.
        raises(lambda: prepare([pass_spec(True)]), "unconditional upload")
        selector = runtime.ndarray("i32", 1)
        branch = dict(
            kind="branch",
            selector="selector",
            selection="nonzero",
            has_default=False,
            lowering=dict(kind="vulkan_host", version=1),
            condition=[dict(kind="fill32", destination="selector", value=0)],
            bodies=[[pass_spec()]],
        )
        raises(lambda: prepare([branch, pass_spec(True)], dict(selector=selector)), "unconditional upload")
        selector.close()
        with closing(runtime.ndarray("f32", 21)) as wrong_usage:
            raises(lambda: prepare([pass_spec()], dict(vertices=wrong_usage)), "usage")
        with closing(runtime.ndarray("f32", 14, usage=("vertex",))) as short:
            raises(lambda: prepare([pass_spec()], dict(vertices=short)), "too small")
        bad = pass_spec()
        bad["parameters"]["draws"][0]["buffers"] = []
        raises(lambda: prepare([bad]), "descriptor")
        bad = pass_spec()
        bad["parameters"]["draws"][0]["expected_pipeline"] = "0" * 64
        raises(lambda: prepare([bad]), "contract mismatch")
        bad = pass_spec(True)
        del bad["parameters"]["draws"][0]["index_max"]
        raises(lambda: prepare([bad]), "explicit index_min")
        raises(lambda: prepare([pass_spec()], dict(source=owners["target"])), "attachment alias")
        with runtime.native_resource(
            "image", "sampled", parameters=dict(shape=[8, 8], format="rgba32f")
        ) as wrong_usage:
            raises(lambda: prepare([pass_spec()], dict(target=wrong_usage)), "attachment usage")

        # Clear covers the entire attachment even though the draw has half-width viewport.
        with prepare([pass_spec(), dict(kind="kernel", entry="sample", arguments=["target", "features"])]) as half:
            half()
            assert read(frame["features"]) == [
                v for i in range(64) for v in ([0.5, 0, 0, 1] if i % 8 < 4 else [0, 0, 1, 1])
            ]
        calls = {}
        for name in ("public", "private", "private_arrays"):
            supplied = {k: available[k] for k in module.execution_plan(name).argument_names}
            start = time.perf_counter()
            call = module.prepare_execution(name, supplied)
            assert call.info["native_resource_bytes"] == (0 if name == "public" else 1296)
            assert call.info["resource_bytes"] == (148 if name == "private_arrays" else 4)
            costs["cases"][name] = dict(prepare_ms=(time.perf_counter() - start) * 1000, info=call.info)
            calls[name] = call
            for color in ([0, 1, 0], [0, 0, 0.25], [0, 1, 0]):
                write(frame["colors"], [1, 0, 0, *color])
                call()
                check(frame, color)
            times, hosts = [], []
            for _ in range(8):
                start = time.perf_counter_ns()
                call()
                hosts.append((time.perf_counter_ns() - start) / 1000)
                runtime.wait()
                times.append((time.perf_counter_ns() - start) / 1000)
            costs["cases"][name].update(host_us=statistics.median(hosts), complete_us=statistics.median(times))
            if name == "public":
                with runtime.load_module(artifact) as sibling_module:
                    with sibling_module.prepare_execution(name, supplied) as sibling:
                        sibling()
                        check(frame, [0, 1, 0])
                call()
                check(frame, [0, 1, 0])
                with closing(runtime.ndarray("f32", 6, usage=("vertex",))) as rebound:
                    write(rebound, [1, 0, 0, 0, 0, 0.5])
                    with module.prepare_execution(name, dict(supplied, colors=rebound)) as alternate:
                        alternate()
                        check(frame, [0, 0, 0.5])
                        call()
                        check(frame, [0, 1, 0])
        # Retiring an externally shared pipeline invalidates its dependent call,
        # while a private reconstructed pipeline remains independently usable.
        owners["pipeline"].close()
        raises(calls["public"])
        calls["private"]()
        check(frame, [0, 1, 0])
        # Private drawing arrays, including the offset vertex view, are rebuilt
        # from archived data and survive closing unrelated public storage.
        frame["vertices"].close()
        frame["positions"].close()
        calls["private_arrays"]()
        check(frame, [0, 1, 0])
        if mode == "borrowed":
            ti.reset()
            raises(calls["private_arrays"])
        else:
            module.close()
            assert calls["private_arrays"].info["native_resource_bytes"] == 0
            assert calls["private_arrays"].info["resource_bytes"] == 0
        for call in calls.values():
            call.close()
        for resource in owners.values():
            resource.close()
        if mode == "standalone":
            assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
