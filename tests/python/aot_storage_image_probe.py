"""Source-independent integer MRT and storage-image kernel consumer."""

import json
import os
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


def pipeline_parameters():
    return dict(
        shader_count=2,
        shaders=[dict(stage="vertex"), dict(stage="fragment")],
        vertices=[dict(binding=0, stride=20)],
        attributes=[
            dict(location=0, binding=0, format="rg32f", offset=0),
            dict(location=1, binding=0, format="rgb32f", offset=8),
        ],
        color_targets=[dict(blending=False)] * 3,
    )


def image_resources():
    return [
        dict(native("image", "sampled", {}, shape=[8, 4], format="rgba32f", attachment=True), name="color"),
        dict(native("image", "storage", {}, shape=[8, 4], format="r32u", attachment=True), name="unsigned"),
        dict(native("image", "storage", {}, shape=[8, 4], format="r32i", attachment=True), name="signed"),
    ]


def operations():
    return [
        native(
            "vulkan_graphics",
            "pass_direct",
            dict(pipeline_0="pipeline", array_0="vertices", color_0="color", color_1="unsigned", color_2="signed"),
            pipeline_count=1,
            array_count=1,
            image_count=0,
            color_count=3,
            colors=[
                dict(clear=True, clear_bits=[1048576000, 1056964608, 1061158912, 1065353216]),
                dict(clear=True, clear_bits=[0xFEDCBA98, 0, 0, 0]),
                dict(clear=True, clear_bits=[(-987654321) & 0xFFFFFFFF, 0, 0, 0]),
            ],
            viewport=[0, 0, 4, 4],
            draws=[dict(pipeline=0, element_count=3, vertices=[dict(binding=0, array=0)], buffers=[], images=[])],
        ),
        dict(kind="kernel", entry="update", arguments=["unsigned", "signed", "delta"]),
        dict(
            kind="kernel", entry="consume", arguments=["color", "unsigned", "signed", "ids", "primitives", "features"]
        ),
        dict(kind="fill32", destination="dfeatures", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["features", "loss"]),
        dict(kind="kernel", entry="backward", arguments=["features", "loss"]),
    ]


def mip_parts():
    return dict(
        resources=[dict(native("image", "storage", {}, shape=[3, 5], format="r32f", mip_levels=3), name="image")],
        operations=[
            native(
                "image", "upload_mips", dict(image="image", pixels_0="m0", pixels_1="m1", pixels_2="m2"), mip_levels=3
            ),
            dict(kind="kernel", entry="mip_update", arguments=["image", "mip_output"]),
            dict(kind="kernel", entry="mip_read", arguments=["image", "mip_output"]),
        ],
    )


def main(mode, artifact, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    costs = {}
    with runtime:
        assert "vulkan_inline" in runtime.native_operation_capabilities("image_mip_upload")
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        costs["load_ms"] = (time.perf_counter() - start) * 1000
        frame = {
            name: runtime.ndarray(dtype, shape)
            for name, dtype, shape in [
                ("delta", "i32", 1),
                ("ids", "u32", 32),
                ("primitives", "i32", 32),
                ("features", "f32", 128),
                ("dfeatures", "f32", 128),
                ("loss", "f32", 1),
            ]
        }
        frame["vertices"] = runtime.ndarray("f32", 15, usage=("vertex",))
        write(frame["vertices"], [v for xy in [(-1, -1), (3, -1), (-1, 3)] for v in [*xy, 1, 0, 0]])
        blobs = [runtime.host_blob((Path(artifact).parent / f"shader_{i}.spv").read_bytes()) for i in range(2)]
        owners = dict(
            pipeline=runtime.native_resource(
                "vulkan_graphics",
                "pipeline",
                parameters=pipeline_parameters(),
                bindings={f"shader_{i}": value for i, value in enumerate(blobs)},
            )
        )
        for value in blobs:
            value.close()
        for spec in image_resources():
            owners[spec["name"]] = runtime.native_resource("image", spec["operation"], parameters=spec["parameters"])
        available = dict(frame, **owners)

        def prepare(ops, resources=(), supplied=None):
            plan = ExecutionPlan(ops, resources=resources)
            data = available if supplied is None else supplied
            return plan.prepare(module, {key: data[key] for key in plan.argument_names})

        # Arbitrary stores do not prove that an entire image has been written.
        raises(lambda: prepare(operations()[1:2]), "unconditional upload")
        wrong = dict(native("image", "sampled", {}, shape=[8, 4], format="r32f"), name="unsigned")
        raises(lambda: prepare(operations()[1:2], [wrong]), "storage image with matching format")
        wrong = dict(native("image", "storage", {}, shape=[8, 4], format="r32i"), name="unsigned")
        raises(lambda: prepare(operations()[1:2], [wrong]), "matching format")
        wrong = dict(native("image", "storage", {}, shape=[32], format="r32u"), name="unsigned")
        raises(lambda: prepare(operations()[1:2], [wrong]), "dimension contract")
        for params, error in [
            (dict(shape=[3, 5], format="depth32f"), "color format"),
            (dict(shape=[3, 5], format="r32u", sampler={}), "field"),
        ]:
            raises(lambda: runtime.native_resource("image", "storage", parameters=params), error)

        calls = []
        for name in ("public", "private"):
            plan = module.execution_plan(name)
            start = time.perf_counter()
            call = plan.prepare(module, {key: available[key] for key in plan.argument_names})
            prepare_ms = (time.perf_counter() - start) * 1000
            assert call.info["native_resource_bytes"] == (768 if name == "private" else 0)
            host, complete = [], []
            for delta in (0, 7, -3):
                write(frame["delta"], [delta], "i")
                start = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                host.append((submitted - start) / 1000)
                complete.append((time.perf_counter_ns() - start) / 1000)
                assert read(frame["ids"], "I") == [(0xF1234567 if i % 8 < 4 else 0xFEDCBA98) + delta for i in range(32)]
                assert read(frame["primitives"], "i") == [
                    (-123456789 if i % 8 < 4 else -987654321) - delta for i in range(32)
                ]
                expected = [v for i in range(32) for v in ([4, 0, 0, 0.5] if i % 8 < 4 else [0.25, 0.5, 0.75, 1])]
                assert read(frame["features"]) == expected
                assert read(frame["dfeatures"]) == expected
                assert abs(read(frame["loss"])[0] - sum(v * v * 0.5 for v in expected)) < 1e-4
            costs[name] = dict(
                prepare_ms=prepare_ms,
                host_us=statistics.median(host),
                complete_us=statistics.median(complete),
                info=call.info,
            )
            calls.append(call)

        mip = {
            name: runtime.ndarray("f32", shape)
            for name, shape in [("m0", (3, 5)), ("m1", (1, 2)), ("m2", (1, 1)), ("mip_output", 5)]
        }
        for name, values in [("m0", range(15)), ("m1", [7, 9]), ("m2", [21])]:
            write(mip[name], values)
        plan = module.execution_plan("mip")
        call = plan.prepare(module, mip)
        assert call.info["native_resource_bytes"] == 72
        for _ in range(2):
            call()
            runtime.wait()
            assert read(mip["mip_output"]) == [1, 2, 10, 12, 21]
        costs["mip_info"] = call.info
        calls.append(call)
        legacy = mip_parts()
        legacy["resources"][0]["operation"] = "sampled"
        legacy["resources"][0]["parameters"]["storage"] = True
        with ExecutionPlan(**legacy).prepare(module, mip) as legacy_call:
            legacy_call()
            runtime.wait()
            assert read(mip["mip_output"])[:4] == [1, 2, 10, 12]
        seed = runtime.ndarray("u32", (2, 2))
        write(seed, [0xF1234567, 2, 3, 4], "I")
        seeded = runtime.native_resource(
            "image", "storage", parameters=dict(shape=[2, 2], format="r32u"), bindings=dict(pixels=seed)
        )
        seed.close()  # The cold upload has completed; the image owns its pixels.
        with prepare(
            [dict(kind="kernel", entry="seeded_update", arguments=["image", "ids"])],
            supplied=dict(image=seeded, ids=frame["ids"]),
        ) as seeded_call:
            for iteration in range(2):
                seeded_call()
                runtime.wait()
                assert read(frame["ids"], "I")[0] == 0xF1234568 + iteration
        seeded.close()
        invalid = mip_parts()
        invalid["resources"][0]["parameters"]["mip_levels"] = 1
        raises(
            lambda: ExecutionPlan([invalid["operations"][1]], resources=invalid["resources"]).prepare(
                module, {"mip_output": mip["mip_output"]}
            ),
            "mip level",
        )
        # A storage-only image cannot silently become a floating sampled image.
        wrong = dict(native("image", "storage", {}, shape=[8, 4], format="rgba32f"), name="color")
        raises(lambda: prepare(operations()[2:3], [wrong]), "floating sampled image")
        call()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for call in calls:
            raises(call)
            if mode == "standalone":
                assert call.info["native_resource_bytes"] == 0
            call.close()
        for value in [*owners.values(), *frame.values(), *mip.values()]:
            value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
