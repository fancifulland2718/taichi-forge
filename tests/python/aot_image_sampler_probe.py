"""Minimal runtime-only sampler consumer; JIT is imported only in borrowed mode."""

import json
import os
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_optix_probe import raises, read, write

PIXELS = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]


def sampler_parameters(kind):
    return dict(
        min_filter="nearest" if kind == "compare" else "linear",
        mag_filter="nearest" if kind == "compare" else "linear",
        address_mode_u="clamp_to_edge",
        address_mode_v="mirrored_repeat",
        mip_filter="nearest" if kind == "compare" else "linear",
        lod_bias=0.25,
        min_lod=0.0,
        max_lod=0.0,
        max_anisotropy=1.0,
        compare_op="less" if kind == "compare" else None,
    )


def expected(kind, flip=False):
    values = [0, 0, 1, 1] if kind == "compare" else [0.375, 1, 0.75, 1]
    if flip:
        values = [1, 1, 1, 0] if kind == "compare" else [1 - v for v in values]
    return values


def check(frame, values):
    assert all(abs(a - b) < 1e-6 for a, b in zip(read(frame["result"]), values))
    assert all(abs(a - b) < 1e-6 for a, b in zip(read(frame["dresult"]), values))
    assert abs(read(frame["loss"])[0] - sum(v * v * 0.5 for v in values)) < 1e-6


def main(mode, artifact, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    costs = {}
    with runtime:
        module = runtime.load_module(artifact)
        for kind in ("color", "compare"):
            fmt = "depth32f" if kind == "compare" else "r32f"
            params = dict(shape=[4, 2], format=fmt, sampler=sampler_parameters(kind))
            for recipe in ("ordinary", "selected"):
                frame = {
                    name: runtime.ndarray("f32", (1,) if name == "loss" else (4,))
                    for name in ("result", "dresult", "loss")
                }
                pixels = runtime.ndarray("f32", (4, 2))
                write(pixels, PIXELS)
                for private in (True, False):
                    name = kind + "_" + recipe + ("_private" if private else "_public")
                    if private:
                        arguments = dict(frame, pixels=pixels)
                    elif mode == "borrowed":
                        jit_pixels = ti.ndarray(ti.f32, (4, 2))
                        alias = runtime.import_ndarray(jit_pixels)
                        write(alias, PIXELS)
                        alias.close()
                        texture = ti.Texture(
                            getattr(ti.Format, fmt),
                            (4, 2),
                            sampler=ti.hardware.sampling.SamplerConfig(**sampler_parameters(kind)),
                        )
                        texture.from_ndarray(jit_pixels)
                        image = runtime.import_texture(texture)
                        arguments = dict(frame, public_image=image)
                    else:
                        image = runtime.native_resource(
                            "image", "sampled", parameters=params, bindings=dict(pixels=pixels)
                        )
                        arguments = dict(frame, public_image=image)
                    start = time.perf_counter()
                    call = module.prepare_execution(name, arguments)
                    call.initialize()
                    costs[name] = dict(ready_ms=(time.perf_counter() - start) * 1000)
                    call()
                    check(frame, expected(kind))
                    timings = []
                    for _ in range(8):
                        start = time.perf_counter_ns()
                        call()
                        runtime.wait()
                        timings.append((time.perf_counter_ns() - start) / 1000)
                    costs[name]["complete_us"] = statistics.median(timings)
                    if not private:
                        upload = ExecutionPlan(
                            [
                                dict(
                                    kind="native",
                                    provider="image",
                                    operation="upload",
                                    version=1,
                                    parameters={},
                                    bindings=dict(image="image", pixels="pixels"),
                                )
                            ]
                        ).prepare(module, dict(image=image, pixels=pixels))
                        write(pixels, [1 - v for v in PIXELS])
                        upload()
                        call()
                        check(frame, expected(kind, True))
                        image.close()
                        raises(call)
                        raises(upload)
                    call.close()
            for changed, message in (
                (dict(filter="linear"), "mixed"),
                (dict(sampler=dict(max_lod=-2)), "max_lod"),
                (dict(sampler=dict(compare_op="unknown")), "compare_op"),
                (dict(sampler=dict(min_lod=-1)), "LOD clamps"),
                (dict(sampler=dict(max_anisotropy=0.5)), "max_anisotropy"),
                (dict(format="r32f", sampler=dict(compare_op="less")), "Comparison sampling requires"),
            ):
                raises(lambda: runtime.native_resource("image", "sampled", parameters=dict(params, **changed)), message)
            if kind == "color":
                write(pixels, [0.5] * 8)
                try:
                    image = runtime.native_resource(
                        "image",
                        "sampled",
                        bindings=dict(pixels=pixels),
                        parameters=dict(
                            params, sampler=dict(params["sampler"], min_filter="nearest", max_anisotropy=2)
                        ),
                    )
                except RuntimeError as error:
                    if "anisotropy is not enabled" not in str(error) and "within [1, 1]" not in str(error):
                        raise
                    costs["anisotropy"] = dict(supported=False)
                else:
                    call = module.prepare_execution("color_selected_public", dict(frame, public_image=image))
                    call()
                    check(frame, [0.5] * 4)
                    costs["anisotropy"] = dict(supported=True)
                    call.close()
                    image.close()
        if mode == "borrowed":
            ti.reset()
        else:
            assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
