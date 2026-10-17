"""Small mip-chain consumer with no producer imports in standalone mode."""

import json
import os
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_optix_probe import raises, read, write

SHAPES = [(3, 5), (1, 2), (1, 1)]
VALUES = [[i / 16 for i in range(15)], [0.25, 0.5], [0.75]]


def sampler_parameters():
    return dict(min_filter="nearest", mag_filter="nearest", mip_filter="linear")


def expected(flip=False):
    values = [7 / 16, 0.5, 0.75, 0, 0.25, 0.75, (7 / 16 + 0.5) / 2]
    return [1 - v for v in values] if flip else values


def native(operation, bindings, **parameters):
    return dict(
        kind="native", provider="image", operation=operation, version=1, parameters=parameters, bindings=bindings
    )


def check(frame, flip=False):
    values = expected(flip)
    for name in ("result", "dresult"):
        assert all(abs(a - b) < 1e-6 for a, b in zip(read(frame[name]), values)), (name, read(frame[name]), values)
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
        for recipe in ("ordinary", "selected"):
            for kind in ("private", "public", "depth"):
                frame = {
                    name: runtime.ndarray("f32", (1,) if name == "loss" else (7,))
                    for name in ("result", "dresult", "loss")
                }
                seeds = {f"seed{i}": runtime.ndarray("f32", shape) for i, shape in enumerate(SHAPES)}
                for source, values in zip(seeds.values(), VALUES):
                    write(source, values)
                arguments = dict(frame, **seeds)
                if kind == "private":
                    plan = module.execution_plan(recipe + "_private")
                    if recipe == "ordinary":
                        # A conditional upload cannot initialize a later unconditional sampler.
                        document = json.loads(plan.to_json())
                        document["version"] = max(12, document["version"])
                        document["initialization"] = [
                            dict(
                                kind="branch",
                                selector="selector",
                                selection="nonzero",
                                has_default=False,
                                lowering=dict(kind="vulkan_host", version=1),
                                condition=[dict(kind="fill32", destination="selector", value=0)],
                                bodies=[document["initialization"]],
                            )
                        ]
                        selector = runtime.ndarray("i32", (1,))
                        raises(
                            lambda: ExecutionPlan.from_json(json.dumps(document)).prepare(
                                module, dict(arguments, selector=selector)
                            ),
                            "unconditional upload",
                        )
                        selector.close()
                else:
                    fmt = "depth32f" if kind == "depth" else "r32f"
                    params = dict(shape=list(SHAPES[0]), mip_levels=3, format=fmt, sampler=sampler_parameters())
                    if mode == "borrowed":
                        texture = ti.Texture(
                            getattr(ti.Format, fmt),
                            SHAPES[0],
                            mip_levels=3,
                            sampler=ti.hardware.sampling.SamplerConfig(**sampler_parameters()),
                        )
                        image = runtime.import_texture(texture)
                    else:
                        image = runtime.native_resource("image", "sampled", parameters=params)
                    arguments["public_image"] = image
                    original = module.execution_plan(recipe + "_public")
                    upload = native(
                        "upload_mips",
                        dict(image="public_image", **{f"pixels_{i}": f"seed{i}" for i in range(3)}),
                        mip_levels=3,
                    )
                    document = json.loads(original.to_json())
                    document["initialization"] = [upload]
                    document["version"] = max(2, document["version"])
                    plan = ExecutionPlan.from_json(json.dumps(document))
                    if mode == "standalone":
                        raises(lambda: original.prepare(module, dict(frame, public_image=image)), "initial pixels")
                    # Legacy base-only copies cannot claim full-chain initialization.
                    raises(
                        lambda: ExecutionPlan([native("upload", dict(image="image", pixels="pixels"))]).prepare(
                            module, dict(image=image, pixels=seeds["seed0"])
                        ),
                        "upload_mips",
                    )
                    incomplete = native("upload_mips", dict(image="image", pixels_0="pixels"), mip_levels=1)
                    raises(
                        lambda: ExecutionPlan([incomplete]).prepare(module, dict(image=image, pixels=seeds["seed0"])),
                        "every allocated level",
                    )
                    bad_seeds = dict(arguments, seed1=runtime.ndarray("f32", (2, 1)))
                    raises(lambda: plan.prepare(module, bad_seeds))
                    bad_seeds["seed1"].close()
                start = time.perf_counter()
                call = plan.prepare(module, arguments)
                call.initialize()
                assert call.info["native_resource_bytes"] == (72 if kind == "private" else 0)
                costs[recipe + "_" + kind] = dict(ready_ms=(time.perf_counter() - start) * 1000)
                costs[recipe + "_" + kind]["resource_bytes"] = call.info["resource_bytes"]
                call()
                check(frame)
                timings = []
                for _ in range(6):
                    start = time.perf_counter_ns()
                    call()
                    runtime.wait()
                    timings.append((time.perf_counter_ns() - start) / 1000)
                costs[recipe + "_" + kind]["complete_us"] = statistics.median(timings)
                if kind == "private":
                    # Initialization-only seed owners are released after completion.
                    for source in seeds.values():
                        source.close()
                    call()
                    check(frame)
                else:
                    update = ExecutionPlan([upload]).prepare(module, dict(public_image=image, **seeds))
                    for flip in (True, False):
                        for source, values in zip(seeds.values(), VALUES):
                            write(source, [1 - v for v in values] if flip else values)
                        update()
                        call()
                        check(frame, flip)
                    image.close()
                    raises(call)
                    raises(update)
                call.close()
        for changed, message in (
            (dict(mip_levels=4), "complete chain"),
            (dict(mip_levels=0), "positive uint32"),
            (dict(shape=[4]), "2D Vulkan"),
        ):
            raises(lambda: runtime.native_resource("image", "sampled", parameters=dict(params, **changed)), message)
        raises(
            lambda: runtime.native_resource(
                "image", "sampled", parameters=params, bindings=dict(pixels=runtime.ndarray("f32", SHAPES[0]))
            ),
            "upload_mips",
        )
        if mode == "borrowed":
            ti.reset()
        else:
            assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(costs, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
