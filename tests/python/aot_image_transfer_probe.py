"""Source-independent region transfers with finite views and downstream AD."""

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


def region(width, height, *, x=0, y=0, mip=0):
    return dict(offset=[x, y, 0], extent=[width, height, 1], mip_level=mip)


def plan_parts(private=True):
    resources = [
        dict(kind="ndarray_view", name="input_view", array="input", offset=2, shape=[9]),
        dict(kind="ndarray_view", name="output_view", array="padded", offset=2, shape=[24]),
    ]
    if private:
        for name, shape, bindings in (("a", [3, 2], {}), ("b", [4, 3], dict(pixels="seed")), ("c", [4, 4], {})):
            resources.append(dict(native("image", "sampled", bindings, shape=shape, format="r32f"), name=name))
    return dict(
        resources=resources,
        gradients=dict(output="doutput", loss="dloss"),
        operations=[
            native(
                "image",
                "buffer_to_image",
                dict(image="a", buffer="input_view"),
                region=region(3, 2),
                byte_offset=4,
                row_length=5,
                image_height=4,
            ),
            native(
                "image",
                "copy_region",
                dict(source="a", destination="b"),
                source_region=region(2, 2, x=1),
                destination_region=region(2, 2, x=1, y=1),
            ),
            native(
                "image",
                "blit",
                dict(source="a", destination="c"),
                source_region=region(2, 2, x=1),
                destination_region=region(4, 4),
                filter="nearest",
            ),
            native(
                "image",
                "image_to_buffer",
                dict(image="c", buffer="output_view"),
                region=region(4, 4),
                byte_offset=4,
                row_length=6,
            ),
            native("image", "image_to_buffer", dict(image="b", buffer="copied"), region=region(4, 3)),
            dict(kind="kernel", entry="sample", arguments=["c", "output"]),
            dict(kind="fill32", destination="doutput", value=0),
            dict(kind="fill32", destination="loss", value=0),
            dict(kind="fill32", destination="dloss", value=1065353216),
            dict(kind="kernel", entry="energy", arguments=["output", "loss"]),
            dict(kind="kernel", entry="backward", arguments=["output", "loss"]),
        ],
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
            name: runtime.ndarray("f32", (4, 3) if name == "seed" else (n,))
            for name, n in dict(input=32, seed=12, padded=32, copied=12, output=16, doutput=16, loss=1, dloss=1).items()
        }
        write(frame["seed"], [-7] * 12)
        write(frame["padded"], [-99] * 32)
        for kind in ("image_copy_region", "image_blit", "buffer_to_image", "image_to_buffer"):
            assert runtime.native_operation_capabilities(kind)
        retained = []
        for policy in ("private", "public"):
            bindings = dict(frame)
            images = []
            if policy == "public":
                for name, shape in (("a", [3, 2]), ("b", [4, 3]), ("c", [4, 4])):
                    image = runtime.native_resource(
                        "image",
                        "sampled",
                        parameters=dict(shape=shape, format="r32f"),
                        bindings=dict(pixels=frame["seed"]) if name == "b" else {},
                    )
                    images.append(image)
                    bindings[name] = image
            plan = module.execution_plan(policy)
            bindings = {n: bindings[n] for n in plan.argument_names}
            start = time.perf_counter()
            call = plan.prepare(module, bindings)
            prepare_ms = (time.perf_counter() - start) * 1000
            assert call.info["execution_kind"] == "vulkan_commands"
            if policy == "private":
                assert call.info["native_resource_bytes"] == 136

            def replay(scale):
                source = [-88] * 32
                source[3:6], source[8:11] = [scale, 2 * scale, 3 * scale], [4 * scale, 5 * scale, 6 * scale]
                write(frame["input"], source)
                call()
                expected = [v * scale for v in [2, 2, 3, 3] * 2 + [5, 5, 6, 6] * 2]
                assert read(frame["output"]) == expected
                assert read(frame["doutput"]) == expected
                assert read(frame["loss"]) == [sum(x * x / 2 for x in expected)]
                padded = [-99] * 32
                for y in range(4):
                    padded[3 + y * 6 : 7 + y * 6] = expected[y * 4 : y * 4 + 4]
                assert read(frame["padded"]) == padded
                assert read(frame["copied"]) == [-7] * 5 + [2 * scale, 3 * scale, -7, -7, 5 * scale, 6 * scale, -7]

            for scale in (1, 0, -2):
                replay(scale)
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
                raises(lambda: bad.prepare(module, {n: bindings[n] for n in bad.argument_names}), message)

            reject(lambda d: d["resources"][0].update(shape=[8]), "finite buffer view")
            reject(lambda d: d["operations"][0]["parameters"].update(byte_offset=2), "aligned")
            reject(
                lambda d: (
                    d["operations"][1]["bindings"].update(destination="a"),
                    d["operations"][1]["parameters"].update(destination_region=region(2, 2, x=1)),
                ),
                "alias",
            )
            reject(lambda d: d["operations"][0]["parameters"]["region"].update(layer_count=2), "array layers")
            reject(lambda d: d["operations"].pop(0), "unconditional upload")
            if policy == "private":
                reject(lambda d: d["resources"][3].update(bindings={}), "unconditional upload")
                retained.append(call)
            else:
                sibling = plan.prepare(module, bindings)
                sibling()
                images[0].close()
                raises(call)
                raises(sibling)
                call.close()
                sibling.close()
                for image in images[1:]:
                    image.close()

        # A single mip write cannot initialize the entire chain.
        mip = runtime.native_resource("image", "sampled", parameters=dict(shape=[4, 4], format="r32f", mip_levels=3))
        seeds = [runtime.ndarray("f32", (n, n)) for n in (4, 2, 1)]
        for seed in seeds:
            write(seed, [9] * (seed.nbytes // 4))
        out = runtime.ndarray("f32", (4,))
        ops = [
            native(
                "image", "upload_mips", dict(image="mip", pixels_0="s0", pixels_1="s1", pixels_2="s2"), mip_levels=3
            ),
            native("image", "buffer_to_image", dict(image="mip", buffer="s1"), region=region(2, 2, mip=1)),
            native("image", "image_to_buffer", dict(image="mip", buffer="out"), region=region(2, 2, mip=1)),
        ]
        args = dict(mip=mip, s0=seeds[0], s1=seeds[1], s2=seeds[2], out=out)
        uninitialized = ExecutionPlan(ops[1:])
        raises(
            lambda: uninitialized.prepare(module, {n: args[n] for n in uninitialized.argument_names}),
            "unconditional upload",
        )
        mip_call = ExecutionPlan(ops).prepare(module, args)
        mip_call()
        assert read(out) == [9] * 4
        mip_call.close()
        mip.close()
        out.close()
        for seed in seeds:
            seed.close()

        # Both depth formats use graphics queues and four-byte buffer offsets.
        for fmt, dtype, code, values in (
            ("depth32f", "f32", "f", [123, 0, 1, 0.25, 0.5]),
            ("depth16", "u16", "H", [123, 456, 0, 65535, 16384, 32768]),
        ):
            depth_image = runtime.native_resource("image", "sampled", parameters=dict(shape=[2, 2], format=fmt))
            depth_input, depth_output = (runtime.ndarray(dtype, (len(values),)) for _ in range(2))
            write(depth_input, values, code)
            write(depth_output, [99] * len(values), code)
            depth_ops = [
                native(
                    "image", "buffer_to_image", dict(image="image", buffer="input"), region=region(2, 2), byte_offset=4
                ),
                native(
                    "image", "image_to_buffer", dict(image="image", buffer="output"), region=region(2, 2), byte_offset=4
                ),
            ]
            depth_args = dict(image=depth_image, input=depth_input, output=depth_output)
            bad_depth = json.loads(ExecutionPlan(depth_ops).to_json())
            bad_depth["operations"][0]["parameters"]["byte_offset"] = 2
            raises(lambda: ExecutionPlan.from_json(json.dumps(bad_depth)).prepare(module, depth_args), "aligned")
            depth_call = ExecutionPlan(depth_ops).prepare(module, depth_args)
            depth_call()
            assert read(depth_output, code) == [99] * (len(values) - 4) + values[-4:]
            if fmt == "depth16":
                seed = runtime.ndarray(dtype, (2, 2))
                write(seed, values[-4:], code)
                seeded = runtime.native_resource(
                    "image", "sampled", parameters=dict(shape=[2, 2], format=fmt), bindings=dict(pixels=seed)
                )
                seeded_call = ExecutionPlan(depth_ops[1:]).prepare(module, dict(image=seeded, output=depth_output))
                seed.close()
                seeded_call()
                assert read(depth_output, code)[-4:] == values[-4:]
                seeded_call.close()
                seeded.close()
                # Construction-time pixels must obey the same absolute aspect
                # alignment, including a subview hidden behind a resource port.
                misaligned_seed = ExecutionPlan(
                    depth_ops[1:],
                    resources=[
                        dict(kind="ndarray_view", name="seed_view", array="input", offset=1, shape=[2, 2]),
                        dict(
                            native("image", "sampled", dict(pixels="seed_view"), shape=[2, 2], format=fmt), name="image"
                        ),
                    ],
                )
                raises(lambda: misaligned_seed.prepare(module, dict(input=depth_input, output=depth_output)), "aligned")
            for value in (depth_call, depth_image, depth_input, depth_output):
                value.close()

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
