"""Independent scalar reference for loaded CUDA warp/Vulkan subgroup kernels."""

import array
import os
import sys

from aot_consumer_probe import NoFrontend, raises


def read(value, code="f"):
    result = array.array(code)
    result.frombytes(value.read())
    return result.tolist()


def check_exchange(output, delta, scale):
    values = read(output, "i")
    checked = 0
    for i in range(137):
        lane, width, *actual = values[i * 6 : i * 6 + 6]
        assert 0 <= lane < width, (i, lane, width)
        for value, source in zip(
            actual, (lane ^ delta, lane - delta, lane + delta, lane ^ delta)
        ):
            origin = i - lane + source
            # CUDA out-of-range returns and Vulkan undefined sources differ.
            # Compare only the defined, active source lanes on both backends.
            if 0 <= source < width and 0 <= origin < 137:
                assert value == (origin * 3 - 17) * scale, (i, source, value)
                checked += 1
    assert checked > 400


def main(arch, mode, artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime import aot

        kwargs = (
            dict(device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0)))
            if arch == "vulkan"
            else {}
        )
        runtime = aot.Runtime(arch, **kwargs)
    else:
        import taichi_forge as ti
        from taichi_forge_runtime import aot

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    with runtime:
        modules = [runtime.load_module(artifact) for _ in range(2)]
        frames = [
            dict(
                out=runtime.ndarray("i32", (137, 6)),
                values=runtime.ndarray("f32", 128),
                gradient=runtime.ndarray("f32", 128),
                loss=runtime.ndarray("f32", 1),
            )
            for _ in modules
        ]
        for module, frame in zip(modules, frames):
            module.kernel("exchange")(
                frame["out"], aot.Scalar("u32", 2), aot.Scalar("i32", -2)
            )
            check_exchange(frame["out"], 2, -2)
        for index, (module, frame) in enumerate(zip(modules, frames)):
            for delta, scale in ((0, 1), (2, -2), (1, 0)):
                bindings = dict(
                    frame,
                    delta=aot.Scalar("u32", delta),
                    scale=aot.Scalar("i32", scale),
                )
                frame["values"].write(array.array("f", [-99] * 128))
                call = module.prepare_execution("simt", bindings)
                assert read(frame["values"]) == [-99] * 128, "prepare executed work"
                assert (
                    call.info["execution_kind"]
                    == {"cuda": "cuda_graph", "vulkan": "vulkan_commands"}[arch]
                )
                for _ in range(2):
                    call()
                    check_exchange(frame["out"], delta, scale)
                    expected = [(i // 32 * 32 + 15.5) * scale for i in range(128)]
                    assert read(frame["values"]) == expected
                    assert read(frame["gradient"]) == expected
                    assert (
                        abs(read(frame["loss"])[0] - sum(x * x * 0.5 for x in expected))
                        < 0.01
                    )
                call.close()
            # Exercise enqueue-before-close without keeping producer objects alive.
            call = module.prepare_execution("simt", bindings)
            call()
            module.close()
            raises(RuntimeError, call)
            call.close()
            assert read(frame["values"]) == [0] * 128
            if index == 0:
                modules[1].kernel("exchange")(
                    frames[1]["out"], aot.Scalar("u32", 2), aot.Scalar("i32", 3)
                )
                check_exchange(frames[1]["out"], 2, 3)
        for frame in frames:
            for value in frame.values():
                value.close()
    if mode == "standalone":
        assert not {"taichi_forge", "numpy"}.intersection(sys.modules)
    else:
        ti.reset()
    print("AOT SIMT OK", arch, mode)


if __name__ == "__main__":
    main(*sys.argv[1:])
