"""Actual JIT/AOT queue sharing, sampler preservation and leased retirement."""

import gc
import json
from pathlib import Path
import statistics
import sys
import time
import weakref

import numpy as np
import taichi_forge as ti
from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from aot_optix_probe import raises, read, write


def main(arch, artifact, result):
    ti.init(arch=getattr(ti, arch), offline_cache=False, debug=arch == "vulkan")

    @ti.kernel
    def paint(pixels: ti.types.ndarray(ti.f32, 2), scale: ti.f32):
        for i, j in pixels:
            pixels[i, j] = scale * (i * 5 + j)

    @ti.kernel
    def sample(
        image: ti.types.texture(num_dimensions=2), output: ti.types.ndarray(ti.f32, 1)
    ):
        output[0] = image.fetch(ti.Vector([0, 4]), 0).x
        output[1] = image.fetch(ti.Vector([2, 1]), 0).x
        output[2] = image.sample_lod(ti.Vector([0.5, 0.25]), 0.0).x
        output[3] = image.sample_lod(ti.Vector([1.1, 0.9]), 0.0).x
        output[4] = image.fetch(ti.Vector([1, 2]), 0).x

    pixels = ti.ndarray(ti.f32, (3, 5))
    output = ti.ndarray(ti.f32, 5)
    sampler = ti.hardware.sampling.SamplerConfig(
        min_filter="nearest",
        mag_filter="nearest",
        address_mode_u="repeat",
        address_mode_v="repeat",
    )
    texture = ti.Texture(ti.Format.r32f, (3, 5), sampler=sampler)
    paint(pixels, 1)
    texture.from_ndarray(pixels)
    sample(texture, output)
    expected = output.to_numpy().tolist()
    assert expected[:3] == ([4, 11, 6] if arch == "cuda" else [12, 5, 4])
    with Runtime(arch) as standalone:
        raises(lambda: standalone.import_texture(texture), "borrow_runtime")
    runtime, second = ti.aot.borrow_runtime(), ti.aot.borrow_runtime()
    start = time.perf_counter()
    image = runtime.import_texture(texture)
    alias = runtime.import_texture(texture)
    import_ms = (time.perf_counter() - start) * 1000
    other_image = second.import_texture(texture)
    assert image._handle != alias._handle != other_image._handle
    module, other_module = runtime.load_module(artifact), second.load_module(artifact)

    def prepare(owner, mod, image, alias):
        frame = {
            name: owner.ndarray("f32", (1,) if name in ("loss", "dloss") else (5,))
            for name in ("output", "doutput", "loss", "dloss")
        }
        call = mod.prepare_execution("step", dict(frame, image=image, alias=alias))
        return call, frame

    def check(call, frame, scale):
        call()
        np.testing.assert_allclose(read(frame["output"]), np.array(expected) * scale)
        np.testing.assert_allclose(read(frame["doutput"]), np.array(expected) * scale)
        np.testing.assert_allclose(
            read(frame["loss"]), [0.5 * sum(x * x for x in expected) * scale**2]
        )

    start = time.perf_counter()
    call, first = prepare(runtime, module, image, alias)
    other, frame = prepare(second, other_module, other_image, other_image)
    prepare_ms = (time.perf_counter() - start) * 1000
    check(call, first, 1)
    check(other, frame, 1)
    # Device-written JIT pixels and the existing upload precede AOT consumption.
    paint(pixels, 2)
    texture.from_ndarray(pixels)
    check(call, first, 2)
    check(other, frame, 2)

    # AOT upload changes the existing JIT texture; no new sampler or storage.
    source = runtime.ndarray("f32", (3, 5))
    write(source, [3 * i for i in range(15)])
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
    ).prepare(module, dict(image=image, pixels=source))
    upload()
    sample(texture, output)
    np.testing.assert_allclose(output.to_numpy(), np.array(expected) * 3)
    check(other, frame, 3)
    image.close()
    raises(call)
    raises(upload)
    call.close()
    upload.close()
    # Independent imports own independent leases on the same Program texture.
    direct = module.kernel("sample").prepare(alias, alias, first["output"])
    direct()
    np.testing.assert_allclose(read(first["output"]), np.array(expected) * 3)
    runtime.close()
    sample(texture, output)
    np.testing.assert_allclose(output.to_numpy(), np.array(expected) * 3)
    check(other, frame, 3)

    # Dropping the source wrapper removes its public JIT view, not the lease.
    raw = texture.tex
    texture_ref = weakref.ref(texture)
    del texture
    gc.collect()
    assert texture_ref() is None
    raises(lambda: second._borrower.import_texture(raw), "stale or retired")
    check(other, frame, 3)
    submit, complete = [], []
    for _ in range(8):
        start = time.perf_counter_ns()
        other()
        submit.append((time.perf_counter_ns() - start) / 1000)
        second.wait()
        complete.append((time.perf_counter_ns() - start) / 1000)
    # Closing one lease after in-flight use leaves its Program and peer alive.
    other()
    other_image.close()
    raises(other)
    other.close()
    if arch == "vulkan":
        mip = ti.Texture(ti.Format.r32f, (4, 4), mip_levels=2)
        raises(lambda: second.import_texture(mip), "single-level")
        del mip

    # Owner reset retires calls and leases before destroying backend storage.
    texture = ti.Texture(ti.Format.r32f, (3, 5), sampler=sampler)
    paint(pixels, 4)
    texture.from_ndarray(pixels)
    image = second.import_texture(texture)
    final, _ = prepare(second, other_module, image, image)
    final()
    ti.reset()
    raises(final)
    raises(lambda: second.import_texture(texture))
    final.close()
    second.close()
    Path(result).write_text(
        json.dumps(
            dict(
                import_two_ms=import_ms,
                prepare_two_ms=prepare_ms,
                submit_us=statistics.median(submit),
                complete_us=statistics.median(complete),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
