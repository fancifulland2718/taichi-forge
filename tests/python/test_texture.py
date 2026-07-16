import gc
from io import BytesIO

import numpy as np
import pytest
import requests
from PIL import Image
from taichi_forge.lang import impl

import taichi_forge as ti
from tests import test_utils

supported_archs_texture = [ti.vulkan]
supported_archs_texture_excluding_load_store = [ti.vulkan, ti.opengl]

integer_storage_image_cases = [
    (ti.Format.r16u, ti.u32, 1, (1, 0, 0, 0)),
    (ti.Format.rg16u, ti.u32, 2, (1, 65535, 0, 0)),
    (ti.Format.rgba16u, ti.u32, 4, (1, 65535, 32768, 7)),
    (ti.Format.r16i, ti.i32, 1, (-32768, 0, 0, 0)),
    (ti.Format.rg16i, ti.i32, 2, (-32768, -1, 0, 0)),
    (ti.Format.rgba16i, ti.i32, 4, (-32768, -1, 1234, 32767)),
    (ti.Format.r32u, ti.u32, 1, (1, 0, 0, 0)),
    (ti.Format.rg32u, ti.u32, 2, (1, 0xFFFFFFFF, 0, 0)),
    (ti.Format.rgba32u, ti.u32, 4, (1, 0xFFFFFFFF, 0x80000000, 7)),
    (ti.Format.r32i, ti.i32, 1, (-0x80000000, 0, 0, 0)),
    (ti.Format.rg32i, ti.i32, 2, (-0x80000000, -1, 0, 0)),
    (ti.Format.rgba32i, ti.i32, 4, (-0x80000000, -1, 1234, 0x7FFFFFFF)),
]


@ti.func
def taichi_logo(pos: ti.template(), scale: float = 1 / 1.11):
    p = (pos - 0.5) / scale + 0.5
    ret = -1
    if not (p - 0.50).norm_sqr() <= 0.52**2:
        if ret == -1:
            ret = 0
    if not (p - 0.50).norm_sqr() <= 0.495**2:
        if ret == -1:
            ret = 1
    if (p - ti.Vector([0.50, 0.25])).norm_sqr() <= 0.08**2:
        if ret == -1:
            ret = 1
    if (p - ti.Vector([0.50, 0.75])).norm_sqr() <= 0.08**2:
        if ret == -1:
            ret = 0
    if (p - ti.Vector([0.50, 0.25])).norm_sqr() <= 0.25**2:
        if ret == -1:
            ret = 0
    if (p - ti.Vector([0.50, 0.75])).norm_sqr() <= 0.25**2:
        if ret == -1:
            ret = 1
    if p[0] < 0.5:
        if ret == -1:
            ret = 1
    else:
        if ret == -1:
            ret = 0
    return 1 - ret


@ti.kernel
def make_texture_2d_r32f(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0), n: ti.i32):
    for i, j in ti.ndrange(n, n):
        ret = ti.cast(taichi_logo(ti.Vector([i, j]) / n), ti.f32)
        tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))


@ti.kernel
def make_texture_2d_rgba8(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.rgba8, lod=0), n: ti.i32):
    for i, j in ti.ndrange(n, n):
        ret = ti.cast(taichi_logo(ti.Vector([i, j]) / n), ti.f32)
        tex.store(ti.Vector([i, j]), ti.Vector([ret, 0.0, 0.0, 0.0]))


@ti.kernel
def make_texture_3d(tex: ti.types.rw_texture(num_dimensions=3, fmt=ti.Format.r32f, lod=0), n: ti.i32):
    for i, j, k in ti.ndrange(n, n, n):
        div = ti.cast(i / n, ti.f32)
        if div > 0.5:
            tex.store(ti.Vector([i, j, k]), ti.Vector([1.0, 0.0, 0.0, 0.0]))
        else:
            tex.store(ti.Vector([i, j, k]), ti.Vector([0.5, 0.0, 0.0, 0.0]))


@test_utils.test(arch=supported_archs_texture)
def test_texture_compiled_functions():
    res = (512, 512)
    pixels = ti.Vector.field(3, dtype=float, shape=res)

    @ti.kernel
    def paint(t: ti.f32, tex: ti.types.texture(num_dimensions=2), n: ti.i32):
        for i, j in pixels:
            uv = ti.Vector([i / res[0], j / res[1]])
            warp_uv = uv + ti.Vector([ti.cos(t + uv.x * 5.0), ti.sin(t + uv.y * 5.0)]) * 0.1
            c = ti.math.vec4(0.0)
            if uv.x > 0.5:
                c = tex.sample_lod(warp_uv, 0.0)
            else:
                c = tex.fetch(ti.cast(warp_uv * n, ti.i32), 0)
            pixels[i, j] = [c.r, c.r, c.r]

    n1 = 128
    texture1 = ti.Texture(ti.Format.r32f, (n1, n1))
    n2 = 256
    texture2 = ti.Texture(ti.Format.r32f, (n2, n2))
    texture3 = ti.Texture(ti.Format.rgba8, (n1, n1))

    make_texture_2d_r32f(texture1, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    make_texture_2d_r32f(texture2, n2)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    make_texture_2d_rgba8(texture3, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 2

    paint(0.1, texture1, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 3

    paint(0.2, texture2, n2)
    assert impl.get_runtime().get_num_compiled_functions() == 3

    # (penguinliong) Remember that non-RW textures don't enforce a format so
    # it's the same as the first call to `paint`.
    paint(0.3, texture3, n1)
    assert impl.get_runtime().get_num_compiled_functions() == 3


@test_utils.test(arch=supported_archs_texture_excluding_load_store)
def test_texture_from_field():
    res = (128, 128)
    f = ti.Vector.field(2, ti.f32, res)
    tex = ti.Texture(ti.Format.r32f, res)

    @ti.kernel
    def init_taichi_logo_field():
        for i, j in f:
            f[i, j] = [taichi_logo(ti.Vector([i / res[0], j / res[1]])), 0]

    init_taichi_logo_field()
    tex.from_field(f)


@test_utils.test(arch=supported_archs_texture_excluding_load_store)
def test_texture_from_ndarray():
    res = (128, 128)
    f = ti.Vector.ndarray(2, ti.f32, res)
    tex = ti.Texture(ti.Format.r32f, res)

    @ti.kernel
    def init_taichi_logo_ndarray(f: ti.types.ndarray(ndim=2)):
        for i, j in f:
            f[i, j] = [taichi_logo(ti.Vector([i / res[0], j / res[1]])), 0]

    init_taichi_logo_ndarray(f)
    tex.from_ndarray(f)


@test_utils.test(arch=supported_archs_texture)
def test_texture_3d():
    res = (32, 32, 32)
    tex = ti.Texture(ti.Format.r32f, res)

    make_texture_3d(tex, res[0])


@test_utils.test(arch=supported_archs_texture)
def test_from_to_image():
    url = "https://github.com/taichi-dev/taichi/blob/master/misc/logo.png?raw=true"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    tex = ti.Texture(ti.Format.rgba8, img.size)

    tex.from_image(img)
    out = tex.to_image()

    assert (np.asarray(out) == np.asarray(img.convert("RGB"))).all()


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_2d_struct_for():
    res = (128, 128)
    tex = ti.Texture(ti.Format.r32f, res)
    arr = ti.ndarray(ti.f32, res)

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    @ti.kernel
    def read(tex: ti.types.texture(num_dimensions=2), arr: ti.types.ndarray()):
        for i, j in arr:
            arr[i, j] = tex.fetch(ti.Vector([i, j]), 0).x

    write(tex)
    read(tex, arr)
    assert arr.to_numpy().sum() == 128 * 128


@pytest.mark.parametrize(
    "fmt,dtype,num_channels,values", integer_storage_image_cases
)
@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_integer_sampled_types(fmt, dtype, num_channels, values):
    tex = ti.Texture(fmt, (1, 1))
    out = ti.ndarray(dtype=dtype, shape=4)

    @ti.kernel
    def write(
        tex: ti.types.rw_texture(num_dimensions=2, fmt=fmt, lod=0),
        v0: dtype,
        v1: dtype,
        v2: dtype,
        v3: dtype,
    ):
        tex.store(
            ti.Vector([0, 0]),
            ti.Vector([v0, v1, v2, v3]),
        )

    @ti.kernel
    def read(
        tex: ti.types.rw_texture(num_dimensions=2, fmt=fmt, lod=0),
        out: ti.types.ndarray(dtype=dtype, ndim=1),
    ):
        value = tex.load(ti.Vector([0, 0]))
        for i in ti.static(range(4)):
            out[i] = value[i]

    write(tex, *values)
    read(tex, out)
    np.testing.assert_array_equal(
        out.to_numpy()[:num_channels],
        np.asarray(
            values[:num_channels],
            dtype=np.uint32 if dtype == ti.u32 else np.int32,
        ),
    )


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_2d_struct_for_dim_check():
    tex = ti.Texture(ti.Format.r32f, (32, 32, 32))

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="RWTextureType dimension mismatch for argument tex: expected 2, got 3",
    ) as e:
        write(tex)


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_wrong_fmt():
    tex = ti.Texture(ti.Format.rgba8, (32, 32))

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="RWTextureType format mismatch for argument tex: expected Format.r32f, got Format.rgba8",
    ) as e:
        write(tex)


@test_utils.test(arch=supported_archs_texture)
def test_rw_texture_wrong_ndim():
    tex = ti.Texture(ti.Format.rgba8, (32, 32))

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=1, fmt=ti.Format.rgba8, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="RWTextureType dimension mismatch for argument tex: expected 1, got 2",
    ) as e:
        write(tex)


@test_utils.test(arch=supported_archs_texture)
def test_texture_registry_keeps_inflight_launch_alive_until_sync():
    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        tex.store(ti.Vector([0, 0]), ti.Vector([7.0, 0.0, 0.0, 0.0]))

    warmup = ti.Texture(ti.Format.r32f, (1, 1))
    write(warmup)
    ti.sync()
    del warmup
    gc.collect()
    ti.sync()

    prog = impl.get_runtime().prog
    baseline = prog._debug_texture_resource_stats()
    tex = ti.Texture(ti.Format.r32f, (1, 1))
    created = prog._debug_texture_resource_stats()
    assert created["created_total"] == baseline["created_total"] + 1
    assert created["live"] == baseline["live"] + 1
    assert created["views"] == baseline["views"] + 1
    assert created["leases"] == baseline["leases"] + 1

    write(tex)
    launched = prog._debug_texture_resource_stats()
    assert launched["inflight"] == baseline["inflight"] + 1
    assert launched["leases"] == baseline["leases"] + 2

    del tex
    gc.collect()
    retired = prog._debug_texture_resource_stats()
    assert retired["views"] == baseline["views"]
    assert retired["retiring"] == baseline["retiring"] + 1
    assert retired["retired_total"] == baseline["retired_total"] + 1

    ti.sync()
    completed = prog._debug_texture_resource_stats()
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key]
    assert completed["created_total"] == baseline["created_total"] + 1
    assert completed["retired_total"] == baseline["retired_total"] + 1
    assert completed["released_total"] == baseline["released_total"] + 1


@test_utils.test(arch=supported_archs_texture)
def test_texture_launch_context_rejects_stale_resource_generation():
    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        tex.store(ti.Vector([0, 0]), ti.Vector([3.0, 0.0, 0.0, 0.0]))

    old = ti.Texture(ti.Format.r32f, (1, 1))
    primal = write._primal
    key = primal.ensure_compiled(old)
    kernel_cpp = primal.compiled_kernels[key]
    prog = impl.get_runtime().prog
    compiled = prog.compile_kernel(
        prog.config(), prog.get_device_caps(), kernel_cpp
    )
    old_identity = prog._debug_texture_resource_identity(old.tex)

    old_native = old.tex
    old._invalidate_runtime()
    prog.delete_texture(old_native)

    replacement = ti.Texture(ti.Format.r32f, (1, 1))
    replacement_identity = prog._debug_texture_resource_identity(
        replacement.tex
    )
    assert replacement_identity["domain"] == old_identity["domain"]
    assert replacement_identity["index"] == old_identity["index"]
    assert replacement_identity["generation"] != old_identity["generation"]

    launch_ctx = kernel_cpp.make_launch_context()
    launch_ctx.set_arg_rw_texture([0], replacement.tex)
    launch_ctx._debug_set_texture_resource_handle(
        [0],
        old_identity["domain"],
        old_identity["kind"],
        old_identity["index"],
        old_identity["generation"],
    )
    with pytest.raises(RuntimeError, match="stale or retired Texture"):
        prog.launch_kernel(compiled, launch_ctx)


@pytest.mark.run_in_serial
@test_utils.test(arch=supported_archs_texture)
def test_texture_registry_resize_churn_conserves_resources():
    prog = impl.get_runtime().prog
    runtime = impl.get_runtime()
    baseline = prog._debug_texture_resource_stats()
    baseline_runtime_objects = len(runtime._runtime_object_refs)

    iterations = 256
    for i in range(iterations):
        extent = 1 << (i % 5)
        tex = ti.Texture(ti.Format.rgba8, (extent, extent))
        del tex
        if i % 32 == 0:
            gc.collect()
    gc.collect()

    completed = prog._debug_texture_resource_stats()
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key]
    assert completed["created_total"] - baseline["created_total"] == iterations
    assert completed["retired_total"] - baseline["retired_total"] == iterations
    assert completed["released_total"] - baseline["released_total"] == iterations
    assert len(runtime._runtime_object_refs) == baseline_runtime_objects
