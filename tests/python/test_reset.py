import gc

import pytest

import taichi_forge as ti
from tests import test_utils
from taichi_forge.lang import impl


@test_utils.test(arch=[ti.vulkan])
def test_reset_invalidates_live_runtime_wrappers():
    arr = ti.ndarray(ti.i32, shape=32)
    arr.fill(7)
    vec = ti.Vector.ndarray(2, ti.f32, shape=8)
    vec.fill([1.0, 2.0])
    tex = ti.Texture(ti.Format.rgba8, (8, 8))
    pack_type = ti.types.argpack(a=ti.i32, b=ti.f32)
    pack = pack_type(a=1, b=2.0)

    ti.reset()
    del arr, vec, tex, pack
    gc.collect()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_reset_retires_inflight_argpack():
    sink = ti.field(ti.i32, shape=())
    pack_type = ti.types.argpack(value=ti.i32)
    pack = pack_type(value=9)

    @ti.kernel
    def consume(value: pack_type):
        sink[None] += value.value

    consume(pack)
    arch = impl.current_cfg().arch
    ti.reset()

    # A new Program may reuse native addresses, but an invalidated Python view
    # must fail before it can resolve against that Program's registry.
    ti.init(arch=arch)

    @ti.kernel
    def consume_after_reset(value: pack_type) -> ti.i32:
        return value.value

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="Cannot submit an ArgPack after its Taichi runtime has been reset",
    ):
        consume_after_reset(pack)
    del pack
    gc.collect()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_reset_rejects_invalidated_ndarray_submission():
    arr = ti.ndarray(ti.i32, shape=1)
    arr.fill(9)
    arch = impl.current_cfg().arch
    ti.reset()
    ti.init(arch=arch)

    @ti.kernel
    def consume_after_reset(value: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        value[0] += 1

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="Cannot submit an Ndarray after its Taichi runtime has been reset",
    ):
        consume_after_reset(arr)
    del arr
    gc.collect()


@test_utils.test(arch=[ti.vulkan])
def test_reset_rejects_invalidated_texture_submission():
    tex = ti.Texture(ti.Format.rgba8, (1, 1))
    ti.reset()
    ti.init(arch=ti.vulkan)

    @ti.kernel
    def consume_after_reset(
        value: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.rgba8, lod=0
        ),
    ):
        value.store(ti.Vector([0, 0]), ti.Vector([1.0, 0.0, 0.0, 1.0]))

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="Cannot submit a Texture after its Taichi runtime has been reset",
    ):
        consume_after_reset(tex)
    del tex
    gc.collect()
