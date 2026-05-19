import gc

import taichi_forge as ti
from tests import test_utils


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
