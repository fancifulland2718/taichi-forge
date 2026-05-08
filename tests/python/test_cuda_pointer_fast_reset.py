from taichi_forge.lang import impl

import taichi_forge as ti
from tests import test_utils


def _kernel_profile_names():
    prog = impl.get_runtime().prog
    prog.sync_kernel_profiler()
    prog.update_kernel_profiler()
    return [rec.name for rec in prog.get_kernel_profiler_records()]


@test_utils.test(
    arch=ti.cuda,
    offline_cache=False,
    kernel_profiler=True,
    cuda_pointer_deterministic_slot=True,
    cuda_pointer_fast_reset=True,
    cuda_listgen_reuse=True,
)
def test_cuda_deterministic_pointer_fast_reset_skips_gc():
    n = 32
    block = 8
    x = ti.field(dtype=ti.f32)
    ptr = ti.root.pointer(ti.ijk, n // block)
    leaf = ptr.bitmasked(ti.ijk, block)
    leaf.place(x)

    @ti.kernel
    def fill(v: ti.f32):
        for i, j, k in ti.ndrange(n, n, n):
            if (i * 13 + j * 7 + k * 3) % 29 == 0:
                x[i, j, k] = v

    @ti.kernel
    def sum_x() -> ti.f32:
        s = 0.0
        for I in ti.grouped(x):
            s += x[I]
        return s

    fill(1.0)
    before = sum_x()
    assert before > 0.0
    ti.sync()

    ti.profiler.clear_kernel_profiler_info()
    ptr.deactivate_all()
    ti.sync()
    names = _kernel_profile_names()
    assert names
    assert not any("gc_" in name for name in names)

    fill(2.0)
    after = sum_x()
    assert abs(after - before * 2.0) < 1e-3
