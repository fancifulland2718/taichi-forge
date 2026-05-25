import taichi_forge as ti
from tests import test_utils


_D3_GLOBAL_SCALE = 2


@test_utils.test()
def test_ti_func_global_lookup_reads_current_module_global():
    global _D3_GLOBAL_SCALE
    _D3_GLOBAL_SCALE = 7

    @ti.func
    def add_global(x):
        return x + _D3_GLOBAL_SCALE

    _D3_GLOBAL_SCALE = 11

    @ti.kernel
    def run() -> ti.i32:
        return add_global(5)

    assert run() == 16


@test_utils.test()
def test_ti_func_global_lookup_reads_current_closure_cell():
    scale = 3

    @ti.func
    def add_closure(x):
        return x + scale

    scale = 13

    @ti.kernel
    def run() -> ti.i32:
        return add_closure(5)

    assert run() == 18
