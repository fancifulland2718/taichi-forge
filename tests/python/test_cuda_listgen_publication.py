"""Listgen decisions stay stable across a grid and repeated topology changes."""

import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("reuse", [True, False])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_multilevel_listgen_late_active_parents(reuse):
    ti.cfg.cuda_listgen_reuse = reuse
    # Small payload, but many parent containers. Empty low-index parents let
    # early blocks finish before later blocks have visited the active region.
    n = 8192
    values = ti.field(ti.i32)
    blocks = ti.root.pointer(ti.i, n)
    blocks.dense(ti.i, 1).bitmasked(ti.i, 2).place(values)

    @ti.kernel
    def populate(begin: ti.i32, end: ti.i32, scale: ti.i32):
        for i in range(begin, end):
            values[2 * i] = scale

    @ti.kernel
    def clear():
        for i in range(n):
            ti.deactivate(blocks, i)

    @ti.kernel
    def inspect() -> ti.types.vector(2, ti.i32):
        count = 0
        total = 0
        for i in values:
            count += 1
            total += values[i]
        return ti.Vector([count, total])

    for begin, end, scale in ((n // 2, n, 3), (0, n // 4, 5), (n - 7, n, 11)):
        populate(begin, end, scale)
        expected = [end - begin, (end - begin) * scale]
        # Exercise both rebuild and unchanged-topology reuse.
        assert list(inspect()) == expected
        assert list(inspect()) == expected
        clear()
        assert list(inspect()) == [0, 0]

