"""Compact child capacity bounds live parents rather than past activations."""

import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("active_list", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False, hash_snode_compact_child_pool=True)
def test_compact_hash_recycles_retired_parent_slots(active_list):
    ti.lang.impl.current_cfg().hash_snode_active_list = active_list
    values = ti.field(ti.i32)
    outer = ti.root.hash(ti.i, 64, expected_active=2)
    outer.hash(ti.j, 8, capacity=4).place(values)

    @ti.kernel
    def rotate(previous: ti.i32, current: ti.i32, value: ti.i32):
        ti.deactivate(outer, previous)
        values[current, 1] = value

    @ti.kernel
    def gather() -> ti.i32:
        result = 0
        for i, j in values:
            result += values[i, j]
        return result

    previous = 0
    for step, current in enumerate((3, 17, 1, 39, 12, 56, 3, 17)):
        rotate(previous, current, step + 1)
        assert gather() == step + 1
        assert values[previous, 1] == 0
        previous = current


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False, hash_snode_compact_child_pool=True)
def test_compact_hash_recycles_parallel_slots_and_clears_children():
    # Cross a bitmap word boundary, including a partial final word. All slots
    # are live together; the next generation must reuse every retired slot.
    capacity = 33
    values = ti.field(ti.i32)
    outer = ti.root.hash(ti.i, 512, expected_active=capacity)
    outer.hash(ti.j, 8, capacity=4).place(values)

    @ti.kernel
    def fill(base: ti.i32, generation: ti.i32):
        for i in range(capacity):
            values[base + i, generation % 2] = generation * 100 + i + 1

    @ti.kernel
    def retire(base: ti.i32):
        for i in range(capacity):
            ti.deactivate(outer, base + i)

    @ti.kernel
    def verify(base: ti.i32, generation: ti.i32) -> ti.i32:
        errors = 0
        for i in range(capacity):
            errors += values[base + i, generation % 2] != generation * 100 + i + 1
            errors += values[base + i, (generation + 1) % 2] != 0
        return errors

    @ti.kernel
    def gather() -> ti.i32:
        result = 0
        for i, j in values:
            result += values[i, j]
        return result

    for generation in range(4):
        base = generation * 64
        fill(base, generation)
        assert verify(base, generation) == 0
        assert gather() == generation * 100 * capacity + capacity * (capacity + 1) // 2
        retire(base)
        assert gather() == 0
