import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_independent_local_sums_with_partial_subgroup(dtype):
    caps = ti.lang.impl.get_runtime().prog.get_device_caps()
    double_add = caps.get("spirv_has_atomic_float64_add") or caps.get("spirv_has_atomic_int64")
    if dtype == ti.f64 and not (caps.get("spirv_has_float64") and double_add):
        pytest.skip("device lacks float64 storage-buffer atomic add or its integer fallback")

    @ti.kernel
    def totals(n: ti.template(), seed: dtype) -> ti.types.vector(2, dtype):
        a = seed
        b = -seed
        for i in range(n):
            a += ti.cast(i % 3, dtype)
            b += ti.cast(i % 5, dtype)
        return ti.Vector([a, b])

    for n in (127, 32768, 65539):
        indices = np.arange(n)
        expected = [7 + (indices % 3).sum(), -7 + (indices % 5).sum()]
        np.testing.assert_array_equal(totals(n, 7), expected)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_observed_local_atomic_add_returns_unique_old_values():
    n = 32771
    tickets = ti.field(ti.f32, shape=n)

    @ti.kernel
    def assign_tickets() -> ti.f32:
        count = 0.0
        for i in range(n):
            tickets[i] = ti.atomic_add(count, 1.0)
        return count

    assert assign_tickets() == n
    np.testing.assert_array_equal(np.sort(tickets.to_numpy()), np.arange(n))
