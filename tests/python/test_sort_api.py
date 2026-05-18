import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu])
def test_sort_entrypoint_auto_uses_host_stable_fallback():
    keys = ti.field(ti.i32, 8)
    values = ti.field(ti.i32, 8)

    @ti.kernel
    def fill():
        for i in keys:
            keys[i] = 8 - i
            values[i] = i

    fill()
    ti.algorithms.sort(keys, values)

    assert keys.to_numpy().tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert values.to_numpy().tolist() == [7, 6, 5, 4, 3, 2, 1, 0]


@test_utils.test(arch=[ti.cpu])
def test_sort_by_key_single_part_routes_to_sort():
    keys = ti.field(ti.i32, 4)
    values = ti.field(ti.i32, 4)

    @ti.kernel
    def fill():
        keys[0] = 3
        keys[1] = 1
        keys[2] = 2
        keys[3] = 0
        for i in values:
            values[i] = i

    fill()
    ti.algorithms.sort_by_key([keys], values)

    assert keys.to_numpy().tolist() == [0, 1, 2, 3]
    assert values.to_numpy().tolist() == [3, 1, 2, 0]


@test_utils.test(arch=[ti.cpu])
def test_sort_by_key_multi_part_is_stable():
    primary = ti.field(ti.i32, 6)
    secondary = ti.field(ti.i32, 6)
    values = ti.field(ti.i32, 6)

    @ti.kernel
    def fill():
        primary[0], secondary[0], values[0] = 1, 2, 0
        primary[1], secondary[1], values[1] = 0, 3, 1
        primary[2], secondary[2], values[2] = 1, 1, 2
        primary[3], secondary[3], values[3] = 0, 2, 3
        primary[4], secondary[4], values[4] = 1, 1, 4
        primary[5], secondary[5], values[5] = 0, 2, 5

    fill()
    ti.algorithms.sort_by_key([primary, secondary], values)

    assert primary.to_numpy().tolist() == [0, 0, 0, 1, 1, 1]
    assert secondary.to_numpy().tolist() == [2, 2, 3, 1, 1, 2]
    assert values.to_numpy().tolist() == [3, 5, 1, 2, 4, 0]


@test_utils.test(arch=[ti.cpu])
def test_sort_auto_is_stable_for_duplicate_keys():
    keys = ti.field(ti.i32, 8)
    values = ti.field(ti.i32, 8)

    @ti.kernel
    def fill():
        for i in keys:
            keys[i] = i % 2
            values[i] = i

    fill()
    ti.algorithms.sort(keys, values)

    assert keys.to_numpy().tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert values.to_numpy().tolist() == [0, 2, 4, 6, 1, 3, 5, 7]


@test_utils.test(arch=[ti.cpu])
def test_sort_auto_keys_only():
    keys = ti.field(ti.i32, 8)

    @ti.kernel
    def fill():
        keys[0] = 4
        keys[1] = 1
        keys[2] = 3
        keys[3] = 1
        keys[4] = 2
        keys[5] = 0
        keys[6] = 4
        keys[7] = 2

    fill()
    ti.algorithms.sort(keys)

    assert keys.to_numpy().tolist() == [0, 1, 1, 2, 2, 3, 4, 4]


@pytest.mark.parametrize("dtype", [ti.i64, ti.u64, ti.f64])
@test_utils.test(arch=[ti.cpu])
def test_sort_auto_supports_wide_key_dtypes(dtype):
    keys = ti.field(dtype, 4)
    values = ti.field(ti.i32, 4)

    @ti.kernel
    def fill():
        keys[0] = 3
        keys[1] = 1
        keys[2] = 2
        keys[3] = 0
        for i in values:
            values[i] = i

    fill()
    ti.algorithms.sort(keys, values)

    assert keys.to_numpy().tolist() == [0, 1, 2, 3]
    assert values.to_numpy().tolist() == [3, 1, 2, 0]


def test_sort_workspace_metadata_only():
    workspace = ti.algorithms.SortWorkspace(max_items=16, device="test")
    assert workspace.max_items == 16
    assert workspace.device == "test"
    assert workspace.workspace_bytes_current == 0
    assert workspace.workspace_bytes_peak == 0

    assert workspace.reserve(dtype=ti.i32, value_dtype=ti.i32, n=8) is workspace
    workspace.clear()
    assert workspace.workspace_bytes_current == 0
    assert workspace.workspace_bytes_peak == 0

    with pytest.raises(ValueError):
        workspace.reserve(n=17)


@test_utils.test(arch=[ti.cpu])
def test_sort_rejects_unimplemented_modes():
    keys = ti.field(ti.i32, 4)

    with pytest.raises(NotImplementedError):
        ti.algorithms.sort(keys, stable=False)
    with pytest.raises(NotImplementedError):
        ti.algorithms.sort(keys, descending=True)
    with pytest.raises(RuntimeError):
        ti.algorithms.sort(keys, method="cuda_cub_native")
    with pytest.raises(RuntimeError):
        ti.algorithms.sort(keys, method="cuda_cub_u32")
    with pytest.raises(NotImplementedError):
        ti.algorithms.sort(keys, precision="approx")
    with pytest.raises(ValueError):
        ti.algorithms.sort(keys, nan_policy="middle")
    with pytest.raises(RuntimeError):
        ti.algorithms.sort(keys, method="radix_u32")
    with pytest.raises(NotImplementedError):
        ti.algorithms.sort_by_key([keys, keys], method="legacy")
