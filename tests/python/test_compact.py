import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


_COMPACT_DTYPES = [
    (ti.u32, np.uint32),
    (ti.i32, np.int32),
    (ti.f32, np.float32),
    (ti.u64, np.uint64),
    (ti.i64, np.int64),
    (ti.f64, np.float64),
]


def _compact_values(n, np_dtype):
    if np.issubdtype(np_dtype, np.floating):
        return (
            ((np.arange(n, dtype=np.float64) % 97) - 48) * np.float64(0.125)
        ).astype(np_dtype)
    if np.issubdtype(np_dtype, np.unsignedinteger):
        return ((np.arange(n, dtype=np.uint64) * 11 + 5) % 4294967291).astype(
            np_dtype
        )
    return (np.arange(n, dtype=np.int64) * 7 - 13).astype(np_dtype)


def _assert_compact_matches(actual, expected):
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    else:
        assert np.array_equal(actual, expected)


def _run_ndarray_compact(dtype, np_dtype, method):
    n = 4096
    values = ti.ndarray(dtype, shape=n)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(dtype, shape=n)
    count = ti.ndarray(ti.i32, shape=1)

    values_np = _compact_values(n, np_dtype)
    flags_np = (
        ((np.arange(n) % 3 == 0) | (np.arange(n) % 17 == 0)).astype(np.int32)
    )
    values.from_numpy(values_np)
    flags.from_numpy(flags_np)
    output.fill(0)
    count.from_numpy(np.array([-1], dtype=np.int32))

    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    ti.algorithms.experimental_compact(
        values, flags, output, count, method=method, workspace=workspace
    )

    expected = values_np[flags_np != 0]
    assert count.to_numpy()[0] == expected.shape[0]
    _assert_compact_matches(output.to_numpy()[: expected.shape[0]], expected)
    return workspace


def _run_vector_ndarray_compact(method):
    n = 2048
    values = ti.Vector.ndarray(3, ti.f32, shape=n)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.Vector.ndarray(3, ti.f32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)
    values_np = (
        ((np.arange(n * 3, dtype=np.float32).reshape(n, 3) % 97) - 48)
        * np.float32(0.125)
    )
    flags_np = (
        ((np.arange(n) % 4 == 0) | (np.arange(n) % 19 == 0)).astype(np.int32)
    )
    values.from_numpy(values_np)
    flags.from_numpy(flags_np)
    output.fill(0)
    count.from_numpy(np.array([-1], dtype=np.int32))

    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    ti.algorithms.experimental_compact(
        values, flags, output, count, method=method, workspace=workspace
    )

    expected = values_np[flags_np != 0]
    assert count.to_numpy()[0] == expected.shape[0]
    np.testing.assert_allclose(
        output.to_numpy()[: expected.shape[0]], expected, rtol=1e-6, atol=1e-6
    )
    return workspace


def _run_struct_tensor_member_compact(method, n=256, workspace=None):
    payload = ti.types.struct(vec=ti.types.vector(2, ti.i32), tag=ti.i32)
    values = ti.ndarray(payload, shape=n)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(payload, shape=n)
    count = ti.ndarray(ti.i32, shape=1)
    values_np = np.zeros((n,), dtype=values.numpy_dtype)
    output_np = np.zeros((n,), dtype=output.numpy_dtype)
    values_np["vec"] = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 97) - 48
    values_np["tag"] = np.arange(n, dtype=np.int32) * 5 + 1
    output_np["tag"] = np.arange(n, dtype=np.int32) * 11 + 7
    flags_np = ((np.arange(n) % 3 == 0) | (np.arange(n) % 19 == 0)).astype(np.int32)
    values.from_numpy(values_np)
    output.from_numpy(output_np)
    flags.from_numpy(flags_np)
    count.from_numpy(np.array([-1], dtype=np.int32))
    if workspace is None:
        workspace = ti.algorithms.CompactWorkspace(max_items=n)

    ti.algorithms.experimental_compact(
        values.field("vec"),
        flags,
        output.field("vec"),
        count,
        method=method,
        workspace=workspace,
    )

    selected = flags_np != 0
    selected_count = int(np.count_nonzero(selected))
    result = output.to_numpy()
    assert count.to_numpy()[0] == selected_count
    assert np.array_equal(result["vec"][:selected_count], values_np["vec"][selected])
    assert np.array_equal(result["tag"], output_np["tag"])
    return workspace


@test_utils.test(arch=[ti.cpu])
def test_experimental_compact_cpu_native_struct_tensor_member_views():
    workspace = _run_struct_tensor_member_compact("cpu_native")
    assert workspace.workspace_bytes_peak >= 256 * 2 * 4
    copy_workspace = workspace._order_apply_indexed_copy_workspace
    assert copy_workspace is not None
    assert copy_workspace._native_indexed_copy_plan is not None
    assert (
        copy_workspace._native_indexed_copy_plan.method_name
        == "cpu_gather_strided_ndarray"
    )


@test_utils.test(arch=[ti.cpu])
def test_experimental_compact_struct_tensor_member_order_pair_exact_size_cache():
    workspace = ti.algorithms.CompactWorkspace(max_items=64)
    _run_struct_tensor_member_compact("cpu_native", n=64, workspace=workspace)
    first_pair = workspace._order_apply_pair
    assert first_pair["in"].shape[0] == 64

    _run_struct_tensor_member_compact("cpu_native", n=32, workspace=workspace)
    second_pair = workspace._order_apply_pair
    assert second_pair is not first_pair
    assert second_pair["in"].shape[0] == 32

    _run_struct_tensor_member_compact("cpu_native", n=64, workspace=workspace)
    assert workspace._order_apply_pair is first_pair


@test_utils.test(arch=[ti.cuda])
def test_experimental_compact_cuda_cub_struct_tensor_member_views():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_cub_select_available")
        and prog.cuda_cub_select_available()
        and hasattr(prog, "cuda_device_indexed_copy_available")
        and prog.cuda_device_indexed_copy_available()
        and hasattr(prog, "cuda_toolkit_transform_available")
        and prog.cuda_toolkit_transform_available()
    ):
        pytest.skip("CUDA CUB compact, indexed copy, or strided transform is unavailable.")
    _run_struct_tensor_member_compact("cuda_cub")


@test_utils.test(arch=[ti.cuda])
def test_experimental_compact_cuda_auto_never_selects_cub_reference():
    prog = impl.get_runtime().prog
    if not prog.cuda_device_compact_available():
        pytest.skip("CUDA Driver compact is unavailable in this build/runtime.")

    workspace = _run_ndarray_compact(ti.i32, np.int32, "auto")

    assert workspace._native_compact_plan.backend == "cuda_device"
    assert workspace._cuda_device_active
    assert not workspace._cuda_cub_active


@test_utils.test(arch=[ti.cuda])
def test_experimental_compact_cuda_device_large_tiled_direct():
    n = (1 << 20) + 257
    prog = impl.get_runtime().prog
    if not prog.cuda_device_compact_available():
        pytest.skip("CUDA Driver compact is unavailable in this build/runtime.")

    values_np = np.arange(n, dtype=np.int32) * 3 - 17
    flags_np = np.where(
        (np.arange(n) % 5 == 0) | (np.arange(n) % 7 == 0), -3, 0
    ).astype(np.int32)
    values = ti.ndarray(ti.i32, shape=n)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)
    values.from_numpy(values_np)
    flags.from_numpy(flags_np)
    count.fill(-1)

    prog.cuda_device_compact_ndarray(
        values.arr, flags.arr, output.arr, count.arr, 0
    )

    expected = values_np[flags_np != 0]
    actual_count = int(count.to_numpy()[0])
    assert actual_count == expected.shape[0]
    np.testing.assert_array_equal(output.to_numpy()[:actual_count], expected)


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_vulkan_native_struct_tensor_member_views():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_compact_available")
        and prog.vulkan_compact_available()
    ):
        pytest.skip("Vulkan native compact is unavailable.")
    _run_struct_tensor_member_compact("vulkan_native")


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_field_scan():
    n = 2048
    values = ti.field(ti.i32, shape=n)
    flags = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=n)
    count = ti.field(ti.i32, shape=())

    @ti.kernel
    def fill():
        for i in range(n):
            values[i] = i * 3 - 17
            flags[i] = 1 if i % 5 == 0 or i % 7 == 0 else 0
            output[i] = -1
        count[None] = -1

    fill()
    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    ti.algorithms.experimental_compact(
        values, flags, output, count, method="field_scan", workspace=workspace
    )

    values_np = np.arange(n, dtype=np.int32) * 3 - 17
    flags_np = ((np.arange(n) % 5 == 0) | (np.arange(n) % 7 == 0))
    expected = values_np[flags_np]
    assert count[None] == expected.shape[0]
    assert np.array_equal(output.to_numpy()[: expected.shape[0]], expected)
    if impl.current_cfg().arch == ti.cpu:
        assert workspace.workspace_bytes_peak == 0
    elif impl.current_cfg().arch == ti.cuda:
        assert workspace.workspace_bytes_peak > 0
    else:
        assert workspace.workspace_bytes_peak >= n * 4


@test_utils.test(arch=[ti.cpu])
def test_experimental_compact_cpu_dense_field_native_avoids_helper_policy():
    n = 64
    values = ti.field(ti.i32, shape=n)
    flags = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=n)
    count = ti.field(ti.i32, shape=())

    @ti.kernel
    def fill():
        for i in range(n):
            values[i] = i * 2 + 1
            flags[i] = 1 if i % 3 == 0 else 0
            output[i] = -1
        count[None] = -1

    fill()
    ti.algorithms.clear_legacy_helper_fallback_counts()
    ti.algorithms.set_legacy_helper_auto_fallback_enabled(False)
    try:
        ti.algorithms.experimental_compact(
            values, flags, output, count, method="auto"
        )
        assert ti.algorithms.get_legacy_helper_fallback_counts() == {}

        ti.algorithms.experimental_compact(
            values, flags, output, count, method="field_scan"
        )
        assert ti.algorithms.get_legacy_helper_fallback_counts() == {}

        ti.algorithms.set_legacy_helper_fallback_counting_enabled(True)
        ti.algorithms.experimental_compact(
            values, flags, output, count, method="field_scan"
        )
        counts = ti.algorithms.get_legacy_helper_fallback_counts(reset=True)
        assert counts == {}
    finally:
        ti.algorithms.reset_legacy_helper_auto_fallback_policy()
        ti.algorithms.set_legacy_helper_fallback_counting_enabled(False, clear=True)
        ti.algorithms.clear_legacy_helper_fallback_counts()

    expected = (np.arange(n, dtype=np.int32) * 2 + 1)[np.arange(n) % 3 == 0]
    assert count[None] == expected.shape[0]
    assert np.array_equal(output.to_numpy()[: expected.shape[0]], expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_dense_field_native_f32_no_helper():
    n = 512
    values_np = _compact_values(n, np.float32)
    flags_np = ((np.arange(n) % 4 == 0) | (np.arange(n) % 9 == 0)).astype(
        np.int32
    )
    values = ti.field(ti.f32, shape=n)
    flags = ti.field(ti.i32, shape=n)
    output = ti.field(ti.f32, shape=n)
    count = ti.field(ti.i32, shape=())
    values.from_numpy(values_np)
    flags.from_numpy(flags_np)
    output.fill(np.float32(-999.0))
    count[None] = -1

    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    ti.algorithms.clear_legacy_helper_fallback_counts()
    ti.algorithms.set_legacy_helper_auto_fallback_enabled(False)
    try:
        try:
            ti.algorithms.experimental_compact(
                values, flags, output, count, method="auto", workspace=workspace
            )
        except RuntimeError as exc:
            pytest.skip(str(exc))
        assert ti.algorithms.get_legacy_helper_fallback_counts() == {}
    finally:
        ti.algorithms.reset_legacy_helper_auto_fallback_policy()
        ti.algorithms.clear_legacy_helper_fallback_counts()

    expected = values_np[flags_np != 0]
    assert count[None] == expected.shape[0]
    _assert_compact_matches(output.to_numpy()[: expected.shape[0]], expected)


@test_utils.test(arch=[ti.cuda])
def test_experimental_compact_cuda_field_scan_uses_driver_workspace():
    n = 4096
    values = ti.field(ti.i32, shape=n)
    flags = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=n)
    count = ti.field(ti.i32, shape=())

    prog = impl.get_runtime().prog
    if not prog.cuda_device_compact_available():
        pytest.skip("CUDA Driver compact is unavailable in this build/runtime.")

    @ti.kernel
    def fill():
        for i in range(n):
            values[i] = i - 5
            flags[i] = 1 if i % 4 == 0 else 0
            output[i] = -1
        count[None] = -1

    fill()
    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    ti.algorithms.experimental_compact(
        values, flags, output, count, method="field_scan", workspace=workspace
    )

    expected = (np.arange(n, dtype=np.int32) - 5)[np.arange(n) % 4 == 0]
    assert count[None] == expected.shape[0]
    assert np.array_equal(output.to_numpy()[: expected.shape[0]], expected)
    assert workspace.workspace_bytes_peak > 0
    assert prog.cuda_device_compact_workspace_bytes() > 0
    workspace.clear()
    assert prog.cuda_device_compact_workspace_bytes() == 0
    assert prog.cuda_device_scan_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_field_scan_single_item():
    values = ti.field(ti.i32, shape=1)
    flags = ti.field(ti.i32, shape=1)
    output = ti.field(ti.i32, shape=1)
    count = ti.field(ti.i32, shape=())

    @ti.kernel
    def fill(flag: ti.i32):
        values[0] = 37
        flags[0] = flag
        output[0] = -1
        count[None] = -1

    workspace = ti.algorithms.CompactWorkspace(max_items=1)
    fill(0)
    ti.algorithms.experimental_compact(
        values, flags, output, count, method="field_scan", workspace=workspace
    )
    assert count[None] == 0

    fill(1)
    ti.algorithms.experimental_compact(
        values, flags, output, count, method="field_scan", workspace=workspace
    )
    assert count[None] == 1
    assert output[0] == 37


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_field_scan_empty_and_full_selection():
    n = 257
    values = ti.field(ti.i32, shape=n)
    flags = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=n)
    count = ti.field(ti.i32, shape=())

    @ti.kernel
    def fill(mode: ti.i32):
        for i in range(n):
            values[i] = i * 9 - 41
            flags[i] = 1 if mode == 1 else 0
            output[i] = -1
        count[None] = -1

    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    fill(0)
    ti.algorithms.experimental_compact(
        values, flags, output, count, method="field_scan", workspace=workspace
    )
    assert count[None] == 0

    fill(1)
    ti.algorithms.experimental_compact(
        values, flags, output, count, method="field_scan", workspace=workspace
    )
    expected = np.arange(n, dtype=np.int32) * 9 - 41
    assert count[None] == n
    assert np.array_equal(output.to_numpy(), expected)


@test_utils.test(arch=[ti.cuda])
def test_experimental_compact_cuda_cub_ndarray_supported_dtypes():
    if not impl.get_runtime().prog.cuda_cub_select_available():
        pytest.skip("CUDA CUB select is unavailable in this build/runtime.")

    for dtype, np_dtype in _COMPACT_DTYPES:
        workspace = _run_ndarray_compact(dtype, np_dtype, "auto")
        assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_compact_cuda_cub_ndarray_vector_payload():
    if not impl.get_runtime().prog.cuda_cub_select_available():
        pytest.skip("CUDA CUB select is unavailable in this build/runtime.")

    workspace = _run_vector_ndarray_compact("cuda_cub")
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_compact_cpu_native_ndarray_supported_dtypes():
    if not impl.get_runtime().prog.cpu_compact_available():
        pytest.skip("CPU native compact is unavailable in this build/runtime.")

    for dtype, np_dtype in _COMPACT_DTYPES:
        workspace = _run_ndarray_compact(dtype, np_dtype, "auto")
        assert workspace.workspace_bytes_peak == 0
    assert impl.get_runtime().prog.cpu_compact_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_compact_cpu_native_ndarray_vector_payload():
    workspace = _run_vector_ndarray_compact("cpu_native")
    assert workspace.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_vulkan_native_ndarray_supported_dtypes():
    if not impl.get_runtime().prog.vulkan_compact_available():
        pytest.skip("Vulkan native compact is unavailable in this build/runtime.")

    for dtype, np_dtype in _COMPACT_DTYPES:
        workspace = _run_ndarray_compact(dtype, np_dtype, "auto")
        assert workspace.workspace_bytes_peak > 0
    assert impl.get_runtime().prog.vulkan_compact_workspace_bytes() > 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_vulkan_native_ndarray_vector_payload():
    if not impl.get_runtime().prog.vulkan_compact_available():
        pytest.skip("Vulkan native compact is unavailable in this build/runtime.")

    workspace = _run_vector_ndarray_compact("vulkan_native")
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_vulkan_native_empty_and_full_i32():
    n = 4096
    values = ti.ndarray(ti.i32, shape=n)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)

    if not impl.get_runtime().prog.vulkan_compact_available():
        pytest.skip("Vulkan native compact is unavailable in this build/runtime.")

    @ti.kernel
    def fill(
        values_arr: ti.types.ndarray(ti.i32, ndim=1),
        flags_arr: ti.types.ndarray(ti.i32, ndim=1),
        output_arr: ti.types.ndarray(ti.i32, ndim=1),
        count_arr: ti.types.ndarray(ti.i32, ndim=1),
        mode: ti.i32,
    ):
        for i in range(n):
            values_arr[i] = i * 5 - 23
            if mode == 1:
                flags_arr[i] = 0
            elif mode == 2:
                flags_arr[i] = 1
            else:
                flags_arr[i] = 1 if i % 4 == 0 or i % 11 == 0 else 0
            output_arr[i] = -1
        count_arr[0] = -1

    workspace = ti.algorithms.CompactWorkspace(max_items=n)
    values_np = np.arange(n, dtype=np.int32) * 5 - 23
    for mode in range(3):
        fill(values, flags, output, count, mode)
        ti.algorithms.experimental_compact(
            values, flags, output, count, method="auto", workspace=workspace
        )
        if mode == 1:
            flags_np = np.zeros(n, dtype=bool)
        elif mode == 2:
            flags_np = np.ones(n, dtype=bool)
        else:
            flags_np = ((np.arange(n) % 4 == 0) | (np.arange(n) % 11 == 0))
        expected = values_np[flags_np]
        assert count.to_numpy()[0] == expected.shape[0]
        assert np.array_equal(output.to_numpy()[: expected.shape[0]], expected)
    assert workspace.workspace_bytes_peak > 0
    assert impl.get_runtime().prog.vulkan_compact_workspace_bytes() > 0


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_compact_vulkan_native_reset_with_live_ndarray():
    n = 1024
    values = ti.ndarray(ti.i32, shape=n)
    flags = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)

    if not impl.get_runtime().prog.vulkan_compact_available():
        pytest.skip("Vulkan native compact is unavailable in this build/runtime.")

    @ti.kernel
    def fill(
        values_arr: ti.types.ndarray(ti.i32, ndim=1),
        flags_arr: ti.types.ndarray(ti.i32, ndim=1),
        output_arr: ti.types.ndarray(ti.i32, ndim=1),
        count_arr: ti.types.ndarray(ti.i32, ndim=1),
    ):
        for i in range(n):
            values_arr[i] = i - 31
            flags_arr[i] = 1 if i % 9 == 0 else 0
            output_arr[i] = -1
        count_arr[0] = -1

    fill(values, flags, output, count)
    ti.algorithms.experimental_compact(values, flags, output, count, method="vulkan_native")
    expected = (np.arange(n, dtype=np.int32) - 31)[np.arange(n) % 9 == 0]
    assert count.to_numpy()[0] == expected.shape[0]
    assert np.array_equal(output.to_numpy()[: expected.shape[0]], expected)
    ti.reset()
    del values, flags, output, count
    gc.collect()
