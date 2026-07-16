import gc

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils

_HISTOGRAM_DTYPES = [(ti.i32, np.int32), (ti.u32, np.uint32)]
_HISTOGRAM_BIN_DTYPES = [(ti.i32, np.int32), (ti.i64, np.int64)]
_HISTOGRAM_VALUE_TYPE = {ti.i32: 0, ti.u32: 2}
_HISTOGRAM_BIN_TYPE = {ti.i32: 0, ti.i64: 4}


def _histogram_values(n, num_bins, np_dtype, mode):
    if mode == 2:
        if np.issubdtype(np_dtype, np.unsignedinteger):
            return (np.arange(n, dtype=np.uint64) % (num_bins + 4)).astype(
                np_dtype
            )
        return (np.arange(n, dtype=np.int64) % (num_bins + 4) - 2).astype(
            np_dtype
        )
    if mode == 1:
        return np.full(n, 7, dtype=np_dtype)
    return ((np.arange(n, dtype=np.uint64) * 13 + 9) % num_bins).astype(np_dtype)


def _histogram_expected(values_np, num_bins, bin_np_dtype=np.int32):
    values_i64 = values_np.astype(np.int64)
    valid = values_i64[(0 <= values_i64) & (values_i64 < num_bins)]
    return np.bincount(valid, minlength=num_bins).astype(bin_np_dtype)


def _vulkan_histogram_dtype_available(value_dtype, bin_dtype):
    prog = impl.get_runtime().prog
    if not prog.vulkan_histogram_available():
        return False
    if not hasattr(prog, "vulkan_histogram_value_type_available"):
        return value_dtype == ti.i32 and bin_dtype == ti.i32
    return prog.vulkan_histogram_value_type_available(
        _HISTOGRAM_VALUE_TYPE[value_dtype], _HISTOGRAM_BIN_TYPE[bin_dtype]
    )


def _run_dense_field_histogram(value_dtype, value_np_dtype, bin_dtype, bin_np_dtype, method):
    n = 4096
    num_bins = 64
    values = ti.field(value_dtype, shape=n)
    bins = ti.field(bin_dtype, shape=num_bins)
    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
    for mode in range(3):
        values_np = _histogram_values(n, num_bins, value_np_dtype, mode)
        values.from_numpy(values_np)
        bins.from_numpy(np.full(num_bins, -1, dtype=bin_np_dtype))
        ti.algorithms.experimental_histogram(
            values, bins, method=method, workspace=workspace
        )
        assert np.array_equal(
            bins.to_numpy(), _histogram_expected(values_np, num_bins, bin_np_dtype)
        )
    assert workspace._native_histogram_plan is not None
    plan = workspace._native_histogram_plan
    assert "histogram_dense_field" in plan.method_name
    values_np = _histogram_values(n, num_bins, value_np_dtype, 0)
    values.from_numpy(values_np)
    bins.from_numpy(np.full(num_bins, -1, dtype=bin_np_dtype))
    ti.algorithms.experimental_histogram(
        values, bins, method=method, workspace=workspace
    )
    assert workspace._native_histogram_plan is plan
    assert np.array_equal(
        bins.to_numpy(), _histogram_expected(values_np, num_bins, bin_np_dtype)
    )


def _run_struct_member_histogram(value_dtype, value_np_dtype, bin_dtype, bin_np_dtype, method):
    n = 4096
    num_bins = 64
    value_payload = ti.types.struct(value=value_dtype, tag=ti.i32)
    bin_payload = ti.types.struct(count=bin_dtype, tag=ti.i32)
    values = ti.ndarray(value_payload, shape=n)
    bins = ti.ndarray(bin_payload, shape=num_bins)
    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
    for mode in range(3):
        values_np = _histogram_values(n, num_bins, value_np_dtype, mode)
        values_host = np.zeros((n,), dtype=values.numpy_dtype)
        values_host["value"] = values_np
        values_host["tag"] = np.arange(n, dtype=np.int32) * 7 + 3
        bins_host = np.zeros((num_bins,), dtype=bins.numpy_dtype)
        bins_host["count"] = np.full(num_bins, -1, dtype=bin_np_dtype)
        bins_host["tag"] = np.arange(num_bins, dtype=np.int32) * 11 - 5
        values.from_numpy(values_host)
        bins.from_numpy(bins_host)
        ti.algorithms.experimental_histogram(
            values.field("value"),
            bins.field("count"),
            method=method,
            workspace=workspace,
        )
        result = bins.to_numpy()
        assert np.array_equal(
            result["count"], _histogram_expected(values_np, num_bins, bin_np_dtype)
        )
        np.testing.assert_array_equal(result["tag"], bins_host["tag"])
        np.testing.assert_array_equal(values.to_numpy()["tag"], values_host["tag"])
    assert len(workspace._staged_histogram_plan_groups) >= 1


@test_utils.test(arch=[ti.cpu])
def test_experimental_histogram_rejects_struct_tensor_member_views():
    n = 8
    payload = ti.types.struct(vec=ti.types.vector(2, ti.i32), tag=ti.i32)
    values = ti.ndarray(payload, shape=n)
    bins = ti.ndarray(ti.i32, shape=4)
    with pytest.raises(NotImplementedError, match="scalar quantities"):
        ti.algorithms.experimental_histogram(values.field("vec"), bins)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_histogram_field_atomic():
    n = 4096
    num_bins = 37
    values = ti.field(ti.i32, shape=n)
    bins = ti.field(ti.i32, shape=num_bins)

    @ti.kernel
    def fill(mode: ti.i32):
        for i in range(n):
            if mode == 2:
                values[i] = (i % (num_bins + 4)) - 2
            elif mode == 1:
                values[i] = 3
            else:
                values[i] = (i * 11 + 5) % num_bins
        for i in range(num_bins):
            bins[i] = -1

    for method in ("field_atomic", "field_direct", "field_private"):
        workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
        for mode in range(3):
            fill(mode)
            ti.algorithms.experimental_histogram(
                values, bins, method=method, workspace=workspace
            )
            if mode == 2:
                raw = (np.arange(n) % (num_bins + 4)) - 2
                expected_values = raw[(0 <= raw) & (raw < num_bins)].astype(np.int32)
            elif mode == 1:
                expected_values = np.full(n, 3, dtype=np.int32)
            else:
                expected_values = ((np.arange(n) * 11 + 5) % num_bins).astype(np.int32)
            expected = np.bincount(expected_values, minlength=num_bins).astype(np.int32)
            assert np.array_equal(bins.to_numpy(), expected)
        if method == "field_private":
            assert workspace.workspace_bytes_peak > 0
        else:
            assert workspace.workspace_bytes_peak == 0

@test_utils.test(arch=[ti.cuda])
def test_experimental_histogram_cuda_device_storage_and_dtypes():
    n = 8192
    num_bins = 64
    prog = impl.get_runtime().prog
    if not prog.cuda_device_histogram_available():
        pytest.skip("CUDA Driver histogram is unavailable in this build/runtime.")

    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
    for dtype, np_dtype in _HISTOGRAM_DTYPES:
        for bin_dtype, bin_np_dtype in _HISTOGRAM_BIN_DTYPES:
            values = ti.ndarray(dtype, shape=n)
            bins = ti.ndarray(bin_dtype, shape=num_bins)
            for mode in range(3):
                values_np = _histogram_values(n, num_bins, np_dtype, mode)
                values.from_numpy(values_np)
                ti.algorithms.experimental_histogram(values, bins, method="cuda_device", workspace=workspace)
                assert np.array_equal(
                    bins.to_numpy(),
                    _histogram_expected(values_np, num_bins, bin_np_dtype),
                )
            _run_dense_field_histogram(dtype, np_dtype, bin_dtype, bin_np_dtype, "cuda_device")
            _run_struct_member_histogram(dtype, np_dtype, bin_dtype, bin_np_dtype, "cuda_device")
    assert workspace._cuda_device_active
    assert workspace._native_histogram_plan.backend == "cuda_device"


@test_utils.test(arch=[ti.cuda])
def test_experimental_histogram_cuda_cub_ndarray():
    n = 8192
    num_bins = 64

    if not impl.get_runtime().prog.cuda_cub_histogram_available():
        pytest.skip("CUDA CUB histogram is unavailable in this build/runtime.")

    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
    for dtype, np_dtype in _HISTOGRAM_DTYPES:
        for bin_dtype, bin_np_dtype in _HISTOGRAM_BIN_DTYPES:
            values = ti.ndarray(dtype, shape=n)
            bins = ti.ndarray(bin_dtype, shape=num_bins)
            for mode in range(2):
                values_np = _histogram_values(n, num_bins, np_dtype, mode)
                values.from_numpy(values_np)
                bins.from_numpy(np.full(num_bins, -1, dtype=bin_np_dtype))
                ti.algorithms.experimental_histogram(values, bins, method="cuda_cub", workspace=workspace)
                assert np.array_equal(
                    bins.to_numpy(),
                    _histogram_expected(values_np, num_bins, bin_np_dtype),
                )
                bins.from_numpy(np.full(num_bins, -1, dtype=bin_np_dtype))
                ti.algorithms.experimental_histogram(values, bins, method="two_level", workspace=workspace)
                assert np.array_equal(
                    bins.to_numpy(),
                    _histogram_expected(values_np, num_bins, bin_np_dtype),
                )
            _run_dense_field_histogram(dtype, np_dtype, bin_dtype, bin_np_dtype, "cuda_two_level")
            _run_struct_member_histogram(dtype, np_dtype, bin_dtype, bin_np_dtype, "cuda_two_level")
    assert workspace._cuda_cub_active
    assert workspace._native_histogram_plan is not None


@test_utils.test(arch=[ti.cpu])
def test_experimental_histogram_cpu_native_ndarray():
    n = 8192
    num_bins = 64

    if not impl.get_runtime().prog.cpu_histogram_available():
        pytest.skip("CPU native histogram is unavailable in this build/runtime.")

    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
    for dtype, np_dtype in _HISTOGRAM_DTYPES:
        for bin_dtype, bin_np_dtype in _HISTOGRAM_BIN_DTYPES:
            values = ti.ndarray(dtype, shape=n)
            bins = ti.ndarray(bin_dtype, shape=num_bins)
            for mode in range(3):
                values_np = _histogram_values(n, num_bins, np_dtype, mode)
                values.from_numpy(values_np)
                bins.from_numpy(np.full(num_bins, -1, dtype=bin_np_dtype))
                ti.algorithms.experimental_histogram(
                    values, bins, method="auto", workspace=workspace
                )
                assert np.array_equal(
                    bins.to_numpy(),
                    _histogram_expected(values_np, num_bins, bin_np_dtype),
                )
                bins.from_numpy(np.full(num_bins, -1, dtype=bin_np_dtype))
                ti.algorithms.experimental_histogram(
                    values, bins, method="two_level", workspace=workspace
                )
                assert np.array_equal(
                    bins.to_numpy(),
                    _histogram_expected(values_np, num_bins, bin_np_dtype),
                )
            _run_dense_field_histogram(
                dtype, np_dtype, bin_dtype, bin_np_dtype, "cpu_two_level"
            )
            _run_struct_member_histogram(
                dtype, np_dtype, bin_dtype, bin_np_dtype, "cpu_two_level"
            )
    assert workspace.workspace_bytes_peak == 0
    assert impl.get_runtime().prog.cpu_histogram_workspace_bytes() == 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_histogram_vulkan_native_ndarray():
    n = 8192
    num_bins = 64

    if not impl.get_runtime().prog.vulkan_histogram_available():
        pytest.skip("Vulkan native histogram is unavailable in this build/runtime.")

    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)
    for dtype, np_dtype in _HISTOGRAM_DTYPES:
        for bin_dtype, bin_np_dtype in _HISTOGRAM_BIN_DTYPES:
            if not _vulkan_histogram_dtype_available(dtype, bin_dtype):
                continue
            values = ti.ndarray(dtype, shape=n)
            bins = ti.ndarray(bin_dtype, shape=num_bins)
            for mode in range(3):
                values_np = _histogram_values(n, num_bins, np_dtype, mode)
                values.from_numpy(values_np)
                bins.from_numpy(np.full(num_bins, -1, dtype=bin_np_dtype))
                ti.algorithms.experimental_histogram(
                    values, bins, method="auto", workspace=workspace
                )
                assert np.array_equal(
                    bins.to_numpy(),
                    _histogram_expected(values_np, num_bins, bin_np_dtype),
                )
                bins.from_numpy(np.full(num_bins, -1, dtype=bin_np_dtype))
                ti.algorithms.experimental_histogram(
                    values, bins, method="two_level", workspace=workspace
                )
                assert np.array_equal(
                    bins.to_numpy(),
                    _histogram_expected(values_np, num_bins, bin_np_dtype),
                )
            _run_dense_field_histogram(
                dtype, np_dtype, bin_dtype, bin_np_dtype, "vulkan_two_level"
            )
            _run_struct_member_histogram(
                dtype, np_dtype, bin_dtype, bin_np_dtype, "vulkan_two_level"
            )
    assert impl.get_runtime().prog.vulkan_histogram_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_histogram_ndarray_rejects_float_values():
    n = 128
    values = ti.ndarray(ti.f32, shape=n)
    bins = ti.ndarray(ti.i32, shape=8)
    with pytest.raises(TypeError, match="i32/ti.u32 values"):
        ti.algorithms.experimental_histogram(values, bins, method="cpu_native")


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_histogram_vulkan_native_i64_bins_capability():
    n = 128
    num_bins = 8
    values = ti.ndarray(ti.u32, shape=n)
    bins = ti.ndarray(ti.i64, shape=num_bins)

    if not impl.get_runtime().prog.vulkan_histogram_available():
        pytest.skip("Vulkan native histogram is unavailable in this build/runtime.")

    if not _vulkan_histogram_dtype_available(ti.u32, ti.i64):
        with pytest.raises(RuntimeError, match="vulkan_native"):
            ti.algorithms.experimental_histogram(values, bins, method="vulkan_native")
        return

    values_np = _histogram_values(n, num_bins, np.uint32, 2)
    values.from_numpy(values_np)
    bins.from_numpy(np.full(num_bins, -1, dtype=np.int64))
    ti.algorithms.experimental_histogram(values, bins, method="vulkan_native")
    assert np.array_equal(
        bins.to_numpy(), _histogram_expected(values_np, num_bins, np.int64)
    )


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_histogram_vulkan_native_resource_replay_ring_wrap(monkeypatch):
    monkeypatch.setenv("TI_VULKAN_RESOURCE_REPLAY_RING_SIZE", "2")
    n = 8192
    num_bins = 64
    values = ti.ndarray(ti.i32, shape=n)
    bins = ti.ndarray(ti.i32, shape=num_bins)

    if not impl.get_runtime().prog.vulkan_histogram_available():
        pytest.skip("Vulkan native histogram is unavailable in this build/runtime.")

    workspace = ti.algorithms.HistogramWorkspace(max_items=n, max_bins=num_bins)

    # Warm allocation/pipeline creation before observing ring wraps. Rebinding
    # a pinned resource-set wrapper must allocate/reuse the exact descriptor
    # set through the RHI cache, not force a Program/device synchronization.
    values.from_numpy(_histogram_values(n, num_bins, np.int32, 0))
    bins.fill(0)
    ti.algorithms.experimental_histogram(
        values, bins, method="vulkan_native", workspace=workspace
    )
    ti.sync()
    waits_before = impl.get_runtime().prog._runtime_statistics_snapshot()[
        "synchronization"
    ]["backend_waits"]
    for _ in range(8):
        ti.algorithms.experimental_histogram(
            values, bins, method="vulkan_native", workspace=workspace
        )
    waits_after = impl.get_runtime().prog._runtime_statistics_snapshot()[
        "synchronization"
    ]["backend_waits"]
    if waits_before is not None and waits_after is not None:
        assert waits_after == waits_before

    for step in range(6):
        values_np = _histogram_values(n, num_bins, np.int32, step % 3)
        values.from_numpy(values_np)
        bins.from_numpy(np.full(num_bins, -1, dtype=np.int32))
        ti.algorithms.experimental_histogram(
            values, bins, method="vulkan_native", workspace=workspace
        )
        assert np.array_equal(
            bins.to_numpy(), _histogram_expected(values_np, num_bins, np.int32)
        )

    alt_values = ti.ndarray(ti.i32, shape=n)
    alt_bins = ti.ndarray(ti.i32, shape=num_bins)
    for step in range(6):
        active_values = values if step % 2 == 0 else alt_values
        active_bins = bins if step % 2 == 0 else alt_bins
        values_np = _histogram_values(n, num_bins, np.int32, (step + 1) % 3)
        active_values.from_numpy(values_np)
        active_bins.from_numpy(np.full(num_bins, -1, dtype=np.int32))
        ti.algorithms.experimental_histogram(
            active_values, active_bins, method="vulkan_native", workspace=workspace
        )
        assert np.array_equal(
            active_bins.to_numpy(),
            _histogram_expected(values_np, num_bins, np.int32),
        )


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_histogram_vulkan_native_reset_with_live_ndarray():
    n = 1024
    num_bins = 64
    values = ti.ndarray(ti.i32, shape=n)
    bins = ti.ndarray(ti.i32, shape=num_bins)

    if not impl.get_runtime().prog.vulkan_histogram_available():
        pytest.skip("Vulkan native histogram is unavailable in this build/runtime.")

    @ti.kernel
    def fill(
        values_arr: ti.types.ndarray(ti.i32, ndim=1),
        bins_arr: ti.types.ndarray(ti.i32, ndim=1),
    ):
        for i in range(n):
            values_arr[i] = (i * 7 + 1) % num_bins
        for i in range(num_bins):
            bins_arr[i] = -1

    fill(values, bins)
    ti.algorithms.experimental_histogram(values, bins, method="vulkan_native")
    expected_values = ((np.arange(n) * 7 + 1) % num_bins).astype(np.int32)
    expected = np.bincount(expected_values, minlength=num_bins).astype(np.int32)
    assert np.array_equal(bins.to_numpy(), expected)
    ti.reset()
    del values, bins
    gc.collect()
