import numpy as np

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from tests import test_utils


def _allocation_calls():
    return ti_core.query_int64("cuda_async_allocation_calls") + ti_core.query_int64(
        "cuda_sync_allocation_fallback_calls"
    )


def _free_calls():
    return ti_core.query_int64("cuda_async_free_calls") + ti_core.query_int64(
        "cuda_sync_free_fallback_calls"
    )


def _jit_snapshot():
    return {
        key: int(ti_core.query_int64(key))
        for key in (
            "cuda_jit_module_load_calls",
            "cuda_jit_ptx_bytes",
            "cuda_jit_host_wall_ns",
            "cuda_jit_driver_wall_us",
            "cuda_jit_diagnostic_loads",
            "cuda_jit_info_log_bytes",
            "cuda_jit_error_log_bytes",
        )
    }


def _retained_launch_snapshot():
    return {
        key: int(ti_core.query_int64(key))
        for key in (
            "cuda_retained_launch_current_bytes",
            "cuda_retained_launch_peak_bytes",
            "cuda_retained_launch_allocation_calls",
            "cuda_retained_launch_release_calls",
        )
    }


@test_utils.test(arch=ti.cuda)
def test_cuda_jit_diagnostics_are_opt_in_and_accounted(monkeypatch):
    monkeypatch.setenv("TI_CUDA_JIT_DIAGNOSTICS", "1")
    before = _jit_snapshot()

    values = ti.field(dtype=ti.i32, shape=16)

    @ti.kernel
    def diagnostic_kernel(scale: ti.i32):
        for i in values:
            values[i] = i * scale + 3

    with ti.compile_profile() as profile:
        diagnostic_kernel(7)
        ti.sync()
    after = _jit_snapshot()

    assert after["cuda_jit_module_load_calls"] > before["cuda_jit_module_load_calls"]
    assert after["cuda_jit_ptx_bytes"] > before["cuda_jit_ptx_bytes"]
    assert after["cuda_jit_host_wall_ns"] > before["cuda_jit_host_wall_ns"]
    assert after["cuda_jit_diagnostic_loads"] > before["cuda_jit_diagnostic_loads"]
    # Driver wall time and info-log length may legitimately be zero on a
    # cache hit. Error logs must remain empty for a successful load.
    assert after["cuda_jit_driver_wall_us"] >= before["cuda_jit_driver_wall_us"]
    assert after["cuda_jit_info_log_bytes"] >= before["cuda_jit_info_log_bytes"]
    assert after["cuda_jit_error_log_bytes"] == before["cuda_jit_error_log_bytes"]
    profile_paths = [row["path"] for row in profile.records(include_python=False)]
    assert any("cuda_driver_module_load" in path for path in profile_paths)
    assert any("cuda_driver_function_lookup" in path for path in profile_paths)


@test_utils.test(arch=ti.cuda)
def test_cuda_void_field_launch_avoids_temporary_result_buffer():
    counter = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def step():
        ti.atomic_add(counter[None], 1)

    @ti.kernel
    def answer() -> ti.i32:
        return 7

    @ti.kernel
    def increment_host_array(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    # Compile and materialize before the allocation baseline. A void kernel
    # that only touches Taichi fields no longer needs a device result buffer.
    step()
    ti.sync()
    counter[None] = 0
    ti.sync()
    allocations_before = _allocation_calls()
    frees_before = _free_calls()
    for _ in range(128):
        step()
    ti.sync()
    assert _allocation_calls() == allocations_before
    assert _free_calls() == frees_before
    assert counter[None] == 128

    # Results and host arrays still need the lazy result channel and retain
    # their old behavior.
    assert answer() == 7
    values = np.zeros(32, dtype=np.int32)
    increment_host_array(values)
    ti.sync()
    np.testing.assert_array_equal(values, np.ones(32, dtype=np.int32))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_scalar_launch_reuses_retained_argument_and_result_buffers():
    counter = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def add(amount: ti.i32):
        ti.atomic_add(counter[None], amount)

    @ti.kernel
    def answer(scale: ti.i32) -> ti.i32:
        return counter[None] * scale + 5

    # Warm every supported retained argument slot before taking the allocation
    # baseline. The default ring may be tuned within the bounded internal cap.
    for _ in range(8):
        add(1)
    assert answer(2) == 21
    ti.sync()
    allocations_before = _allocation_calls()
    frees_before = _free_calls()
    retained_before = _retained_launch_snapshot()

    for _ in range(128):
        add(1)
    for _ in range(16):
        assert answer(2) >= 7
    ti.sync()

    retained_after = _retained_launch_snapshot()
    assert _allocation_calls() == allocations_before
    assert _free_calls() == frees_before
    assert retained_after["cuda_retained_launch_current_bytes"] > 0
    assert (
        retained_after["cuda_retained_launch_peak_bytes"]
        >= retained_after["cuda_retained_launch_current_bytes"]
    )
    assert (
        retained_after["cuda_retained_launch_allocation_calls"]
        == retained_before["cuda_retained_launch_allocation_calls"]
    )
    assert counter[None] == 136


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_retained_launch_buffers_release_with_snode_tree():
    baseline = _retained_launch_snapshot()
    builder = ti.FieldsBuilder()
    values = ti.field(dtype=ti.i32)
    builder.dense(ti.i, 32).place(values)
    tree = builder.finalize()

    @ti.kernel
    def update(amount: ti.i32):
        for i in values:
            values[i] += amount

    update(3)
    ti.sync()
    materialized = _retained_launch_snapshot()
    assert (
        materialized["cuda_retained_launch_current_bytes"]
        > baseline["cuda_retained_launch_current_bytes"]
    )

    tree.destroy()
    ti.sync()
    retired = _retained_launch_snapshot()
    assert (
        retired["cuda_retained_launch_current_bytes"]
        == baseline["cuda_retained_launch_current_bytes"]
    )
    assert (
        retired["cuda_retained_launch_release_calls"]
        > materialized["cuda_retained_launch_release_calls"]
    )
