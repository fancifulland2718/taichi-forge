import gc

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from taichi_forge.lang import impl


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_dlpack_view_binds_numpy_storage_without_copy():
    @ti.kernel
    def add_bias(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in values:
            values[i] += 2.5

    values = np.arange(16, dtype=np.float32)
    program = impl.get_runtime().prog
    resources_before = program._debug_external_dense_storage_stats()
    bindings_before = program._debug_dense_storage_binding_stats()

    view = ti.interop.from_dlpack(values)
    assert view.provider == "dlpack"
    assert view.device == (1, 0)
    assert view.descriptor.owner_kind == "kExternalManaged"
    assert view.description.properties["compact_contiguous"]
    assert view.allocation_bytes == values.nbytes
    assert not view.closed

    resources_imported = program._debug_external_dense_storage_stats()
    assert resources_imported["live"] == resources_before["live"] + 1
    assert resources_imported["created_total"] == (
        resources_before["created_total"] + 1
    )

    add_bias(view)
    np.testing.assert_allclose(
        values, np.arange(16, dtype=np.float32) + 2.5
    )
    bindings_after = program._debug_dense_storage_binding_stats()
    assert bindings_after["external_bindings"] == (
        bindings_before["external_bindings"] + 1
    )
    assert bindings_after["temporary_allocations"] == 0
    assert bindings_after["temporary_bytes"] == 0

    view.close()
    assert view.closed
    with pytest.raises(RuntimeError, match="stale or retired"):
        add_bias(view)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_historical_numpy_kernel_argument_uses_managed_storage_internally():
    @ti.kernel
    def increment(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += i + 1

    values = np.zeros(32, dtype=np.int32)
    program = impl.get_runtime().prog
    resources_before = program._debug_external_dense_storage_stats()
    bindings_before = program._debug_dense_storage_binding_stats()

    increment(values)
    gc.collect()

    np.testing.assert_array_equal(
        values, np.arange(32, dtype=np.int32) + 1
    )
    resources_after = program._debug_external_dense_storage_stats()
    bindings_after = program._debug_dense_storage_binding_stats()
    assert resources_after["created_total"] == (
        resources_before["created_total"] + 1
    )
    assert resources_after["released_total"] == (
        resources_before["released_total"] + 1
    )
    assert resources_after["live"] == resources_before["live"]
    assert bindings_after["external_bindings"] == (
        bindings_before["external_bindings"] + 1
    )
    assert bindings_after["temporary_allocations"] == 0
    assert bindings_after["temporary_bytes"] == 0


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_historical_fortran_numpy_argument_keeps_copy_fallback():
    @ti.kernel
    def increment(values: ti.types.ndarray(dtype=ti.i32, ndim=2)):
        for i, j in values:
            values[i, j] += i * 10 + j

    values = np.asfortranarray(np.zeros((4, 3), dtype=np.int32))
    program = impl.get_runtime().prog
    resources_before = program._debug_external_dense_storage_stats()

    increment(values)

    expected = np.fromfunction(
        lambda i, j: i * 10 + j, (4, 3), dtype=np.int32
    ).astype(np.int32)
    np.testing.assert_array_equal(values, expected)
    resources_after = program._debug_external_dense_storage_stats()
    assert resources_after["created_total"] == (
        resources_before["created_total"] + 1
    )
    assert resources_after["released_total"] == (
        resources_before["released_total"] + 1
    )


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_dlpack_view_rejects_pointer_only_vulkan_import():
    values = np.zeros(4, dtype=np.float32)
    with pytest.raises(
        BufferError, match="cannot import DLPack storage without copying"
    ):
        ti.interop.from_dlpack(values)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_dlpack_capabilities_require_initialized_runtime():
    # This test documents the public query shape without depending on a
    # particular optional GPU backend.
    capabilities = ti.interop.capabilities()
    assert capabilities["schema_version"] == 1
    assert capabilities["provider"] == "dlpack"
    assert capabilities["zero_copy"]
    assert capabilities["copy_fallback"] is False
    assert capabilities["devices"] == ("cpu", "cuda_host")
