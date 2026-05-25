import gc

import numpy as np
import pytest

import taichi_forge as ti
import taichi_forge.algorithms._algorithms as alg_impl
from taichi_forge.lang import impl
from tests import test_utils


def _source_values(n, np_dtype):
    if np.issubdtype(np_dtype, np.floating):
        return (
            ((np.arange(n, dtype=np.float64) % 67) - 33) * np.float64(0.25)
        ).astype(np_dtype)
    if np.issubdtype(np_dtype, np.unsignedinteger):
        return ((np.arange(n, dtype=np.uint64) * 3 + 7) % 4294967291).astype(
            np_dtype
        )
    return (np.arange(n, dtype=np.int32) % 97 - 48).astype(np_dtype)


def _reverse_indices(n):
    return (n - 1 - np.arange(n, dtype=np.int32)).astype(np.int32)


def _assert_matches(actual, expected):
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    else:
        assert np.array_equal(actual, expected)


_INDEXED_COPY_DTYPES = [
    (ti.u32, np.uint32),
    (ti.i32, np.int32),
    (ti.f32, np.float32),
    (ti.u64, np.uint64),
    (ti.i64, np.int64),
    (ti.f64, np.float64),
]


def _run_ndarray_indexed_copy(dtype, np_dtype, method, scatter):
    n = 4096
    src = ti.ndarray(dtype, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(dtype, shape=n)
    data = _source_values(n, np_dtype).astype(np_dtype)
    index_data = _reverse_indices(n)
    src.from_numpy(data)
    indices.from_numpy(index_data)
    dst.fill(0)
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=n)

    if scatter:
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=method, workspace=workspace
        )
    else:
        ti.algorithms.experimental_gather(
            src, indices, dst, method=method, workspace=workspace
        )

    _assert_matches(dst.to_numpy(), data[index_data])
    return workspace


def _run_vector_ndarray_indexed_copy(method, scatter):
    n = 2048
    src = ti.Vector.ndarray(3, ti.f32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.Vector.ndarray(3, ti.f32, shape=n)
    data = (
        ((np.arange(n * 3, dtype=np.float32).reshape(n, 3) % 67) - 33)
        * np.float32(0.25)
    ).astype(np.float32)
    index_data = _reverse_indices(n)
    src.from_numpy(data)
    indices.from_numpy(index_data)
    dst.fill(0)
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=n)

    if scatter:
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=method, workspace=workspace
        )
    else:
        ti.algorithms.experimental_gather(
            src, indices, dst, method=method, workspace=workspace
        )

    np.testing.assert_allclose(dst.to_numpy(), data[index_data], rtol=1e-6, atol=1e-6)
    assert workspace.workspace_bytes_peak == 0
    return workspace


def _run_matrix_ndarray_indexed_copy(method, scatter):
    n = 1024
    src = ti.Matrix.ndarray(2, 2, ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.Matrix.ndarray(2, 2, ti.i32, shape=n)
    data = (np.arange(n * 4, dtype=np.int32).reshape(n, 2, 2) % 97) - 48
    index_data = _reverse_indices(n)
    src.from_numpy(data)
    indices.from_numpy(index_data)
    dst.fill(0)
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=n)

    if scatter:
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=method, workspace=workspace
        )
    else:
        ti.algorithms.experimental_gather(
            src, indices, dst, method=method, workspace=workspace
        )

    assert np.array_equal(dst.to_numpy(), data[index_data])
    assert workspace.workspace_bytes_peak == 0
    return workspace


def _run_dense_field_indexed_copy(dtype, np_dtype, method, scatter):
    n = 2048
    src = ti.field(dtype, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.field(dtype, shape=n)
    data = _source_values(n, np_dtype).astype(np_dtype)
    index_data = _reverse_indices(n)
    src.from_numpy(data)
    indices.from_numpy(index_data)
    dst.fill(0)
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=n)

    if scatter:
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=method, workspace=workspace
        )
    else:
        ti.algorithms.experimental_gather(
            src, indices, dst, method=method, workspace=workspace
        )

    _assert_matches(dst.to_numpy(), data[index_data])
    return workspace


def _native_indexed_copy_method_for_current_arch(scatter):
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.cpu:
        method_name = "cpu_scatter_dense_field" if scatter else "cpu_gather_dense_field"
        if not hasattr(prog, method_name):
            pytest.skip("CPU dense field indexed-copy is unavailable.")
        return "cpu_native", method_name
    if arch == ti.cuda:
        method_name = (
            "cuda_device_scatter_dense_field"
            if scatter
            else "cuda_device_gather_dense_field"
        )
        if not (
            hasattr(prog, method_name)
            and hasattr(prog, "cuda_device_indexed_copy_payload_available")
            and prog.cuda_device_indexed_copy_payload_available(4)
        ):
            pytest.skip("CUDA dense field indexed-copy is unavailable.")
        return "cuda_device", method_name
    if arch == ti.vulkan:
        method_name = "vulkan_scatter_dense_field" if scatter else "vulkan_gather_dense_field"
        if not (
            hasattr(prog, method_name)
            and hasattr(prog, "vulkan_indexed_copy_available")
            and prog.vulkan_indexed_copy_available()
        ):
            pytest.skip("Vulkan dense field indexed-copy is unavailable.")
        return "vulkan_native", method_name
    pytest.skip("native indexed-copy is unavailable on this arch.")


def _run_dense_matrix_field_indexed_copy(scatter):
    method, method_name = _native_indexed_copy_method_for_current_arch(scatter)
    n = 64
    m = 17
    src_shape = m if scatter else n
    dst_shape = n if scatter else m
    src = ti.Vector.field(2, ti.i32, shape=src_shape)
    dst = ti.Vector.field(2, ti.i32, shape=dst_shape)
    indices = ti.ndarray(ti.i32, shape=m)
    data = (np.arange(src_shape * 2, dtype=np.int32).reshape(src_shape, 2) % 97) - 48
    index_data = ((np.arange(m, dtype=np.int32) * 7 + 3) % n).astype(np.int32)
    src.from_numpy(data)
    indices.from_numpy(index_data)
    dst.fill(0)
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=m)

    if scatter:
        expected = np.zeros((dst_shape, 2), dtype=np.int32)
        expected[index_data] = data
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=method, workspace=workspace
        )
    else:
        expected = data[index_data]
        ti.algorithms.experimental_gather(
            src, indices, dst, method=method, workspace=workspace
        )

    np.testing.assert_array_equal(dst.to_numpy(), expected)
    assert len(workspace._native_indexed_copy_plans) == 2
    assert workspace._native_indexed_copy_plan.method_name == method_name
    assert workspace.workspace_bytes_peak <= 64
    assert len(workspace._native_indexed_copy_plan_groups) == 1

    dst.fill(0)
    if scatter:
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=method, workspace=workspace
        )
    else:
        ti.algorithms.experimental_gather(
            src, indices, dst, method=method, workspace=workspace
        )
    np.testing.assert_array_equal(dst.to_numpy(), expected)
    assert len(workspace._native_indexed_copy_plan_groups) == 1


def _run_dense_field_indexed_copy_replay(
    dtype, np_dtype, method, scatter, backend, method_name
):
    n = 2048
    src = ti.field(dtype, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.field(dtype, shape=n)
    index_data = _reverse_indices(n)
    data = _source_values(n, np_dtype).astype(np_dtype)
    indices.from_numpy(index_data)
    src.from_numpy(data)
    dst.fill(0)
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=n)

    if scatter:
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=method, workspace=workspace
        )
    else:
        ti.algorithms.experimental_gather(
            src, indices, dst, method=method, workspace=workspace
        )
    first_plan = workspace._native_indexed_copy_plan
    assert first_plan is not None
    assert first_plan.backend == backend
    assert first_plan.method_name == method_name

    data = (data * np_dtype(2) + np_dtype(3)).astype(np_dtype)
    src.from_numpy(data)
    dst.fill(0)
    if scatter:
        ti.algorithms.experimental_scatter(
            src, indices, dst, method=method, workspace=workspace
        )
    else:
        ti.algorithms.experimental_gather(
            src, indices, dst, method=method, workspace=workspace
        )

    assert workspace._native_indexed_copy_plan is first_plan
    _assert_matches(dst.to_numpy(), data[index_data])
    return workspace


def _run_struct_tensor_member_indexed_copy(method, scatter):
    n = 2048
    payload = ti.types.struct(
        vec=ti.types.vector(2, ti.i32),
        mat=ti.types.matrix(2, 2, ti.i32),
        tag=ti.i32,
    )
    src = ti.ndarray(payload, shape=n)
    dst = ti.ndarray(payload, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    index_data = _reverse_indices(n)
    host = np.zeros((n,), dtype=src.numpy_dtype)
    dst_host = np.zeros((n,), dtype=dst.numpy_dtype)
    host["vec"] = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 97) - 48
    host["mat"] = (np.arange(n * 4, dtype=np.int32).reshape(n, 2, 2) % 101) - 50
    host["tag"] = np.arange(n, dtype=np.int32) * 3 + 1
    dst_host["tag"] = np.arange(n, dtype=np.int32) * 7 + 5
    src.from_numpy(host)
    dst.from_numpy(dst_host)
    indices.from_numpy(index_data)
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=n)

    if scatter:
        ti.algorithms.experimental_scatter(
            src.field("vec"), indices, dst.field("vec"), method=method, workspace=workspace
        )
        ti.algorithms.experimental_scatter(
            src.field("mat"), indices, dst.field("mat"), method=method, workspace=workspace
        )
    else:
        ti.algorithms.experimental_gather(
            src.field("vec"), indices, dst.field("vec"), method=method, workspace=workspace
        )
        ti.algorithms.experimental_gather(
            src.field("mat"), indices, dst.field("mat"), method=method, workspace=workspace
        )

    result = dst.to_numpy()
    assert np.array_equal(result["vec"], host["vec"][index_data])
    assert np.array_equal(result["mat"], host["mat"][index_data])
    assert np.array_equal(result["tag"], dst_host["tag"])
    assert np.array_equal(src.to_numpy()["tag"], host["tag"])
    return workspace


def _make_struct_scalar_indexed_copy_case(n=256):
    payload = ti.types.struct(val=ti.i32, tag=ti.i32)
    src = ti.ndarray(payload, shape=n)
    dst = ti.ndarray(payload, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    index_data = _reverse_indices(n)
    host = np.zeros((n,), dtype=src.numpy_dtype)
    dst_host = np.zeros((n,), dtype=dst.numpy_dtype)
    host["val"] = (np.arange(n, dtype=np.int32) * 5 + 3) % 127 - 63
    host["tag"] = np.arange(n, dtype=np.int32) * 11 + 7
    dst_host["tag"] = np.arange(n, dtype=np.int32) * 13 + 9
    src.from_numpy(host)
    dst.from_numpy(dst_host)
    indices.from_numpy(index_data)
    return src, dst, indices, index_data, host, dst_host


def _make_struct_tensor_indexed_copy_case(n=256):
    payload = ti.types.struct(
        vec=ti.types.vector(2, ti.i32),
        tag=ti.i32,
    )
    src = ti.ndarray(payload, shape=n)
    dst = ti.ndarray(payload, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    index_data = _reverse_indices(n)
    host = np.zeros((n,), dtype=src.numpy_dtype)
    dst_host = np.zeros((n,), dtype=dst.numpy_dtype)
    host["vec"] = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 97) - 48
    host["tag"] = np.arange(n, dtype=np.int32) * 17 + 3
    dst_host["tag"] = np.arange(n, dtype=np.int32) * 19 + 5
    src.from_numpy(host)
    dst.from_numpy(dst_host)
    indices.from_numpy(index_data)
    return src, dst, indices, index_data, host, dst_host


@test_utils.test(arch=ti.cpu)
def test_primitive_descriptor_key_stable_for_rebuilt_dense_views():
    n = 16
    payload = ti.types.struct(
        val=ti.i32,
        vec=ti.types.vector(2, ti.i32),
    )
    struct_arr = ti.ndarray(payload, shape=n)

    scalar_a = struct_arr.field("val")
    scalar_b = struct_arr.field("val")
    assert scalar_a is not scalar_b
    assert alg_impl._primitive_plan_object_key(
        scalar_a
    ) == alg_impl._primitive_plan_object_key(scalar_b)

    tensor_a = struct_arr.field("vec")
    tensor_b = struct_arr.field("vec")
    assert tensor_a is not tensor_b
    assert alg_impl._primitive_plan_object_key(
        tensor_a
    ) == alg_impl._primitive_plan_object_key(tensor_b)

    component_a = struct_arr.field("vec", component=0)
    component_b = struct_arr.field("vec", component=0)
    assert component_a is not component_b
    assert alg_impl._primitive_plan_object_key(
        component_a
    ) == alg_impl._primitive_plan_object_key(component_b)

    matrix_field = ti.Vector.field(2, ti.i32, shape=n)
    field_component_a = matrix_field.get_scalar_field(0)
    field_component_b = matrix_field.get_scalar_field(0)
    assert alg_impl._primitive_plan_object_key(
        field_component_a
    ) == alg_impl._primitive_plan_object_key(field_component_b)


def _run_invalid_index_ndarray(method, scatter):
    src = ti.ndarray(ti.i32, shape=4)
    indices = ti.ndarray(ti.i32, shape=4)
    dst = ti.ndarray(ti.i32, shape=4)
    data = np.array([10, 20, 30, 40], dtype=np.int32)
    index_data = np.array([3, -1, 99, 0], dtype=np.int32)
    src.from_numpy(data)
    indices.from_numpy(index_data)
    dst.from_numpy(np.full(4, -7, dtype=np.int32))

    if scatter:
        ti.algorithms.experimental_scatter(src, indices, dst, method=method)
        out = dst.to_numpy()
        assert out[0] == 40
        assert out[3] == 10
        return
    else:
        ti.algorithms.experimental_gather(src, indices, dst, method=method)
        expected = np.array([40, 0, 0, 10], dtype=np.int32)
    assert np.array_equal(dst.to_numpy(), expected)


@test_utils.test(arch=ti.cpu)
def test_experimental_gather_cpu_native_struct_scalar_member_workspace_replay():
    src, dst, indices, index_data, host, dst_host = _make_struct_scalar_indexed_copy_case()
    src_val = src.field("val")
    dst_val = dst.field("val")
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=src.shape[0])

    ti.algorithms.experimental_gather(
        src_val, indices, dst_val, method="cpu_native", workspace=workspace
    )
    first_plan = workspace._native_indexed_copy_plan
    assert first_plan is not None
    assert first_plan.backend == "cpu_native"
    assert first_plan.method_name == "cpu_gather_strided_ndarray"
    assert first_plan.object_keys is not None

    host["val"] = host["val"] * np.int32(-2) + np.int32(5)
    dst_host["val"] = np.full(src.shape[0], -1000, dtype=np.int32)
    src.from_numpy(host)
    dst.from_numpy(dst_host)
    ti.algorithms.experimental_gather(
        src.field("val"),
        indices,
        dst.field("val"),
        method="cpu_native",
        workspace=workspace,
    )

    assert workspace._native_indexed_copy_plan is first_plan
    result = dst.to_numpy()
    assert np.array_equal(result["val"], host["val"][index_data])
    assert np.array_equal(result["tag"], dst_host["tag"])


@test_utils.test(arch=ti.cpu)
def test_experimental_indexed_copy_cpu_native_struct_member_multi_plan_cache():
    src, dst, indices, index_data, host, dst_host = _make_struct_scalar_indexed_copy_case()
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=src.shape[0])

    ti.algorithms.experimental_gather(
        src.field("val"),
        indices,
        dst.field("val"),
        method="cpu_native",
        workspace=workspace,
    )
    gather_plan = workspace._native_indexed_copy_plan

    dst.from_numpy(dst_host)
    ti.algorithms.experimental_scatter(
        src.field("val"),
        indices,
        dst.field("val"),
        method="cpu_native",
        workspace=workspace,
    )
    scatter_plan = workspace._native_indexed_copy_plan
    assert scatter_plan is not gather_plan
    assert len(workspace._native_indexed_copy_plans) == 2

    dst.from_numpy(dst_host)
    ti.algorithms.experimental_gather(
        src.field("val"),
        indices,
        dst.field("val"),
        method="cpu_native",
        workspace=workspace,
    )
    assert workspace._native_indexed_copy_plan is gather_plan
    result = dst.to_numpy()
    assert np.array_equal(result["val"], host["val"][index_data])
    assert np.array_equal(result["tag"], dst_host["tag"])


@test_utils.test(arch=ti.cpu)
def test_experimental_scatter_cpu_native_struct_tensor_member_workspace_replay():
    src, dst, indices, index_data, host, dst_host = _make_struct_tensor_indexed_copy_case()
    src_vec = src.field("vec")
    dst_vec = dst.field("vec")
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=src.shape[0])

    ti.algorithms.experimental_scatter(
        src_vec, indices, dst_vec, method="cpu_native", workspace=workspace
    )
    first_plan = workspace._native_indexed_copy_plan
    assert first_plan is not None
    assert first_plan.backend == "cpu_native"
    assert first_plan.method_name == "cpu_scatter_strided_ndarray"
    assert first_plan.object_keys is not None

    host["vec"] = host["vec"] * np.int32(3) - np.int32(4)
    dst_host["vec"] = np.full((src.shape[0], 2), -1000, dtype=np.int32)
    src.from_numpy(host)
    dst.from_numpy(dst_host)
    ti.algorithms.experimental_scatter(
        src.field("vec"),
        indices,
        dst.field("vec"),
        method="cpu_native",
        workspace=workspace,
    )

    assert workspace._native_indexed_copy_plan is first_plan
    result = dst.to_numpy()
    assert np.array_equal(result["vec"], host["vec"][index_data])
    assert np.array_equal(result["tag"], dst_host["tag"])


@test_utils.test(arch=[ti.cuda])
def test_experimental_gather_scatter_cuda_device_ndarray_supported_dtypes():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_indexed_copy_available")
        and prog.cuda_device_indexed_copy_available()
    ):
        pytest.skip("CUDA indexed-copy is unavailable in this runtime.")

    for dtype, np_dtype in _INDEXED_COPY_DTYPES:
        gather_ws = _run_ndarray_indexed_copy(dtype, np_dtype, "auto", scatter=False)
        scatter_ws = _run_ndarray_indexed_copy(
            dtype, np_dtype, "cuda_device", scatter=True
        )
        assert gather_ws.workspace_bytes_peak == 0
        assert scatter_ws.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_gather_scatter_cuda_device_ndarray_vector_payloads():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_indexed_copy_payload_available")
        and prog.cuda_device_indexed_copy_payload_available(12)
        and prog.cuda_device_indexed_copy_payload_available(16)
    ):
        pytest.skip("CUDA indexed-copy wide payload support is unavailable.")

    _run_vector_ndarray_indexed_copy("cuda_device", scatter=False)
    _run_vector_ndarray_indexed_copy("cuda_device", scatter=True)
    _run_matrix_ndarray_indexed_copy("cuda_device", scatter=False)
    _run_matrix_ndarray_indexed_copy("cuda_device", scatter=True)


@test_utils.test(arch=[ti.cuda])
def test_experimental_gather_scatter_cuda_device_struct_tensor_member_views():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_indexed_copy_payload_available")
        and prog.cuda_device_indexed_copy_payload_available(4)
    ):
        pytest.skip("CUDA indexed-copy strided support is unavailable.")

    gather_ws = _run_struct_tensor_member_indexed_copy("cuda_device", scatter=False)
    scatter_ws = _run_struct_tensor_member_indexed_copy("cuda_device", scatter=True)
    assert gather_ws.workspace_bytes_peak == 0
    assert scatter_ws.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_gather_scatter_cuda_device_dense_fields():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_gather_dense_field")
        and hasattr(prog, "cuda_device_scatter_dense_field")
        and hasattr(prog, "cuda_device_indexed_copy_payload_available")
        and prog.cuda_device_indexed_copy_payload_available(8)
    ):
        pytest.skip("CUDA dense field indexed-copy is unavailable.")

    for dtype, np_dtype in _INDEXED_COPY_DTYPES:
        gather_ws = _run_dense_field_indexed_copy(
            dtype, np_dtype, "cuda_device", scatter=False
        )
        scatter_ws = _run_dense_field_indexed_copy(
            dtype, np_dtype, "cuda_device", scatter=True
        )
        assert gather_ws.workspace_bytes_peak == 0
        assert scatter_ws.workspace_bytes_peak == 0

    gather_ws = _run_dense_field_indexed_copy_replay(
        ti.i32,
        np.int32,
        "cuda_device",
        False,
        "cuda_device",
        "cuda_device_gather_dense_field",
    )
    scatter_ws = _run_dense_field_indexed_copy_replay(
        ti.i32,
        np.int32,
        "cuda_device",
        True,
        "cuda_device",
        "cuda_device_scatter_dense_field",
    )
    assert gather_ws.workspace_bytes_peak == 0
    assert scatter_ws.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_indexed_copy_cuda_device_invalid_indices_are_ignored():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_indexed_copy_available")
        and prog.cuda_device_indexed_copy_available()
    ):
        pytest.skip("CUDA driver indexed-copy is unavailable in this runtime.")

    _run_invalid_index_ndarray("cuda_device", scatter=False)
    _run_invalid_index_ndarray("cuda_device", scatter=True)


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_gather_scatter_vulkan_native_ndarray_supported_dtypes():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_indexed_copy_available")
        and prog.vulkan_indexed_copy_available()
    ):
        pytest.skip("Vulkan native indexed-copy is unavailable in this runtime.")

    for dtype, np_dtype in _INDEXED_COPY_DTYPES:
        gather_ws = _run_ndarray_indexed_copy(dtype, np_dtype, "auto", scatter=False)
        scatter_ws = _run_ndarray_indexed_copy(
            dtype, np_dtype, "vulkan_native", scatter=True
        )
        assert gather_ws.workspace_bytes_peak == 0
        assert scatter_ws.workspace_bytes_peak == 0
    assert prog.vulkan_indexed_copy_workspace_bytes() == 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_gather_scatter_vulkan_native_ndarray_vector_payloads():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_indexed_copy_available")
        and prog.vulkan_indexed_copy_available()
    ):
        pytest.skip("Vulkan native indexed-copy is unavailable in this runtime.")

    _run_vector_ndarray_indexed_copy("vulkan_native", scatter=False)
    _run_vector_ndarray_indexed_copy("vulkan_native", scatter=True)
    _run_matrix_ndarray_indexed_copy("vulkan_native", scatter=False)
    _run_matrix_ndarray_indexed_copy("vulkan_native", scatter=True)
    assert prog.vulkan_indexed_copy_workspace_bytes() == 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_gather_scatter_vulkan_native_struct_tensor_member_views():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_indexed_copy_available")
        and prog.vulkan_indexed_copy_available()
    ):
        pytest.skip("Vulkan native indexed-copy is unavailable in this runtime.")

    gather_ws = _run_struct_tensor_member_indexed_copy("vulkan_native", scatter=False)
    scatter_ws = _run_struct_tensor_member_indexed_copy("vulkan_native", scatter=True)
    assert gather_ws.workspace_bytes_peak >= 28
    assert scatter_ws.workspace_bytes_peak >= 28


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_gather_scatter_vulkan_native_dense_fields():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_gather_dense_field")
        and hasattr(prog, "vulkan_scatter_dense_field")
        and hasattr(prog, "vulkan_indexed_copy_available")
        and prog.vulkan_indexed_copy_available()
    ):
        pytest.skip("Vulkan dense field indexed-copy is unavailable.")

    for dtype, np_dtype in _INDEXED_COPY_DTYPES:
        gather_ws = _run_dense_field_indexed_copy(
            dtype, np_dtype, "vulkan_native", scatter=False
        )
        scatter_ws = _run_dense_field_indexed_copy(
            dtype, np_dtype, "vulkan_native", scatter=True
        )
        assert gather_ws.workspace_bytes_peak == 0
        assert scatter_ws.workspace_bytes_peak == 0

    gather_ws = _run_dense_field_indexed_copy_replay(
        ti.i32,
        np.int32,
        "vulkan_native",
        False,
        "vulkan_native",
        "vulkan_gather_dense_field",
    )
    scatter_ws = _run_dense_field_indexed_copy_replay(
        ti.i32,
        np.int32,
        "vulkan_native",
        True,
        "vulkan_native",
        "vulkan_scatter_dense_field",
    )
    assert gather_ws.workspace_bytes_peak == 0
    assert scatter_ws.workspace_bytes_peak == 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_indexed_copy_vulkan_native_invalid_indices_are_ignored():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_indexed_copy_available")
        and prog.vulkan_indexed_copy_available()
    ):
        pytest.skip("Vulkan native indexed-copy is unavailable in this runtime.")

    _run_invalid_index_ndarray("vulkan_native", scatter=False)
    _run_invalid_index_ndarray("vulkan_native", scatter=True)


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_indexed_copy_vulkan_reset_with_live_ndarray():
    n = 1024
    src = ti.ndarray(ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_indexed_copy_available")
        and prog.vulkan_indexed_copy_available()
    ):
        pytest.skip("Vulkan native indexed-copy is unavailable in this runtime.")

    data = _source_values(n, np.int32)
    index_data = _reverse_indices(n)
    src.from_numpy(data)
    indices.from_numpy(index_data)
    ti.algorithms.experimental_gather(src, indices, dst, method="vulkan_native")
    assert dst.to_numpy()[0] == data[-1]
    ti.reset()
    del src, indices, dst
    gc.collect()


@test_utils.test(arch=[ti.cpu])
def test_experimental_gather_scatter_cpu_native_ndarray_supported_dtypes():
    for dtype, np_dtype in _INDEXED_COPY_DTYPES:
        gather_ws = _run_ndarray_indexed_copy(dtype, np_dtype, "auto", scatter=False)
        scatter_ws = _run_ndarray_indexed_copy(dtype, np_dtype, "cpu_native", scatter=True)
        assert gather_ws.workspace_bytes_peak == 0
        assert scatter_ws.workspace_bytes_peak == 0
    assert impl.get_runtime().prog.cpu_indexed_copy_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_gather_scatter_cpu_native_ndarray_vector_payloads():
    _run_vector_ndarray_indexed_copy("cpu_native", scatter=False)
    _run_vector_ndarray_indexed_copy("cpu_native", scatter=True)
    _run_matrix_ndarray_indexed_copy("cpu_native", scatter=False)
    _run_matrix_ndarray_indexed_copy("cpu_native", scatter=True)
    assert impl.get_runtime().prog.cpu_indexed_copy_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_gather_scatter_cpu_native_struct_tensor_member_views():
    gather_ws = _run_struct_tensor_member_indexed_copy("cpu_native", scatter=False)
    scatter_ws = _run_struct_tensor_member_indexed_copy("cpu_native", scatter=True)
    assert gather_ws.workspace_bytes_peak == 0
    assert scatter_ws.workspace_bytes_peak == 0
    assert impl.get_runtime().prog.cpu_indexed_copy_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_gather_scatter_cpu_native_dense_fields():
    prog = impl.get_runtime().prog
    assert hasattr(prog, "cpu_gather_dense_field")
    assert hasattr(prog, "cpu_scatter_dense_field")

    for dtype, np_dtype in _INDEXED_COPY_DTYPES:
        gather_ws = _run_dense_field_indexed_copy(
            dtype, np_dtype, "auto", scatter=False
        )
        scatter_ws = _run_dense_field_indexed_copy(
            dtype, np_dtype, "cpu_native", scatter=True
        )
        assert gather_ws.workspace_bytes_peak == 0
        assert scatter_ws.workspace_bytes_peak == 0

    gather_ws = _run_dense_field_indexed_copy_replay(
        ti.i32,
        np.int32,
        "cpu_native",
        False,
        "cpu_native",
        "cpu_gather_dense_field",
    )
    scatter_ws = _run_dense_field_indexed_copy_replay(
        ti.i32,
        np.int32,
        "cpu_native",
        True,
        "cpu_native",
        "cpu_scatter_dense_field",
    )
    assert gather_ws.workspace_bytes_peak == 0
    assert scatter_ws.workspace_bytes_peak == 0
    assert prog.cpu_indexed_copy_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_gather_scatter_native_dense_matrix_field_components():
    _run_dense_matrix_field_indexed_copy(scatter=False)
    _run_dense_matrix_field_indexed_copy(scatter=True)


@test_utils.test(arch=[ti.cpu])
def test_experimental_indexed_copy_cpu_native_invalid_indices_are_ignored():
    _run_invalid_index_ndarray("cpu_native", scatter=False)
    _run_invalid_index_ndarray("cpu_native", scatter=True)


@test_utils.test(arch=[ti.cpu])
def test_experimental_indexed_copy_ndarray_rejects_mismatched_element_shape():
    n = 8
    src = ti.Vector.ndarray(3, ti.f32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.Vector.ndarray(2, ti.f32, shape=n)

    with pytest.raises(TypeError, match="element_shape"):
        ti.algorithms.experimental_gather(src, indices, dst, method="cpu_native")


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.cpu], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_gather_scatter_field_kernel_i32_f32():
    n = 2048
    indices_np = _reverse_indices(n)

    for dtype, np_dtype in [(ti.i32, np.int32), (ti.f32, np.float32)]:
        src = ti.field(dtype, shape=n)
        indices = ti.field(ti.i32, shape=n)
        dst = ti.field(dtype, shape=n)
        data = _source_values(n, np_dtype).astype(np_dtype)
        src.from_numpy(data)
        indices.from_numpy(indices_np)
        dst.fill(0)
        ti.algorithms.experimental_gather(src, indices, dst, method="field_kernel")
        _assert_matches(dst.to_numpy(), data[indices_np])

        dst.fill(0)
        ti.algorithms.experimental_scatter(src, indices, dst, method="field_kernel")
        _assert_matches(dst.to_numpy(), data[indices_np])


@test_utils.test(arch=[ti.cpu])
def test_experimental_indexed_copy_field_kernel_rejects_wide_dtype():
    n = 8
    src = ti.field(ti.i64, shape=n)
    indices = ti.field(ti.i32, shape=n)
    dst = ti.field(ti.i64, shape=n)

    with pytest.raises(RuntimeError, match="Wider scalar values require"):
        ti.algorithms.experimental_gather(src, indices, dst, method="field_kernel")
