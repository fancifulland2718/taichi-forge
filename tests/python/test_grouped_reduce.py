import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


_GROUPED_DTYPES = [
    (ti.i32, np.int32),
    (ti.f32, np.float32),
    (ti.u32, np.uint32),
    (ti.u64, np.uint64),
    (ti.i64, np.int64),
    (ti.f64, np.float64),
]
_GROUPED_VALUE_TYPE = {
    ti.i32: 0,
    ti.f32: 1,
    ti.u32: 2,
    ti.u64: 3,
    ti.i64: 4,
    ti.f64: 5,
}


def _grouped_reduce_input(n, groups, np_dtype):
    keys = (np.arange(n, dtype=np.int32) * 37 + 11) % groups
    if np.issubdtype(np_dtype, np.floating):
        values = np.full(n, np.float32(0.5), dtype=np_dtype)
    elif np.issubdtype(np_dtype, np.unsignedinteger):
        values = (np.arange(n, dtype=np.uint64) % 17).astype(np_dtype)
    else:
        values = (np.arange(n, dtype=np.int64) % 17 - 8).astype(np_dtype)
    expected = np.zeros(groups, dtype=np_dtype)
    np.add.at(expected, keys, values)
    return keys, values, expected


def _assert_matches(actual, expected):
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        assert np.array_equal(actual, expected)


def _run_ndarray_grouped_reduce(dtype, np_dtype, method):
    n = 4096
    groups = 257
    keys = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(dtype, shape=n)
    output = ti.ndarray(dtype, shape=groups)
    keys_np, values_np, expected = _grouped_reduce_input(n, groups, np_dtype)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    output.fill(np_dtype(3))
    workspace = ti.algorithms.GroupedReduceWorkspace(max_items=n, max_groups=groups)
    ti.algorithms.experimental_grouped_reduce(
        keys, values, output, method=method, workspace=workspace
    )
    _assert_matches(output.to_numpy(), expected)
    assert workspace.workspace_bytes_peak >= 0


def _run_struct_member_grouped_reduce(dtype, np_dtype, method):
    n = 2048
    groups = 127
    key_payload = ti.types.struct(key=ti.i32, key_tag=ti.i32)
    payload = ti.types.struct(value=dtype, tag=ti.i32)
    keys = ti.ndarray(key_payload, shape=n)
    values = ti.ndarray(payload, shape=n)
    output = ti.ndarray(payload, shape=groups)
    keys_np, values_np, expected = _grouped_reduce_input(n, groups, np_dtype)
    keys_host = np.zeros((n,), dtype=keys.numpy_dtype)
    keys_host["key"] = keys_np
    keys_host["key_tag"] = np.arange(n, dtype=np.int32) * 7 + 3
    host = np.zeros((n,), dtype=values.numpy_dtype)
    host["value"] = values_np
    host["tag"] = np.arange(n, dtype=np.int32) * 5 - 11
    output_host = np.zeros((groups,), dtype=output.numpy_dtype)
    output_host["value"] = np_dtype(3)
    output_host["tag"] = np.arange(groups, dtype=np.int32) * 13 + 9
    keys.from_numpy(keys_host)
    values.from_numpy(host)
    output.from_numpy(output_host)
    workspace = ti.algorithms.GroupedReduceWorkspace(max_items=n, max_groups=groups)
    ti.algorithms.experimental_grouped_reduce(
        keys.field("key"),
        values.field("value"),
        output.field("value"),
        method=method,
        workspace=workspace,
    )
    result = output.to_numpy()
    _assert_matches(result["value"], expected)
    np.testing.assert_array_equal(keys.to_numpy()["key_tag"], keys_host["key_tag"])
    np.testing.assert_array_equal(result["tag"], output_host["tag"])
    np.testing.assert_array_equal(values.to_numpy()["tag"], host["tag"])


def _run_struct_nested_component_grouped_reduce(method):
    n = 1024
    groups = 67
    key_payload = ti.types.struct(meta=ti.types.struct(key=ti.i32, tag=ti.i32))
    value_payload = ti.types.struct(vec=ti.types.vector(2, ti.i32), tag=ti.i32)
    output_payload = ti.types.struct(total=ti.i32, tag=ti.i32)
    keys = ti.ndarray(key_payload, shape=n)
    values = ti.ndarray(value_payload, shape=n)
    output = ti.ndarray(output_payload, shape=groups)

    keys_np, values_np, expected = _grouped_reduce_input(n, groups, np.int32)
    keys_host = np.zeros((n,), dtype=keys.numpy_dtype)
    keys_host["meta"]["key"] = keys_np
    keys_host["meta"]["tag"] = np.arange(n, dtype=np.int32) * 11
    values_host = np.zeros((n,), dtype=values.numpy_dtype)
    values_host["vec"][:, 0] = values_np * 3
    values_host["vec"][:, 1] = values_np
    values_host["tag"] = np.arange(n, dtype=np.int32) * 5 + 9
    output_host = np.zeros((groups,), dtype=output.numpy_dtype)
    output_host["total"] = -123
    output_host["tag"] = np.arange(groups, dtype=np.int32) * 7
    keys.from_numpy(keys_host)
    values.from_numpy(values_host)
    output.from_numpy(output_host)

    ti.algorithms.experimental_grouped_reduce(
        keys.field("meta.key"),
        values.field("vec", component=1),
        output.field("total"),
        method=method,
    )

    result = output.to_numpy()
    np.testing.assert_array_equal(result["total"], expected)
    np.testing.assert_array_equal(keys.to_numpy()["meta"]["tag"], keys_host["meta"]["tag"])
    np.testing.assert_array_equal(values.to_numpy()["vec"][:, 0], values_host["vec"][:, 0])
    np.testing.assert_array_equal(values.to_numpy()["tag"], values_host["tag"])
    np.testing.assert_array_equal(result["tag"], output_host["tag"])


def _run_struct_tensor_member_grouped_reduce(method):
    n = 2048
    groups = 127
    key_payload = ti.types.struct(key=ti.i32, key_tag=ti.i32)
    value_payload = ti.types.struct(
        vec=ti.types.vector(2, ti.i32),
        mat=ti.types.matrix(2, 2, ti.i32),
        tag=ti.i32,
    )
    keys = ti.ndarray(key_payload, shape=n)
    values = ti.ndarray(value_payload, shape=n)
    output = ti.ndarray(value_payload, shape=groups)

    keys_np = (np.arange(n, dtype=np.int32) * 37 + 11) % groups
    vec_np = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 17) - 8
    mat_np = (np.arange(n * 4, dtype=np.int32).reshape(n, 2, 2) % 13) - 6
    expected_vec = np.zeros((groups, 2), dtype=np.int32)
    expected_mat = np.zeros((groups, 2, 2), dtype=np.int32)
    np.add.at(expected_vec, keys_np, vec_np)
    np.add.at(expected_mat, keys_np, mat_np)

    keys_host = np.zeros((n,), dtype=keys.numpy_dtype)
    keys_host["key"] = keys_np
    keys_host["key_tag"] = np.arange(n, dtype=np.int32) * 7 + 3
    values_host = np.zeros((n,), dtype=values.numpy_dtype)
    values_host["vec"] = vec_np
    values_host["mat"] = mat_np
    values_host["tag"] = np.arange(n, dtype=np.int32) * 5 - 11
    output_host = np.zeros((groups,), dtype=output.numpy_dtype)
    output_host["vec"] = -999
    output_host["mat"] = -777
    output_host["tag"] = np.arange(groups, dtype=np.int32) * 13 + 9
    keys.from_numpy(keys_host)
    values.from_numpy(values_host)
    output.from_numpy(output_host)

    ti.algorithms.experimental_grouped_reduce(
        keys.field("key"),
        values.field("vec"),
        output.field("vec"),
        method=method,
    )
    ti.algorithms.experimental_grouped_reduce(
        keys.field("key"),
        values.field("mat"),
        output.field("mat"),
        method=method,
    )

    result = output.to_numpy()
    np.testing.assert_array_equal(result["vec"], expected_vec)
    np.testing.assert_array_equal(result["mat"], expected_mat)
    np.testing.assert_array_equal(keys.to_numpy()["key_tag"], keys_host["key_tag"])
    np.testing.assert_array_equal(values.to_numpy()["tag"], values_host["tag"])
    np.testing.assert_array_equal(result["tag"], output_host["tag"])


@test_utils.test(arch=[ti.cuda])
def test_experimental_grouped_reduce_cuda_device_ndarray_wide_dtypes():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cuda_device_grouped_reduce_available")
        and prog.cuda_device_grouped_reduce_available()
    ):
        pytest.skip("CUDA toolkit grouped reduce is unavailable in this runtime.")

    for dtype, np_dtype in _GROUPED_DTYPES:
        _run_ndarray_grouped_reduce(dtype, np_dtype, "cuda_device")
        _run_struct_member_grouped_reduce(dtype, np_dtype, "cuda_device")
        _run_ndarray_grouped_reduce(dtype, np_dtype, "cuda_segmented")
        _run_struct_member_grouped_reduce(dtype, np_dtype, "cuda_segmented")
    _run_ndarray_grouped_reduce(ti.i32, np.int32, "auto")
    _run_struct_nested_component_grouped_reduce("cuda_device")
    _run_struct_nested_component_grouped_reduce("cuda_segmented")
    _run_struct_tensor_member_grouped_reduce("cuda_device")
    _run_struct_tensor_member_grouped_reduce("cuda_segmented")


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_grouped_reduce_vulkan_native_ndarray_types():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_grouped_reduce_available")
        and prog.vulkan_grouped_reduce_available()
    ):
        pytest.skip("Vulkan native grouped reduce is unavailable in this runtime.")

    _run_ndarray_grouped_reduce(ti.i32, np.int32, "auto")
    _run_ndarray_grouped_reduce(ti.i32, np.int32, "vulkan_native")
    _run_struct_member_grouped_reduce(ti.i32, np.int32, "vulkan_native")
    _run_struct_nested_component_grouped_reduce("vulkan_native")
    _run_struct_tensor_member_grouped_reduce("vulkan_native")
    _run_ndarray_grouped_reduce(ti.u32, np.uint32, "vulkan_native")
    _run_struct_member_grouped_reduce(ti.u32, np.uint32, "vulkan_native")
    for dtype, np_dtype in _GROUPED_DTYPES:
        value_type = _GROUPED_VALUE_TYPE[dtype]
        if hasattr(prog, "vulkan_grouped_reduce_value_type_available") and not (
            prog.vulkan_grouped_reduce_value_type_available(value_type)
        ):
            continue
        _run_ndarray_grouped_reduce(dtype, np_dtype, "vulkan_segmented")
    f32_atomic_native = (
        hasattr(prog, "vulkan_grouped_reduce_atomic_value_type_available")
        and prog.vulkan_grouped_reduce_atomic_value_type_available(1)
    )
    if f32_atomic_native:
        for _ in range(3):
            _run_ndarray_grouped_reduce(ti.f32, np.float32, "vulkan_native")
            _run_struct_member_grouped_reduce(ti.f32, np.float32, "vulkan_native")
    else:
        with pytest.raises(RuntimeError, match="native grouped-reduce shaders"):
            _run_ndarray_grouped_reduce(ti.f32, np.float32, "vulkan_native")
        with pytest.raises(RuntimeError, match="native grouped-reduce shaders"):
            _run_struct_member_grouped_reduce(ti.f32, np.float32, "vulkan_native")
    for value_type, dtype, np_dtype in [
        (3, ti.u64, np.uint64),
        (4, ti.i64, np.int64),
        (5, ti.f64, np.float64),
    ]:
        if (
            hasattr(prog, "vulkan_grouped_reduce_atomic_value_type_available")
            and prog.vulkan_grouped_reduce_atomic_value_type_available(value_type)
        ):
            for _ in range(3):
                _run_ndarray_grouped_reduce(dtype, np_dtype, "vulkan_native")
                _run_struct_member_grouped_reduce(dtype, np_dtype, "vulkan_native")
        else:
            with pytest.raises(RuntimeError, match="native grouped-reduce shaders"):
                _run_ndarray_grouped_reduce(dtype, np_dtype, "vulkan_native")
            with pytest.raises(RuntimeError, match="native grouped-reduce shaders"):
                _run_struct_member_grouped_reduce(dtype, np_dtype, "vulkan_native")


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_grouped_reduce_vulkan_reset_with_live_ndarray():
    n = 1024
    groups = 64
    keys = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=groups)
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_grouped_reduce_available")
        and prog.vulkan_grouped_reduce_available()
    ):
        pytest.skip("Vulkan native grouped reduce is unavailable in this runtime.")

    keys_np, values_np, expected = _grouped_reduce_input(n, groups, np.int32)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    output.fill(np.int32(-777))
    ti.algorithms.experimental_grouped_reduce(keys, values, output, method="vulkan_native")
    assert np.array_equal(output.to_numpy(), expected)
    ti.reset()
    del keys, values, output
    gc.collect()


@test_utils.test(arch=[ti.cpu])
def test_experimental_grouped_reduce_cpu_native_ndarray_wide_dtypes():
    for dtype, np_dtype in _GROUPED_DTYPES:
        _run_ndarray_grouped_reduce(dtype, np_dtype, "cpu_native")
        _run_struct_member_grouped_reduce(dtype, np_dtype, "cpu_native")
    _run_ndarray_grouped_reduce(ti.i32, np.int32, "auto")
    _run_struct_nested_component_grouped_reduce("cpu_native")
    _run_struct_nested_component_grouped_reduce("segmented")
    _run_struct_tensor_member_grouped_reduce("cpu_native")
    _run_struct_tensor_member_grouped_reduce("segmented")


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.cpu], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_grouped_reduce_field_kernel_i32():
    n = 2048
    groups = 129
    keys = ti.field(ti.i32, shape=n)
    values = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=groups)
    keys_np, values_np, expected = _grouped_reduce_input(n, groups, np.int32)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)
    output.fill(np.int32(-777))
    ti.algorithms.experimental_grouped_reduce(keys, values, output, method="field_kernel")
    assert np.array_equal(output.to_numpy(), expected)


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.cpu], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_grouped_reduce_invalid_keys_are_ignored():
    keys = ti.ndarray(ti.i32, shape=5)
    values = ti.ndarray(ti.i32, shape=5)
    output = ti.ndarray(ti.i32, shape=4)
    keys.from_numpy(np.array([0, -1, 2, 99, 2], dtype=np.int32))
    values.from_numpy(np.array([10, 20, 30, 40, 5], dtype=np.int32))
    output.fill(np.int32(-777))
    ti.algorithms.experimental_grouped_reduce(keys, values, output, method="auto")
    assert np.array_equal(output.to_numpy(), np.array([10, 0, 35, 0], dtype=np.int32))


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_grouped_reduce_vulkan_native_i64_invalid_keys():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_grouped_reduce_atomic_value_type_available")
        and prog.vulkan_grouped_reduce_atomic_value_type_available(3)
        and prog.vulkan_grouped_reduce_atomic_value_type_available(4)
    ):
        pytest.skip("Vulkan native i64/u64 grouped reduce atomics are unavailable.")

    for dtype, np_dtype in [(ti.u64, np.uint64), (ti.i64, np.int64)]:
        keys = ti.ndarray(ti.i32, shape=5)
        values = ti.ndarray(dtype, shape=5)
        output = ti.ndarray(dtype, shape=4)
        keys.from_numpy(np.array([0, -1, 2, 99, 2], dtype=np.int32))
        values.from_numpy(np.array([10, 20, 30, 40, 5], dtype=np_dtype))
        output.fill(np_dtype(777))
        ti.algorithms.experimental_grouped_reduce(
            keys, values, output, method="vulkan_native"
        )
        assert np.array_equal(
            output.to_numpy(), np.array([10, 0, 35, 0], dtype=np_dtype)
        )


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_grouped_reduce_vulkan_native_f64_invalid_and_special_keys():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_grouped_reduce_atomic_value_type_available")
        and prog.vulkan_grouped_reduce_atomic_value_type_available(5)
    ):
        pytest.skip("Vulkan native f64 grouped reduce atomics are unavailable.")

    keys = ti.ndarray(ti.i32, shape=6)
    values = ti.ndarray(ti.f64, shape=6)
    output = ti.ndarray(ti.f64, shape=4)
    keys.from_numpy(np.array([0, 1, 2, 2, 3, 99], dtype=np.int32))
    values.from_numpy(
        np.array([np.nan, np.inf, 1.25, 2.75, -np.inf, 99.0], dtype=np.float64)
    )
    output.fill(np.float64(777.0))
    ti.algorithms.experimental_grouped_reduce(
        keys, values, output, method="vulkan_native"
    )
    out = output.to_numpy()
    assert np.isnan(out[0])
    assert np.isposinf(out[1])
    assert np.isclose(out[2], 4.0)
    assert np.isneginf(out[3])


@test_utils.test(arch=[ti.cpu])
def test_experimental_grouped_reduce_rejects_non_sum_op():
    keys = ti.ndarray(ti.i32, shape=4)
    values = ti.ndarray(ti.i32, shape=4)
    output = ti.ndarray(ti.i32, shape=2)
    with pytest.raises(ValueError, match="grouped reduce op"):
        ti.algorithms.experimental_grouped_reduce(keys, values, output, op="max")
