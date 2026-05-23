import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


_REDUCE_DTYPE_CASES = (
    (ti.u32, np.uint32, 2),
    (ti.i32, np.int32, 0),
    (ti.f32, np.float32, 1),
    (ti.u64, np.uint64, 3),
    (ti.i64, np.int64, 4),
    (ti.f64, np.float64, 5),
)


def _values_np(n, np_dtype):
    index = np.arange(n, dtype=np.int64)
    if np.issubdtype(np_dtype, np.unsignedinteger):
        return ((index.astype(np.uint64) * 17 + 5) % 1021).astype(np_dtype)
    if np.issubdtype(np_dtype, np.floating):
        return (((index % 41) - 20).astype(np.float64) * 0.25).astype(np_dtype)
    return (index % 97 - 48).astype(np_dtype)


def _expected(values_np, op):
    if op == "min":
        return np.min(values_np)
    if op == "max":
        return np.max(values_np)
    return np.sum(values_np, dtype=values_np.dtype).astype(values_np.dtype)


def _assert_reduce_output(actual, expected, np_dtype):
    if np.issubdtype(np_dtype, np.floating):
        assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)
    else:
        assert actual == expected


def _run_ndarray_reduce_case(n, dtype, np_dtype, method, workspace):
    values = ti.ndarray(dtype, shape=n)
    output = ti.ndarray(dtype, shape=1)
    values_np = _values_np(n, np_dtype)
    values.from_numpy(values_np)
    for op in ("sum", "min", "max"):
        output.from_numpy(np.array([0], dtype=np_dtype))
        ti.algorithms.experimental_reduce(
            values, output, op=op, method=method, workspace=workspace
        )
        _assert_reduce_output(output.to_numpy()[0], _expected(values_np, op), np_dtype)


def _run_dense_field_reduce_case(n, dtype, np_dtype, method, workspace):
    values = ti.field(dtype, shape=n)
    output = ti.field(dtype, shape=())
    values_np = _values_np(n, np_dtype)
    values.from_numpy(values_np)
    for op in ("sum", "min", "max"):
        output[None] = 0
        ti.algorithms.experimental_reduce(
            values, output, op=op, method=method, workspace=workspace
        )
        _assert_reduce_output(output[None], _expected(values_np, op), np_dtype)


def _native_reduce_method_for_current_arch():
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.cpu:
        if not (hasattr(prog, "cpu_reduce_available") and prog.cpu_reduce_available()):
            pytest.skip("CPU native reduce is unavailable.")
        return "cpu_native", "cpu_reduce_dense_field"
    if arch == ti.cuda:
        if not (
            hasattr(prog, "cuda_cub_reduce_available")
            and prog.cuda_cub_reduce_available()
        ):
            pytest.skip("CUDA CUB reduce is unavailable.")
        return "cuda_cub", "cuda_cub_reduce_dense_field"
    if arch == ti.vulkan:
        if not (
            hasattr(prog, "vulkan_reduce_available")
            and prog.vulkan_reduce_available()
        ):
            pytest.skip("Vulkan native reduce is unavailable.")
        return "vulkan_native", "vulkan_reduce_dense_field"
    pytest.skip("native reduce is unavailable on this arch.")


def _run_dense_matrix_field_reduce_case():
    n = 128
    method, expected_method = _native_reduce_method_for_current_arch()
    values = ti.Vector.field(2, ti.i32, shape=n)
    output = ti.Vector.field(2, ti.i32, shape=())
    values_np = (np.arange(n * 2, dtype=np.int32).reshape(n, 2) % 17) - 8
    values.from_numpy(values_np)
    output.fill(0)
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)

    ti.algorithms.experimental_reduce(
        values, output, op="sum", method=method, workspace=workspace
    )

    np.testing.assert_array_equal(
        output.to_numpy(), np.sum(values_np, axis=0, dtype=np.int32)
    )
    assert len(workspace._native_reduce_plans) == 2
    assert workspace._native_reduce_plan["method_name"] == expected_method
    assert len(workspace._native_reduce_plan_groups) == 1

    output.fill(0)
    ti.algorithms.experimental_reduce(
        values, output, op="sum", method=method, workspace=workspace
    )
    np.testing.assert_array_equal(
        output.to_numpy(), np.sum(values_np, axis=0, dtype=np.int32)
    )
    assert len(workspace._native_reduce_plan_groups) == 1


def _run_struct_member_reduce_case(n, dtype, np_dtype, method, workspace):
    payload = ti.types.struct(value=dtype, tag=ti.i32)
    values = ti.ndarray(payload, shape=n)
    output = ti.ndarray(dtype, shape=1)
    values_np = _values_np(n, np_dtype)
    host = np.zeros((n,), dtype=values.numpy_dtype)
    host["value"] = values_np
    host["tag"] = np.arange(n, dtype=np.int32) * 3 + 1
    values.from_numpy(host)
    for op in ("sum", "min", "max"):
        output.from_numpy(np.array([0], dtype=np_dtype))
        ti.algorithms.experimental_reduce(
            values.field("value"), output, op=op, method=method, workspace=workspace
        )
        _assert_reduce_output(output.to_numpy()[0], _expected(values_np, op), np_dtype)
    assert np.array_equal(values.to_numpy()["tag"], host["tag"])


def _run_struct_tensor_member_reduce_case(n, dtype, np_dtype, method, workspace):
    payload = ti.types.struct(
        vec=ti.types.vector(2, dtype),
        mat=ti.types.matrix(2, 2, dtype),
        tag=ti.i32,
    )
    values = ti.ndarray(payload, shape=n)
    output = ti.ndarray(payload, shape=1)
    vec_np = _values_np(n * 2, np_dtype).reshape(n, 2)
    mat_np = _values_np(n * 4, np_dtype).reshape(n, 2, 2)
    host = np.zeros((n,), dtype=values.numpy_dtype)
    out_host = np.zeros((1,), dtype=output.numpy_dtype)
    host["vec"] = vec_np
    host["mat"] = mat_np
    host["tag"] = np.arange(n, dtype=np.int32) * 3 + 1
    out_host["tag"] = np.array([12345], dtype=np.int32)
    values.from_numpy(host)
    for op in ("sum", "min", "max"):
        output.from_numpy(out_host)
        ti.algorithms.experimental_reduce(
            values.field("vec"),
            output.field("vec"),
            op=op,
            method=method,
            workspace=workspace,
        )
        ti.algorithms.experimental_reduce(
            values.field("mat"),
            output.field("mat"),
            op=op,
            method=method,
            workspace=workspace,
        )
        result = output.to_numpy()
        expected_vec = np.asarray([_expected(vec_np[:, lane], op) for lane in range(2)])
        expected_mat = np.asarray(
            [
                [_expected(mat_np[:, row, col], op) for col in range(2)]
                for row in range(2)
            ]
        )
        np.testing.assert_allclose(result["vec"][0], expected_vec, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(result["mat"][0], expected_mat, rtol=1e-5, atol=1e-6)
        assert result["tag"][0] == out_host["tag"][0]
    assert len(workspace._native_reduce_plan_groups) >= 6
    assert np.array_equal(values.to_numpy()["tag"], host["tag"])


@test_utils.test(arch=[ti.cuda])
def test_experimental_reduce_cuda_cub_ndarray_dtypes():
    n = 4096
    prog = impl.get_runtime().prog
    if not prog.cuda_cub_reduce_available():
        pytest.skip("CUDA CUB reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for dtype, np_dtype, _value_type in _REDUCE_DTYPE_CASES:
        _run_ndarray_reduce_case(n, dtype, np_dtype, "cuda_cub", workspace)
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_reduce_cuda_cub_struct_member_view():
    n = 4096
    prog = impl.get_runtime().prog
    if not prog.cuda_cub_reduce_available():
        pytest.skip("CUDA CUB reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for dtype, np_dtype, _value_type in _REDUCE_DTYPE_CASES:
        _run_struct_member_reduce_case(n, dtype, np_dtype, "cuda_cub", workspace)
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_reduce_cuda_cub_dense_field_dtypes():
    n = 4096
    prog = impl.get_runtime().prog
    if not prog.cuda_cub_reduce_available():
        pytest.skip("CUDA CUB reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for dtype, np_dtype, _value_type in _REDUCE_DTYPE_CASES:
        _run_dense_field_reduce_case(n, dtype, np_dtype, "cuda_cub", workspace)
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.cuda])
def test_experimental_reduce_cuda_cub_dense_field_workspace_replay():
    n = 128
    prog = impl.get_runtime().prog
    if not prog.cuda_cub_reduce_available():
        pytest.skip("CUDA CUB reduce is unavailable in this build/runtime.")

    values = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=())
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for base in (0, 17):
        values_np = (np.arange(n, dtype=np.int32) + base).astype(np.int32)
        values.from_numpy(values_np)
        output[None] = 0
        ti.algorithms.experimental_reduce(
            values, output, op="sum", method="cuda_cub", workspace=workspace
        )
        assert output[None] == np.sum(values_np, dtype=np.int32)
    assert workspace._native_reduce_plan is not None
    assert workspace._native_reduce_plan["backend"] == "cuda_cub"


@test_utils.test(arch=[ti.cuda])
def test_experimental_reduce_cuda_cub_struct_tensor_member_view():
    n = 4096
    prog = impl.get_runtime().prog
    if not prog.cuda_cub_reduce_available():
        pytest.skip("CUDA CUB reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for dtype, np_dtype, _value_type in (
        (ti.i32, np.int32, 0),
        (ti.f32, np.float32, 1),
    ):
        _run_struct_tensor_member_reduce_case(
            n, dtype, np_dtype, "cuda_cub", workspace
        )
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_native_dense_matrix_field_components():
    _run_dense_matrix_field_reduce_case()


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_vulkan_native_ndarray_dtypes():
    n = 8192
    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_reduce_available") or not prog.vulkan_reduce_available():
        pytest.skip("Vulkan native reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    tested = 0
    for dtype, np_dtype, value_type in _REDUCE_DTYPE_CASES:
        if hasattr(prog, "vulkan_reduce_value_type_available"):
            if not prog.vulkan_reduce_value_type_available(value_type):
                continue
        elif dtype != ti.i32:
            continue
        _run_ndarray_reduce_case(n, dtype, np_dtype, "vulkan_native", workspace)
        tested += 1
    assert tested >= 3
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_vulkan_native_dense_field_dtypes():
    n = 8192
    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_reduce_available") or not prog.vulkan_reduce_available():
        pytest.skip("Vulkan native reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    tested = 0
    for dtype, np_dtype, value_type in _REDUCE_DTYPE_CASES:
        if hasattr(prog, "vulkan_reduce_value_type_available"):
            if not prog.vulkan_reduce_value_type_available(value_type):
                continue
        elif dtype != ti.i32:
            continue
        _run_dense_field_reduce_case(n, dtype, np_dtype, "vulkan_native", workspace)
        tested += 1
    assert tested >= 3
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_vulkan_native_dense_field_workspace_replay():
    n = 128
    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_reduce_available") or not prog.vulkan_reduce_available():
        pytest.skip("Vulkan native reduce is unavailable in this build/runtime.")

    values = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=())
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for base in (0, 17):
        values_np = (np.arange(n, dtype=np.int32) + base).astype(np.int32)
        values.from_numpy(values_np)
        output[None] = 0
        ti.algorithms.experimental_reduce(
            values, output, op="sum", method="vulkan_native", workspace=workspace
        )
        assert output[None] == np.sum(values_np, dtype=np.int32)
    assert workspace._native_reduce_plan is not None
    assert workspace._native_reduce_plan["backend"] == "vulkan_native"


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_vulkan_native_struct_member_view():
    n = 8192
    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_reduce_available") or not prog.vulkan_reduce_available():
        pytest.skip("Vulkan native reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    tested = 0
    for dtype, np_dtype, value_type in _REDUCE_DTYPE_CASES:
        if hasattr(prog, "vulkan_reduce_value_type_available"):
            if not prog.vulkan_reduce_value_type_available(value_type):
                continue
        elif dtype != ti.i32:
            continue
        _run_struct_member_reduce_case(n, dtype, np_dtype, "vulkan_native", workspace)
        tested += 1
    assert tested >= 3
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_vulkan_native_struct_tensor_member_view():
    n = 8192
    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_reduce_available") or not prog.vulkan_reduce_available():
        pytest.skip("Vulkan native reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    tested = 0
    for dtype, np_dtype, value_type in (
        (ti.i32, np.int32, 0),
        (ti.f32, np.float32, 1),
    ):
        if hasattr(prog, "vulkan_reduce_value_type_available"):
            if not prog.vulkan_reduce_value_type_available(value_type):
                continue
        _run_struct_tensor_member_reduce_case(
            n, dtype, np_dtype, "vulkan_native", workspace
        )
        tested += 1
    assert tested >= 2
    assert workspace.workspace_bytes_peak > 0


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_vulkan_native_reset_with_live_ndarray():
    n = 8192
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=1)

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_reduce_available") or not prog.vulkan_reduce_available():
        pytest.skip("Vulkan native reduce is unavailable in this build/runtime.")

    values_np = (np.arange(n, dtype=np.int64) % 17 - 8).astype(np.int32)
    values.from_numpy(values_np)
    ti.algorithms.experimental_reduce(values, output, op="sum", method="vulkan_native")
    expected = np.sum(values_np, dtype=np.int32).astype(np.int32)
    assert output.to_numpy()[0] == expected
    ti.reset()


@test_utils.test(arch=[ti.cpu])
def test_experimental_reduce_cpu_native_ndarray_dtypes():
    n = 131072
    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_reduce_available") or not prog.cpu_reduce_available():
        pytest.skip("CPU native reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for dtype, np_dtype, _value_type in _REDUCE_DTYPE_CASES:
        _run_ndarray_reduce_case(n, dtype, np_dtype, "cpu_native", workspace)
    assert workspace.workspace_bytes_peak > 0
    assert impl.get_runtime().prog.cpu_reduce_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_reduce_cpu_native_dense_field_i32_f32():
    n = 4096
    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_reduce_available") or not prog.cpu_reduce_available():
        pytest.skip("CPU native reduce is unavailable in this build/runtime.")

    for dtype, np_dtype in [(ti.i32, np.int32), (ti.f32, np.float32)]:
        values = ti.field(dtype, shape=n)
        output = ti.field(dtype, shape=())
        data = (np.arange(n, dtype=np_dtype) % 7).astype(np_dtype)
        values.from_numpy(data)
        workspace = ti.algorithms.ReduceWorkspace(max_items=n)
        ti.algorithms.experimental_reduce(
            values, output, op="sum", method="cpu_native", workspace=workspace
        )
        if np.issubdtype(np_dtype, np.floating):
            np.testing.assert_allclose(output[None], np.sum(data), rtol=1e-6, atol=1e-6)
        else:
            assert output[None] == np.sum(data)
        assert workspace.workspace_bytes_peak == 0
    assert impl.get_runtime().prog.cpu_reduce_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_reduce_cpu_native_dense_field_workspace_replay():
    n = 128
    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_reduce_available") or not prog.cpu_reduce_available():
        pytest.skip("CPU native reduce is unavailable in this build/runtime.")

    values = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=())
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for base in (0, 17):
        values_np = (np.arange(n, dtype=np.int32) + base).astype(np.int32)
        values.from_numpy(values_np)
        output[None] = 0
        ti.algorithms.experimental_reduce(
            values, output, op="sum", method="cpu_native", workspace=workspace
        )
        assert output[None] == np.sum(values_np, dtype=np.int32)
    assert workspace._native_reduce_plan is not None
    assert workspace._native_reduce_plan["backend"] == "cpu_native"


@test_utils.test(arch=[ti.cpu])
def test_experimental_reduce_cpu_native_ndarray_workspace_replay():
    n = 128
    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_reduce_available") or not prog.cpu_reduce_available():
        pytest.skip("CPU native reduce is unavailable in this build/runtime.")

    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=1)
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    first_plan = None
    for base in (0, 17):
        values_np = (np.arange(n, dtype=np.int32) + base).astype(np.int32)
        values.from_numpy(values_np)
        output.fill(0)
        ti.algorithms.experimental_reduce(
            values, output, op="sum", method="cpu_native", workspace=workspace
        )
        assert output.to_numpy()[0] == np.sum(values_np, dtype=np.int32)
        if first_plan is None:
            first_plan = workspace._native_reduce_plan
        else:
            assert workspace._native_reduce_plan is first_plan
    assert workspace._native_reduce_plan["backend"] == "cpu_native"
    assert workspace._native_reduce_plan["method_name"] == "cpu_reduce_ndarray"


@test_utils.test(arch=[ti.cpu])
def test_experimental_reduce_cpu_native_struct_member_workspace_replay():
    n = 128
    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_reduce_available") or not prog.cpu_reduce_available():
        pytest.skip("CPU native reduce is unavailable in this build/runtime.")

    payload = ti.types.struct(value=ti.i32, tag=ti.i32)
    values = ti.ndarray(payload, shape=n)
    values_member = values.field("value")
    output = ti.ndarray(ti.i32, shape=1)
    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    first_plan = None
    for base in (0, 17):
        values_np = (np.arange(n, dtype=np.int32) + base).astype(np.int32)
        host = np.zeros((n,), dtype=values.numpy_dtype)
        host["value"] = values_np
        host["tag"] = np.arange(n, dtype=np.int32) * 3 + 1
        values.from_numpy(host)
        output.fill(0)
        ti.algorithms.experimental_reduce(
            values_member, output, op="sum", method="cpu_native", workspace=workspace
        )
        assert output.to_numpy()[0] == np.sum(values_np, dtype=np.int32)
        if first_plan is None:
            first_plan = workspace._native_reduce_plan
        else:
            assert workspace._native_reduce_plan is first_plan
    assert workspace._native_reduce_plan["backend"] == "cpu_native"
    assert (
        workspace._native_reduce_plan["method_name"]
        == "cpu_reduce_strided_ndarray"
    )


@test_utils.test(arch=[ti.cpu])
def test_experimental_reduce_cpu_native_struct_member_view():
    n = 131072
    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_reduce_available") or not prog.cpu_reduce_available():
        pytest.skip("CPU native reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for dtype, np_dtype, _value_type in _REDUCE_DTYPE_CASES:
        _run_struct_member_reduce_case(n, dtype, np_dtype, "cpu_native", workspace)
    assert workspace.workspace_bytes_peak > 0
    assert impl.get_runtime().prog.cpu_reduce_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_reduce_cpu_native_struct_tensor_member_view():
    n = 131072
    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_reduce_available") or not prog.cpu_reduce_available():
        pytest.skip("CPU native reduce is unavailable in this build/runtime.")

    workspace = ti.algorithms.ReduceWorkspace(max_items=n)
    for dtype, np_dtype, _value_type in (
        (ti.i32, np.int32, 0),
        (ti.f32, np.float32, 1),
    ):
        _run_struct_tensor_member_reduce_case(
            n, dtype, np_dtype, "cpu_native", workspace
        )
    assert workspace.workspace_bytes_peak > 0
    assert impl.get_runtime().prog.cpu_reduce_workspace_bytes() == 0


@test_utils.test(arch=[ti.cpu])
def test_experimental_reduce_struct_member_view_rejections():
    payload = ti.types.struct(value=ti.f32, tag=ti.i32)
    values = ti.ndarray(payload, shape=8)
    scalar_output = ti.ndarray(ti.f32, shape=1)
    struct_output = ti.ndarray(payload, shape=1)

    with pytest.raises(TypeError, match="does not support StructNdarray"):
        ti.algorithms.experimental_reduce(values, scalar_output, method="cpu_native")
    with pytest.raises(TypeError, match="does not support StructNdarray"):
        ti.algorithms.experimental_reduce(
            values.field("value"), struct_output, method="cpu_native"
        )


@test_utils.test(arch=[ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_field_atomic_i32():
    n = 2048
    values = ti.field(ti.i32, shape=n)
    output = ti.field(ti.i32, shape=())

    @ti.kernel
    def fill():
        for i in range(n):
            values[i] = i % 53 - 26
        output[None] = -777

    fill()
    values_np = np.array([i % 53 - 26 for i in range(n)], dtype=np.int32)
    for op, expected in (
        ("sum", np.sum(values_np, dtype=np.int64).astype(np.int32)),
        ("min", np.min(values_np)),
        ("max", np.max(values_np)),
    ):
        ti.algorithms.experimental_reduce(values, output, op=op, method="field_atomic")
        assert output[None] == expected


@test_utils.test(arch=[ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_experimental_reduce_field_atomic_f32():
    n = 1024
    values = ti.field(ti.f32, shape=n)
    output = ti.field(ti.f32, shape=())

    @ti.kernel
    def fill():
        for i in range(n):
            values[i] = ti.cast(i % 37 - 18, ti.f32) * 0.5
        output[None] = -777.0

    fill()
    values_np = (np.arange(n, dtype=np.int32) % 37 - 18).astype(np.float32) * 0.5
    ti.algorithms.experimental_reduce(values, output, op="sum", method="field_atomic")
    assert output[None] == pytest.approx(float(np.sum(values_np)), rel=1e-5)
    ti.algorithms.experimental_reduce(values, output, op="min", method="field_atomic")
    assert output[None] == pytest.approx(float(np.min(values_np)))
    ti.algorithms.experimental_reduce(values, output, op="max", method="field_atomic")
    assert output[None] == pytest.approx(float(np.max(values_np)))
