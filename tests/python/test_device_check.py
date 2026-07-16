import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _skip_if_native_check_unavailable():
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.cuda:
        if not (
            hasattr(prog, "cuda_device_check_count_available")
            and prog.cuda_device_check_count_available()
        ):
            pytest.skip("CUDA Driver check_count is unavailable.")
        return
    if arch == ti.vulkan:
        if not (
            hasattr(prog, "vulkan_check_count_available")
            and prog.vulkan_check_count_available()
        ):
            pytest.skip("Vulkan native check_count is unavailable.")
        if (
            hasattr(prog, "vulkan_check_count_value_type_available")
            and not prog.vulkan_check_count_value_type_available(1)
        ):
            pytest.skip("Vulkan native check_count f32 is unavailable.")
        return
    if not (
        hasattr(prog, "cpu_check_count_available")
        and prog.cpu_check_count_available()
    ):
        pytest.skip("CPU native check_count is unavailable.")


def _skip_if_native_metric_unavailable(value_type=1):
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.cuda:
        if not (
            hasattr(prog, "cuda_device_metric_reduce_available")
            and prog.cuda_device_metric_reduce_available()
        ):
            pytest.skip("CUDA Driver metric_reduce is unavailable.")
        if (
            hasattr(prog, "cuda_device_metric_reduce_value_type_available")
            and not prog.cuda_device_metric_reduce_value_type_available(value_type)
        ):
            pytest.skip("CUDA Driver metric_reduce dtype is unavailable.")
        return
    if arch == ti.vulkan:
        if not (
            hasattr(prog, "vulkan_metric_reduce_available")
            and prog.vulkan_metric_reduce_available()
        ):
            pytest.skip("Vulkan native metric_reduce is unavailable.")
        if (
            hasattr(prog, "vulkan_metric_reduce_value_type_available")
            and not prog.vulkan_metric_reduce_value_type_available(value_type)
        ):
            pytest.skip("Vulkan native metric_reduce dtype is unavailable.")
        return
    if not (
        hasattr(prog, "cpu_metric_reduce_available")
        and prog.cpu_metric_reduce_available()
    ):
        pytest.skip("CPU native metric_reduce is unavailable.")
    if (
        hasattr(prog, "cpu_metric_reduce_value_type_available")
        and not prog.cpu_metric_reduce_value_type_available(value_type)
    ):
        pytest.skip("CPU native metric_reduce dtype is unavailable.")


def _check_count_value_available(value_type):
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.vulkan and hasattr(prog, "vulkan_check_count_value_type_available"):
        return prog.vulkan_check_count_value_type_available(value_type)
    return True


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_check_predicates():
    _skip_if_native_check_unavailable()
    flags = ti.ndarray(ti.i32, shape=5)
    flags.from_numpy(np.array([0, 1, 2, 0, -3], dtype=np.int32))
    workspace = ti.algorithms.CheckWorkspace(max_items=5)

    count = ti.algorithms.count_if(flags, workspace=workspace)
    assert count.to_int() == 3
    assert ti.algorithms.any_if(flags, workspace=workspace).to_bool()
    assert not ti.algorithms.all_if(flags, workspace=workspace).to_bool()

    flags.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    assert ti.algorithms.all_if(flags, workspace=workspace).to_bool()
    assert workspace.workspace_bytes_peak >= 4


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_check_basic_numeric_dtypes():
    _skip_if_native_check_unavailable()
    cases = [
        (ti.i32, 0, np.array([0, 1, -2, 3], dtype=np.int32), 3),
        (ti.u32, 2, np.array([0, 1, 2, 3], dtype=np.uint32), 3),
        (ti.u64, 3, np.array([0, 1, 2, 3], dtype=np.uint64), 3),
        (ti.i64, 4, np.array([0, 1, -2, 3], dtype=np.int64), 3),
        (ti.f32, 1, np.array([0, 1.5, -2.0, 3.0], dtype=np.float32), 3),
        (ti.f64, 5, np.array([0, 1.5, -2.0, 3.0], dtype=np.float64), 3),
    ]
    for dtype, value_type, host, expected in cases:
        if not _check_count_value_available(value_type):
            continue
        values = ti.ndarray(dtype, shape=host.shape[0])
        values.from_numpy(host)
        workspace = ti.algorithms.CheckWorkspace(max_items=host.shape[0])
        assert ti.algorithms.count_if(values, workspace=workspace).to_int() == expected


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_check_floating_values():
    _skip_if_native_check_unavailable()
    values = ti.ndarray(ti.f32, shape=6)
    values.from_numpy(
        np.array([0.0, np.nan, np.inf, -np.inf, 4.0, -5.0], dtype=np.float32)
    )
    workspace = ti.algorithms.CheckWorkspace(max_items=6)

    assert ti.algorithms.nan_count(values, workspace=workspace).to_int() == 1
    assert ti.algorithms.inf_count(values, workspace=workspace).to_int() == 2
    finite = ti.algorithms.all_finite(values, workspace=workspace)
    assert not finite.to_bool()
    assert not finite.ok()

    values.from_numpy(np.array([0.0, 1.0, -2.0, 3.5, 4.0, -5.0], dtype=np.float32))
    finite = ti.algorithms.all_finite(values, workspace=workspace)
    assert finite.to_bool()
    assert finite.ok()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_check_dense_field_and_struct_member():
    _skip_if_native_check_unavailable()
    dense = ti.field(ti.i32, shape=6)
    dense.from_numpy(np.array([0, 1, 0, -2, 3, 0], dtype=np.int32))
    workspace = ti.algorithms.CheckWorkspace(max_items=6)
    assert ti.algorithms.count_if(dense, workspace=workspace).to_int() == 3
    assert not ti.algorithms.all_if(dense, workspace=workspace).to_bool()

    payload = ti.types.struct(value=ti.i32, tag=ti.i32)
    values = ti.ndarray(payload, shape=6)
    host = np.zeros((6,), dtype=values.numpy_dtype)
    host["value"] = np.array([0, 1, 0, -2, 3, 0], dtype=np.int32)
    host["tag"] = np.arange(6, dtype=np.int32) + 10
    values.from_numpy(host)
    assert ti.algorithms.count_if(values.field("value"), workspace=workspace).to_int() == 3
    np.testing.assert_array_equal(values.to_numpy()["tag"], host["tag"])


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_check_index_bounds():
    _skip_if_native_check_unavailable()
    indices = ti.ndarray(ti.i32, shape=5)
    indices.from_numpy(np.array([-1, 0, 2, 4, 7], dtype=np.int32))
    workspace = ti.algorithms.CheckWorkspace(max_items=5)

    invalid = ti.algorithms.index_bounds_check(
        indices, lower=0, upper=4, workspace=workspace
    )
    assert invalid.to_int() == 3
    assert not invalid.ok()

    indices.from_numpy(np.array([0, 1, 2, 3, 0], dtype=np.int32))
    invalid = ti.algorithms.index_bounds_check(
        indices, lower=0, upper=4, workspace=workspace
    )
    assert invalid.to_int() == 0
    assert invalid.ok()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_check_result_graph_replay():
    _skip_if_native_check_unavailable()
    flags = ti.ndarray(ti.i32, shape=6)
    workspace = ti.algorithms.CheckWorkspace(max_items=6)
    flags.from_numpy(np.array([0, 1, 0, 2, 3, 0], dtype=np.int32))
    result = ti.algorithms.count_if(flags, workspace=workspace)
    assert result.to_int() == 3

    builder = ti.graph.GraphBuilder()
    assert builder.append_native(result) is builder
    graph = builder.compile()

    flags.from_numpy(np.array([1, 1, 0, 0, 0, 0], dtype=np.int32))
    graph.run({})
    assert result.to_int() == 2


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_metric_f32():
    _skip_if_native_metric_unavailable(value_type=1)
    values = ti.ndarray(ti.f32, shape=5)
    ref = ti.ndarray(ti.f32, shape=5)
    values.from_numpy(np.array([-2.0, 3.5, -4.25, 0.5, 1.0], dtype=np.float32))
    ref.from_numpy(np.array([-1.0, 1.0, -3.0, -0.5, 5.0], dtype=np.float32))
    workspace = ti.algorithms.MetricWorkspace(max_items=5)

    assert ti.algorithms.max_abs(values, workspace=workspace).to_float() == pytest.approx(
        4.25
    )
    assert ti.algorithms.max_abs_delta(
        values, ref, workspace=workspace
    ).to_float() == pytest.approx(4.0)
    assert workspace.workspace_bytes_peak >= 4


@test_utils.test(arch=[ti.cuda])
def test_cuda_device_diagnostics_use_driver_provider():
    _skip_if_native_check_unavailable()
    _skip_if_native_metric_unavailable(value_type=1)
    flags = ti.ndarray(ti.i32, shape=5)
    values = ti.ndarray(ti.f32, shape=5)
    flags.from_numpy(np.array([0, 1, 0, 2, 3], dtype=np.int32))
    values.from_numpy(np.array([0.0, -2.0, np.nan, 3.5, -1.0], dtype=np.float32))
    check_workspace = ti.algorithms.CheckWorkspace(max_items=5)
    metric_workspace = ti.algorithms.MetricWorkspace(max_items=5)

    count = ti.algorithms.count_if(
        flags, method="cuda_device", workspace=check_workspace
    )
    metric = ti.algorithms.max_abs(
        values, method="cuda_device", workspace=metric_workspace
    )
    assert count.to_int() == 3
    assert np.isinf(metric.to_float())
    assert check_workspace._native_check_plan.backend == "cuda_device"
    assert metric_workspace._native_metric_plan.backend == "cuda_device"
    assert check_workspace._cuda_device_active
    assert metric_workspace._cuda_device_active
    check_workspace.clear()
    metric_workspace.clear()
    assert not check_workspace._cuda_device_active
    assert not metric_workspace._cuda_device_active


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_metric_dense_field_and_struct_member_f32():
    _skip_if_native_metric_unavailable(value_type=1)
    dense = ti.field(ti.f32, shape=5)
    ref_dense = ti.field(ti.f32, shape=5)
    dense.from_numpy(np.array([-2.0, 3.5, -4.25, 0.5, 1.0], dtype=np.float32))
    ref_dense.from_numpy(np.array([-1.0, 1.0, -3.0, -0.5, 5.0], dtype=np.float32))
    workspace = ti.algorithms.MetricWorkspace(max_items=5)
    assert ti.algorithms.max_abs(dense, workspace=workspace).to_float() == pytest.approx(
        4.25
    )
    assert ti.algorithms.max_abs_delta(
        dense, ref_dense, workspace=workspace
    ).to_float() == pytest.approx(4.0)

    payload = ti.types.struct(value=ti.f32, tag=ti.i32)
    values = ti.ndarray(payload, shape=5)
    ref = ti.ndarray(payload, shape=5)
    host = np.zeros((5,), dtype=values.numpy_dtype)
    ref_host = np.zeros((5,), dtype=ref.numpy_dtype)
    host["value"] = np.array([-2.0, 3.5, -4.25, 0.5, 1.0], dtype=np.float32)
    ref_host["value"] = np.array([-1.0, 1.0, -3.0, -0.5, 5.0], dtype=np.float32)
    host["tag"] = np.arange(5, dtype=np.int32) + 100
    ref_host["tag"] = np.arange(5, dtype=np.int32) + 200
    values.from_numpy(host)
    ref.from_numpy(ref_host)
    assert ti.algorithms.max_abs(
        values.field("value"), workspace=workspace
    ).to_float() == pytest.approx(4.25)
    assert ti.algorithms.max_abs_delta(
        values.field("value"), ref.field("value"), workspace=workspace
    ).to_float() == pytest.approx(4.0)
    np.testing.assert_array_equal(values.to_numpy()["tag"], host["tag"])


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_metric_mixed_dense_array_f32():
    _skip_if_native_metric_unavailable(value_type=1)
    dense = ti.field(ti.f32, shape=5)
    ref = ti.ndarray(ti.f32, shape=5)
    dense.from_numpy(np.array([-2.0, 3.5, -4.25, 0.5, 1.0], dtype=np.float32))
    ref.from_numpy(np.array([-1.0, 1.0, -3.0, -0.5, 5.0], dtype=np.float32))
    workspace = ti.algorithms.MetricWorkspace(max_items=5)

    assert ti.algorithms.max_abs_delta(
        dense, ref, workspace=workspace
    ).to_float() == pytest.approx(4.0)
    assert ti.algorithms.max_abs_delta(
        ref, dense, workspace=workspace
    ).to_float() == pytest.approx(4.0)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_metric_mixed_dense_struct_member_graph_replay_f32():
    _skip_if_native_metric_unavailable(value_type=1)
    dense = ti.field(ti.f32, shape=5)
    payload = ti.types.struct(value=ti.f32, tag=ti.i32)
    values = ti.ndarray(payload, shape=5)
    workspace = ti.algorithms.MetricWorkspace(max_items=5)

    dense.from_numpy(np.array([-2.0, 3.5, -4.25, 0.5, 1.0], dtype=np.float32))
    host = np.zeros((5,), dtype=values.numpy_dtype)
    host["value"] = np.array([-1.0, 1.0, -3.0, -0.5, 5.0], dtype=np.float32)
    host["tag"] = np.arange(5, dtype=np.int32) + 100
    values.from_numpy(host)
    result = ti.algorithms.max_abs_delta(
        dense, values.field("value"), workspace=workspace
    )
    assert result.to_float() == pytest.approx(4.0)

    builder = ti.graph.GraphBuilder()
    assert builder.append_native(result) is builder
    graph = builder.compile()

    dense.from_numpy(np.array([0.0, 1.0, 8.0, -2.0, 3.0], dtype=np.float32))
    host["value"] = np.array([0.0, -3.0, 1.0, -8.0, 3.5], dtype=np.float32)
    values.from_numpy(host)
    graph.run({})
    assert result.to_float() == pytest.approx(7.0)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_metric_result_graph_replay_struct_member_f32():
    _skip_if_native_metric_unavailable(value_type=1)
    payload = ti.types.struct(value=ti.f32, tag=ti.i32)
    values = ti.ndarray(payload, shape=5)
    ref = ti.ndarray(payload, shape=5)
    workspace = ti.algorithms.MetricWorkspace(max_items=5)

    host = np.zeros((5,), dtype=values.numpy_dtype)
    ref_host = np.zeros((5,), dtype=ref.numpy_dtype)
    host["value"] = np.array([-2.0, 3.5, -4.25, 0.5, 1.0], dtype=np.float32)
    ref_host["value"] = np.array([-1.0, 1.0, -3.0, -0.5, 5.0], dtype=np.float32)
    values.from_numpy(host)
    ref.from_numpy(ref_host)
    result = ti.algorithms.max_abs_delta(
        values.field("value"), ref.field("value"), workspace=workspace
    )
    assert result.to_float() == pytest.approx(4.0)

    builder = ti.graph.GraphBuilder()
    assert builder.append_native(result) is builder
    graph = builder.compile()

    host["value"] = np.array([0.0, 1.0, 8.0, -2.0, 3.0], dtype=np.float32)
    ref_host["value"] = np.array([0.0, -3.0, 1.0, -8.0, 3.5], dtype=np.float32)
    values.from_numpy(host)
    ref.from_numpy(ref_host)
    graph.run({})
    assert result.to_float() == pytest.approx(7.0)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")]
)
def test_native_device_check_metric_primitive_sequence_graph_replay():
    _skip_if_native_check_unavailable()
    _skip_if_native_metric_unavailable(value_type=1)
    flags = ti.ndarray(ti.i32, shape=5)
    values = ti.ndarray(ti.f32, shape=5)
    check_workspace = ti.algorithms.CheckWorkspace(max_items=5)
    metric_workspace = ti.algorithms.MetricWorkspace(max_items=5)

    seq = ti.algorithms.primitive_sequence()
    seq.index_bounds_check(flags, upper=4, workspace=check_workspace).max_abs(
        values, workspace=metric_workspace
    )
    builder = ti.graph.GraphBuilder()
    assert builder.append_native(seq) is builder
    graph = builder.compile()

    flags.from_numpy(np.array([-1, 0, 1, 4, 9], dtype=np.int32))
    values.from_numpy(np.array([-1.0, 2.0, -5.5, 0.0, 3.0], dtype=np.float32))
    graph.run({})
    assert int(check_workspace._get_result_i32_ndarray().to_numpy()[0]) == 3
    assert float(metric_workspace._get_result_ndarray(ti.f32).to_numpy()[0]) == pytest.approx(
        5.5
    )


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_native_device_metric_f64():
    _skip_if_native_metric_unavailable(value_type=5)
    values = ti.ndarray(ti.f64, shape=4)
    ref = ti.ndarray(ti.f64, shape=4)
    values.from_numpy(np.array([-2.0, 3.5, -4.25, 0.5], dtype=np.float64))
    ref.from_numpy(np.array([-1.0, 1.0, -3.0, -0.5], dtype=np.float64))
    workspace = ti.algorithms.MetricWorkspace(max_items=4)

    assert ti.algorithms.max_abs(values, workspace=workspace).to_float() == pytest.approx(
        4.25
    )
    assert ti.algorithms.max_abs_delta(
        values, ref, workspace=workspace
    ).to_float() == pytest.approx(2.5)


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_native_device_metric_mixed_dense_array_f64():
    _skip_if_native_metric_unavailable(value_type=5)
    dense = ti.field(ti.f64, shape=4)
    ref = ti.ndarray(ti.f64, shape=4)
    dense.from_numpy(np.array([-2.0, 3.5, -4.25, 0.5], dtype=np.float64))
    ref.from_numpy(np.array([-1.0, 1.0, -3.0, -0.5], dtype=np.float64))
    workspace = ti.algorithms.MetricWorkspace(max_items=4)

    assert ti.algorithms.max_abs_delta(
        dense, ref, workspace=workspace
    ).to_float() == pytest.approx(2.5)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_native_graph_aot_rejects_native_nodes():
    _skip_if_native_check_unavailable()
    flags = ti.ndarray(ti.i32, shape=4)
    flags.from_numpy(np.array([0, 1, 0, 2], dtype=np.int32))
    result = ti.algorithms.count_if(flags)
    builder = ti.graph.GraphBuilder()
    builder.append_native(result)
    graph = builder.compile()

    module = ti.aot.Module()
    with pytest.raises(Exception, match="Native graph replay is JIT-only"):
        module.add_graph("native_check", graph)
