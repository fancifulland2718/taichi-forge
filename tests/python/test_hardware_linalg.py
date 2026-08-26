import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from taichi_forge.hardware._retained import retained_execution_contract
from tests import test_utils
from tests.python.hardware_provider_lifecycle_qualification import (
    stress_iterations,
)
from tests.python.hardware_process_memory import ProcessMemoryPlateau


@test_utils.test(arch=ti.cpu)
def test_cublas_gemm_contract_rejects_non_cuda_runtime_and_bad_arguments():
    with pytest.raises(ValueError, match="rows"):
        ti.hardware.linalg.CublasGemmRecording(0, 3, 4)
    with pytest.raises(ValueError, match="finite"):
        ti.hardware.linalg.CublasGemmRecording(2, 3, 4, alpha=np.inf)
    with pytest.raises(ValueError, match="unique"):
        ti.hardware.linalg.CublasGemmRecording(
            2, 3, 4, a="input", b="input", output="output"
        )

    recording = ti.hardware.linalg.CublasGemmRecording(
        2, 3, 4, alpha=2.0, beta=0.5
    )
    assert recording.backend == "cuda"
    assert recording.stream_binding == "runtime_ordered"
    assert recording.workspace_ownership == "none"
    assert recording.no_host_readback
    retained = retained_execution_contract(recording)
    assert retained.identity is None
    assert retained.automatic_selection_policy == "forbidden"
    assert retained is retained_execution_contract(
        ti.hardware.linalg.CublasGemmRecording(2, 3, 4)
    )
    assert tuple(
        (item.name, item.amortization_scope) for item in retained.cost_model.fixed_costs
    ) == (
        ("provider_library_load", "process"),
        ("provider_handle", "runtime_generation"),
    )
    assert retained.cost_model.scale_costs[0].dimensions == (
        "rows",
        "columns",
        "inner",
    )
    assert tuple(
        (effect.resource, effect.access) for effect in recording.resource_effects
    ) == (
        ("a", GraphAccess.READ),
        ("b", GraphAccess.READ),
        ("output", GraphAccess.READ_WRITE),
    )
    with pytest.raises(RuntimeError, match="requires the CUDA backend"):
        recording.execute({"a": object(), "b": object(), "output": object()})
    with pytest.raises(RuntimeError, match="compiled for cuda"):
        ti.graph.GraphBuilder().append_native(recording, admission="auto")

    descriptor = ti.hardware.capability("linalg.gemm.cublas")
    assert descriptor.implementation_status == "existing_public"
    assert descriptor.graph_integration == "root_ordered"
    assert descriptor.public_api == "ti.hardware.linalg.gemm_f32"

    matrix = ti.linalg.SparseMatrix(n=4, m=4, dtype=ti.f32)
    with pytest.raises(RuntimeError, match="requires a CUDA SparseMatrix"):
        ti.hardware.linalg.CusparseSpmvRecording(matrix)
    with pytest.raises(TypeError, match="must be a SparseMatrix"):
        ti.hardware.linalg.CusparseSpmvRecording(object())
    with pytest.raises(TypeError, match="must be a SparseMatrix"):
        ti.hardware.linalg.CusparseSpmmRecording(object(), 4)
    with pytest.raises(ValueError, match="rhs_count"):
        ti.hardware.linalg.CusparseSpmmRecording(matrix, 1)
    with pytest.raises(ValueError, match="algorithm"):
        ti.hardware.linalg.CusparseSpmmRecording(matrix, 4, algorithm="fastest_magic")
    with pytest.raises(TypeError, match="must be a SparseMatrix"):
        ti.hardware.linalg.CusparseSpsvRecording(object())
    with pytest.raises(ValueError, match="fill_mode"):
        ti.hardware.linalg.CusparseSpsvRecording(matrix, fill_mode="full")
    with pytest.raises(TypeError, match="unit_diagonal"):
        ti.hardware.linalg.CusparseSpsvRecording(matrix, unit_diagonal=1)
    with pytest.raises(TypeError, match="transpose"):
        ti.hardware.linalg.CusparseSpsvRecording(matrix, transpose=1)
    with pytest.raises(ValueError, match="rhs_count"):
        ti.hardware.linalg.CusparseSpsmRecording(matrix, 1)
    with pytest.raises(ValueError, match="algorithm"):
        ti.hardware.linalg.CusparseSpsmRecording(matrix, 2, algorithm="fast")

    spmv_descriptor = ti.hardware.capability(
        "linalg.spmv.cusparse_explicit"
    )
    assert spmv_descriptor.graph_integration == "root_ordered"
    assert spmv_descriptor.public_api == "ti.hardware.linalg.spmv_f32"
    spmm_descriptor = ti.hardware.capability("linalg.spmm.cusparse_explicit")
    assert spmm_descriptor.graph_integration == "root_ordered"
    assert spmm_descriptor.public_api == "ti.hardware.linalg.spmm_f32"
    spsv_descriptor = ti.hardware.capability("linalg.spsv.cusparse_explicit")
    assert spsv_descriptor.graph_integration == "root_ordered"
    assert spsv_descriptor.activation_mode == "explicit_hardware_api"
    assert spsv_descriptor.public_api == "ti.hardware.linalg.spsv_f32"
    spsm_descriptor = ti.hardware.capability("linalg.spsm.cusparse_explicit")
    assert spsm_descriptor.graph_integration == "root_ordered"
    assert spsm_descriptor.activation_mode == "explicit_hardware_api"
    assert spsm_descriptor.public_api == "ti.hardware.linalg.spsm_f32"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cublas_gemm_executes_directly_and_through_graph():
    if not ti.hardware.linalg.is_available():
        pytest.skip("a compatible user-provided cuBLAS shared library is unavailable")

    rows, inner, columns = 17, 13, 9
    rng = np.random.default_rng(20260823)
    a_values = (rng.standard_normal((rows, inner)) * 0.25).astype(np.float32)
    b_values = (rng.standard_normal((inner, columns)) * 0.25).astype(np.float32)
    initial = (rng.standard_normal((rows, columns)) * 0.1).astype(np.float32)
    a = ti.ndarray(ti.f32, shape=a_values.shape)
    b = ti.ndarray(ti.f32, shape=b_values.shape)
    output = ti.ndarray(ti.f32, shape=initial.shape)
    a.from_numpy(a_values)
    b.from_numpy(b_values)
    output.from_numpy(initial)

    ti.hardware.linalg.gemm_f32(a, b, output, alpha=1.5, beta=0.25)
    ti.sync()
    expected = 1.5 * (a_values @ b_values) + 0.25 * initial
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-5, atol=2e-5)

    output.from_numpy(initial)
    recording = ti.hardware.linalg.CublasGemmRecording(
        rows, columns, inner, alpha=-0.75, beta=0.5
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    graph.run({"a": a, "b": b, "output": output})
    ti.sync()
    expected = -0.75 * (a_values @ b_values) + 0.5 * initial
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-5, atol=2e-5)
    assert graph._debug_info["optimization"]["backend_command_nodes"] == 1

    resolved = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "linalg.gemm.cublas"
    )
    assert resolved.discovery == "available"
    assert resolved.enablement == "enabled"
    assert resolved.selection == "eligible"

    square = ti.ndarray(ti.f32, shape=(4, 4))
    with pytest.raises(RuntimeError, match="must not alias"):
        ti.hardware.linalg.gemm_f32(square, square, square)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparse_spmv_executes_directly_and_through_graph():
    if not ti.hardware.linalg.cusparse_is_available():
        pytest.skip(
            "a compatible user-provided cuSPARSE shared library is unavailable"
        )

    n = 4
    builder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=8)

    @ti.kernel
    def fill(a: ti.types.sparse_matrix_builder()):
        for i in range(n):
            a[i, i] += ti.cast(i + 2, ti.f32)
            if i + 1 < n:
                a[i, i + 1] += 0.5

    fill(builder)
    matrix = builder.build()
    input_values = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    expected = np.array([1.0, -5.75, 3.5, 15.0], dtype=np.float32)
    input_array = ti.ndarray(ti.f32, shape=n)
    output = ti.ndarray(ti.f32, shape=n)
    input_array.from_numpy(input_values)

    ti.hardware.linalg.spmv_f32(matrix, input_array, output)
    ti.sync()
    np.testing.assert_allclose(output.to_numpy(), expected)

    output.fill(0)
    recording = ti.hardware.linalg.CusparseSpmvRecording(matrix)
    assert recording.replay_mode == "stream_capture"
    assert recording.workspace_ownership == "provider_generation"
    retained = retained_execution_contract(recording)
    assert retained is retained_execution_contract(
        ti.hardware.linalg.CusparseSpmvRecording(matrix)
    )
    assert retained.identity.operation_id == "linalg.spmv.cusparse_explicit"
    assert retained.identity.provider_id == "cusparse"
    assert not retained.identity.persistent_cache_safe
    assert retained.workspace_ownership == "provider_generation"
    assert retained.concurrency_policy == "runtime_ordered"
    assert retained.automatic_selection_policy == "qualification_gated"
    assert retained.cost_model.scale_costs[0].dimensions == ("rows", "nonzeros")
    assert tuple(
        (effect.resource, effect.access) for effect in recording.resource_effects
    ) == (
        ("input", GraphAccess.READ),
        ("output", GraphAccess.WRITE),
    )
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.append_native(recording, admission="auto")
    graph = graph_builder.compile()
    assert len(graph._graph_stats) == 1
    graph.run({"input": input_array, "output": output})
    ti.sync()
    np.testing.assert_allclose(output.to_numpy(), expected)
    optimization = graph._debug_info["optimization"]
    assert optimization["mixed_backend_regions"] == 0
    assert graph._debug_info["native_count"] == 1
    assert "backend_command_nodes" not in optimization
    graph_stats = graph._graph_stats[0]
    assert graph_stats["captures"] == 0, {
        key: graph_stats.get(key)
        for key in (
            "attempts",
            "captures",
            "last_path",
            "last_fallback_reason",
            "fallbacks",
        )
    }
    assert graph_stats["last_path"] == "ordinary_fallback"
    assert graph_stats["last_fallback_reason"] == "structural_unsupported"
    assert graph._spec.lifetime_leases
    stats = matrix._debug_runtime_stats()
    assert stats["operations"]["spmv_calls"] == 2
    assert stats["operations"]["spmv_handle_creations"] == 1
    assert stats["operations"]["spmv_plan_builds"] == 1
    assert stats["operations"]["spmv_plan_reuses"] == 1
    if stats["provider"]["spmv_preprocess_available"]:
        assert stats["operations"]["spmv_preprocess_builds"] == 1
        assert stats["operations"]["spmv_preprocess_reuses"] == 1

    updated_values = ti.ndarray(ti.f32, shape=stats["identity"]["nnz"])
    updated_values.fill(1.0)
    matrix.update_values(updated_values)
    assert retained is retained_execution_contract(
        ti.hardware.linalg.CusparseSpmvRecording(matrix)
    )

    wrong = ti.ndarray(ti.f32, shape=n + 1)
    with pytest.raises(RuntimeError, match="shape"):
        ti.hardware.linalg.spmv_f32(matrix, wrong, output)
    with pytest.raises(RuntimeError, match="must not alias"):
        ti.hardware.linalg.spmv_f32(matrix, input_array, input_array)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparse_spmm_retains_plans_and_executes_directly_and_through_graph():
    if not ti.hardware.linalg.cusparse_spmm_is_available():
        pytest.skip("the loaded cuSPARSE library lacks generic SpMM symbols")

    rows, columns, rhs_count = 7, 5, 4
    builder = ti.linalg.SparseMatrixBuilder(rows, columns, max_num_triplets=rows * 2)

    @ti.kernel
    def fill(a: ti.types.sparse_matrix_builder()):
        for i in range(rows):
            a[i, i % columns] += ti.cast(i + 1, ti.f32)
            a[i, (i + 2) % columns] += 0.25

    fill(builder)
    matrix = builder.build()
    dense_matrix = np.zeros((rows, columns), dtype=np.float32)
    for i in range(rows):
        dense_matrix[i, i % columns] += i + 1
        dense_matrix[i, (i + 2) % columns] += 0.25
    input_values = np.arange(columns * rhs_count, dtype=np.float32).reshape(columns, rhs_count) / 7.0 - 0.5
    expected = dense_matrix @ input_values
    input_array = ti.ndarray(ti.f32, shape=input_values.shape)
    output = ti.ndarray(ti.f32, shape=expected.shape)
    input_array.from_numpy(input_values)

    ti.hardware.linalg.spmm_f32(matrix, input_array, output)
    ti.sync()
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-5, atol=2e-5)

    recording = ti.hardware.linalg.CusparseSpmmRecording(matrix, rhs_count)
    retained = retained_execution_contract(recording)
    assert retained.identity.operation_id == "linalg.spmm.cusparse_explicit"
    assert retained.identity.to_dict()["problem_scope"]["rhs_count"] == rhs_count
    assert retained.concurrency_policy == "single_inflight"
    assert retained.automatic_selection_policy == "forbidden"
    assert retained.cost_model.scale_costs[0].dimensions == (
        "rows",
        "nonzeros",
        "rhs_count",
    )
    assert tuple(item.name for item in retained.cost_model.fixed_costs) == (
        "provider_library_load",
        "handle_and_descriptors",
        "workspace_allocation",
        "graph_capture",
    )
    assert retained is retained_execution_contract(ti.hardware.linalg.CusparseSpmmRecording(matrix, rhs_count))

    output.fill(0)
    result = ti.ndarray(ti.f32, shape=expected.shape)

    @ti.kernel
    def finish(
        source: ti.types.ndarray(dtype=ti.f32, ndim=2),
        destination: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in source:
            destination[i, j] = source[i, j] + 1.0

    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2)
    result_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=2)
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.append_native(recording, admission="auto")
    graph_builder.dispatch(finish, output_arg, result_arg)
    graph = graph_builder.compile()
    graph.run({"input": input_array, "output": output, "result": result})
    ti.sync()
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(result.to_numpy(), expected + 1.0, rtol=2e-5, atol=2e-5)
    assert graph._graph_stats[0]["last_path"] == "cuda_capture"
    assert graph._graph_stats[0]["structural_fallbacks"] == 0

    deterministic_output = ti.ndarray(ti.f32, shape=expected.shape)
    ti.hardware.linalg.spmm_f32(
        matrix,
        input_array,
        deterministic_output,
        algorithm="deterministic",
    )
    ti.sync()
    np.testing.assert_allclose(deterministic_output.to_numpy(), expected, rtol=2e-5, atol=2e-5)

    stats = matrix._debug_runtime_stats()
    assert stats["provider"]["spmm_f32_available"]
    assert stats["operations"]["spmm_plan_builds"] == 2
    assert stats["operations"]["spmm_plan_reuses"] >= 1
    assert stats["resources"]["spmm_plan_count"] == 2
    assert stats["resources"]["spmm_dense_matrix_descriptor_count"] == 4
    assert stats["resources"]["spmm_workspace_reserved_bytes"] >= 0
    report = recording.memory_report()
    assert report.ownership_scope == "sparse_matrix_rhs_algorithm_generation"

    wrong = ti.ndarray(ti.f32, shape=(columns, rhs_count + 1))
    with pytest.raises(RuntimeError, match="shape"):
        ti.hardware.linalg.spmm_f32(matrix, wrong, output)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparse_triangular_retains_analysis_updates_values_and_captures_graph():
    if not (
        ti.hardware.linalg.cusparse_spsv_is_available()
        and ti.hardware.linalg.cusparse_spsm_is_available()
    ):
        pytest.skip("the loaded cuSPARSE library lacks retained SpSV/SpSM symbols")

    n, rhs_count = 6, 3
    builder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=2 * n - 1)

    @ti.kernel
    def fill(a: ti.types.sparse_matrix_builder()):
        for i in range(n):
            a[i, i] += ti.cast(i + 2, ti.f32)
            if i > 0:
                a[i, i - 1] += ti.cast(i, ti.f32) * 0.25

    fill(builder)
    matrix = builder.build()
    dense = np.diag(np.arange(2, n + 2, dtype=np.float32))
    dense[np.arange(1, n), np.arange(n - 1)] = np.arange(1, n, dtype=np.float32) * 0.25

    rhs_values = np.linspace(-1.25, 2.0, n, dtype=np.float32)
    rhs = ti.ndarray(ti.f32, shape=n)
    solution = ti.ndarray(ti.f32, shape=n)
    rhs.from_numpy(rhs_values)
    expected = np.linalg.solve(dense, rhs_values)

    ti.hardware.linalg.spsv_f32(matrix, rhs, solution)
    ti.sync()
    np.testing.assert_allclose(solution.to_numpy(), expected, rtol=2e-5, atol=2e-5)

    spsv_recording = ti.hardware.linalg.CusparseSpsvRecording(matrix)
    retained = retained_execution_contract(spsv_recording)
    assert retained.identity.operation_id == "linalg.spsv.cusparse_explicit"
    assert retained.identity.to_dict()["execution_scope"] == {
        "algorithm": "default",
        "capture_compatible": True,
        "fill_mode": "lower",
        "stream_binding": "runtime_ordered",
        "transpose": False,
        "unit_diagonal": False,
        "value_update": "in_place_retained_plan_update",
    }
    assert tuple(item.name for item in retained.cost_model.fixed_costs) == (
        "provider_library_load",
        "handle_and_descriptors",
        "triangular_analysis",
        "workspace_allocation",
        "graph_capture",
    )
    assert retained.cost_model.scale_costs[0].dimensions == (
        "rows",
        "nonzeros",
        "dependency_depth",
    )
    assert retained.concurrency_policy == "single_inflight"
    assert retained.automatic_selection_policy == "forbidden"

    transposed_solution = ti.ndarray(ti.f32, shape=n)
    ti.hardware.linalg.spsv_f32(
        matrix,
        rhs,
        transposed_solution,
        transpose=True,
    )
    ti.sync()
    np.testing.assert_allclose(
        transposed_solution.to_numpy(),
        np.linalg.solve(dense.T, rhs_values),
        rtol=2e-5,
        atol=2e-5,
    )
    unit_solution = ti.ndarray(ti.f32, shape=n)
    unit_dense = dense.copy()
    np.fill_diagonal(unit_dense, 1.0)
    ti.hardware.linalg.spsv_f32(
        matrix,
        rhs,
        unit_solution,
        unit_diagonal=True,
    )
    ti.sync()
    np.testing.assert_allclose(
        unit_solution.to_numpy(),
        np.linalg.solve(unit_dense, rhs_values),
        rtol=2e-5,
        atol=2e-5,
    )

    rhs_matrix_values = np.arange(n * rhs_count, dtype=np.float32).reshape(n, rhs_count) / 5.0 - 1.0
    rhs_matrix = ti.ndarray(ti.f32, shape=rhs_matrix_values.shape)
    solution_matrix = ti.ndarray(ti.f32, shape=rhs_matrix_values.shape)
    rhs_matrix.from_numpy(rhs_matrix_values)
    expected_matrix = np.linalg.solve(dense, rhs_matrix_values)
    ti.hardware.linalg.spsm_f32(matrix, rhs_matrix, solution_matrix)
    ti.sync()
    np.testing.assert_allclose(solution_matrix.to_numpy(), expected_matrix, rtol=2e-5, atol=2e-5)

    spsm_recording = ti.hardware.linalg.CusparseSpsmRecording(matrix, rhs_count)
    spsm_retained = retained_execution_contract(spsm_recording)
    assert spsm_retained.automatic_selection_policy == "forbidden"
    assert spsm_retained.cost_model.scale_costs[0].dimensions == (
        "rows",
        "nonzeros",
        "dependency_depth",
        "rhs_count",
    )
    matrix_graph_result = ti.ndarray(ti.f32, shape=rhs_matrix_values.shape)

    @ti.kernel
    def finish_matrix(
        source: ti.types.ndarray(dtype=ti.f32, ndim=2),
        destination: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i, j in source:
            destination[i, j] = source[i, j] - 1.0

    matrix_output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "matrix_output", ti.f32, ndim=2
    )
    matrix_result_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "matrix_result", ti.f32, ndim=2
    )
    spsm_graph_builder = ti.graph.GraphBuilder()
    spsm_graph_builder.append_native(
        ti.hardware.linalg.CusparseSpsmRecording(
            matrix,
            rhs_count,
            input="matrix_input",
            output="matrix_output",
        ),
        admission="auto",
    )
    spsm_graph_builder.dispatch(
        finish_matrix, matrix_output_arg, matrix_result_arg
    )
    spsm_graph = spsm_graph_builder.compile()
    solution_matrix.fill(0)
    spsm_graph.run(
        {
            "matrix_input": rhs_matrix,
            "matrix_output": solution_matrix,
            "matrix_result": matrix_graph_result,
        }
    )
    ti.sync()
    np.testing.assert_allclose(
        solution_matrix.to_numpy(), expected_matrix, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        matrix_graph_result.to_numpy(), expected_matrix - 1.0, rtol=2e-5, atol=2e-5
    )
    assert spsm_graph._graph_stats[0]["last_path"] == "cuda_capture"

    graph_result = ti.ndarray(ti.f32, shape=n)

    @ti.kernel
    def finish(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in source:
            destination[i] = source[i] + 1.0

    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    result_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=1)
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.append_native(spsv_recording, admission="auto")
    graph_builder.dispatch(finish, output_arg, result_arg)
    graph = graph_builder.compile()
    solution.fill(0)
    graph.run({"input": rhs, "output": solution, "result": graph_result})
    ti.sync()
    first_graph_result = solution.to_numpy()
    np.testing.assert_allclose(first_graph_result, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(graph_result.to_numpy(), expected + 1.0, rtol=2e-5, atol=2e-5)
    assert graph._graph_stats[0]["last_path"] == "cuda_capture", {
        key: graph._graph_stats[0].get(key)
        for key in (
            "attempts",
            "last_path",
            "last_fallback_reason",
            "structural_fallbacks",
            "runtime_fallbacks",
        )
    }
    assert graph._graph_stats[0]["structural_fallbacks"] == 0

    graph.run({"input": rhs, "output": solution, "result": graph_result})
    ti.sync()
    assert np.array_equal(solution.to_numpy(), first_graph_result)

    # CSR order is [diag(0), subdiag(1), diag(1), ...]. Updating the existing
    # value buffer must preserve analysis state and an already captured Graph.
    updated_host_values = np.empty(2 * n - 1, dtype=np.float32)
    updated_host_values[0] = 3.0
    for i in range(1, n):
        updated_host_values[2 * i - 1] = 0.125 * i
        updated_host_values[2 * i] = i + 3.0
    updated_values = ti.ndarray(ti.f32, shape=2 * n - 1)
    updated_values.from_numpy(updated_host_values)
    matrix.update_values(updated_values)
    updated_dense = np.diag(np.arange(3, n + 3, dtype=np.float32))
    updated_dense[np.arange(1, n), np.arange(n - 1)] = np.arange(1, n, dtype=np.float32) * 0.125
    updated_expected = np.linalg.solve(updated_dense, rhs_values)
    graph.run({"input": rhs, "output": solution, "result": graph_result})
    ti.sync()
    np.testing.assert_allclose(solution.to_numpy(), updated_expected, rtol=2e-5, atol=2e-5)
    updated_expected_matrix = np.linalg.solve(updated_dense, rhs_matrix_values)
    spsm_graph.run(
        {
            "matrix_input": rhs_matrix,
            "matrix_output": solution_matrix,
            "matrix_result": matrix_graph_result,
        }
    )
    ti.sync()
    np.testing.assert_allclose(
        solution_matrix.to_numpy(), updated_expected_matrix, rtol=2e-5, atol=2e-5
    )

    stats = matrix._debug_runtime_stats()
    assert stats["provider"]["spsv_f32_available"]
    assert stats["provider"]["spsm_f32_available"]
    assert stats["provider"]["spsv_value_update_available"]
    assert stats["provider"]["spsm_value_update_available"]
    assert stats["operations"]["spsv_plan_builds"] == 3
    assert stats["operations"]["spsv_plan_reuses"] >= 2
    assert stats["operations"]["spsm_plan_builds"] == 1
    assert stats["operations"]["spsm_plan_reuses"] >= 2
    assert stats["operations"]["spsv_value_updates"] == 3
    assert stats["operations"]["spsm_value_updates"] == 1
    assert stats["resources"]["spsv_plan_count"] == 3
    assert stats["resources"]["spsm_plan_count"] == 1
    assert spsv_recording.memory_report().ownership_scope == "sparse_matrix_triangle_rhs_generation"

    with pytest.raises(RuntimeError, match="must not alias"):
        ti.hardware.linalg.spsv_f32(matrix, rhs, rhs)
    wrong = ti.ndarray(ti.f32, shape=n + 1)
    with pytest.raises(RuntimeError, match="shape"):
        ti.hardware.linalg.spsv_f32(matrix, wrong, solution)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_vendor_commands_preserve_cross_provider_graph_order():
    if not (
        ti.hardware.linalg.cublas_is_available()
        and ti.hardware.linalg.cusparse_is_available()
    ):
        pytest.skip("compatible cuBLAS and cuSPARSE libraries are required")

    n = 4
    sparse_builder = ti.linalg.SparseMatrixBuilder(
        n, n, max_num_triplets=n
    )

    @ti.kernel
    def fill_sparse(a: ti.types.sparse_matrix_builder()):
        for i in range(n):
            a[i, i] += ti.cast(i + 1, ti.f32)

    @ti.kernel
    def reduce_rows(
        matrix: ti.types.ndarray(dtype=ti.f32, ndim=2),
        vector: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in vector:
            total = 0.0
            for j in ti.static(range(n)):
                total += matrix[i, j]
            vector[i] = total

    @ti.kernel
    def finish(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1),
        destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in source:
            destination[i] = source[i] + 1.0

    fill_sparse(sparse_builder)
    sparse = sparse_builder.build()
    a_values = np.arange(1, n * n + 1, dtype=np.float32).reshape(n, n)
    b_values = np.eye(n, dtype=np.float32) * 0.5
    a = ti.ndarray(ti.f32, shape=(n, n))
    b = ti.ndarray(ti.f32, shape=(n, n))
    dense_output = ti.ndarray(ti.f32, shape=(n, n))
    row_sums = ti.ndarray(ti.f32, shape=n)
    sparse_output = ti.ndarray(ti.f32, shape=n)
    result = ti.ndarray(ti.f32, shape=n)
    a.from_numpy(a_values)
    b.from_numpy(b_values)

    dense_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "dense_output", ti.f32, ndim=2
    )
    rows_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "row_sums", ti.f32, ndim=1
    )
    sparse_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "sparse_output", ti.f32, ndim=1
    )
    result_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(
        ti.hardware.linalg.CublasGemmRecording(
            n, n, n, output="dense_output"
        ),
        admission="auto",
    )
    builder.dispatch(reduce_rows, dense_arg, rows_arg)
    builder.append_native(
        ti.hardware.linalg.CusparseSpmvRecording(
            sparse, input="row_sums", output="sparse_output"
        ),
        admission="auto",
    )
    builder.dispatch(finish, sparse_arg, result_arg)
    graph = builder.compile()
    assert len(graph._graph_stats) == 1
    graph.run(
        {
            "a": a,
            "b": b,
            "dense_output": dense_output,
            "row_sums": row_sums,
            "sparse_output": sparse_output,
            "result": result,
        }
    )
    ti.sync()

    expected_dense = a_values @ b_values
    expected_rows = expected_dense.sum(axis=1)
    expected = expected_rows * np.arange(1, n + 1, dtype=np.float32) + 1
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6)
    optimization = graph._debug_info["optimization"]
    # cuBLAS remains an ordinary backend command while the two Taichi kernels
    # and fixed-plan cuSPARSE dispatch were inserted into one CUDA Graph during
    # builder assembly, before the later mixed-region lowering pass.
    assert optimization["backend_command_nodes"] == 1
    assert optimization["mixed_backend_regions"] == 0
    assert graph._debug_info["native_count"] == 2
    assert graph._graph_stats[0]["captures"] == 1
    assert graph._graph_stats[0]["last_path"] == "cuda_capture"


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_runtime_owned_provider_replay_plateaus():
    """Stress runtime-owned handles without pretending they have close()."""

    if not (
        ti.hardware.linalg.cublas_is_available()
        and ti.hardware.linalg.cusparse_is_available()
    ):
        pytest.skip("compatible cuBLAS and cuSPARSE libraries are required")

    iterations = stress_iterations(64)
    n = 4
    a = ti.ndarray(ti.f32, shape=(n, n))
    b = ti.ndarray(ti.f32, shape=(n, n))
    product = ti.ndarray(ti.f32, shape=(n, n))
    a_values = np.arange(n * n, dtype=np.float32).reshape(n, n) / 16.0
    b_values = np.eye(n, dtype=np.float32) * 2.0
    a.from_numpy(a_values)
    b.from_numpy(b_values)

    sparse_builder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=n)

    @ti.kernel
    def fill_diagonal(builder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            builder[i, i] += ti.cast(i + 1, ti.f32)

    fill_diagonal(sparse_builder)
    matrix = sparse_builder.build()
    vector = ti.ndarray(ti.f32, shape=n)
    spmv_output = ti.ndarray(ti.f32, shape=n)
    vector_values = np.arange(1, n + 1, dtype=np.float32)
    vector.from_numpy(vector_values)
    gemm = ti.hardware.linalg.CublasGemmRecording(n, n, n)
    spmv = ti.hardware.linalg.CusparseSpmvRecording(matrix)
    spmm = None
    spmm_input = None
    spmm_output = None
    bindings_spmm = None
    if ti.hardware.linalg.cusparse_spmm_is_available():
        spmm_input = ti.ndarray(ti.f32, shape=(n, 2))
        spmm_output = ti.ndarray(ti.f32, shape=(n, 2))
        spmm_input.from_numpy(np.column_stack((vector_values, -vector_values)))
        spmm = ti.hardware.linalg.CusparseSpmmRecording(matrix, 2)
        bindings_spmm = {"input": spmm_input, "output": spmm_output}
    spsv = None
    spsv_output = None
    bindings_spsv = None
    spsm = None
    spsm_input = None
    spsm_output = None
    bindings_spsm = None
    if (
        ti.hardware.linalg.cusparse_spsv_is_available()
        and ti.hardware.linalg.cusparse_spsm_is_available()
    ):
        spsv_output = ti.ndarray(ti.f32, shape=n)
        spsv = ti.hardware.linalg.CusparseSpsvRecording(matrix)
        bindings_spsv = {"input": vector, "output": spsv_output}
        spsm_input = ti.ndarray(ti.f32, shape=(n, 2))
        spsm_input.from_numpy(np.column_stack((vector_values, -vector_values)))
        spsm_output = ti.ndarray(ti.f32, shape=(n, 2))
        spsm = ti.hardware.linalg.CusparseSpsmRecording(matrix, 2)
        bindings_spsm = {"input": spsm_input, "output": spsm_output}
    bindings_gemm = {"a": a, "b": b, "output": product}
    bindings_spmv = {"input": vector, "output": spmv_output}

    gemm.execute(bindings_gemm)
    spmv.execute(bindings_spmv)
    if spmm is not None:
        spmm.execute(bindings_spmm)
    if spsv is not None:
        spsv.execute(bindings_spsv)
        spsm.execute(bindings_spsm)
    ti.sync()
    program = ti.lang.impl.get_runtime().prog
    baseline_memory = program._runtime_statistics_snapshot()["memory"]
    baseline_sparse = matrix._debug_runtime_stats()["operations"]
    process_memory = ProcessMemoryPlateau(
        "cuda-runtime-owned-providers", ("cuda-cublas", "cuda-cusparse")
    )
    process_memory.capture("before")
    assert baseline_sparse["spmv_handle_creations"] == 1
    assert baseline_sparse["spmv_plan_builds"] == 1
    if spmm is not None:
        assert baseline_sparse["spmm_plan_builds"] == 1
        assert baseline_sparse["spmm_plan_reuses"] == 0
    if spsv is not None:
        assert baseline_sparse["spsv_plan_builds"] == 1
        assert baseline_sparse["spsv_plan_reuses"] == 0
        assert baseline_sparse["spsm_plan_builds"] == 1
        assert baseline_sparse["spsm_plan_reuses"] == 0

    midpoint = None
    for iteration in range(iterations):
        gemm.execute(bindings_gemm)
        spmv.execute(bindings_spmv)
        if spmm is not None:
            spmm.execute(bindings_spmm)
        if spsv is not None:
            spsv.execute(bindings_spsv)
            spsm.execute(bindings_spsm)
        if (iteration + 1) % 64 == 0:
            ti.sync()
        if iteration + 1 == max(1, iterations // 2):
            ti.sync()
            midpoint = program._runtime_statistics_snapshot()["memory"]
            process_memory.capture("midpoint")
    ti.sync()

    final_memory = program._runtime_statistics_snapshot()["memory"]
    process_memory.capture("after")
    process_memory.finish(iterations)
    final_sparse = matrix._debug_runtime_stats()["operations"]
    for key in ("live_resources", "retiring_resources", "inflight_resources"):
        assert midpoint[key] == baseline_memory[key]
        assert final_memory[key] == baseline_memory[key]
    assert final_sparse["spmv_handle_creations"] == 1
    assert final_sparse["spmv_plan_builds"] == 1
    assert final_sparse["spmv_plan_reuses"] >= iterations
    if spmm is not None:
        assert final_sparse["spmm_plan_builds"] == 1
        assert final_sparse["spmm_plan_reuses"] >= iterations
    if spsv is not None:
        assert final_sparse["spsv_plan_builds"] == 1
        assert final_sparse["spsv_plan_reuses"] >= iterations
        assert final_sparse["spsm_plan_builds"] == 1
        assert final_sparse["spsm_plan_reuses"] >= iterations
    np.testing.assert_allclose(product.to_numpy(), a_values @ b_values, rtol=1e-6)
    np.testing.assert_allclose(
        spmv_output.to_numpy(),
        vector_values * np.arange(1, n + 1, dtype=np.float32),
        rtol=1e-6,
    )
    if spmm is not None:
        expected_spmm = np.column_stack(
            (
                vector_values * np.arange(1, n + 1, dtype=np.float32),
                -vector_values * np.arange(1, n + 1, dtype=np.float32),
            )
        )
        np.testing.assert_allclose(spmm_output.to_numpy(), expected_spmm, rtol=1e-6)
    if spsv is not None:
        expected_spsv = vector_values / np.arange(1, n + 1, dtype=np.float32)
        np.testing.assert_allclose(spsv_output.to_numpy(), expected_spsv, rtol=1e-6)
        expected_spsm = np.column_stack((expected_spsv, -expected_spsv))
        np.testing.assert_allclose(spsm_output.to_numpy(), expected_spsm, rtol=1e-6)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_vendor_graphs_fail_closed_after_runtime_reset():
    if not (
        ti.hardware.linalg.cublas_is_available()
        and ti.hardware.linalg.cusparse_is_available()
    ):
        pytest.skip("compatible cuBLAS and cuSPARSE libraries are required")

    n = 2
    sparse_builder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=n)

    @ti.kernel
    def fill_diagonal(builder: ti.types.sparse_matrix_builder()):
        for i in range(n):
            builder[i, i] += 1.0

    fill_diagonal(sparse_builder)
    matrix = sparse_builder.build()
    gemm_builder = ti.graph.GraphBuilder()
    gemm_builder.append_native(
        ti.hardware.linalg.CublasGemmRecording(n, n, n), admission="auto"
    )
    gemm_graph = gemm_builder.compile()
    spmv_builder = ti.graph.GraphBuilder()
    spmv_builder.append_native(
        ti.hardware.linalg.CusparseSpmvRecording(matrix), admission="auto"
    )
    spmv_graph = spmv_builder.compile()
    spmm_graph = None
    if ti.hardware.linalg.cusparse_spmm_is_available():
        spmm_builder = ti.graph.GraphBuilder()
        spmm_builder.append_native(
            ti.hardware.linalg.CusparseSpmmRecording(matrix, 2),
            admission="auto",
        )
        spmm_graph = spmm_builder.compile()
    spsv_graph = None
    spsm_graph = None
    if (
        ti.hardware.linalg.cusparse_spsv_is_available()
        and ti.hardware.linalg.cusparse_spsm_is_available()
    ):
        spsv_builder = ti.graph.GraphBuilder()
        spsv_builder.append_native(
            ti.hardware.linalg.CusparseSpsvRecording(matrix),
            admission="auto",
        )
        spsv_graph = spsv_builder.compile()
        spsm_builder = ti.graph.GraphBuilder()
        spsm_builder.append_native(
            ti.hardware.linalg.CusparseSpsmRecording(matrix, 2),
            admission="auto",
        )
        spsm_graph = spsm_builder.compile()
    matrix_array = ti.ndarray(ti.f32, shape=(n, n))
    spmm_output = ti.ndarray(ti.f32, shape=(n, n))
    vector = ti.ndarray(ti.f32, shape=n)

    ti.reset()

    with pytest.raises(RuntimeError, match="compiled before ti.reset"):
        gemm_graph.run(
            {"a": matrix_array, "b": matrix_array, "output": matrix_array}
        )
    with pytest.raises(RuntimeError, match="compiled before ti.reset"):
        spmv_graph.run({"input": vector, "output": vector})
    if spmm_graph is not None:
        with pytest.raises(RuntimeError, match="compiled before ti.reset"):
            spmm_graph.run({"input": matrix_array, "output": spmm_output})
    if spsv_graph is not None:
        with pytest.raises(RuntimeError, match="compiled before ti.reset"):
            spsv_graph.run({"input": vector, "output": vector})
        with pytest.raises(RuntimeError, match="compiled before ti.reset"):
            spsm_graph.run({"input": matrix_array, "output": spmm_output})
