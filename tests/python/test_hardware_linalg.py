import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from tests import test_utils


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
    assert descriptor.graph_support == "recordable"
    assert descriptor.public_api == "ti.hardware.linalg.gemm_f32"

    matrix = ti.linalg.SparseMatrix(n=4, m=4, dtype=ti.f32)
    with pytest.raises(RuntimeError, match="requires a CUDA SparseMatrix"):
        ti.hardware.linalg.CusparseSpmvRecording(matrix)
    with pytest.raises(TypeError, match="must be a SparseMatrix"):
        ti.hardware.linalg.CusparseSpmvRecording(object())

    spmv_descriptor = ti.hardware.capability(
        "linalg.spmv.cusparse_explicit"
    )
    assert spmv_descriptor.graph_support == "recordable"
    assert spmv_descriptor.public_api == "ti.hardware.linalg.spmv_f32"


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
    assert recording.workspace_ownership == "provider_generation"
    assert tuple(
        (effect.resource, effect.access) for effect in recording.resource_effects
    ) == (
        ("input", GraphAccess.READ),
        ("output", GraphAccess.WRITE),
    )
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.append_native(recording, admission="auto")
    graph = graph_builder.compile()
    graph.run({"input": input_array, "output": output})
    ti.sync()
    np.testing.assert_allclose(output.to_numpy(), expected)
    assert graph._debug_info["optimization"]["backend_command_nodes"] == 1
    assert graph._spec.lifetime_leases

    wrong = ti.ndarray(ti.f32, shape=n + 1)
    with pytest.raises(RuntimeError, match="shape"):
        ti.hardware.linalg.spmv_f32(matrix, wrong, output)
    with pytest.raises(RuntimeError, match="must not alias"):
        ti.hardware.linalg.spmv_f32(matrix, input_array, input_array)
