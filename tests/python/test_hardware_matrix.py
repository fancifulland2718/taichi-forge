import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_cuda_matrix_mma_rejects_non_cuda_runtime_and_bad_contracts():
    with pytest.raises(ValueError, match="batch_count"):
        ti.hardware.matrix.CudaMatrixMmaRecording(0)
    with pytest.raises(ValueError, match="unique"):
        ti.hardware.matrix.CudaMatrixMmaRecording(
            1, a="value", b="value", output="output"
        )

    recording = ti.hardware.matrix.CudaMatrixMmaRecording(1)
    assert recording.backend == "cuda"
    assert recording.command_count == 1
    assert recording.queue == "compute"
    assert recording.stream_binding == "runtime_ordered"
    assert recording.workspace_ownership == "none"
    assert recording.no_host_readback
    assert tuple(
        (effect.resource, effect.access) for effect in recording.resource_effects
    ) == (
        ("a", GraphAccess.READ),
        ("b", GraphAccess.READ),
        ("output", GraphAccess.WRITE),
    )

    with pytest.raises(RuntimeError, match="requires the CUDA backend"):
        recording.execute({"a": object(), "b": object(), "output": object()})
    with pytest.raises(RuntimeError, match="compiled for cuda"):
        ti.graph.GraphBuilder().append_native(recording, admission="auto")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_matrix_mma_executes_driver_ptx_directly_and_through_graph():
    if not ti.hardware.matrix.is_available():
        pytest.skip("CUDA WMMA requires NVIDIA compute capability 7.0 or newer")

    resolved = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "matrix.mma.cuda"
    )
    assert resolved.discovery == "available"
    assert resolved.enablement == "enabled"
    assert resolved.selection == "eligible"

    batch_count = 2
    rng = np.random.default_rng(20260823)
    a_values = (rng.standard_normal((batch_count, 16, 16)) * 0.125).astype(np.float16)
    b_values = (rng.standard_normal((batch_count, 16, 16)) * 0.125).astype(np.float16)
    expected = np.matmul(a_values.astype(np.float32), b_values.astype(np.float32))

    a = ti.ndarray(ti.f16, shape=(batch_count, 16, 16))
    b = ti.ndarray(ti.f16, shape=(batch_count, 16, 16))
    output = ti.ndarray(ti.f32, shape=(batch_count, 16, 16))
    a.from_numpy(a_values)
    b.from_numpy(b_values)

    ti.hardware.matrix.mma_f16_f32(a, b, output)
    ti.sync()
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-2, atol=2e-3)

    output.fill(0)
    recording = ti.hardware.matrix.CudaMatrixMmaRecording(batch_count)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    graph.run({"a": a, "b": b, "output": output})
    ti.sync()
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-2, atol=2e-3)

    assert graph._debug_info["optimization"] == {
        "backend": "cuda",
        "input_segments": 1,
        "output_segments": 1,
        "mixed_backend_regions": 0,
        "lowered_native_nodes": 0,
        "opaque_native_nodes": 0,
        "backend_command_nodes": 1,
    }

    ti.reset()
    with pytest.raises(RuntimeError, match="before ti.reset"):
        graph.run({"a": a, "b": b, "output": output})
