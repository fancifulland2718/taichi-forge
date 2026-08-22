import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_cufft_plan_rejects_non_cuda_runtime_and_bad_contracts():
    assert not ti.hardware.fft.is_available()
    with pytest.raises(ValueError, match="length"):
        ti.hardware.fft.CufftPlan1D(0)
    with pytest.raises(RuntimeError, match="requires the CUDA backend"):
        ti.hardware.fft.CufftPlan1D(8)

    descriptor = ti.hardware.capability("fft.transform.cufft")
    assert descriptor.dependency_tier == "lazy_external"
    assert descriptor.implementation_status == "existing_public"
    assert descriptor.scopes == ("python", "graph")
    assert descriptor.execution_kind == "external_library"
    assert descriptor.graph_support == "recordable"
    assert descriptor.workspace_ownership == "provider_owned"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_c2c_executes_directly_and_through_graph():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    batch_count = 2
    length = 8
    rng = np.random.default_rng(20260823)
    complex_values = (
        rng.standard_normal((batch_count, length))
        + 1j * rng.standard_normal((batch_count, length))
    ).astype(np.complex64)
    packed_values = np.stack(
        (complex_values.real, complex_values.imag), axis=-1
    ).astype(np.float32)

    source = ti.ndarray(ti.f32, shape=(batch_count, length, 2))
    spectrum = ti.ndarray(ti.f32, shape=(batch_count, length, 2))
    recovered = ti.ndarray(ti.f32, shape=(batch_count, length, 2))
    source.from_numpy(packed_values)

    plan = ti.hardware.fft.CufftPlan1D(length, batch_count=batch_count)
    assert plan.shape == (batch_count, length, 2)
    plan.execute(source, spectrum, direction="forward")
    ti.sync()
    spectrum_values = spectrum.to_numpy()
    spectrum_complex = spectrum_values[..., 0] + 1j * spectrum_values[..., 1]
    np.testing.assert_allclose(
        spectrum_complex,
        np.fft.fft(complex_values, axis=-1),
        rtol=2e-5,
        atol=2e-5,
    )

    resolved = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "fft.transform.cufft"
    )
    assert resolved.discovery == "available"
    assert resolved.enablement == "enabled"
    assert resolved.selection == "eligible"
    assert resolved.provider_abi == "cufft-basic-c2c-dynamic-symbols-v1"

    recording = plan.record(
        direction="inverse", input="spectrum", output="signal"
    )
    assert tuple(
        (effect.resource, effect.access) for effect in recording.resource_effects
    ) == (("spectrum", GraphAccess.READ), ("signal", GraphAccess.WRITE))
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    graph.run({"spectrum": spectrum, "signal": recovered})
    ti.sync()
    np.testing.assert_allclose(
        recovered.to_numpy(), packed_values * length, rtol=2e-5, atol=2e-5
    )
    assert graph._debug_info["optimization"]["backend_command_nodes"] == 1

    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        graph.run({"spectrum": spectrum, "signal": recovered})


@test_utils.test(arch=ti.cuda)
def test_cufft_recording_validation_is_fail_closed():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    plan = ti.hardware.fft.CufftPlan1D(8)
    try:
        with pytest.raises(ValueError, match="direction"):
            plan.record(direction="sideways")
        with pytest.raises(ValueError, match="differ"):
            plan.record(input="data", output="data")

        wrong = ti.ndarray(ti.f32, shape=(8, 3))
        output = ti.ndarray(ti.f32, shape=(8, 2))
        with pytest.raises(RuntimeError, match="shape"):
            plan.execute(wrong, output)
        with pytest.raises(RuntimeError, match="distinct"):
            plan.execute(output, output)
    finally:
        plan.close()
