import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.graph._ir import GraphAccess
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_cufft_plan_rejects_non_cuda_runtime_and_bad_contracts():
    assert not ti.hardware.fft.is_available()
    cache = ti.hardware.fft.cache_statistics()
    assert cache.create_requests == 0
    assert cache.cache_hits == 0
    assert cache.cache_misses == 0
    assert cache.live_handles == 0
    assert cache.live_plans == 0
    assert cache.workspace_bytes_live == 0
    with pytest.raises(ValueError, match="length"):
        ti.hardware.fft.CufftPlan1D(0)
    with pytest.raises(ValueError, match="transform"):
        ti.hardware.fft.CufftPlan1D(8, transform="z2z")
    with pytest.raises(ValueError, match="rank 2 or 3"):
        ti.hardware.fft.CufftPlanND((8,))
    with pytest.raises(ValueError, match="cover"):
        ti.hardware.fft.CufftPlanND(
            (4, 8),
            input_layout=ti.hardware.fft.CufftLayout(embed=(4, 7)),
        )
    with pytest.raises(ValueError, match="overlap"):
        ti.hardware.fft.CufftPlanND(
            (4, 8),
            input_layout=ti.hardware.fft.CufftLayout(batch_distance=1),
        )
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
    assert resolved.provider_abi == "cufft-plan-many-dynamic-symbols-v3"

    recording = plan.record(direction="inverse", input="spectrum", output="signal")
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


@pytest.mark.parametrize("length", (7, 8))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_real_transforms_preserve_hermitian_layout_and_scale(length):
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    batch_count = 2
    rng = np.random.default_rng(20260824 + length)
    values = rng.standard_normal((batch_count, length)).astype(np.float32)
    hermitian_length = length // 2 + 1

    signal = ti.ndarray(ti.f32, shape=(batch_count, length))
    spectrum = ti.ndarray(ti.f32, shape=(batch_count, hermitian_length, 2))
    recovered = ti.ndarray(ti.f32, shape=(batch_count, length))
    signal.from_numpy(values)

    forward = ti.hardware.fft.CufftPlan1D(
        length, batch_count=batch_count, transform="r2c"
    )
    inverse = ti.hardware.fft.CufftPlan1D(
        length, batch_count=batch_count, transform="c2r"
    )
    try:
        assert forward.input_shape == (batch_count, length)
        assert forward.output_shape == (batch_count, hermitian_length, 2)
        assert inverse.input_shape == forward.output_shape
        assert inverse.output_shape == forward.input_shape
        assert inverse.inverse_scale == pytest.approx(1.0 / length)

        forward.execute(signal, spectrum)
        ti.sync()
        packed = spectrum.to_numpy()
        actual_spectrum = packed[..., 0] + 1j * packed[..., 1]
        np.testing.assert_allclose(
            actual_spectrum,
            np.fft.rfft(values, axis=-1),
            rtol=2e-5,
            atol=2e-5,
        )

        recording = inverse.record(input="spectrum", output="signal")
        assert recording.direction == "inverse"
        assert recording.output_scale == pytest.approx(1.0 / length)
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="auto")
        graph = builder.compile()
        graph.run({"spectrum": spectrum, "signal": recovered})
        ti.sync()
        np.testing.assert_allclose(
            recovered.to_numpy() * recording.output_scale,
            values,
            rtol=2e-5,
            atol=2e-5,
        )

        with pytest.raises(ValueError, match="forward"):
            forward.record(direction="inverse")
        with pytest.raises(ValueError, match="inverse"):
            inverse.record(direction="forward")
    finally:
        forward.close()
        inverse.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_batched_nd_compact_transforms_match_numpy():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    rng = np.random.default_rng(20260825)
    dimensions = (3, 5)
    batch_count = 2
    values = rng.standard_normal((batch_count, *dimensions)).astype(np.float32)
    signal = ti.ndarray(ti.f32, shape=values.shape)
    signal.from_numpy(values)
    forward = ti.hardware.fft.CufftPlanND(
        dimensions, batch_count=batch_count, transform="r2c"
    )
    inverse = ti.hardware.fft.CufftPlanND(
        dimensions, batch_count=batch_count, transform="c2r"
    )
    spectrum = ti.ndarray(ti.f32, shape=forward.output_shape)
    recovered = ti.ndarray(ti.f32, shape=inverse.output_shape)
    try:
        forward.execute(signal, spectrum)
        packed = spectrum.to_numpy()
        actual = packed[..., 0] + 1j * packed[..., 1]
        np.testing.assert_allclose(
            actual,
            np.fft.rfftn(values, axes=(-2, -1)),
            rtol=3e-5,
            atol=3e-5,
        )

        builder = ti.graph.GraphBuilder()
        builder.append_native(
            inverse.record(input="spectrum", output="signal"), admission="auto"
        )
        graph = builder.compile()
        graph.run({"spectrum": spectrum, "signal": recovered})
        ti.sync()
        np.testing.assert_allclose(
            recovered.to_numpy() * inverse.inverse_scale,
            values,
            rtol=3e-5,
            atol=3e-5,
        )
    finally:
        forward.close()
        inverse.close()

    dimensions = (2, 3, 4)
    complex_values = (
        rng.standard_normal((batch_count, *dimensions))
        + 1j * rng.standard_normal((batch_count, *dimensions))
    ).astype(np.complex64)
    packed_values = np.stack(
        (complex_values.real, complex_values.imag), axis=-1
    ).astype(np.float32)
    plan = ti.hardware.fft.CufftPlanND(
        dimensions, batch_count=batch_count, transform="c2c"
    )
    source = ti.ndarray(ti.f32, shape=plan.input_shape)
    output = ti.ndarray(ti.f32, shape=plan.output_shape)
    source.from_numpy(packed_values)
    try:
        plan.execute(source, output)
        ti.sync()
        packed = output.to_numpy()
        actual = packed[..., 0] + 1j * packed[..., 1]
        np.testing.assert_allclose(
            actual,
            np.fft.fftn(complex_values, axes=(-3, -2, -1)),
            rtol=3e-5,
            atol=3e-5,
        )
    finally:
        plan.close()


def _layout_offset(index, layout):
    offset = index[0]
    for axis in range(1, len(index)):
        offset = offset * layout.embed[axis] + index[axis]
    return offset * layout.stride


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_nd_explicit_embed_stride_and_batch_distance():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    dimensions = (3, 4)
    batch_count = 2
    input_layout = ti.hardware.fft.CufftLayout(
        embed=(3, 6), stride=2, batch_distance=36
    )
    output_layout = ti.hardware.fft.CufftLayout(
        embed=(3, 5), stride=1, batch_distance=15
    )
    plan = ti.hardware.fft.CufftPlanND(
        dimensions,
        batch_count=batch_count,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    rng = np.random.default_rng(20260826)
    values = (
        rng.standard_normal((batch_count, *dimensions))
        + 1j * rng.standard_normal((batch_count, *dimensions))
    ).astype(np.complex64)
    input_storage = np.full(plan.input_shape, -123.0, dtype=np.float32)
    for batch in range(batch_count):
        for index in np.ndindex(dimensions):
            element = batch * plan.input_layout.batch_distance + _layout_offset(
                index, plan.input_layout
            )
            input_storage[2 * element] = values[(batch, *index)].real
            input_storage[2 * element + 1] = values[(batch, *index)].imag

    source = ti.ndarray(ti.f32, shape=plan.input_shape)
    output = ti.ndarray(ti.f32, shape=plan.output_shape)
    source.from_numpy(input_storage)
    try:
        plan.execute(source, output)
        ti.sync()
        output_storage = output.to_numpy()
        actual = np.empty_like(values)
        for batch in range(batch_count):
            for index in np.ndindex(dimensions):
                element = batch * plan.output_layout.batch_distance + _layout_offset(
                    index, plan.output_layout
                )
                actual[(batch, *index)] = (
                    output_storage[2 * element] + 1j * output_storage[2 * element + 1]
                )
        np.testing.assert_allclose(
            actual,
            np.fft.fft2(values, axes=(-2, -1)),
            rtol=3e-5,
            atol=3e-5,
        )
    finally:
        plan.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cufft_plan_cache_reuses_plan_and_reports_workspace():
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT shared library is unavailable")

    before = ti.hardware.fft.cache_statistics()
    first = ti.hardware.fft.CufftPlanND((16, 16), batch_count=2)
    second = ti.hardware.fft.CufftPlanND((16, 16), batch_count=2)
    after_create = ti.hardware.fft.cache_statistics()
    assert after_create.create_requests == before.create_requests + 2
    assert after_create.cache_misses == before.cache_misses + 1
    assert after_create.cache_hits == before.cache_hits + 1
    assert after_create.live_handles == before.live_handles + 2
    assert after_create.live_plans == before.live_plans + 1

    report = first.memory_report()
    workspace = next(
        component
        for component in report.components
        if component.name == "automatic_workspace"
    )
    assert workspace.requested_bytes_exact
    assert workspace.requested_bytes == after_create.workspace_bytes_live
    assert report.opaque_component_count == 1

    program = ti.lang.impl.get_runtime().prog
    waits_before = program._runtime_statistics_snapshot()["synchronization"][
        "backend_waits"
    ]
    first.close()
    after_first_close = ti.hardware.fft.cache_statistics()
    assert after_first_close.live_handles == before.live_handles + 1
    assert after_first_close.live_plans == before.live_plans + 1
    assert (
        program._runtime_statistics_snapshot()["synchronization"]["backend_waits"]
        == waits_before
    )

    source = ti.ndarray(ti.f32, shape=second.input_shape)
    output = ti.ndarray(ti.f32, shape=second.output_shape)
    second.execute(source, output)
    ti.sync()
    telemetry = ti.hardware.telemetry().providers["cufft"]
    assert telemetry["plan_cache_hits"] == after_create.cache_hits
    assert telemetry["plan_live_plans"] == after_create.live_plans

    second.close()
    after_close = ti.hardware.fft.cache_statistics()
    assert after_close.live_handles == before.live_handles
    assert after_close.live_plans == before.live_plans
    assert after_close.workspace_bytes_live == before.workspace_bytes_live
