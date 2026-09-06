"""Optional Vulkan FFT execution, publication, and resource ownership contracts."""

import gc
import os

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware import _vulkan_fft
from taichi_forge.hardware._external_providers import passive_external_provider_status
from tests import test_utils


def _adapter():
    path = os.environ.get("TI_VKFFT_LIBRARY_PATH")
    if not path:
        pytest.skip("the optional Vulkan FFT adapter is not configured")
    # An explicitly configured but broken adapter must fail, not skip.
    return path


def _input(dimensions, batches):
    shape = ((batches,) if batches > 1 else ()) + dimensions + (2,)
    values = (
        np.random.default_rng(1473).uniform(-0.125, 0.125, shape).astype(np.float32)
    )
    data = ti.ndarray(ti.f32, shape=shape)
    data.from_numpy(values)
    return data, values


def _complex(values):
    return values[..., 0] + 1j * values[..., 1]


@test_utils.test(arch=ti.cpu)
def test_vulkan_fft_cold_contracts_and_passive_discovery(monkeypatch):
    data = ti.ndarray(ti.f32, shape=(8, 2))
    for dimensions in ((), (2, 2, 2, 2), (0,), (True,), (2.5,), (17,)):
        with pytest.raises(ValueError, match="rank|dimension|prime"):
            ti.hardware.fft.VulkanFftPlan(data, dimensions)
    for options in (
        {"batch_count": False},
        {"direction": "backward"},
        {"normalization": "ortho"},
    ):
        with pytest.raises(ValueError):
            ti.hardware.fft.VulkanFftPlan(data, (8,), **options)
    with pytest.raises(RuntimeError, match="Vulkan backend"):
        ti.hardware.fft.VulkanFftPlan(data, (8,))

    def unexpected_load(*args):
        raise AssertionError("passive status must not load or probe an adapter")

    monkeypatch.setattr(_vulkan_fft.ctypes, "CDLL", unexpected_load)
    monkeypatch.delenv("TI_VKFFT_LIBRARY_PATH", raising=False)
    status = passive_external_provider_status("vkfft")
    assert not status["library_loaded"]
    assert not status["native_facts"]["external_component_probed"]
    result = _vulkan_fft.probe_provider()
    assert result["discovery"] == "missing"
    assert not result["external_component_probed"]
    descriptor = ti.hardware.capability("fft.transform.vkfft")
    assert descriptor.backends == ("vulkan",)
    assert descriptor.graph_integration == "root_ordered"
    assert descriptor.update_policy == "immutable"
    assert descriptor.public_api == "ti.hardware.fft.VulkanFftPlan"
    assert ti.hardware.report().external_components_probed is False


@pytest.mark.parametrize(
    "dimensions,batches,normalized",
    [
        ((64,), 1, False),
        ((143,), 2, True),
        ((16, 8), 3, False),
        ((9, 7), 2, True),
        ((4, 4, 4), 2, True),
    ],
)
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_transforms_match_independent_reference(
    dimensions, batches, normalized
):
    adapter = _adapter()
    data, values = _input(dimensions, batches)
    before = passive_external_provider_status("vkfft")
    probe = _vulkan_fft.probe_provider(adapter)
    assert probe["discovery"] == "available", probe
    assert not probe["native_facts"]["execution_qualified"]
    assert passive_external_provider_status("vkfft") == before
    axes = tuple(range(-len(dimensions), 0))
    with ti.hardware.fft.VulkanFftPlan(
        data, dimensions, batch_count=batches, adapter_path=adapter
    ) as forward:
        forward.run()
        np.testing.assert_allclose(
            _complex(data.to_numpy()),
            np.fft.fftn(_complex(values), axes=axes),
            atol=2e-5,
            rtol=2e-5,
        )
        assert passive_external_provider_status("vkfft")["library_loaded"]
        facts = forward.statistics()
        assert facts["adapter_abi"] == 1 and facts["vkfft_version"] == 10304
        assert facts["device_vendor_id"] > 0
        facts["vkfft_version"] = 0
        assert forward.statistics()["vkfft_version"] == 10304
    with ti.hardware.fft.VulkanFftPlan(
        data,
        dimensions,
        batch_count=batches,
        direction="inverse",
        normalization="inverse" if normalized else "none",
        adapter_path=adapter,
    ) as inverse:
        inverse.run()
        scale = 1 if normalized else np.prod(dimensions)
        np.testing.assert_allclose(
            data.to_numpy(), values * scale, atol=2e-5, rtol=3e-5
        )
    with pytest.raises(RuntimeError, match="closed"):
        forward.run()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_rejects_noncompact_or_non_scalar_storage():
    adapter = _adapter()
    for data in (
        ti.ndarray(ti.i32, shape=(64, 2)),
        ti.ndarray(ti.f32, shape=(2, 64, 2)),
        ti.Vector.ndarray(2, ti.f32, shape=(64,)),
    ):
        with pytest.raises(ValueError, match="scalar f32 ndarray of shape"):
            ti.hardware.fft.VulkanFftPlan(data, (64,), adapter_path=adapter)
    data, _ = _input((64,), 1)
    with ti.hardware.fft.VulkanFftPlan(data, (64,), adapter_path=adapter) as plan:
        with pytest.raises(AttributeError):
            plan.direction = "inverse"
        with pytest.raises(ValueError, match="binding name"):
            plan.record(data="")


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_graph_order_publication_and_retained_submission(monkeypatch):
    adapter = _adapter()
    data, values = _input((16, 8), 2)
    other, _ = _input((16, 8), 2)
    forward = ti.hardware.fft.VulkanFftPlan(
        data, (16, 8), batch_count=2, adapter_path=adapter
    )
    inverse = ti.hardware.fft.VulkanFftPlan(
        data,
        (16, 8),
        batch_count=2,
        direction="inverse",
        normalization="inverse",
        adapter_path=adapter,
    )
    first_node = forward.record().compile()
    inverse_node = inverse.record().compile()
    assert (
        first_node.graph_ir_node.semantic_fingerprint
        != inverse_node.graph_ir_node.semantic_fingerprint
    )
    assert first_node.graph_physical_plan_id != inverse_node.graph_physical_plan_id
    with ti.hardware.fft.VulkanFftPlan(
        other, (16, 8), batch_count=2, adapter_path=adapter
    ) as equivalent:
        assert (
            equivalent.record().compile().graph_ir_node.semantic_fingerprint
            == first_node.graph_ir_node.semantic_fingerprint
        )
        assert (
            equivalent.record().compile().graph_physical_plan_id
            == first_node.graph_physical_plan_id
        )

    @ti.kernel
    def scale(array: ti.types.ndarray(dtype=ti.f32)):
        for index in ti.grouped(array):
            array[index] *= 2

    builder = ti.graph.GraphBuilder()
    arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "data", ti.f32, ndim=4)
    builder.dispatch(scale, arg)
    builder.append_native(forward.record())
    builder.dispatch(
        scale, arg
    )  # Same compiled kernel must rebind its pipeline after secondary execution.
    builder.append_native(inverse.record())
    builder.dispatch(scale, arg)
    graph = builder.compile()
    reports = graph._spec.provider_memory_reports()
    assert len(reports) == 2
    assert all(report.provider == "vkfft" for report in reports)
    with pytest.raises(RuntimeError, match="original data"):
        graph.bind({"data": other})
    bindings = graph.bind({"data": data})

    def unexpected_replay_observation(*args):
        raise AssertionError(
            "fixed bindings, lifetime, and memory must not be inspected on replay"
        )

    monkeypatch.setattr(
        _vulkan_fft._Recording, "validate_graph_bindings", unexpected_replay_observation
    )
    for plan in (forward, inverse):
        monkeypatch.setattr(
            plan, "validate_graph_lifetime", unexpected_replay_observation
        )
        monkeypatch.setattr(plan, "memory_report", unexpected_replay_observation)
        monkeypatch.setattr(plan, "statistics", unexpected_replay_observation)
    monkeypatch.setattr(_vulkan_fft, "passive_status", unexpected_replay_observation)
    for index in range(3):
        graph.run(bindings)
        np.testing.assert_allclose(
            data.to_numpy(), values * 8 ** (index + 1), atol=0.002, rtol=5e-5
        )
    graph.run(bindings)
    forward.close()
    inverse.close()
    # No explicit synchronization before close. Submitted native commands own their leases.
    np.testing.assert_allclose(data.to_numpy(), values * 8**4, atol=0.02, rtol=8e-5)
    with pytest.raises(RuntimeError, match="closed"):
        graph.run(bindings)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_reports_workspace_without_claiming_retirement():
    data, _ = _input((262144,), 2)
    with ti.hardware.fft.VulkanFftPlan(
        data, (262144,), batch_count=2, adapter_path=_adapter()
    ) as plan:
        statistics = plan.statistics()
        report = plan.memory_report()
        assert (
            report.known_resident_requested_bytes
            == statistics["persistent_allocation_bytes"]
        )
        assert (
            statistics["initialization_peak_allocation_bytes"]
            >= statistics["persistent_allocation_bytes"]
        )
        assert not report.resident_requested_bytes_complete
        identity = plan._graph_provider_memory_identity()
        plan.run()
    closed = plan.memory_report()
    assert closed.lifecycle_state == "closed"
    assert (
        closed.known_capacity_requested_bytes == report.known_capacity_requested_bytes
    )
    assert (
        not closed.resident_requested_bytes_complete
    )  # Command-buffer retention was not polled.
    assert plan._graph_provider_memory_identity() == identity
    ti.sync()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_runtime_reset_cannot_redirect_old_handle():
    adapter = _adapter()
    data, _ = _input((64,), 1)
    old = ti.hardware.fft.VulkanFftPlan(data, (64,), adapter_path=adapter)
    old.run()
    ti.reset()
    ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=False)
    data, values = _input((64,), 1)
    with ti.hardware.fft.VulkanFftPlan(data, (64,), adapter_path=adapter) as current:
        with pytest.raises(RuntimeError, match="closed"):
            old.run()
        with pytest.raises(RuntimeError, match="previous runtime"):
            old.record()
        old.close()
        current.run()
        np.testing.assert_allclose(
            _complex(data.to_numpy()),
            np.fft.fft(_complex(values)),
            atol=2e-5,
            rtol=2e-5,
        )
    del old
    gc.collect()
