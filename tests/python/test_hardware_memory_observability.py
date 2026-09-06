import gc
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from tests import test_utils


def test_hardware_memory_schema_separates_known_requested_and_opaque_bytes():
    known = ti.hardware.HardwareMemoryComponent(
        "workspace",
        4096,
        True,
        "provider_generation",
        "provider",
        resident=True,
    )
    opaque = ti.hardware.HardwareMemoryComponent(
        "driver_state",
        None,
        False,
        "provider_generation",
        "driver",
        resident=True,
    )
    report = ti.hardware.HardwareMemoryReport(
        schema_version=ti.hardware.HARDWARE_MEMORY_SCHEMA_VERSION,
        provider="test_provider",
        backend="vulkan",
        lifecycle_state="ready",
        ownership_scope="provider_generation",
        components=(known, opaque),
    )

    assert report.known_resident_requested_bytes == 4096
    assert report.known_capacity_requested_bytes == 4096
    assert not report.resident_requested_bytes_complete
    assert report.opaque_component_count == 1
    assert report.to_dict()["components"][1]["requested_bytes"] is None
    with pytest.raises(FrozenInstanceError):
        report.lifecycle_state = "closed"


def test_spmm_plan_memory_missing_observation_and_retired_owner():
    from types import SimpleNamespace

    from taichi_forge.hardware._linalg import CusparseSpmmRecording
    from taichi_forge.lang.exception import TaichiRuntimeError

    native = SimpleNamespace()
    matrix = SimpleNamespace(matrix=native, _ensure_valid=lambda: None)
    # Exercise the report compatibility boundary without initializing CUDA or
    # fabricating an executable recording/retained execution certificate.
    recording = object.__new__(CusparseSpmmRecording)
    for key, value in {
        "matrix": matrix,
        "rhs_count": 8,
        "_algorithm_code": 1,
        "_plan_info_snapshot": None,
        "_memory_resources": {"spmm_workspace_reserved_bytes": 99999},
    }.items():
        object.__setattr__(recording, key, value)
    assert recording.plan_info()["status"] == "unavailable"
    report = recording.memory_report()
    assert report.components[0].requested_bytes is None
    assert not report.resident_requested_bytes_complete
    assert report.known_resident_requested_bytes == 0

    native._cuda_cusparse_spmm_plan_info = lambda rhs, algorithm: {
        "prepared": True,
        "preprocess_attempted": True,
        "preprocessed": False,
        "preprocess_error": 10,
        "workspace_bytes": 4096,
    }
    info = recording.plan_info()
    assert info["preprocess_error"] == 10
    info["workspace_bytes"] = 99999

    def retired():
        raise TaichiRuntimeError("the matrix runtime was reset")

    matrix._ensure_valid = retired
    report = recording.memory_report()
    assert report.lifecycle_state == "runtime_invalid"
    assert report.known_capacity_requested_bytes == 4096
    assert report.known_resident_requested_bytes == 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_provider_memory_reports_do_not_invent_vendor_workspace_bytes():
    opaque_reports = (
        ti.hardware.linalg.CublasGemmRecording(16, 16, 16).memory_report(),
        ti.hardware.matrix.CudaMatrixMmaRecording(1).memory_report(),
    )
    for report in opaque_reports:
        assert report.backend == "cuda"
        assert report.known_resident_requested_bytes == 0
        assert report.resident_requested_bytes_complete
        assert report.opaque_component_count == 0

    if ti.hardware.fft.is_available():
        plan = ti.hardware.fft.CufftPlan1D(8)
        report = plan.memory_report()
        assert report.provider == "cufft_c2c_1d"
        workspace = next(
            component
            for component in report.components
            if component.name == "automatic_workspace"
        )
        assert workspace.requested_bytes_exact
        assert report.known_resident_requested_bytes == workspace.requested_bytes
        assert not report.resident_requested_bytes_complete
        assert report.opaque_component_count == 1
        graph_builder = ti.graph.GraphBuilder()
        graph_builder.append_native(plan.record(), admission="auto")
        graph = graph_builder.compile()
        graph_report = graph.execution_stats()
        assert graph_report.provider_memory == (report,)
        assert graph_report.memory.provider_generation_report_count == 1
        plan.close()
        assert plan.memory_report().lifecycle_state == "closed"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparse_graph_reports_provider_generation_memory_when_available():
    if not ti.hardware.linalg.cusparse_is_available():
        pytest.skip("a compatible user-provided cuSPARSE library is unavailable")

    size = 4
    builder = ti.linalg.SparseMatrixBuilder(size, size, max_num_triplets=size)

    @ti.kernel
    def fill(matrix: ti.types.sparse_matrix_builder()):
        for i in range(size):
            matrix[i, i] += ti.cast(i + 1, ti.f32)

    fill(builder)
    matrix = builder.build()
    recording = ti.hardware.linalg.CusparseSpmvRecording(matrix)
    report = recording.memory_report()
    assert report.provider == "cusparse_spmv_f32"
    assert report.known_resident_requested_bytes > 0
    assert not report.resident_requested_bytes_complete

    graph_builder = ti.graph.GraphBuilder()
    graph_builder.append_native(recording, admission="auto")
    graph_builder.append_native(
        ti.hardware.linalg.CusparseSpmvRecording(matrix), admission="auto"
    )
    graph = graph_builder.compile()
    graph_report = graph.execution_stats()
    assert graph_report.provider_memory == (report,)
    assert graph_report.memory.provider_generation_report_count == 1
    assert (
        graph_report.memory.provider_generation_known_resident_requested_bytes
        == report.known_resident_requested_bytes
    )


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_ray_graph_deduplicates_scene_generation_memory():
    if not ti.hardware.ray.is_available():
        pytest.skip("Vulkan ray-query features are unavailable")

    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    updated = ti.ndarray(ti.f32, shape=(3, 3))
    indices = ti.ndarray(ti.i32, shape=(1, 3))
    rays = ti.ndarray(ti.f32, shape=(1, 8))
    hits = ti.ndarray(ti.f32, shape=(1, 4))
    vertex_values = np.array(
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32
    )
    vertices.from_numpy(vertex_values)
    updated.from_numpy(vertex_values + np.array([0, 0, 0.5], np.float32))
    indices.from_numpy(np.array([[0, 1, 2]], dtype=np.int32))
    rays.from_numpy(
        np.array([[0, 0, 1, 0.001, 0, 0, -1, 100]], dtype=np.float32)
    )

    scene = ti.hardware.ray.TriangleScene(vertices, indices)
    ready = scene.memory_report()
    assert ready.provider == "vulkan_triangle_ray"
    assert ready.known_resident_requested_bytes > 0
    assert ready.known_capacity_requested_bytes == ready.known_resident_requested_bytes
    assert not ready.resident_requested_bytes_complete

    builder = ti.graph.GraphBuilder()
    builder.append_native(scene.record_refit(vertices="positions"), admission="auto")
    builder.append_native(scene.record(1), admission="auto")
    graph = builder.compile()
    graph.run({"positions": updated, "rays": rays, "hits": hits})
    ti.sync()

    report = graph.execution_stats()
    assert report.schema_version == 7
    assert report.provider_memory == (scene.memory_report(),)
    assert report.memory.provider_generation_report_count == 1
    assert report.memory.provider_generation_known_resident_requested_bytes == (
        ready.known_resident_requested_bytes
    )

    scene.close()
    closed = scene.memory_report()
    assert closed.lifecycle_state == "closed"
    assert closed.known_resident_requested_bytes == 0
    assert closed.known_capacity_requested_bytes > 0


@pytest.mark.skipif(not _ti_core.GGUI_AVAILABLE, reason="GGUI Not Available")
@test_utils.test(arch=ti.vulkan)
def test_vulkan_raster_memory_report_keeps_hidden_allocations_opaque():
    raster_pass = ti.hardware.raster.RasterPass((16, 16))
    try:
        report = raster_pass.memory_report()
        assert report.provider == "vulkan_raster_pass"
        assert report.known_resident_requested_bytes == 0
        assert not report.resident_requested_bytes_complete
    finally:
        raster_pass.destroy()
    assert raster_pass.memory_report().lifecycle_state == "closed"


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_vulkan_timestamp_ticket_releases_device_children_across_reinit():
    command = ti.graph.VulkanBufferCommand
    recording = ti.graph.VulkanBufferCommandRecording(
        (
            command.fill_u32("destination", 64, 0),
            # This recording has an explicit barrier policy. Both transfer
            # commands write the same range; a barrier after copy is too late.
            command.buffer_barrier("destination"),
            command.copy("destination", "source", 64),
            command.memory_barrier(),
        )
    )

    for cycle in range(4):
        if cycle:
            ti.init(arch=ti.vulkan, enable_fallback=False)
        source = ti.ndarray(ti.i32, shape=16)
        destination = ti.ndarray(ti.i32, shape=16)
        unrelated = ti.ndarray(ti.i32, shape=16)
        values = np.arange(16, dtype=np.int32) + cycle * 19
        source.from_numpy(values)
        unrelated.fill(cycle)

        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="auto")
        graph = builder.compile()
        ticket = graph.submit(
            {"source": source, "destination": destination},
            telemetry="timestamps",
        )
        ticket.wait()
        pipeline = ticket.pipeline_report()
        np.testing.assert_array_equal(destination.to_numpy(), values)
        completion = impl.get_runtime().prog._debug_runtime_completion_stats()
        assert completion["pending"] == 0
        assert completion["retained_ndarrays"] == 0

        ti.reset()
        assert ticket.pipeline_report() == pipeline
        del graph, ticket, source, destination, unrelated
        gc.collect()

    ti.init(arch=ti.vulkan, enable_fallback=False)
    probe = ti.ndarray(ti.i32, shape=4)
    probe.fill(7)
    np.testing.assert_array_equal(probe.to_numpy(), np.full(4, 7, np.int32))
