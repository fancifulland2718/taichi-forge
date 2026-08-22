from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.hardware import _capabilities
from tests import test_utils


_OPERATION_IDS = (
    "runtime.buffer_commands.vulkan",
    "raster.draw.vulkan",
    "ray.as_build.vulkan",
    "ray.as_refit.vulkan",
    "ray.query.batch.vulkan",
    "ray.query.inline.vulkan",
    "sampling.texture.vulkan",
    "sampling.texture.cuda",
    "matrix.mma.cuda",
    "matrix.mma.vulkan",
    "interop.external_buffer.cuda_vulkan",
    "linalg.gemm.cublas",
    "linalg.spmv.cusparse",
    "linalg.solve.cusolver",
    "fft.transform.cufft",
    "ray.query.batch.optix",
    "algorithms.primitives.cub",
    "internal.tile.async.cuda",
    "internal.raster.mesh_shader.vulkan",
)


def test_static_hardware_catalog_is_complete_immutable_and_schema_separate():
    ti.reset()
    operations = ti.hardware.operations()

    assert tuple(operation.operation_id for operation in operations) == _OPERATION_IDS
    assert all(
        operation.schema_version == ti.hardware.HARDWARE_CAPABILITY_SCHEMA_VERSION == 1 for operation in operations
    )
    assert ti.algorithms.PRIMITIVE_CAPABILITY_SCHEMA_VERSION == 2
    assert ti.hardware.DEPENDENCY_TIERS == (
        "core",
        "lazy_external",
        "build_external",
    )
    assert "wheel_variant" not in ti.hardware.DEPENDENCY_TIERS
    assert ti.hardware.LOAD_MODES == ("built_in", "runtime_lazy", "build_only")
    assert all(operation.backends for operation in operations)
    assert all(operation.scopes for operation in operations)
    assert all(operation.provider_id for operation in operations)

    with pytest.raises(FrozenInstanceError):
        operations[0].dependency_tier = "wheel_variant"
    with pytest.raises(TypeError):
        operations[0].backends[0] = "cpu"
    with pytest.raises(ValueError, match="unqualified implementations"):
        replace(
            ti.hardware.capability("matrix.mma.vulkan"),
            hardware_acceleration="qualified",
        )
    with pytest.raises(ValueError, match="load_mode must match"):
        replace(ti.hardware.capability("linalg.gemm.cublas"), load_mode="built_in")


def test_hardware_catalog_keeps_dependency_and_provider_axes_orthogonal():
    by_id = {operation.operation_id: operation for operation in ti.hardware.operations()}

    optix = by_id["ray.query.batch.optix"]
    cufft = by_id["fft.transform.cufft"]
    cub = by_id["algorithms.primitives.cub"]
    interop = by_id["interop.external_buffer.cuda_vulkan"]

    assert optix.dependency_tier == cufft.dependency_tier == "lazy_external"
    assert optix.provider_class == "vendor_hardware_runtime"
    assert cufft.provider_class == "vendor_algorithm"
    assert cufft.implementation_status == "existing_public"
    assert cufft.graph_support == "recordable"
    assert cufft.public_api == "ti.hardware.fft.CufftPlan1D"
    assert cub.dependency_tier == "build_external"
    assert cub.implementation_status == "reference_only"
    assert interop.dependency_tier == "core"
    assert interop.provider_class == "runtime_interop"

    for operation in by_id.values():
        expected_load_mode = {
            "core": "built_in",
            "lazy_external": "runtime_lazy",
            "build_external": "build_only",
        }[operation.dependency_tier]
        assert operation.load_mode == expected_load_mode
        assert (operation.dependency_name is None) == (operation.dependency_tier == "core")

    for operation in by_id.values():
        if operation.implementation_status in (
            "qualification_required",
            "planned",
            "reference_only",
        ):
            assert operation.hardware_acceleration not in ("guaranteed", "qualified")


def test_kernel_and_executable_scope_contracts_do_not_overlap_accidentally():
    for operation in ti.hardware.operations():
        if "kernel" in operation.scopes:
            assert operation.execution_kind == "kernel_intrinsic"
            assert operation.graph_support == "inline"
        if operation.execution_kind == "external_library":
            assert "kernel" not in operation.scopes
        if operation.operation_id == "ray.query.batch.optix":
            assert operation.scopes == ("python", "graph")
        if operation.operation_id == "ray.query.inline.vulkan":
            assert operation.scopes == ("kernel",)


def test_capability_and_provider_queries_are_stable_and_fail_closed():
    buffer_commands = ti.hardware.capability(
        "runtime.buffer_commands.vulkan"
    )
    assert buffer_commands.implementation_status == "existing_public"
    assert buffer_commands.hardware_acceleration == "qualified"
    assert buffer_commands.scopes == ("python", "graph")
    assert buffer_commands.execution_kind == "native_command"
    assert buffer_commands.public_api == (
        "ti.graph.VulkanBufferCommandRecording"
    )

    texture = ti.hardware.capability("sampling.texture.vulkan")
    assert texture.implementation_status == "existing_public"
    assert texture.hardware_acceleration == "qualified"
    assert texture.scopes == ("kernel",)
    assert texture.execution_kind == "kernel_intrinsic"
    assert texture.graph_support == "inline"
    assert texture.shapes_or_tiles == ("1D", "2D", "3D")
    assert texture.layouts == ("sampled_image", "storage_image")
    assert "SPIR-V OpImageSampleExplicitLod and OpImageFetch" in texture.requirements
    assert texture.deterministic is False

    cuda_texture = ti.hardware.capability("sampling.texture.cuda")
    assert cuda_texture.implementation_status == "planned"
    assert cuda_texture.hardware_acceleration == "implementation_defined"
    assert cuda_texture.notes == (
        "LLVM CUDA TextureOp lowering is not implemented.",
    )

    raster = ti.hardware.capability("raster.draw.vulkan")
    assert raster.semantic_family == "raster.draw"
    assert raster.public_api == "ti.hardware.raster.RasterPass"
    assert raster.implementation_status == "existing_public"
    assert raster.hardware_acceleration == "qualified"
    assert raster.scopes == ("python",)
    assert raster.execution_kind == "native_command"
    assert raster.graph_support == "unsupported"
    assert raster.workspace_ownership == "provider_owned"
    assert raster.layouts == ("mesh", "mesh_instance", "particles", "lines")
    assert raster.deterministic is False
    assert "Kernel calls are impossible" in raster.notes[0]
    assert ti.hardware.capability(raster.operation_id) is raster

    matrix = ti.hardware.capability("matrix.mma.cuda")
    assert matrix.semantic_family == "matrix.mma"
    assert matrix.public_api == "ti.hardware.matrix.mma_f16_f32"
    assert matrix.implementation_status == "existing_public"
    assert matrix.hardware_acceleration == "qualified"
    assert matrix.scopes == ("python", "graph")
    assert matrix.execution_kind == "native_command"
    assert matrix.graph_support == "recordable"
    assert matrix.stream_binding == "runtime_ordered"
    assert matrix.workspace_ownership == "none"
    assert matrix.shapes_or_tiles == ("m16n16k16", "compact batch")
    assert matrix.layouts == (
        "row_major_a",
        "row_major_b",
        "row_major_output",
    )
    assert "kernel calls remain unsupported" in matrix.notes[0]

    cusparse = ti.hardware.capability("linalg.spmv.cusparse")
    assert cusparse.implementation_status == "existing_public"
    assert cusparse.scopes == ("python",)
    assert cusparse.graph_support == "unsupported"
    assert cusparse.stream_binding == "runtime_ordered"
    assert cusparse.workspace_ownership == "provider_owned"
    assert cusparse.public_api == "ti.linalg.SparseMatrix.__matmul__"
    assert "selects cuSPARSE automatically on CUDA" in cusparse.notes[0]

    cusolver = ti.hardware.capability("linalg.solve.cusolver")
    assert cusolver.implementation_status == "existing_public"
    assert cusolver.scopes == ("python",)
    assert cusolver.graph_support == "unsupported"
    assert cusolver.stream_binding == "runtime_ordered"
    assert cusolver.workspace_ownership == "provider_owned"
    assert cusolver.public_api == "ti.linalg.SparseSolver"
    assert "selects this provider automatically" in cusolver.notes[0]
    assert cusolver.requirements == (
        "compatible cuSOLVER shared library",
        "compatible cuSPARSE shared library",
    )

    async_tile = ti.hardware.capability("internal.tile.async.cuda")
    assert async_tile.implementation_status == "existing_internal"
    assert async_tile.hardware_acceleration == "qualified"
    assert async_tile.scopes == ("internal",)
    assert async_tile.execution_kind == "kernel_intrinsic"
    assert async_tile.public_api is None
    assert async_tile.shapes_or_tiles == (
        "compiler-generated struct-for BLS >= 8192 bytes",
    )
    assert "no public cp.async or TMA API" in async_tile.notes[0]

    mesh_shader = ti.hardware.capability(
        "internal.raster.mesh_shader.vulkan"
    )
    assert mesh_shader.implementation_status == "planned"
    assert mesh_shader.hardware_acceleration == "none"
    assert "feature query and device enablement" in mesh_shader.requirements[0]
    assert "headers alone" in mesh_shader.notes[1]

    with pytest.raises(KeyError, match="unknown hardware operation"):
        ti.hardware.capability("missing.operation")
    with pytest.raises(TypeError, match="nonempty string"):
        ti.hardware.capability(None)

    providers = ti.hardware.providers()
    provider_ids = tuple(provider.provider_id for provider in providers)
    assert provider_ids == tuple(sorted(provider_ids))
    assert "cublas" in provider_ids
    assert "cub_reference" in provider_ids
    assert all(provider.operation_ids for provider in providers)
    assert next(provider for provider in providers if provider.provider_id == "cublas").to_dict() == {
        "schema_version": 1,
        "provider_id": "cublas",
        "dependency_tier": "lazy_external",
        "dependency_name": "cuBLAS",
        "load_mode": "runtime_lazy",
        "provider_class": "vendor_algorithm",
        "operation_ids": ("linalg.gemm.cublas",),
    }
    assert len({operation_id for provider in providers for operation_id in provider.operation_ids}) == len(
        ti.hardware.operations()
    )


def test_static_hardware_descriptor_serialization_is_plain_and_complete():
    descriptor = ti.hardware.capability("matrix.mma.vulkan")
    payload = descriptor.to_dict()

    assert payload["schema_version"] == 1
    assert payload["operation_id"] == descriptor.operation_id
    assert payload["requirements"] == ("VK_KHR_cooperative_matrix",)
    assert payload["hardware_acceleration"] == "implementation_defined"
    assert payload["implementation_status"] == "planned"
    assert payload["load_mode"] == "built_in"
    assert payload["resource_effects"] == ()
    assert payload["lifetime_policy"] == "runtime_generation"
    assert payload["update_policy"] == "immutable"
    assert payload["dtypes"] == ()
    assert payload["deterministic"] is None
    assert payload["fallback_provider"] is None
    assert payload["fallback_equivalent"] is None

    payload["requirements"] = ()
    assert descriptor.requirements == ("VK_KHR_cooperative_matrix",)


def test_passive_report_does_not_probe_or_enable_external_components(monkeypatch):
    def reject_implicit_probe(_provider_id):
        raise AssertionError("passive reports must not invoke a native D1 probe")

    monkeypatch.setattr(_capabilities, "_native_external_probe", reject_implicit_probe)
    monkeypatch.setattr(
        _capabilities,
        "_native_external_status",
        lambda provider_id: {
            "provider_id": provider_id,
            "library_loaded": False,
            "provider_abi": None,
            "provider_version": None,
            "native_facts": {
                "status_policy": "passive_existing_loader",
                "external_component_probed": False,
            },
        },
    )
    ti.reset()
    report = ti.hardware.report()

    assert report.schema_version == 1
    assert report.runtime_initialized is False
    assert report.backend is None
    assert report.external_components_probed is False
    assert set(report.compiled_backends) == {"cuda", "vulkan"}
    assert len(report.operations) == len(ti.hardware.operations())

    by_id = {operation.descriptor.operation_id: operation for operation in report.operations}
    cublas = by_id["linalg.gemm.cublas"]
    if cublas.unavailable_reason == "external_probe_not_requested":
        assert cublas.discovery is None
    assert cublas.enablement == "disabled"
    assert cublas.selection == "not_considered"
    assert cublas.unavailable_reason in (
        "external_probe_not_requested",
        "backend_not_compiled",
    )
    assert cublas.native_facts["external_component_probed"] is False
    assert cublas.provider_abi is None
    assert cublas.provider_version is None
    assert cublas.last_error is None
    assert cublas.failure_scope is None


def test_passive_report_observes_an_already_loaded_provider(monkeypatch):
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (True, "cuda", {"cuda": True, "vulkan": True}),
    )

    def provider_status(provider_id):
        return {
            "provider_id": provider_id,
            "library_loaded": provider_id == "cusparse",
            "provider_abi": f"{provider_id}-dynamic-symbols-v1",
            "provider_version": "12.6.3" if provider_id == "cusparse" else None,
            "native_facts": {
                "status_policy": "passive_existing_loader",
                "external_component_probed": False,
                "provider_enablement_changed": False,
                "provider_selection_changed": False,
                "generic_bsr_spmv_available": provider_id == "cusparse",
            },
        }

    monkeypatch.setattr(
        _capabilities, "_native_external_status", provider_status
    )
    report = ti.hardware.report()
    cusparse = next(
        operation
        for operation in report.operations
        if operation.descriptor.provider_id == "cusparse"
    )
    cublas = next(
        operation
        for operation in report.operations
        if operation.descriptor.provider_id == "cublas"
    )

    assert cusparse.discovery == "available"
    assert cusparse.enablement == "enabled"
    assert cusparse.selection == "eligible"
    assert cusparse.unavailable_reason == "none"
    assert cusparse.provider_abi == "cusparse-dynamic-symbols-v1"
    assert cusparse.provider_version == "12.6.3"
    assert cusparse.native_facts["generic_bsr_spmv_available"]
    assert not cusparse.native_facts["external_component_probed"]
    assert cublas.enablement == "disabled"
    assert cublas.selection == "not_considered"


def test_native_passive_status_does_not_load_external_libraries():
    before = {
        provider_id: dict(
            ti_core.cuda_external_library_status(provider_id)
        )
        for provider_id in ("cublas", "cusparse", "cusolver", "cufft")
    }
    report = ti.hardware.report()
    after = {
        provider_id: dict(
            ti_core.cuda_external_library_status(provider_id)
        )
        for provider_id in ("cublas", "cusparse", "cusolver", "cufft")
    }

    assert report.external_components_probed is False
    assert {
        provider_id: status["library_loaded"]
        for provider_id, status in after.items()
    } == {
        provider_id: status["library_loaded"]
        for provider_id, status in before.items()
    }
    assert all(
        status["native_facts"]["status_policy"]
        == "passive_existing_loader"
        for status in after.values()
    )
    assert all(
        not status["native_facts"]["external_component_probed"]
        for status in after.values()
    )


def test_multibackend_core_route_requires_every_backend(monkeypatch):
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (False, None, {"cuda": True, "vulkan": False}),
    )

    report = ti.hardware.report()
    interop = next(
        operation
        for operation in report.operations
        if operation.descriptor.operation_id == "interop.external_buffer.cuda_vulkan"
    )

    assert interop.discovery == "missing"
    assert interop.selection == "rejected"
    assert interop.unavailable_reason == "backend_not_compiled"
    assert interop.native_facts["provider_backends_compiled"] == ("cuda",)


def test_cuda_matrix_capability_rejects_an_unqualified_active_device(
    monkeypatch,
):
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (True, "cuda", {"cuda": True, "vulkan": True}),
    )
    monkeypatch.setattr(
        _capabilities,
        "_passive_core_statuses",
        lambda runtime_initialized, backend: {
            "matrix.mma.cuda": {
                "available": False,
                "native_facts": {"provider_available": False},
            }
        },
    )

    operation = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "matrix.mma.cuda"
    )
    assert operation.discovery == "incompatible"
    assert operation.enablement == "enabled"
    assert operation.selection == "rejected"
    assert operation.unavailable_reason == "hardware_requirement_not_met"
    assert operation.native_facts["provider_available"] is False


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_async_tile_is_automatic_workload_gated_and_reported():
    initial = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    if not initial.native_facts.get("provider_available", False):
        pytest.skip("CUDA async tile requires sm_80 and PTX 7.0 or newer")

    assert initial.selection == "eligible"
    assert initial.native_facts["minimum_bls_bytes"] == 8192
    assert initial.native_facts["lowered_specializations"] == 0

    small_input = ti.field(ti.f32)
    small_output = ti.field(ti.f32)
    ti.root.dense(ti.i, 64).place(small_input, small_output)

    x0 = ti.field(ti.f32)
    x1 = ti.field(ti.f32)
    x2 = ti.field(ti.f32)
    x3 = ti.field(ti.f32)
    x4 = ti.field(ti.f32)
    x5 = ti.field(ti.f32)
    x6 = ti.field(ti.f32)
    x7 = ti.field(ti.f32)
    output = ti.field(ti.f32)
    block = ti.root.pointer(ti.ij, 1).dense(ti.ij, (16, 16))
    block.place(x0, x1, x2, x3, x4, x5, x6, x7, output)

    @ti.kernel
    def initialize():
        for i in small_input:
            small_input[i] = i
        for i, j in ti.ndrange(16, 16):
            base = i * 16 + j
            x0[i, j] = base
            x1[i, j] = base + 1
            x2[i, j] = base + 2
            x3[i, j] = base + 3
            x4[i, j] = base + 4
            x5[i, j] = base + 5
            x6[i, j] = base + 6
            x7[i, j] = base + 7

    @ti.kernel
    def small_copy():
        ti.loop_config(block_dim=64)
        ti.block_local(small_input)
        for i in small_input:
            small_output[i] = small_input[i]

    @ti.kernel
    def admitted_copy():
        ti.loop_config(block_dim=256)
        ti.block_local(x0, x1, x2, x3, x4, x5, x6, x7)
        for i, j in x0:
            output[i, j] = (
                x0[i, j]
                + x1[i, j]
                + x2[i, j]
                + x3[i, j]
                + x4[i, j]
                + x5[i, j]
                + x6[i, j]
                + x7[i, j]
            )

    initialize()
    small_copy()
    ti.sync()
    after_small = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    assert after_small.selection == "eligible"
    assert after_small.native_facts["lowered_specializations"] == 0

    admitted_copy()
    ti.sync()
    base = np.arange(256, dtype=np.float32).reshape(16, 16)
    np.testing.assert_array_equal(output.to_numpy(), base * 8 + 28)

    selected = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    assert selected.discovery == "available"
    assert selected.enablement == "enabled"
    assert selected.selection == "selected"
    assert selected.unavailable_reason == "none"
    assert selected.native_facts["lowered_specializations"] >= 1
    assert selected.native_facts["copy_sites"] >= 8


@test_utils.test(
    arch=ti.cuda, require=ti.extension.data64, offline_cache=False
)
def test_cuda_async_tile_preserves_f64_block_local_values():
    initial = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    if not initial.native_facts.get("provider_available", False):
        pytest.skip("CUDA async tile requires sm_80 and PTX 7.0 or newer")

    x0 = ti.field(ti.f64)
    x1 = ti.field(ti.f64)
    x2 = ti.field(ti.f64)
    x3 = ti.field(ti.f64)
    output = ti.field(ti.f64)
    block = ti.root.pointer(ti.ij, 1).dense(ti.ij, (16, 16))
    block.place(x0, x1, x2, x3, output)

    @ti.kernel
    def initialize():
        for i, j in ti.ndrange(16, 16):
            base = ti.cast(i * 16 + j, ti.f64) * 0.25
            x0[i, j] = base
            x1[i, j] = base + 1.0
            x2[i, j] = base + 2.0
            x3[i, j] = base + 3.0

    @ti.kernel
    def admitted_copy():
        ti.loop_config(block_dim=256)
        ti.block_local(x0, x1, x2, x3)
        for i, j in x0:
            output[i, j] = x0[i, j] + x1[i, j] + x2[i, j] + x3[i, j]

    initialize()
    admitted_copy()
    ti.sync()
    base = np.arange(256, dtype=np.float64).reshape(16, 16) * 0.25
    np.testing.assert_array_equal(output.to_numpy(), base * 4 + 6)

    selected = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    assert selected.selection == "selected"
    assert selected.native_facts["copy_sites"] >= 4


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_async_tile_rejects_read_write_block_local_cache():
    initial = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    if not initial.native_facts.get("provider_available", False):
        pytest.skip("CUDA async tile requires sm_80 and PTX 7.0 or newer")

    values = ti.field(ti.f32)
    ti.root.pointer(ti.ij, 1).dense(ti.ij, (64, 64)).place(values)

    @ti.kernel
    def initialize():
        for i, j in ti.ndrange(64, 64):
            values[i, j] = 1.0

    @ti.kernel
    def update():
        ti.loop_config(block_dim=256)
        ti.block_local(values)
        for i, j in values:
            values[i, j] += 1.0

    initialize()
    update()
    ti.sync()
    np.testing.assert_array_equal(
        values.to_numpy(), np.full((64, 64), 2.0, dtype=np.float32)
    )

    after_update = next(
        operation
        for operation in ti.hardware.report().operations
        if operation.descriptor.operation_id == "internal.tile.async.cuda"
    )
    assert after_update.selection == "eligible"
    assert after_update.native_facts["lowered_specializations"] == 0
    assert after_update.native_facts["copy_sites"] == 0


@test_utils.test(arch=ti.vulkan)
def test_vulkan_buffer_command_route_is_passively_eligible():
    report = ti.hardware.report()
    for operation_id in (
        "runtime.buffer_commands.vulkan",
        "raster.draw.vulkan",
        "sampling.texture.vulkan",
    ):
        operation = next(
            operation
            for operation in report.operations
            if operation.descriptor.operation_id == operation_id
        )

        assert operation.discovery == "available"
        assert operation.enablement == "enabled"
        assert operation.selection == "eligible"
        assert operation.unavailable_reason == "none"
        assert not operation.native_facts["external_component_probed"]


def _native_probe_payload(discovery, unavailable_reason, **overrides):
    payload = {
        "provider_id": "cublas",
        "external_component_probed": True,
        "discovery": discovery,
        "unavailable_reason": unavailable_reason,
        "provider_abi": None,
        "provider_version": None,
        "last_error": None,
        "failure_scope": None,
        "native_facts": {
            "probe_policy": "explicit_transient_load",
            "provider_enablement_changed": False,
            "provider_selection_changed": False,
        },
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    (
        "native_payload",
        "expected_discovery",
        "expected_reason",
        "expected_failure_scope",
    ),
    [
        (
            _native_probe_payload(
                "available",
                "none",
                provider_abi="cublas-dynamic-symbols-v1",
            ),
            "available",
            "none",
            None,
        ),
        (
            _native_probe_payload("missing", "external_library_not_found"),
            "missing",
            "external_library_not_found",
            None,
        ),
        (
            _native_probe_payload(
                "incompatible",
                "required_provider_symbol_missing",
                provider_abi="cublas-dynamic-symbols-v1",
                last_error="required provider symbol missing: cublasCreate_v2",
                failure_scope="provider",
            ),
            "incompatible",
            "required_provider_symbol_missing",
            "provider",
        ),
    ],
)
def test_explicit_external_probe_normalizes_native_facts_without_enabling(
    monkeypatch,
    native_payload,
    expected_discovery,
    expected_reason,
    expected_failure_scope,
):
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (False, None, {"cuda": True, "vulkan": True}),
    )
    monkeypatch.setattr(_capabilities, "_native_external_probe", lambda provider_id: native_payload)

    report = ti.hardware.probe("cublas")
    cublas = next(operation for operation in report.operations if operation.descriptor.provider_id == "cublas")

    assert cublas.discovery == expected_discovery
    assert cublas.unavailable_reason == expected_reason
    assert cublas.failure_scope == expected_failure_scope
    assert cublas.enablement == "disabled"
    assert cublas.selection == "not_considered"
    assert cublas.native_facts["external_component_probed"] is True
    assert cublas.native_facts["provider_enablement_changed"] is False
    assert cublas.native_facts["provider_selection_changed"] is False
    assert report.external_components_probed is True


def test_explicit_external_probe_failures_remain_provider_scoped(monkeypatch):
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (False, None, {"cuda": True, "vulkan": True}),
    )

    def fail_probe(_provider_id):
        raise RuntimeError("isolated loader failure")

    monkeypatch.setattr(_capabilities, "_native_external_probe", fail_probe)
    report = ti.hardware.probe("cusparse")
    operation = next(operation for operation in report.operations if operation.descriptor.provider_id == "cusparse")

    assert operation.discovery == "incompatible"
    assert operation.unavailable_reason == "native_probe_failed"
    assert operation.last_error == "isolated loader failure"
    assert operation.failure_scope == "provider"
    assert operation.enablement == "disabled"
    assert operation.selection == "not_considered"


def test_planned_external_probe_and_invalid_tiers_fail_closed(monkeypatch):
    monkeypatch.setattr(
        _capabilities,
        "_runtime_facts",
        lambda: (False, None, {"cuda": True, "vulkan": True}),
    )

    report = ti.hardware.probe("optix")
    optix = next(operation for operation in report.operations if operation.descriptor.provider_id == "optix")
    assert optix.discovery is None
    assert optix.unavailable_reason == "native_probe_not_implemented"
    assert optix.enablement == "disabled"
    assert optix.selection == "not_considered"
    assert report.external_components_probed is False

    with pytest.raises(ValueError, match="only lazy_external"):
        ti.hardware.probe("vulkan_texture")
    with pytest.raises(ValueError, match="only lazy_external"):
        ti.hardware.probe("cub_reference")
    with pytest.raises(KeyError, match="unknown hardware provider"):
        ti.hardware.probe("missing")
    with pytest.raises(TypeError, match="nonempty string"):
        ti.hardware.probe(None)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_passive_report_on_cpu_rejects_gpu_routes_without_loading_them():
    report = ti.hardware.report()

    assert report.runtime_initialized is True
    assert report.backend in ("x64", "arm64")
    for operation in report.operations:
        assert operation.selection != "selected"
        assert operation.native_facts["probe_policy"] == "passive"
        if operation.descriptor.dependency_tier == "lazy_external":
            if operation.enablement == "enabled":
                assert operation.discovery == "available"
                assert operation.selection == "rejected"
                assert operation.unavailable_reason == "backend_not_active"
            else:
                assert operation.enablement == "disabled"
                assert operation.selection == "not_considered"

    payload = report.to_dict()
    assert payload["external_components_probed"] is False
    assert len(payload["operations"]) == len(report.operations)
