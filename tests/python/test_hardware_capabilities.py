from dataclasses import FrozenInstanceError, replace

import pytest

import taichi_forge as ti
from taichi_forge.hardware import _capabilities
from tests import test_utils


_OPERATION_IDS = (
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
            ti.hardware.capability("matrix.mma.cuda"),
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
    raster = ti.hardware.capability("raster.draw.vulkan")
    assert raster.semantic_family == "raster.draw"
    assert raster.public_api == "ti.hardware.raster"
    assert ti.hardware.capability(raster.operation_id) is raster

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


def test_passive_report_does_not_probe_or_enable_external_components():
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


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_passive_report_on_cpu_rejects_gpu_routes_without_loading_them():
    report = ti.hardware.report()

    assert report.runtime_initialized is True
    assert report.backend in ("x64", "arm64")
    for operation in report.operations:
        assert operation.selection != "selected"
        assert operation.native_facts["probe_policy"] == "passive"
        if operation.descriptor.dependency_tier == "lazy_external":
            assert operation.enablement == "disabled"
            assert operation.selection == "not_considered"

    payload = report.to_dict()
    assert payload["external_components_probed"] is False
    assert len(payload["operations"]) == len(report.operations)
