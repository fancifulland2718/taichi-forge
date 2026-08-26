import pytest

import taichi_forge as ti
from taichi_forge.hardware import _cublaslt, _cudss, _optix
from taichi_forge.hardware._external_providers import (
    external_provider_ids,
    external_provider_spec,
    external_provider_specs,
    passive_external_provider_status,
    probe_external_provider,
)


def test_external_provider_registry_matches_public_lazy_catalog():
    registered = external_provider_ids()
    catalog = tuple(
        provider.provider_id for provider in ti.hardware.providers() if provider.dependency_tier == "lazy_external"
    )

    assert set(registered) == set(catalog)
    assert len(registered) == len(set(registered))
    assert tuple(spec.provider_id for spec in external_provider_specs()) == registered


def test_external_provider_registry_owns_path_and_lifetime_policy():
    assert not external_provider_spec("cublas").supports_library_path
    assert not external_provider_spec("cusparse").supports_library_path
    assert not external_provider_spec("cufft").supports_library_path

    cublaslt = external_provider_spec("cublaslt")
    assert cublaslt.supports_library_path
    assert cublaslt.adapter_kind == "python_dynamic_symbols"
    assert cublaslt.install_owner == "user_cuda_environment"
    assert cublaslt.process_handle_policy == "process_resident"
    assert cublaslt.runtime_resource_policy == "provider_plan"
    assert cublaslt.transitive_dependencies == ("cuda_runtime",)
    assert cublaslt.python_adapter_module == "taichi_forge.hardware._cublaslt"

    cudss = external_provider_spec("cudss")
    assert cudss.supports_library_path
    assert cudss.adapter_kind == "bundled_provider_c_abi"
    assert cudss.install_owner == "forge_runtime_wheel"
    assert cudss.process_handle_policy == "provider_object"
    assert cudss.runtime_resource_policy == "provider_plan"
    assert cudss.transitive_dependencies == ("cublas",)
    assert cudss.native_path_resolver is None
    assert cudss.python_adapter_module == "taichi_forge.hardware._cudss"

    optix = external_provider_spec("optix")
    assert optix.supports_library_path
    assert optix.library_path_policy == "optional"
    assert optix.adapter_kind == "bundled_provider_c_abi"
    assert optix.install_owner == "forge_runtime_wheel"
    assert optix.transitive_dependencies == (
        "cuda_driver",
        "optix_driver_runtime",
    )
    assert optix.runtime_resource_policy == "provider_context"
    assert optix.python_adapter_module == "taichi_forge.hardware._optix"

    expected_runtime_providers = {
        "cusparselt": (
            "taichi_forge.hardware._cusparselt",
            ("cuda_runtime",),
        ),
        "cutensor": (
            "taichi_forge.hardware._cutensor",
            ("cuda_runtime",),
        ),
        "amgx": (
            "taichi_forge.hardware._amgx",
            ("cuda_runtime", "cublas", "cusparse"),
        ),
    }
    for provider_id, (module, dependencies) in expected_runtime_providers.items():
        provider = external_provider_spec(provider_id)
        assert provider.supports_library_path
        assert provider.adapter_kind == "bundled_provider_c_abi"
        assert provider.install_owner == "forge_runtime_wheel"
        assert provider.process_handle_policy == "provider_object"
        assert provider.runtime_resource_policy == "provider_plan"
        assert provider.transitive_dependencies == dependencies
        assert provider.python_adapter_module == module


def test_external_provider_registry_rejects_ambiguous_library_paths():
    with pytest.raises(ValueError, match="not supported for cublas"):
        probe_external_provider("cublas", "unexpected-cublas.dll")
    with pytest.raises(KeyError, match="unknown external hardware provider"):
        external_provider_spec("missing")


def test_external_provider_registry_keeps_optix_status_passive(monkeypatch):
    def unexpected_load(_path):
        raise AssertionError("passive provider status must not load OptiX")

    monkeypatch.setattr(_optix, "_load_library", unexpected_load)
    status = passive_external_provider_status("optix")

    assert status["provider_id"] == "optix"
    assert status["library_loaded"] is False
    assert status["native_facts"]["external_component_probed"] is False


def test_external_provider_registry_keeps_cudss_status_passive(monkeypatch):
    def unexpected_load(_path):
        raise AssertionError("passive provider status must not load cuDSS")

    monkeypatch.setattr(_cudss, "_load_library", unexpected_load)
    status = passive_external_provider_status("cudss")

    assert status["provider_id"] == "cudss"
    assert status["library_loaded"] is False
    assert status["provider_abi"] == "taichi-forge-cudss-provider-c-abi1"
    assert status["native_facts"]["external_component_probed"] is False


def test_external_provider_registry_keeps_cublaslt_status_passive(monkeypatch):
    def unexpected_load(_path=None):
        raise AssertionError("passive provider status must not load cuBLASLt")

    monkeypatch.setattr(_cublaslt, "_load_process_library", unexpected_load)
    status = passive_external_provider_status("cublaslt")

    assert status["provider_id"] == "cublaslt"
    assert status["library_loaded"] is False
    assert status["provider_abi"] == "cublaslt-dynamic-symbols-v1"
    assert status["native_facts"]["external_component_probed"] is False
