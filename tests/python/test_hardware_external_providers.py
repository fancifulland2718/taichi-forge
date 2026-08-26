import pytest

import taichi_forge as ti
from taichi_forge.hardware import _optix
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
        provider.provider_id
        for provider in ti.hardware.providers()
        if provider.dependency_tier == "lazy_external"
    )

    assert set(registered) == set(catalog)
    assert len(registered) == len(set(registered))
    assert tuple(spec.provider_id for spec in external_provider_specs()) == registered


def test_external_provider_registry_owns_path_and_lifetime_policy():
    assert not external_provider_spec("cublas").supports_library_path
    assert not external_provider_spec("cusparse").supports_library_path
    assert not external_provider_spec("cufft").supports_library_path

    cudss = external_provider_spec("cudss")
    assert cudss.supports_library_path
    assert cudss.install_owner == "user_optional_package"
    assert cudss.runtime_resource_policy == "provider_plan"
    assert cudss.transitive_dependencies == ("cublas",)
    assert cudss.native_path_resolver == "cudss_package"

    optix = external_provider_spec("optix")
    assert optix.supports_library_path
    assert optix.library_path_policy == "required"
    assert optix.adapter_kind == "source_provider_c_abi"
    assert optix.runtime_resource_policy == "provider_context"
    assert optix.python_adapter_module == "taichi_forge.hardware._optix"


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
