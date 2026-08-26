from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import taichi_forge as ti
from taichi_forge.hardware import _amgx, _cusparselt, _cutensor
from taichi_forge.hardware import _bundled_runtime_provider as _runtime_provider


_MODULES = (_cusparselt, _cutensor, _amgx)


@pytest.mark.parametrize("module", _MODULES)
def test_optional_runtime_library_path_is_explicit_and_environment_owned(module, tmp_path, monkeypatch):
    name = module.DEFINITION.library_names[0]
    runtime = tmp_path / name
    runtime.write_bytes(b"test-vendor-runtime")

    assert module.resolve_library_path(runtime) == str(runtime.resolve())
    assert module.resolve_library_path(tmp_path) == str(runtime.resolve())

    monkeypatch.setenv(module.DEFINITION.environment_variable, str(runtime))
    assert module.resolve_library_path() == str(runtime.resolve())

    missing = tmp_path / f"missing-{name}"
    monkeypatch.setenv(module.DEFINITION.environment_variable, str(missing))
    assert module.resolve_library_path() == str(missing)


@pytest.mark.parametrize("module", (_cusparselt, _cutensor))
def test_optional_runtime_discovers_installed_vendor_package_files(module, tmp_path, monkeypatch):
    relative = Path("vendor") / "lib" / module.DEFINITION.library_names[0]
    runtime = tmp_path / relative
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"test-package-runtime")
    distribution = SimpleNamespace(files=(relative,), locate_file=lambda item: tmp_path / item)

    def find_distribution(name):
        if name == module.DEFINITION.package_distributions[0]:
            return distribution
        raise _runtime_provider.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(_runtime_provider.importlib.metadata, "distribution", find_distribution)

    assert module.resolve_library_path() == str(runtime.resolve())


@pytest.mark.parametrize("module", _MODULES)
def test_optional_runtime_probe_audits_without_qualifying_execution(module, monkeypatch):
    definition = module.DEFINITION
    info = _runtime_provider._ProviderInfo()  # pylint: disable=W0212
    info.provider_id = definition.provider_id.encode()
    info.provider_name = definition.provider_name.encode()
    info.supported_version_family = definition.supported_version_family.encode()
    info.build_identity = b"test-forge-adapter"
    info.features = _runtime_provider._REQUIRED_FEATURES  # pylint: disable=W0212
    info.required_symbol_count = 17
    loaded = SimpleNamespace(path="test-forge-adapter", api=SimpleNamespace(info=info))

    monkeypatch.setattr(
        _runtime_provider,
        "_bundled_provider_candidates",
        lambda _definition: ("test-forge-adapter",),
    )
    monkeypatch.setattr(_runtime_provider, "_query_provider", lambda _definition, _path: loaded)
    monkeypatch.setattr(
        _runtime_provider,
        "_probe_runtime",
        lambda _definition, _loaded, path: {
            "version_major": 2,
            "version_minor": 5,
            "version_patch": 0,
            "cuda_runtime_version": 12080,
            "library_path": path or "system-default",
            "build_version": "test-build",
        },
    )
    monkeypatch.setattr(_runtime_provider, "_binary_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(
        _runtime_provider,
        "_vendor_dll_directories",
        lambda _definition, _path: nullcontext(),
    )

    result = module.probe_provider("vendor-runtime")

    assert result["provider_id"] == definition.provider_id
    assert result["discovery"] == "available"
    assert result["provider_version"] == "2.5.0"
    assert result["native_facts"]["required_symbol_count"] == 17
    assert result["native_facts"]["library_loaded_transiently"] is True
    assert result["native_facts"]["execution_api_available"] is False
    assert result["native_facts"]["execution_qualified"] is False
    assert result["native_facts"]["execution_resource_created"] is False


@pytest.mark.parametrize("module", _MODULES)
def test_optional_runtime_probe_fails_closed_when_vendor_runtime_is_missing(module, monkeypatch):
    monkeypatch.setattr(
        _runtime_provider,
        "_bundled_provider_candidates",
        lambda _definition: ("test-forge-adapter",),
    )
    monkeypatch.setattr(
        _runtime_provider,
        "_query_provider",
        lambda _definition, path: SimpleNamespace(path=path),
    )

    def missing_runtime(_definition, _loaded, _path):
        raise _runtime_provider._ProviderRuntimeError(  # pylint: disable=W0212
            _runtime_provider._RUNTIME_UNAVAILABLE,  # pylint: disable=W0212
            "test vendor runtime missing",
        )

    monkeypatch.setattr(_runtime_provider, "_probe_runtime", missing_runtime)
    monkeypatch.setattr(
        _runtime_provider,
        "_vendor_dll_directories",
        lambda _definition, _path: nullcontext(),
    )

    result = module.probe_provider("missing-vendor-runtime")

    assert result["discovery"] == "missing"
    assert result["unavailable_reason"] == "external_library_not_found"
    assert result["failure_scope"] == "provider"
    assert result["native_facts"]["library_loaded_transiently"] is False


@pytest.mark.parametrize("module", _MODULES)
def test_optional_runtime_passive_status_never_loads_a_library(module, monkeypatch):
    def unexpected_load(_path):
        raise AssertionError("passive status must not load an optional runtime")

    monkeypatch.setattr(_runtime_provider, "_load_library", unexpected_load)
    status = module.passive_status()

    assert status["provider_id"] == module.DEFINITION.provider_id
    assert status["library_loaded"] is False
    assert status["native_facts"]["external_component_probed"] is False
    assert status["native_facts"]["execution_api_available"] is False


def test_probe_only_provider_capabilities_are_host_only_and_non_executable():
    for provider_id in ("cusparselt", "cutensor", "amgx"):
        descriptor = ti.hardware.capability(f"runtime.probe.{provider_id}")
        assert descriptor.provider_id == provider_id
        assert descriptor.scopes == ("python",)
        assert descriptor.execution_kind == "external_library"
        assert descriptor.graph_integration == "unsupported"
        assert descriptor.hardware_acceleration == "none"
        assert descriptor.hardware_route == "none"
        assert descriptor.implementation_status == "internal_foundation"
        assert "kernel intrinsic is exposed yet" in descriptor.notes[1]


@pytest.mark.parametrize("module", _MODULES)
def test_public_hardware_probe_keeps_probe_only_provider_disabled(module, monkeypatch):
    definition = module.DEFINITION
    monkeypatch.setattr(
        module,
        "probe_provider",
        lambda _path=None: {
            "provider_id": definition.provider_id,
            "external_component_probed": True,
            "discovery": "available",
            "unavailable_reason": "none",
            "provider_abi": definition.provider_abi_name,
            "provider_version": "2.5.0",
            "last_error": None,
            "failure_scope": None,
            "native_facts": {
                "execution_qualified": False,
                "execution_api_available": False,
            },
        },
    )

    report = ti.hardware.probe(definition.provider_id)
    operation = next(
        item for item in report.operations if item.descriptor.operation_id == f"runtime.probe.{definition.provider_id}"
    )

    assert report.external_components_probed is True
    assert operation.discovery == "available"
    assert operation.enablement == "disabled"
    assert operation.selection == "not_considered"
    assert operation.native_facts["execution_qualified"] is False


def test_optional_runtime_provider_filenames_are_platform_specific():
    for module in _MODULES:
        filename = _runtime_provider._adapter_filename(module.DEFINITION)  # pylint: disable=W0212
        assert Path(filename).suffix == (".dll" if os.name == "nt" else ".so")
