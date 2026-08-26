"""Probe-only mount for a user-managed NVIDIA cuSPARSELt runtime."""

import os

from taichi_forge.hardware._bundled_runtime_provider import (
    BundledRuntimeProviderDefinition,
    passive_status as _passive_status,
    probe_provider as _probe_provider,
    resolve_library_path as _resolve_library_path,
)


DEFINITION = BundledRuntimeProviderDefinition(
    provider_id="cusparselt",
    provider_name="NVIDIA cuSPARSELt",
    adapter_stem="taichi_forge_cusparselt_provider_abi1_api040_090",
    query_symbol="taichi_forge_cusparselt_provider_query",
    provider_abi_name="taichi-forge-cusparselt-provider-c-abi1",
    environment_variable="TI_CUSPARSELT_LIBRARY_PATH",
    library_names=(
        ("cusparseLt64_0.dll", "cusparseLt.dll") if os.name == "nt" else ("libcusparseLt.so.0", "libcusparseLt.so")
    ),
    package_distributions=("nvidia-cusparselt-cu13", "nvidia-cusparselt-cu12"),
    supported_version_family="0.4.x-0.9.x",
)


def resolve_library_path(library_path=None):
    return _resolve_library_path(DEFINITION, library_path)


def probe_provider(library_path=None):
    return _probe_provider(DEFINITION, library_path)


def passive_status():
    return _passive_status(DEFINITION)


__all__ = ("DEFINITION", "passive_status", "probe_provider", "resolve_library_path")
