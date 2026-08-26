"""Probe-only mount for a user-managed NVIDIA AmgX runtime."""

import os

from taichi_forge.hardware._bundled_runtime_provider import (
    BundledRuntimeProviderDefinition,
    passive_status as _passive_status,
    probe_provider as _probe_provider,
    resolve_library_path as _resolve_library_path,
)


DEFINITION = BundledRuntimeProviderDefinition(
    provider_id="amgx",
    provider_name="NVIDIA AmgX",
    adapter_stem="taichi_forge_amgx_provider_abi1_stable_c",
    query_symbol="taichi_forge_amgx_provider_query",
    provider_abi_name="taichi-forge-amgx-provider-c-abi1",
    environment_variable="TI_AMGX_LIBRARY_PATH",
    library_names=(("amgxsh.dll",) if os.name == "nt" else ("libamgxsh.so",)),
    package_distributions=(),
    supported_version_family="stable C API",
)


def resolve_library_path(library_path=None):
    return _resolve_library_path(DEFINITION, library_path)


def probe_provider(library_path=None):
    return _probe_provider(DEFINITION, library_path)


def passive_status():
    return _passive_status(DEFINITION)


__all__ = ("DEFINITION", "passive_status", "probe_provider", "resolve_library_path")
