"""Probe-only mount for a user-managed NVIDIA cuTENSOR runtime."""

import os

from taichi_forge.hardware._bundled_runtime_provider import (
    BundledRuntimeProviderDefinition,
    passive_status as _passive_status,
    probe_provider as _probe_provider,
    resolve_library_path as _resolve_library_path,
)


DEFINITION = BundledRuntimeProviderDefinition(
    provider_id="cutensor",
    provider_name="NVIDIA cuTENSOR",
    adapter_stem="taichi_forge_cutensor_provider_abi1_api200_207",
    query_symbol="taichi_forge_cutensor_provider_query",
    provider_abi_name="taichi-forge-cutensor-provider-c-abi1",
    environment_variable="TI_CUTENSOR_LIBRARY_PATH",
    library_names=(("cutensor64_2.dll", "cutensor.dll") if os.name == "nt" else ("libcutensor.so.2", "libcutensor.so")),
    package_distributions=("cutensor-cu13", "cutensor-cu12"),
    supported_version_family="2.0.x-2.7.x",
)


def resolve_library_path(library_path=None):
    return _resolve_library_path(DEFINITION, library_path)


def probe_provider(library_path=None):
    return _probe_provider(DEFINITION, library_path)


def passive_status():
    return _passive_status(DEFINITION)


__all__ = ("DEFINITION", "passive_status", "probe_provider", "resolve_library_path")
