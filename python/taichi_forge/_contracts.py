"""Versioned compatibility contracts shared by Forge runtime surfaces.

Build/source identities are provenance only.  A split Python shim and native
runtime are compatible when the ABI revision, bootstrap schema, compiler ABI,
and every required schema agree; they do not need to originate from the same
commit.
"""

from __future__ import annotations

from types import MappingProxyType

from taichi_forge._contract_constants import (
    FORGE_CONTRACT_MANIFEST_SCHEMA_VERSION,
    FORGE_NATIVE_ABI_REVISION,
)
from taichi_forge._lib import core as _ti_core


DYNAMIC_WORK_SCHEMA_VERSION = 5
STRUCTURED_CONTROL_SCHEMA_VERSION = 5
GRAPH_PIPELINE_SCHEMA_VERSION = 2
SOLVER_CAPABILITY_SCHEMA_VERSION = 5


def _native_contract_manifest():
    query = getattr(_ti_core, "get_runtime_contract_manifest", None)
    if query is None:
        return {
            "schema_version": 0,
            "native_abi_revision": 0,
            "runtime_statistics_schema": None,
            "source_id": _ti_core.get_commit_hash(),
            "legacy_runtime": True,
        }
    return dict(query())


def runtime_contract_manifest():
    """Return an immutable, backend-independent shim/runtime contract.

    The manifest deliberately contains no active-device probes and is safe to
    inspect before ``ti.init()``.  Backend capabilities remain in their
    existing runtime queries.
    """

    native = _native_contract_manifest()
    return MappingProxyType(
        {
            "schema_version": FORGE_CONTRACT_MANIFEST_SCHEMA_VERSION,
            "native_abi_revision": int(native["native_abi_revision"]),
            "required_native_abi_revision": FORGE_NATIVE_ABI_REVISION,
            "schemas": MappingProxyType(
                {
                    "dynamic_work": DYNAMIC_WORK_SCHEMA_VERSION,
                    "structured_control": STRUCTURED_CONTROL_SCHEMA_VERSION,
                    "graph_pipeline": GRAPH_PIPELINE_SCHEMA_VERSION,
                    "solver": SOLVER_CAPABILITY_SCHEMA_VERSION,
                    "runtime_statistics": native.get(
                        "runtime_statistics_schema"
                    ),
                }
            ),
            "runtime": MappingProxyType(
                {
                    "manifest_schema_version": int(native["schema_version"]),
                    "source_id": str(native["source_id"]),
                    "legacy_runtime": bool(native.get("legacy_runtime", False)),
                }
            ),
            "shim": MappingProxyType(
                {
                    # The native ``get_commit_hash()`` symbol intentionally
                    # identifies the runtime DLL in a split installation.  A
                    # distinct shim build id is supplied by the binding-side
                    # manifest when available; do not falsely label the
                    # runtime id as the shim id.
                    "source_id": native.get("shim_source_id"),
                    "version": _ti_core.get_version_string(),
                    "compiler_abi": native.get("shim_compiler_abi"),
                }
            ),
            "features": MappingProxyType(dict(native.get("features", {}))),
            "build_profile": MappingProxyType(
                dict(
                    native.get(
                        "build_profile", {"schema_version": 1, "kind": "unknown"}
                    )
                )
            ),
            "compiler_compatibility": MappingProxyType(
                {
                    "runtime": native.get("runtime_compiler_abi"),
                    "shim": native.get("shim_compiler_abi"),
                }
            ),
        }
    )


def validate_runtime_contract(*, require_native_manifest=True):
    """Validate compatibility without requiring equal source commits."""

    manifest = runtime_contract_manifest()
    runtime = manifest["runtime"]
    if require_native_manifest and runtime["legacy_runtime"]:
        raise RuntimeError(
            "installed native runtime does not expose a Forge contract manifest"
        )
    actual = manifest["native_abi_revision"]
    required = manifest["required_native_abi_revision"]
    if actual != required:
        raise RuntimeError(
            "installed native runtime ABI is incompatible with this shim: "
            f"required={required}, actual={actual}"
        )
    if not runtime["legacy_runtime"]:
        manifest_schema = runtime["manifest_schema_version"]
        if manifest_schema != FORGE_CONTRACT_MANIFEST_SCHEMA_VERSION:
            raise RuntimeError(
                "installed native runtime manifest schema is incompatible "
                f"with this shim: required="
                f"{FORGE_CONTRACT_MANIFEST_SCHEMA_VERSION}, "
                f"actual={manifest_schema}"
            )
        compiler = manifest["compiler_compatibility"]
        if (
            not compiler["runtime"]
            or not compiler["shim"]
            or compiler["runtime"] != compiler["shim"]
        ):
            raise RuntimeError(
                "installed native runtime compiler ABI is incompatible with "
                f"this shim: runtime={compiler['runtime']!r}, "
                f"shim={compiler['shim']!r}"
            )
    return manifest


__all__ = [
    "DYNAMIC_WORK_SCHEMA_VERSION",
    "FORGE_CONTRACT_MANIFEST_SCHEMA_VERSION",
    "FORGE_NATIVE_ABI_REVISION",
    "GRAPH_PIPELINE_SCHEMA_VERSION",
    "SOLVER_CAPABILITY_SCHEMA_VERSION",
    "STRUCTURED_CONTROL_SCHEMA_VERSION",
    "runtime_contract_manifest",
    "validate_runtime_contract",
]
