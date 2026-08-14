#!/usr/bin/env python3
"""Shared helpers for split-runtime private ABI export manifests.

The CPython shim and the platform-independent runtime are separate binaries,
but their C++ link surface is a package-private ABI rather than a public API.
This module keeps the export-set construction and audit metadata identical
across PE/COFF and ELF tooling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import hashlib


SCHEMA_VERSION = 2
ABI_REVISION = 1
DEFAULT_EXPORT_LIMIT = 32_768
EXPLICIT_ABI_SEEDS = ("taichi_runtime_anchor",)
_COLLISION_PROBE_OWNER_FRAGMENTS = (
    "taichi::lang::Program::",
    "taichi::lang::Kernel::",
    "taichi::lang::SNode::",
    "taichi::lang::CompiledGraph::",
    "taichi::lang::Graph",
)


def export_digest(symbols: Iterable[str]) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(symbols).encode("utf-8"))
    return digest.hexdigest()


def select_private_abi_collision_probes(
    symbols: Iterable[str],
    demangled: dict[str, str],
    *,
    anchor: str,
    limit: int = 16,
) -> list[str]:
    """Choose stable probes used to reject an earlier global Taichi ABI."""

    candidates = []
    for symbol in symbols:
        name = demangled.get(symbol, symbol)
        rank = next(
            (
                index
                for index, fragment in enumerate(
                    _COLLISION_PROBE_OWNER_FRAGMENTS
                )
                if fragment in name
            ),
            None,
        )
        if rank is not None:
            candidates.append((rank, symbol))
    selected = [symbol for _, symbol in sorted(candidates)[: max(0, limit - 1)]]
    return sorted(set([anchor, *selected]))


def build_export_closure(
    raw_symbols: set[str],
    undefined_symbols: set[str],
    *,
    platform: str,
    normalize_reference: Callable[[str], str] = lambda symbol: symbol,
    classify: Callable[[str], str] = lambda symbol: "package_private_abi",
    seeds: tuple[str, ...] = EXPLICIT_ABI_SEEDS,
    additional_required_symbols: set[str] | None = None,
) -> tuple[list[str], dict]:
    """Return the exact runtime symbols consumed by the shim plus ABI seeds."""

    normalized_undefined = {
        normalize_reference(symbol) for symbol in undefined_symbols
    }
    directly_required = raw_symbols.intersection(normalized_undefined)
    additional_required = raw_symbols.intersection(
        additional_required_symbols or set()
    ) - directly_required
    required = directly_required.union(additional_required)
    exports = sorted(required.union(seeds))
    classifications: dict[str, int] = {}
    for symbol in exports:
        role = classify(symbol)
        classifications[role] = classifications.get(role, 0) + 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "abi_revision": ABI_REVISION,
        "platform": platform,
        "raw_defined_symbol_count": len(raw_symbols),
        "shim_undefined_symbol_count": len(undefined_symbols),
        "shim_direct_runtime_symbol_count": len(directly_required),
        "shim_shared_odr_symbol_count": len(additional_required),
        "shim_required_runtime_symbol_count": len(required),
        "explicit_abi_seed_count": len(seeds),
        "exported_symbol_count": len(exports),
        "dropped_raw_symbol_count": len(raw_symbols - required),
        "classifications": dict(sorted(classifications.items())),
        "explicit_abi_seeds": list(seeds),
        "exports": exports,
        "export_set_sha256": export_digest(exports),
    }
    return exports, manifest


def add_binary_audit(
    manifest: dict,
    actual_symbols: set[str],
    *,
    audit_kind: str,
) -> dict:
    """Attach a canonical post-link export audit to a closure manifest."""

    requested = set(manifest.get("exports", ()))
    configured_limit = int(manifest["configured_export_limit"])
    missing = sorted(requested - actual_symbols)
    if missing:
        raise RuntimeError(
            "linked runtime is missing requested exports: "
            + ", ".join(missing[:8])
        )
    if not actual_symbols:
        raise RuntimeError("linked runtime has no exports")
    if len(actual_symbols) > configured_limit:
        raise RuntimeError(
            f"linked runtime has {len(actual_symbols)} exports, exceeding "
            f"the safety limit {configured_limit}"
        )
    actual_exports = sorted(actual_symbols)
    audited = dict(manifest)
    audited.update(
        {
            "binary_audited": True,
            "binary_audit_kind": audit_kind,
            "actual_exported_symbol_count": len(actual_exports),
            "implicit_exported_symbol_count": len(actual_symbols - requested),
            "actual_export_set_sha256": export_digest(actual_exports),
            "actual_exports": actual_exports,
        }
    )
    return audited
