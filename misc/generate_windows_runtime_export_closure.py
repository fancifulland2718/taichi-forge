#!/usr/bin/env python3
"""Generate the split-runtime Windows export closure from shim objects."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


_UNDEFINED_EXTERNAL = re.compile(
    r"\bUNDEF\b.*?\bExternal\b[^|\r\n]*\|\s*(?P<symbol>\S+)"
)
_DLL_EXPORT = re.compile(
    r"^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(?P<symbol>\S+)"
)
_PRIVATE_PREFIXES = ("?$TSS", "??_7", "??_R")
_DEFAULT_EXPORT_LIMIT = 32_768
_EXPLICIT_ABI_SEEDS = ("taichi_runtime_anchor",)


def parse_def_symbols(text: str) -> set[str]:
    symbols = set()
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped == "EXPORTS" or stripped.startswith(";"):
            continue
        symbols.add(stripped.split(None, 1)[0])
    return symbols


def parse_dumpbin_undefined(text: str) -> set[str]:
    result = set()
    for line in text.splitlines():
        match = _UNDEFINED_EXTERNAL.search(line)
        if match:
            result.add(match.group("symbol").strip())
    return result


def parse_dumpbin_exports(text: str) -> set[str]:
    result = set()
    for line in text.splitlines():
        match = _DLL_EXPORT.match(line)
        if match:
            result.add(match.group("symbol"))
    return result


def classify(symbol: str) -> str:
    if symbol.startswith(_PRIVATE_PREFIXES):
        return "compiler_private_metadata"
    if "@std@@" in symbol:
        return "taichi_signature_with_std"
    if symbol.startswith("??_"):
        return "compiler_generated_callable"
    if symbol.startswith("?"):
        return "decorated_cpp"
    return "c_abi"


def normalize_import_reference(symbol: str) -> str:
    # COFF undefined references emitted for dllimport declarations point at
    # import-address-table thunks.  A .def exports the underlying decorated
    # symbol, not the __imp_ indirection.
    if symbol.startswith("__imp_"):
        return symbol[len("__imp_") :]
    return symbol


def _dumpbin_path() -> str:
    configured = os.environ.get("DUMPBIN")
    if configured:
        return configured
    discovered = shutil.which("dumpbin") or shutil.which("dumpbin.exe")
    if discovered is None:
        raise RuntimeError("dumpbin was not found in the MSVC build environment")
    return discovered


def collect_undefined(object_paths: list[Path], dumpbin: str) -> set[str]:
    undefined = set()
    for path in object_paths:
        if not path.is_file():
            raise FileNotFoundError(f"shim object does not exist: {path}")
        completed = subprocess.run(
            [dumpbin, "/nologo", "/symbols", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"dumpbin failed for {path} with {completed.returncode}:\n"
                f"{completed.stdout}"
            )
        undefined.update(parse_dumpbin_undefined(completed.stdout))
    return undefined


def collect_dll_exports(dll_path: Path, dumpbin: str) -> set[str]:
    if not dll_path.is_file():
        raise FileNotFoundError(f"runtime DLL does not exist: {dll_path}")
    completed = subprocess.run(
        [dumpbin, "/nologo", "/exports", str(dll_path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"dumpbin failed for {dll_path} with {completed.returncode}:\n"
            f"{completed.stdout}"
        )
    return parse_dumpbin_exports(completed.stdout)


def build_closure(
    raw_symbols: set[str],
    undefined_symbols: set[str],
    seeds: tuple[str, ...] = _EXPLICIT_ABI_SEEDS,
) -> tuple[list[str], dict]:
    normalized_undefined = {
        normalize_import_reference(symbol) for symbol in undefined_symbols
    }
    required = raw_symbols.intersection(normalized_undefined)
    exports = sorted(required.union(seeds))
    classifications: dict[str, int] = {}
    for symbol in exports:
        role = classify(symbol)
        classifications[role] = classifications.get(role, 0) + 1
    manifest = {
        "schema_version": 1,
        "raw_defined_symbol_count": len(raw_symbols),
        "shim_undefined_symbol_count": len(undefined_symbols),
        "shim_required_runtime_symbol_count": len(required),
        "explicit_abi_seed_count": len(seeds),
        "exported_symbol_count": len(exports),
        "dropped_raw_symbol_count": len(raw_symbols - required),
        "classifications": dict(sorted(classifications.items())),
        "explicit_abi_seeds": list(seeds),
        "exports": exports,
    }
    digest = hashlib.sha256()
    digest.update("\n".join(exports).encode("utf-8"))
    manifest["export_set_sha256"] = digest.hexdigest()
    return exports, manifest


def add_dll_audit(manifest: dict, actual_symbols: set[str]) -> dict:
    requested = set(manifest.get("exports", ()))
    configured_limit = int(manifest["configured_export_limit"])
    missing = sorted(requested - actual_symbols)
    if missing:
        raise RuntimeError(
            "linked runtime DLL is missing requested exports: "
            + ", ".join(missing[:8])
        )
    if not actual_symbols:
        raise RuntimeError("linked runtime DLL has no exports")
    if len(actual_symbols) > configured_limit:
        raise RuntimeError(
            f"linked runtime DLL has {len(actual_symbols)} exports, exceeding "
            f"the safety limit {configured_limit}"
        )
    actual_exports = sorted(actual_symbols)
    digest = hashlib.sha256()
    digest.update("\n".join(actual_exports).encode("utf-8"))
    audited = dict(manifest)
    audited.update(
        {
            "dll_audited": True,
            "actual_exported_symbol_count": len(actual_exports),
            "implicit_exported_symbol_count": len(actual_symbols - requested),
            "actual_export_set_sha256": digest.hexdigest(),
            "actual_exports": actual_exports,
        }
    )
    return audited


def audit_dll(dll_path: Path, manifest_path: Path) -> None:
    dumpbin = _dumpbin_path()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audited = add_dll_audit(
        manifest, collect_dll_exports(dll_path, dumpbin)
    )
    manifest_path.write_text(
        json.dumps(audited, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "--audit-dll":
        audit_dll(Path(argv[2]), Path(argv[3]))
        return 0
    if len(argv) != 5:
        print(
            "usage: generate_windows_runtime_export_closure.py "
            "<raw.def> <shim-objects.txt> <output.def> <manifest.json>\n"
            "   or: generate_windows_runtime_export_closure.py "
            "--audit-dll <runtime.dll> <manifest.json>",
            file=sys.stderr,
        )
        return 2
    raw_path, objects_path, output_path, manifest_path = map(Path, argv[1:])
    object_paths = [
        Path(line.strip())
        for line in objects_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not object_paths:
        raise RuntimeError("the shim object list is empty")
    raw_symbols = parse_def_symbols(
        raw_path.read_text(encoding="utf-8", errors="ignore")
    )
    undefined = collect_undefined(object_paths, _dumpbin_path())
    exports, manifest = build_closure(raw_symbols, undefined)
    if (
        not raw_symbols
        or not undefined
        or manifest["shim_required_runtime_symbol_count"] == 0
    ):
        raise RuntimeError(
            "runtime export closure is unexpectedly empty; refusing to link"
        )
    configured_limit = int(
        os.environ.get("TI_WINDOWS_RUNTIME_EXPORT_LIMIT", _DEFAULT_EXPORT_LIMIT)
    )
    if len(exports) > configured_limit:
        raise RuntimeError(
            f"runtime export closure has {len(exports)} symbols, exceeding "
            f"the safety limit {configured_limit}"
        )
    output_path.write_text(
        "EXPORTS\n" + "\n".join(f"\t{symbol}" for symbol in exports) + "\n",
        encoding="utf-8",
    )
    manifest.update(
        {
            "object_count": len(object_paths),
            "configured_export_limit": configured_limit,
            "dumpbin": _dumpbin_path(),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
