#!/usr/bin/env python3
"""Generate the split-runtime Windows export closure from shim objects."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import ctypes
from pathlib import Path

try:
    from runtime_export_closure import (
        DEFAULT_EXPORT_LIMIT,
        EXPLICIT_ABI_SEEDS,
        add_binary_audit,
        build_export_closure,
        select_private_abi_collision_probes,
    )
except ModuleNotFoundError:
    # importlib-based unit tests do not automatically add the script directory
    # to sys.path, while direct CMake execution does.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime_export_closure import (  # type: ignore[no-redef]
        DEFAULT_EXPORT_LIMIT,
        EXPLICIT_ABI_SEEDS,
        add_binary_audit,
        build_export_closure,
        select_private_abi_collision_probes,
    )


_UNDEFINED_EXTERNAL = re.compile(
    r"\bUNDEF\b.*?\bExternal\b[^|\r\n]*\|\s*(?P<symbol>\S+)"
)
_DLL_EXPORT = re.compile(
    r"^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(?P<symbol>\S+)"
)
_PRIVATE_PREFIXES = ("?$TSS", "??_7", "??_R")
_C_FORBIDDEN_PREFIXES = (
    ("llvm", ("LLVM",)),
    ("glfw", ("glfw",)),
    ("vulkan_loader", ("volk", "vk")),
    ("spirv", ("spv", "SPIRV")),
    ("allocator", ("mi_",)),
)
_CPP_OWNER_NAMESPACES = (
    ("taichi", "taichi::"),
    ("llvm", "llvm::"),
    ("llvm", "clang::"),
    ("spirv", "glslang::"),
    ("spirv", "spvtools::"),
    ("spirv", "spirv_cross::"),
    ("ui", "ImGui::"),
    ("logging", "fmt::"),
    ("logging", "spdlog::"),
    ("allocator", "mimalloc::"),
    ("binding", "pybind11::"),
)
_UNDECORATE_SYMBOL_NAME = None


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


def _undecorate(symbol: str) -> str:
    global _UNDECORATE_SYMBOL_NAME
    if not symbol.startswith("?") or not hasattr(ctypes, "WinDLL"):
        return symbol
    if _UNDECORATE_SYMBOL_NAME is None:
        dbghelp = ctypes.WinDLL("dbghelp")  # type: ignore[attr-defined]
        _UNDECORATE_SYMBOL_NAME = dbghelp.UnDecorateSymbolName
        _UNDECORATE_SYMBOL_NAME.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        _UNDECORATE_SYMBOL_NAME.restype = ctypes.c_uint32
    buffer = ctypes.create_string_buffer(max(4096, len(symbol) * 4))
    if (
        _UNDECORATE_SYMBOL_NAME(
            symbol.encode("ascii"), buffer, len(buffer), 0
        )
        == 0
    ):
        return symbol
    return buffer.value.decode("utf-8", errors="replace")


def forbidden_export_family(symbol: str, undecorated: str) -> str | None:
    for family, prefixes in _C_FORBIDDEN_PREFIXES:
        if any(symbol.startswith(prefix) for prefix in prefixes):
            return family
    owner_region = undecorated.split("(", 1)[0]
    owners = [
        (owner_region.rfind(namespace), family)
        for family, namespace in _CPP_OWNER_NAMESPACES
        if namespace in owner_region
    ]
    if not owners:
        return None
    _, owner = max(owners)
    return None if owner == "taichi" else owner


def audit_forbidden_exports(symbols: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for symbol in sorted(symbols):
        family = forbidden_export_family(symbol, _undecorate(symbol))
        if family is not None:
            result.setdefault(family, []).append(symbol)
    return result


def build_closure(
    raw_symbols: set[str],
    undefined_symbols: set[str],
    seeds: tuple[str, ...] = EXPLICIT_ABI_SEEDS,
) -> tuple[list[str], dict]:
    return build_export_closure(
        raw_symbols,
        undefined_symbols,
        platform="windows-msvc",
        normalize_reference=normalize_import_reference,
        classify=classify,
        seeds=seeds,
    )


def add_dll_audit(manifest: dict, actual_symbols: set[str]) -> dict:
    audited = add_binary_audit(
        manifest, actual_symbols, audit_kind="pe-coff-dll"
    )
    # Retain the schema-v1 marker while wheel tooling migrates to the common
    # binary audit fields.
    audited["dll_audited"] = True
    forbidden = audit_forbidden_exports(actual_symbols)
    if forbidden:
        samples = ", ".join(
            f"{family}={values[:3]}" for family, values in sorted(forbidden.items())
        )
        raise RuntimeError(
            "Windows runtime exports bundled third-party APIs: " + samples
        )
    audited["forbidden_export_families"] = []
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
        os.environ.get("TI_WINDOWS_RUNTIME_EXPORT_LIMIT", DEFAULT_EXPORT_LIMIT)
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
            "private_abi_collision_probe_symbols": (
                select_private_abi_collision_probes(
                    exports,
                    {symbol: _undecorate(symbol) for symbol in exports},
                    anchor="taichi_runtime_anchor",
                )
            ),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
