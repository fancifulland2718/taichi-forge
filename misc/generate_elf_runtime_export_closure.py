#!/usr/bin/env python3
"""Generate and audit the ELF split-runtime private ABI export closure."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

try:
    from runtime_export_closure import (
        DEFAULT_EXPORT_LIMIT,
        EXPLICIT_ABI_SEEDS,
        add_binary_audit,
        build_export_closure,
        select_private_abi_collision_probes,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime_export_closure import (  # type: ignore[no-redef]
        DEFAULT_EXPORT_LIMIT,
        EXPLICIT_ABI_SEEDS,
        add_binary_audit,
        build_export_closure,
        select_private_abi_collision_probes,
    )


_VERSION_NODE = "TAICHI_FORGE_RUNTIME_PRIVATE_1"
_C_SYMBOL_FAMILIES = (
    ("llvm", ("LLVM",)),
    ("glfw", ("glfw",)),
    ("vulkan_loader", ("volk", "vk")),
    ("spirv", ("spv", "SPIRV")),
    ("mimalloc", ("mi_",)),
)
_CPP_NAMESPACE_FAMILIES = (
    ("taichi", ("taichi::",)),
    ("llvm", ("llvm::", "clang::")),
    ("spirv", ("glslang::", "spvtools::", "spv::", "spirv_cross::")),
    ("ui", ("ImGui::", "ImGui_Impl", "GLFW::")),
    ("logging", ("fmt::", "spdlog::")),
    ("allocator", ("mimalloc::",)),
    ("binding", ("pybind11::",)),
)
_DEMANGLED_PREFIXES = (
    "construction vtable for ",
    "covariant return thunk to ",
    "guard variable for ",
    "non-virtual thunk to ",
    "typeinfo for ",
    "typeinfo name for ",
    "virtual thunk to ",
    "vtable for ",
)
_PRIVATE_SCOPE_PROBE_CANDIDATES = (
    "LLVMContextCreate",
    "LLVMParseIRInContext",
    "glfwInit",
    "mi_malloc",
    "volkInitialize",
)


def parse_posix_nm_symbols(text: str) -> set[str]:
    """Parse `nm -P` output for objects, archives, and shared libraries."""

    symbols: set[str] = set()
    for raw in text.splitlines():
        parts = raw.split()
        if parts and parts[0].endswith(":"):
            parts = parts[1:]
        if len(parts) < 2 or len(parts[1]) != 1 or not parts[1].isalpha():
            continue
        symbol = parts[0]
        if symbol:
            symbols.add(symbol)
    return symbols


def strip_elf_symbol_version(symbol: str) -> str:
    return symbol.split("@", 1)[0]


def _tool_path(env_name: str, candidates: tuple[str, ...]) -> str:
    configured = os.environ.get(env_name)
    if configured:
        return configured
    for candidate in candidates:
        discovered = shutil.which(candidate)
        if discovered:
            return discovered
    raise RuntimeError(f"none of {candidates!r} was found")


def _run_nm(path: Path, *, dynamic: bool, undefined: bool) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"symbol input does not exist: {path}")
    nm = _tool_path("TI_RUNTIME_NM", ("llvm-nm", "nm"))
    command = [nm, "-P", "-g"]
    if dynamic:
        command.append("-D")
    command.append("-u" if undefined else "--defined-only")
    command.append(str(path))
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"nm failed for {path} with {completed.returncode}:\n"
            f"{completed.stdout}"
        )
    return parse_posix_nm_symbols(completed.stdout)


def collect_symbols(
    paths: list[Path], *, dynamic: bool = False, undefined: bool = False
) -> set[str]:
    symbols: set[str] = set()
    for path in paths:
        symbols.update(
            _run_nm(path, dynamic=dynamic, undefined=undefined)
        )
    return symbols


def _demangle(symbols: list[str]) -> dict[str, str]:
    if not symbols:
        return {}
    cxxfilt = _tool_path("TI_RUNTIME_CXXFILT", ("llvm-cxxfilt", "c++filt"))
    completed = subprocess.run(
        [cxxfilt],
        input="\n".join(symbols) + "\n",
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"C++ demangler failed with {completed.returncode}:\n"
            f"{completed.stdout}"
        )
    values = completed.stdout.splitlines()
    if len(values) != len(symbols):
        raise RuntimeError("C++ demangler returned an inconsistent symbol count")
    return dict(zip(symbols, values, strict=True))


def forbidden_export_family(symbol: str, demangled: str) -> str | None:
    symbol = strip_elf_symbol_version(symbol)
    for family, prefixes in _C_SYMBOL_FAMILIES:
        if any(symbol.startswith(prefix) for prefix in prefixes):
            return family

    canonical = demangled
    changed = True
    while changed:
        changed = False
        for prefix in _DEMANGLED_PREFIXES:
            if canonical.startswith(prefix):
                canonical = canonical[len(prefix) :]
                changed = True
                break
    owner_region = canonical.split("(", 1)[0]
    owners = [
        (owner_region.rfind(namespace), family)
        for family, namespaces in _CPP_NAMESPACE_FAMILIES
        for namespace in namespaces
        if namespace in owner_region
    ]
    if not owners:
        return None
    _, owner = max(owners)
    return None if owner == "taichi" else owner


def audit_forbidden_exports(symbols: set[str]) -> dict[str, list[str]]:
    ordered = sorted(symbols)
    demangled = _demangle(ordered)
    result: dict[str, list[str]] = {}
    for symbol in ordered:
        family = forbidden_export_family(symbol, demangled[symbol])
        if family is not None:
            result.setdefault(family, []).append(symbol)
    return result


def _read_path_list(path: Path) -> list[Path]:
    values = [
        Path(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise RuntimeError(f"symbol input list is empty: {path}")
    return values


def _write_version_script(path: Path, exports: list[str]) -> None:
    lines = [f"{_VERSION_NODE} {{", "  global:"]
    lines.extend(f"    {symbol};" for symbol in exports)
    lines.extend(("  local:", "    *;", "};", ""))
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def generate(
    runtime_inputs_path: Path,
    shim_inputs_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> None:
    runtime_inputs = _read_path_list(runtime_inputs_path)
    shim_inputs = _read_path_list(shim_inputs_path)
    raw_symbols = collect_symbols(runtime_inputs)
    undefined = collect_symbols(shim_inputs, undefined=True)
    exports, manifest = build_export_closure(
        raw_symbols,
        undefined,
        platform="linux-elf",
        seeds=EXPLICIT_ABI_SEEDS,
    )
    if not raw_symbols or not undefined or not manifest[
        "shim_required_runtime_symbol_count"
    ]:
        raise RuntimeError(
            "runtime export closure is unexpectedly empty; refusing to link"
        )
    forbidden = audit_forbidden_exports(set(exports))
    if forbidden:
        samples = ", ".join(
            f"{family}={values[:3]}" for family, values in sorted(forbidden.items())
        )
        raise RuntimeError(
            "the shim private ABI depends on bundled third-party symbols: "
            + samples
        )
    configured_limit = int(
        os.environ.get("TI_POSIX_RUNTIME_EXPORT_LIMIT", DEFAULT_EXPORT_LIMIT)
    )
    if len(exports) > configured_limit:
        raise RuntimeError(
            f"runtime export closure has {len(exports)} symbols, exceeding "
            f"the safety limit {configured_limit}"
        )
    manifest.update(
        {
            "configured_export_limit": configured_limit,
            "runtime_input_count": len(runtime_inputs),
            "shim_object_count": len(shim_inputs),
            "version_node": _VERSION_NODE,
            "forbidden_export_families": [],
            "global_scope_probe_symbols": [
                symbol
                for symbol in _PRIVATE_SCOPE_PROBE_CANDIDATES
                if symbol in raw_symbols
            ],
            "private_abi_collision_probe_symbols": (
                select_private_abi_collision_probes(
                    exports,
                    _demangle(exports),
                    anchor="taichi_runtime_anchor",
                )
            ),
            "nm": _tool_path("TI_RUNTIME_NM", ("llvm-nm", "nm")),
        }
    )
    _write_version_script(output_path, exports)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_elf(runtime_path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    versioned_actual = collect_symbols([runtime_path], dynamic=True)
    actual = {strip_elf_symbol_version(symbol) for symbol in versioned_actual}
    audited = add_binary_audit(
        manifest, actual, audit_kind="elf-dynamic-symbol-table"
    )
    requested = set(manifest["exports"])
    unexpected = actual - requested - {_VERSION_NODE}
    if unexpected:
        raise RuntimeError(
            "ELF version script leaked unexpected runtime exports: "
            + ", ".join(sorted(unexpected)[:8])
        )
    forbidden = audit_forbidden_exports(actual)
    if forbidden:
        samples = ", ".join(
            f"{family}={values[:3]}" for family, values in sorted(forbidden.items())
        )
        raise RuntimeError(
            "ELF runtime exports bundled third-party APIs: " + samples
        )
    audited.update(
        {
            "elf_audited": True,
            "forbidden_export_families": [],
            "unexpected_export_count": len(unexpected),
        }
    )
    manifest_path.write_text(
        json.dumps(audited, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "--audit-elf":
        audit_elf(Path(argv[2]), Path(argv[3]))
        return 0
    if len(argv) != 5:
        print(
            "usage: generate_elf_runtime_export_closure.py "
            "<runtime-inputs.txt> <shim-objects.txt> <output.map> "
            "<manifest.json>\n"
            "   or: generate_elf_runtime_export_closure.py --audit-elf "
            "<libtaichi_runtime.so> <manifest.json>",
            file=sys.stderr,
        )
        return 2
    generate(*(Path(value) for value in argv[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
