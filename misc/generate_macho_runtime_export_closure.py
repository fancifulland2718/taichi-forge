#!/usr/bin/env python3
"""Generate and audit the Mach-O split-runtime private ABI export closure."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

try:
    import generate_elf_runtime_export_closure as posix_support
    from runtime_export_closure import (
        DEFAULT_EXPORT_LIMIT,
        add_binary_audit,
        build_export_closure,
        select_private_abi_collision_probes,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import generate_elf_runtime_export_closure as posix_support  # type: ignore[no-redef]
    from runtime_export_closure import (  # type: ignore[no-redef]
        DEFAULT_EXPORT_LIMIT,
        add_binary_audit,
        build_export_closure,
        select_private_abi_collision_probes,
    )


_ABI_SEEDS = ("_taichi_runtime_anchor",)
_PRIVATE_SCOPE_PROBE_CANDIDATES = (
    "LLVMContextCreate",
    "LLVMParseIRInContext",
    "glfwInit",
    "mi_malloc",
    "volkInitialize",
)


def _run_nm(path: Path, *, undefined: bool) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"symbol input does not exist: {path}")
    nm = posix_support._tool_path("TI_RUNTIME_NM", ("llvm-nm", "nm"))
    # Apple nm uses -u for undefined-only and -U for defined-only. Unlike ELF,
    # the external name spelling includes Mach-O's leading underscore.
    command = [nm, "-P", "-g", "-u" if undefined else "-U", str(path)]
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
    return posix_support.parse_posix_nm_symbols(completed.stdout)


def collect_symbols(paths: list[Path], *, undefined: bool = False) -> set[str]:
    symbols: set[str] = set()
    for path in paths:
        symbols.update(_run_nm(path, undefined=undefined))
    return symbols


def _demangle(symbols: list[str]) -> dict[str, str]:
    # Mach-O prepends one underscore to external C names. Itanium C++ names
    # therefore begin with __Z; c++filt expects the underlying _Z spelling.
    normalized = [symbol[1:] if symbol.startswith("__Z") else symbol for symbol in symbols]
    values = posix_support._demangle(normalized)
    return {
        original: values[normalized_symbol]
        for original, normalized_symbol in zip(symbols, normalized, strict=True)
    }


def _without_macho_prefix(symbol: str) -> str:
    return symbol[1:] if symbol.startswith("_") else symbol


def audit_forbidden_exports(symbols: set[str]) -> dict[str, list[str]]:
    ordered = sorted(symbols)
    demangled = _demangle(ordered)
    result: dict[str, list[str]] = {}
    for symbol in ordered:
        family = posix_support.forbidden_export_family(
            _without_macho_prefix(symbol), demangled[symbol]
        )
        if family is not None:
            result.setdefault(family, []).append(symbol)
    return result


def _write_export_list(path: Path, exports: list[str]) -> None:
    path.write_text("\n".join(exports) + "\n", encoding="utf-8", newline="\n")


def generate(
    runtime_inputs_path: Path,
    shim_inputs_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> None:
    runtime_inputs = posix_support._read_path_list(runtime_inputs_path)
    shim_inputs = posix_support._read_path_list(shim_inputs_path)
    raw_symbols = collect_symbols(runtime_inputs)
    undefined = collect_symbols(shim_inputs, undefined=True)
    exports, manifest = build_export_closure(
        raw_symbols,
        undefined,
        platform="macos-macho",
        seeds=_ABI_SEEDS,
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
            "forbidden_export_families": [],
            "global_scope_probe_symbols": [
                symbol
                for symbol in _PRIVATE_SCOPE_PROBE_CANDIDATES
                if f"_{symbol}" in raw_symbols
            ],
            "private_abi_collision_probe_symbols": sorted(
                {
                    _without_macho_prefix(symbol)
                    for symbol in select_private_abi_collision_probes(
                        exports,
                        _demangle(exports),
                        anchor="_taichi_runtime_anchor",
                    )
                }
            ),
            "nm": posix_support._tool_path(
                "TI_RUNTIME_NM", ("llvm-nm", "nm")
            ),
        }
    )
    _write_export_list(output_path, exports)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit_macho(runtime_path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = collect_symbols([runtime_path])
    audited = add_binary_audit(
        manifest, actual, audit_kind="macho-external-symbol-table"
    )
    requested = set(manifest["exports"])
    unexpected = actual - requested
    if unexpected:
        raise RuntimeError(
            "Mach-O export list leaked unexpected runtime exports: "
            + ", ".join(sorted(unexpected)[:8])
        )
    forbidden = audit_forbidden_exports(actual)
    if forbidden:
        samples = ", ".join(
            f"{family}={values[:3]}" for family, values in sorted(forbidden.items())
        )
        raise RuntimeError(
            "Mach-O runtime exports bundled third-party APIs: " + samples
        )
    audited.update(
        {
            "macho_audited": True,
            "forbidden_export_families": [],
            "unexpected_export_count": len(unexpected),
        }
    )
    manifest_path.write_text(
        json.dumps(audited, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "--audit-macho":
        audit_macho(Path(argv[2]), Path(argv[3]))
        return 0
    if len(argv) != 5:
        print(
            "usage: generate_macho_runtime_export_closure.py "
            "<runtime-inputs.txt> <shim-objects.txt> <output.list> "
            "<manifest.json>\n"
            "   or: generate_macho_runtime_export_closure.py --audit-macho "
            "<libtaichi_runtime.dylib> <manifest.json>",
            file=sys.stderr,
        )
        return 2
    generate(*(Path(value) for value in argv[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
