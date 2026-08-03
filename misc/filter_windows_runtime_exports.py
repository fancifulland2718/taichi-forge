#!/usr/bin/env python3
"""Filter CMake's auto-generated Windows exports for taichi_runtime."""

from __future__ import annotations

import sys
from pathlib import Path


_MSVC_PRIVATE_METADATA_PREFIXES = (
    "?$TSS",  # thread-safe static initialization guards
    "??_7",  # vftables
    "??_R",  # RTTI descriptors and hierarchy metadata
)


def keep_export(line: str) -> bool:
    symbol = line.strip().split(None, 1)[0] if line.strip() else ""
    if not symbol or symbol == "EXPORTS":
        return False
    if "@taichi@@" not in symbol:
        return False
    # The split shim imports callable Taichi symbols and a small set of named
    # data objects. Exporting compiler-owned RTTI/vftable/TSS records is not an
    # ABI contract, and can overflow the 65,535-member MSVC import-library
    # limit even though the shim has no undefined references to these records.
    if symbol.startswith(_MSVC_PRIVATE_METADATA_PREFIXES):
        return False
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: filter_windows_runtime_exports.py <input.def> <output.def>",
            file=sys.stderr,
        )
        return 2

    input_path = Path(argv[1])
    output_path = Path(argv[2])

    seen: set[str] = set()
    kept: list[str] = []
    for line in input_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not keep_export(line):
            continue
        stripped = line.strip()
        if stripped in seen:
            continue
        seen.add(stripped)
        kept.append(f"\t{stripped}")

    output_path.write_text("EXPORTS\n" + "\n".join(kept) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
