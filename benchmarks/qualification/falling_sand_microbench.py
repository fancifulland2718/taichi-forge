"""THIN-008: deterministic falling-sand keyed-claim qualification entry."""
from __future__ import annotations

import sys
from typing import Sequence

try:
    from .single_kernel_microbench import main as _shared_main
except ImportError:  # Direct execution from this directory.
    from single_kernel_microbench import main as _shared_main


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--operation" in arguments:
        raise ValueError(
            "falling_sand_microbench.py fixes --operation=falling_sand")
    return _shared_main(["--operation", "falling_sand", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
