"""THIN-001: one-operation native reduction qualification entry point.

Forge uses its reusable native reduction workspace. Vanilla uses the declared
equivalent i32 atomic-sum kernel. This is a thin-capability comparison, not an
identical-public-API comparison.
"""
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
        raise ValueError("native_reduce_microbench.py fixes --operation=native_reduce")
    return _shared_main(["--operation", "native_reduce", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
