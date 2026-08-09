"""DIRECT-001: one-operation PrefixSumExecutor qualification entry point.

This intentionally exposes no aggregate case selection. The shared A/B driver
still launches each runtime in a fresh, adjacent, non-overlapping process.
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
        raise ValueError("prefix_sum_microbench.py fixes --operation=prefix_sum")
    return _shared_main(["--operation", "prefix_sum", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
