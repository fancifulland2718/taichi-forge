"""DIRECT-002: identical-public-API legacy parallel-sort microbenchmark."""
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
        raise ValueError("parallel_sort_microbench.py fixes --operation=parallel_sort")
    return _shared_main(["--operation", "parallel_sort", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
