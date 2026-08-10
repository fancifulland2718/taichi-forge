"""DIRECT-004-CONCURRENT: simultaneous live SNodeTree capacity entry point."""
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
            "snode_concurrent_microbench.py fixes --operation=snode_concurrent")
    return _shared_main(["--operation", "snode_concurrent", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
