"""THIN-005: 2-D particle spatial-hash qualification entry point."""
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
            "particle_spatial_hash_microbench.py fixes "
            "--operation=particle_spatial_hash")
    return _shared_main(["--operation", "particle_spatial_hash", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
