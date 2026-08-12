"""THIN-004-CURRENT: current-contract active-grid MPM qualification."""

from __future__ import annotations

import sys
from typing import Sequence

try:
    from .single_kernel_microbench import main as _shared_main
except ImportError:  # Direct execution from this directory.
    from single_kernel_microbench import main as _shared_main


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--operation" in arguments or "--contract-profile" in arguments:
        raise ValueError("active-grid current-contract entry point fixes its operation and profile")
    return _shared_main(
        [
            "--operation",
            "active_grid_mpm",
            "--contract-profile",
            "current",
            *arguments,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
