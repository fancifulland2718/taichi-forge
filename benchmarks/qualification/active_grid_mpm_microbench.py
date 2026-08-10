"""THIN-004: stationary 2-D active-grid MLS-MPM qualification entry point."""
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
            "active_grid_mpm_microbench.py fixes --operation=active_grid_mpm")
    return _shared_main(["--operation", "active_grid_mpm", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
