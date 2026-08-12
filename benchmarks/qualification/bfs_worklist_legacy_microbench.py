"""THIN-007-LEGACY: retained staged/telemetry BFS diagnostic."""

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
        raise ValueError("legacy BFS entry point fixes its operation and profile")
    return _shared_main(
        [
            "--operation",
            "bfs_worklist",
            "--contract-profile",
            "legacy",
            *arguments,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
