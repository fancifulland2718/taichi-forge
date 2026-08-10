"""DIRECT-003 control: same 2-D MLS-MPM frame as direct kernel calls."""
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
        raise ValueError("mpm_direct_control.py fixes --operation=mpm_direct")
    return _shared_main(["--operation", "mpm_direct", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
