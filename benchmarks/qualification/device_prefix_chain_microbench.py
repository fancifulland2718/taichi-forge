"""THIN-003: device-resident stable compact plus scan chain entry point."""
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
            "device_prefix_chain_microbench.py fixes "
            "--operation=device_prefix_chain")
    return _shared_main(["--operation", "device_prefix_chain", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
