"""DIRECT-005 moving sparse block-stencil qualification entry point."""

import sys

try:
    from .single_kernel_microbench import main as _shared_main
except ImportError:
    from single_kernel_microbench import main as _shared_main


def main() -> int:
    arguments = sys.argv[1:]
    return _shared_main(["--operation", "sparse_block_stencil", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
