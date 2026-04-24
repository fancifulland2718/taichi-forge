"""
Taichi Forge — build shim for scikit-build-core.

Project metadata and build configuration now live in ``pyproject.toml``.
This file exists only so that legacy invocations like

    python setup.py bdist_wheel
    python setup.py clean

do not hard-fail with an uninformative traceback. It forwards the common
commands to PEP 517 equivalents (``python -m build``) and otherwise tells
the user to use pip / build directly.

If you really need the historical scikit-build logic (custom Clean, egg
tag-build, apple M1 codesign, manifest filter), see ``setup.py.legacy``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGACY_DIRS = (
    "_skbuild",
    "bin",
    "build",
    "dist",  # caller can keep LLVM snapshots inside; see README
    "python/taichi_forge/assets",
    "python/taichi_forge/_lib/runtime",
    "python/taichi_forge/_lib/c_api",
    "taichi_forge.egg-info",
    "python/taichi_forge.egg-info",
)


def _do_clean() -> int:
    print("[setup.py shim] running `clean` — removing scikit-build artifacts")
    for rel in LEGACY_DIRS:
        p = ROOT / rel
        if not p.is_dir():
            continue
        if rel == "dist":
            # Preserve locally-built toolchain snapshots (taichi-llvm-*) that
            # `scripts/build_llvm20_local.ps1` places under dist/.
            for child in p.iterdir():
                if child.name.startswith("taichi-llvm"):
                    print(f"  keep {child}")
                    continue
                print(f"  rm -rf {child}")
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    try:
                        child.unlink()
                    except OSError:
                        pass
            continue
        print(f"  rm -rf {p}")
        shutil.rmtree(p, ignore_errors=True)
    return 0


def _do_bdist_wheel(extra: list[str]) -> int:
    print(
        "[setup.py shim] delegating `bdist_wheel` to `python -m build -w`. "
        "See pyproject.toml for scikit-build-core configuration."
    )
    cmd = [sys.executable, "-m", "build", "-w"]
    cmd.extend(extra)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 0
    action = sys.argv[1]
    if action == "clean":
        return _do_clean()
    if action == "bdist_wheel":
        return _do_bdist_wheel(sys.argv[2:])
    if action == "build_ext":
        return subprocess.call([sys.executable, "-m", "pip", "install", "-e", "."], cwd=ROOT)
    print(
        f"[setup.py shim] subcommand {action!r} is no longer supported. "
        "Use `pip install .` or `python -m build -w` instead."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
