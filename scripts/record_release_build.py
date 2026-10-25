"""Record actual runtime build inputs beside each validated platform wheel."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import sys

from scripts.release_artifacts import digest, git, public_url, write_json


def record(root: Path, platform: str, output: Path) -> None:
    caches = list((root / "_skbuild").glob("*/CMakeCache.txt"))
    if len(caches) != 1:
        raise ValueError("Expected one runtime CMakeCache.txt")
    cache = caches[0]
    compiler_files = list(cache.parent.glob("CMakeFiles/*/CMakeCXXCompiler.cmake"))
    if len(compiler_files) != 1:
        raise ValueError("Expected one CMake C++ compiler record")
    compiler = compiler_files[0].read_text(encoding="utf-8")
    facts = {}
    for name in ("ID", "VERSION", "ARCHITECTURE_ID"):
        match = re.search(rf'set\(CMAKE_CXX_COMPILER_{name} "([^"]*)"\)', compiler)
        if match:
            facts[name.lower()] = match.group(1)
    if not facts.get("id") or not facts.get("version"):
        raise ValueError("Missing actual C++ compiler identity")
    key = "WIN" if platform == "windows" else "LINUX"
    url = public_url(os.environ[f"LLVM20_{key}_URL"])
    sha = os.environ[f"LLVM20_{key}_SHA256"].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid LLVM digest")
    constraints = ["packaging/constraints/release-build.txt", "packaging/constraints/cudss-build.txt"]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {
        "schema": 1, "platform": platform,
        "commit": git(root, "rev-parse", "HEAD").decode().strip(),
        "version": (root / "version.txt").read_text().strip().removeprefix("v"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "llvm": {"url": url, "sha256": sha},
        "cuda_toolkit_version": os.environ["CUDA_TOOLKIT_VERSION"],
        "compiler": facts, "python": sys.version,
        "build_packages": {name: importlib.metadata.version(name) for name in ("build", "scikit-build-core", "cmake", "ninja", "numpy", "pybind11")},
        "constraints": {name: digest(root / name) for name in constraints},
        "cmake_cache_sha256": digest(cache),
        "cmake_options": {m[1]: m[2] for m in re.finditer(r"^(TI_[A-Z_]+):BOOL=(.*)$", cache.read_text(encoding="utf-8"), re.M)},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("windows", "linux"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record(Path.cwd(), args.platform, args.output)
