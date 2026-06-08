from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
RUNTIME_PYPROJECT = ROOT / "packaging" / "runtime" / "pyproject.toml"
VERSION = ROOT / "version.txt"


def _version_parts(version: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"v?([0-9]+)\.([0-9]+)\.([0-9]+)(?:[-+.][0-9A-Za-z.-]+)?", version)
    if match is None:
        raise RuntimeError(f"Unsupported version format: {version!r}")
    return match.group(1), match.group(2), match.group(3)


def _sync_cmake_version(path: Path, major: str, minor: str, patch: str) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "TI_VERSION_MAJOR": major,
        "TI_VERSION_MINOR": minor,
        "TI_VERSION_PATCH": patch,
    }
    for key, value in replacements.items():
        text, count = re.subn(rf'{key} = "[^"]+"', f'{key} = "{value}"', text)
        if count != 1:
            raise RuntimeError(f"Expected one {key} entry in {path}, found {count}")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    version = VERSION.read_text(encoding="utf-8").strip().removeprefix("v")
    major, minor, patch = _version_parts(version)
    text = PYPROJECT.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'"taichi-forge-runtime==[^"]+"',
        f'"taichi-forge-runtime=={version}"',
        text,
    )
    if count != 1:
        raise RuntimeError(f"Expected one taichi-forge-runtime dependency, found {count}")
    PYPROJECT.write_text(updated, encoding="utf-8")
    _sync_cmake_version(PYPROJECT, major, minor, patch)
    _sync_cmake_version(RUNTIME_PYPROJECT, major, minor, patch)
    print(f"Synced taichi-forge-runtime dependency and CMake version to {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
