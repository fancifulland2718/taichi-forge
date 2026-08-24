from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
RUNTIME_PYPROJECT = ROOT / "packaging" / "runtime" / "pyproject.toml"
VERSION = ROOT / "version.txt"
VERSION_HEADER = ROOT / "taichi" / "common" / "version.h"


def _write_text_if_changed(path: Path, text: str) -> None:
    if path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def _normalize_version(version: str) -> str:
    version = version.strip()
    if version.startswith("forge-v"):
        version = version[len("forge-v") :]
    elif version.startswith("v"):
        version = version[1:]
    return version


def _version_parts(version: str) -> tuple[str, str, str]:
    match = re.fullmatch(
        r"([0-9]+)\.([0-9]+)\.([0-9]+)"
        r"(?:(?:a|b|rc)[0-9]+|(?:\.dev|\.post)[0-9]+|[-+][0-9A-Za-z.-]+)?",
        version,
    )
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
    _write_text_if_changed(path, text)


def _sync_version_header(
    path: Path, major: str, minor: str, patch: str
) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "TI_VERSION_MAJOR": major,
        "TI_VERSION_MINOR": minor,
        "TI_VERSION_PATCH": patch,
    }
    for key, value in replacements.items():
        text, count = re.subn(
            rf"#define {key} [0-9]+", f"#define {key} {value}", text
        )
        if count != 1:
            raise RuntimeError(f"Expected one {key} entry in {path}, found {count}")
    _write_text_if_changed(path, text)


def main() -> int:
    version = _normalize_version(VERSION.read_text(encoding="utf-8"))
    major, minor, patch = _version_parts(version)
    text = PYPROJECT.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'"taichi-forge-runtime==[^"]+"',
        f'"taichi-forge-runtime=={version}"',
        text,
    )
    if count != 1:
        raise RuntimeError(f"Expected one taichi-forge-runtime dependency, found {count}")
    _write_text_if_changed(PYPROJECT, updated)
    _sync_cmake_version(PYPROJECT, major, minor, patch)
    _sync_cmake_version(RUNTIME_PYPROJECT, major, minor, patch)
    _sync_version_header(VERSION_HEADER, major, minor, patch)
    print(
        "Synced CMake version to "
        f"{version} and taichi-forge-runtime dependency to the same version"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
