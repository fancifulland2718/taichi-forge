#!/usr/bin/env python3
"""Validate the complete runtime plus CPython shim release set."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from packaging.utils import parse_wheel_filename
from packaging.version import Version

from scripts.validate_runtime_wheel import inspect_runtime_wheel
from scripts.validate_shim_wheel import validate_shim_wheel


EXPECTED_PYTHON_TAGS = frozenset({"cp310", "cp311", "cp312", "cp313", "cp314"})


def validate_release_set(wheel_dir: Path, expected_version: Version) -> None:
    runtime_wheels = sorted(wheel_dir.glob("taichi_forge_runtime-*.whl"))
    shim_wheels = sorted(wheel_dir.glob("taichi_forge-[0-9]*.whl"))
    if len(runtime_wheels) != 2:
        raise RuntimeError(
            f"Expected two runtime wheels, found {[p.name for p in runtime_wheels]}"
        )
    if len(shim_wheels) != 10:
        raise RuntimeError(
            f"Expected ten CPython shim wheels, found {[p.name for p in shim_wheels]}"
        )

    runtime_infos = [
        inspect_runtime_wheel(
            wheel,
            expected_dependency_class="driver-only",
            required_export_manifest_schema=2,
        )
        for wheel in runtime_wheels
    ]
    if {info.platform for info in runtime_infos} != {"windows", "manylinux"}:
        raise RuntimeError("Runtime release set must contain Windows and manylinux")
    if {Version(info.version) for info in runtime_infos} != {expected_version}:
        raise RuntimeError(
            "Runtime release versions do not match the requested version"
        )

    combinations = Counter()
    for wheel in shim_wheels:
        _, filename_version, _, tags = parse_wheel_filename(wheel.name)
        if filename_version != expected_version:
            raise RuntimeError(
                f"Shim version mismatch: expected={expected_version}, wheel={wheel.name}"
            )
        python_tags = {tag.interpreter for tag in tags}
        if len(python_tags) != 1:
            raise RuntimeError(f"Shim wheel has ambiguous Python tags: {wheel.name}")
        python_tag = next(iter(python_tags))
        platform = "windows" if "win_amd64" in wheel.name.lower() else "manylinux"
        validate_shim_wheel(wheel, platform, expected_python_tag=python_tag)
        combinations[(platform, python_tag)] += 1

    expected = {
        (platform, python_tag)
        for platform in ("windows", "manylinux")
        for python_tag in EXPECTED_PYTHON_TAGS
    }
    if set(combinations) != expected or any(
        count != 1 for count in combinations.values()
    ):
        raise RuntimeError(
            "Shim release matrix is incomplete or duplicated: "
            f"{dict(sorted(combinations.items()))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--version", type=Version, required=True)
    args = parser.parse_args()
    try:
        validate_release_set(args.wheel_dir, args.version)
    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Validated complete release set for {args.version}: "
        "2 runtime wheels, 10 CPython shims"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
