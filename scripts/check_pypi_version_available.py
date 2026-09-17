#!/usr/bin/env python3
"""Inspect release availability or stage only missing, hash-checked wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from packaging.version import Version
from packaging.utils import parse_wheel_filename


def _released_versions(payload: dict) -> set[Version]:
    releases = payload.get("releases", {})
    if not isinstance(releases, dict):
        raise RuntimeError("PyPI JSON response has no releases mapping")
    return {Version(raw) for raw in releases}


def version_exists(repository: str, project: str, version: Version) -> bool:
    url = f"{repository.rstrip('/')}/{quote(project)}/json"
    request = Request(url, headers={"User-Agent": "taichi-forge-release-check"})
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    return version in _released_versions(payload)


def prepare_upload(wheel_dir: Path, upload_dir: Path, repository: str) -> int:
    """Compare every candidate before staging; never replace a published file."""
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheel candidates in {wheel_dir}")
    if upload_dir.exists() and any(upload_dir.iterdir()):
        raise RuntimeError(f"Upload staging directory must be empty: {upload_dir}")
    releases = {}
    pending = []
    for wheel in wheels:
        project, version, _, _ = parse_wheel_filename(wheel.name)
        if project not in {"taichi-forge", "taichi-forge-runtime"}:
            raise RuntimeError(f"Unexpected publication project: {project}")
        key = (project, version)
        if key not in releases:
            request = Request(
                f"{repository.rstrip('/')}/{quote(project)}/{version}/json",
                headers={"User-Agent": "taichi-forge-release-check"},
            )
            try:
                with urlopen(request, timeout=30) as response:  # noqa: S310
                    payload = json.load(response)
            except HTTPError as error:
                if error.code != 404:
                    raise
                payload = {"urls": []}
            files = payload.get("urls")
            if not isinstance(files, list):
                raise RuntimeError(f"Invalid release-file response for {project} {version}")
            releases[key] = {entry["filename"]: entry for entry in files}
        existing = releases[key].get(wheel.name)
        if existing is None:
            pending.append(wheel)
            continue
        digest = hashlib.sha256()
        with wheel.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if existing.get("digests", {}).get("sha256") != digest.hexdigest():
            raise RuntimeError(f"Published wheel differs from candidate: {wheel.name}")
        if existing.get("yanked", False):
            raise RuntimeError(f"Published wheel is yanked: {wheel.name}")
        print(f"Already published with matching SHA256: {wheel.name}")
    upload_dir.mkdir(parents=True, exist_ok=True)
    for wheel in pending:
        shutil.copy2(wheel, upload_dir / wheel.name)
    print(f"Prepared {len(pending)} missing wheel(s); {len(wheels) - len(pending)} already match")
    return len(pending)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        choices=("pypi", "testpypi"),
        required=True,
    )
    parser.add_argument("--version", type=Version)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--upload-dir", type=Path)
    parser.add_argument("projects", nargs="*")
    args = parser.parse_args()

    base = {
        "pypi": "https://pypi.org/pypi",
        "testpypi": "https://test.pypi.org/pypi",
    }[args.repository]
    if args.wheel_dir is not None or args.upload_dir is not None:
        if args.wheel_dir is None or args.upload_dir is None or args.version or args.projects:
            parser.error("Use --wheel-dir and --upload-dir together, without --version/projects")
        prepare_upload(args.wheel_dir, args.upload_dir, base)
        return 0
    if args.version is None or not args.projects:
        parser.error("Availability inspection requires --version and projects")
    conflicts = [
        project
        for project in args.projects
        if version_exists(base, project, args.version)
    ]
    if conflicts:
        raise SystemExit(
            f"Version {args.version} already exists on {args.repository} for: "
            + ", ".join(conflicts)
        )
    print(
        f"Version {args.version} is unused on {args.repository} for: "
        + ", ".join(args.projects)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
