#!/usr/bin/env python3
"""Fail before upload when any release project already owns the version."""

from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from packaging.version import Version


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        choices=("pypi", "testpypi"),
        required=True,
    )
    parser.add_argument("--version", type=Version, required=True)
    parser.add_argument("projects", nargs="+")
    args = parser.parse_args()

    base = {
        "pypi": "https://pypi.org/pypi",
        "testpypi": "https://test.pypi.org/pypi",
    }[args.repository]
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
