"""Coordinate immutable, retryable GitHub release assets using the installed gh CLI."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from packaging.version import Version

from scripts.release_artifacts import digest, verify_bundle


def gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True, encoding="utf-8")


def check_run(repository: str, run_id: str, commit: str) -> None:
    if not run_id.isdecimal():
        raise ValueError("candidate_run_id must be numeric")
    run = json.loads(gh("api", f"repos/{repository}/actions/runs/{run_id}"))
    if (run["head_sha"] != commit or run["path"] != ".github/workflows/release.yml"
            or run["head_repository"]["full_name"] != repository
            or run["status"] != "completed" or run["conclusion"] != "success"):
        raise ValueError("Candidate must be a successful release.yml run on this exact source commit")


def synchronize(directory: Path, repository: str, commit: str, version: str, *, publish: bool) -> None:
    verify_bundle(directory, commit, version)
    tag = f"forge-v{version}"
    remote_commit = gh("api", f"repos/{repository}/commits/{tag}", "--jq", ".sha").strip()
    if remote_commit != commit:
        raise ValueError("Remote tag does not identify the release source")
    result = subprocess.run(["gh", "api", f"repos/{repository}/releases/tags/{tag}"], capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        if "HTTP 404" not in result.stderr or publish:
            raise RuntimeError(result.stderr)
        command = ["release", "create", tag, "--repo", repository, "--verify-tag", "--draft",
                   "--title", f"Taichi Forge {version} / 发布说明", "--notes-file", str(directory / "RELEASE_NOTES.md")]
        if Version(version).is_prerelease:
            command.append("--prerelease")
        gh(*command)
        release = json.loads(gh("api", f"repos/{repository}/releases/tags/{tag}"))
    else:
        release = json.loads(result.stdout)
    expected_body = (directory / "RELEASE_NOTES.md").read_text(encoding="utf-8").strip()
    if (release["body"].strip() != expected_body
            or release["name"] != f"Taichi Forge {version} / 发布说明"
            or release["prerelease"] != Version(version).is_prerelease):
        raise ValueError("Existing release metadata differs; do not overwrite it")
    pages = json.loads(gh("api", f"repos/{repository}/releases/{release['id']}/assets?per_page=100", "--paginate", "--slurp"))
    assets = {asset["name"]: asset for page in pages for asset in page}
    expected = {p.name for p in directory.iterdir()}
    if set(assets) - expected:
        raise ValueError("Existing release contains unexpected assets")
    for name in sorted(expected):
        path = directory / name
        if name in assets:
            asset = assets[name]
            if asset["size"] != path.stat().st_size:
                raise ValueError(f"Published asset size differs: {name}")
            checksum = asset.get("digest")
            if not checksum:
                with tempfile.TemporaryFile() as downloaded:
                    subprocess.run(["gh", "api", f"repos/{repository}/releases/assets/{asset['id']}",
                                    "-H", "Accept: application/octet-stream"], stdout=downloaded, check=True)
                    downloaded.seek(0)
                    import hashlib
                    checksum = "sha256:" + hashlib.file_digest(downloaded, "sha256").hexdigest()
            if checksum != "sha256:" + digest(path):
                raise ValueError(f"Published asset content differs: {name}")
        elif publish or not release["draft"]:
            raise ValueError(f"Release is missing asset: {name}")
        else:
            gh("release", "upload", tag, str(path), "--repo", repository)
    if publish and release["draft"]:
        gh("release", "edit", tag, "--repo", repository, "--draft=false", "--latest=false" if Version(version).is_prerelease else "--latest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-run", "stage", "publish"))
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--version")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.command == "check-run":
        check_run(args.repository, args.run_id or "", args.commit)
    else:
        synchronize(args.directory, args.repository, args.commit, args.version, publish=args.command == "publish")


if __name__ == "__main__":
    main()
