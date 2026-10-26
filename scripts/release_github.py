"""Coordinate immutable, retryable GitHub release assets using the installed gh CLI."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile
from urllib.parse import quote

from packaging.version import Version

from scripts.release_artifacts import digest, verify_bundle, write_json
from scripts.check_pypi_version_available import prepare_upload
from scripts.validate_release_wheel_set import validate_release_set


def gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True, encoding="utf-8")


def checked_run(repository: str, run_id: str, workflow: str, *, completed: bool = True) -> dict:
    if not run_id.isdecimal():
        raise ValueError("Workflow run ID must be numeric")
    run = json.loads(gh("api", f"repos/{repository}/actions/runs/{run_id}"))
    if (run["path"] != f".github/workflows/{workflow}"
            or (run.get("head_repository") or {}).get("full_name") != repository
            or run.get("event") not in {"push", "workflow_dispatch"}
            or (completed and (run["status"] != "completed" or run["conclusion"] != "success"))):
        raise ValueError("Artifact must come from the existing publisher in this repository")
    return run


def wheel_facts(directory: Path) -> list[dict]:
    return [{"name": p.name, "sha256": digest(p)} for p in sorted(directory.glob("*.whl"), key=lambda p: p.name)]


def require_job(repository: str, run_id: str, name: str) -> None:
    pages = json.loads(gh("api", f"repos/{repository}/actions/runs/{run_id}/jobs?per_page=100", "--paginate", "--slurp"))
    if not any(j["name"] == name and j["conclusion"] == "success" for page in pages for j in page["jobs"]):
        raise ValueError(f"Required publisher job has not completed successfully: {name}")


def record_runtime(directory: Path, version: str, index: str, output: Path) -> None:
    validate_release_set(directory, Version(version), project="runtime")
    with tempfile.TemporaryDirectory() as temporary:
        base = "https://pypi.org/pypi" if index == "pypi" else "https://test.pypi.org/pypi"
        if prepare_upload(directory, Path(temporary) / "missing", base):
            raise ValueError("Runtime publication is not complete")
    candidate = json.loads((directory / "runtime-build-info.json").read_text(encoding="utf-8"))
    if candidate["version"] != version or candidate["wheels"] != wheel_facts(directory):
        raise ValueError("Runtime publication differs from the validated candidate")
    records = []
    for platform in ("windows", "linux"):
        record = json.loads((directory / f"build-info-{platform}.json").read_text(encoding="utf-8"))
        if record["version"] != version or record["platform"] != platform or record["commit"] != candidate["commit"]:
            raise ValueError("Runtime build record identity differs")
        records.append(record)
    if records != candidate["build_inputs"]:
        raise ValueError("Runtime publication build inputs differ")
    write_json(output, {"schema": 1, "index": index, "version": version,
                       "commit": candidate["commit"], "source_run_id": candidate["source_run_id"],
                       "run_id": os.environ["GITHUB_RUN_ID"],
                       "wheels": wheel_facts(directory), "build_inputs": records})


def record_shim(directory: Path, repository: str, run_id: str, version: str, runtime_version: str) -> None:
    run = checked_run(repository, run_id, "publish_pypi.yml", completed=False)
    validate_release_set(directory, Version(version), project="shim", runtime_version=Version(runtime_version))
    # A re-publication must preserve the original compile provenance, even when
    # the selected source run itself reused an older validated wheel set.
    previous = Path("artifacts/shim-build-info.json")
    if run_id != os.environ.get("GITHUB_RUN_ID"):
        require_job(repository, run_id, "Validate shim wheel set")
        if not previous.is_file():
            raise ValueError("Reused shims require their original shim-build-info.json")
        info = json.loads(previous.read_text(encoding="utf-8"))
        if info["wheels"] != wheel_facts(directory) or info["version"] != version or info["runtime_version"] != runtime_version:
            raise ValueError("Reused shim source record differs")
    else:
        info = {"schema": 1, "commit": run["head_sha"], "version": version,
                "runtime_version": runtime_version, "source_run_id": run_id,
                "wheels": wheel_facts(directory)}
    write_json(directory / "shim-build-info.json", info)


def artifact_json(repository: str, artifact: dict, filename: str) -> dict:
    with tempfile.TemporaryFile() as downloaded:
        subprocess.run(["gh", "api", f"repos/{repository}/actions/artifacts/{artifact['id']}/zip"], stdout=downloaded, check=True)
        downloaded.seek(0)
        with zipfile.ZipFile(downloaded) as archive:
            if archive.namelist() != [filename]:
                raise ValueError("Unexpected publication manifest contents")
            return json.loads(archive.read(filename))


def resolve_sources(directory: Path, repository: str, shim_run_id: str, version: str, runtime_version: str) -> str:
    # The caller may still be running its final GitHub job. Its PyPI job must
    # already have succeeded; a merely successful build/dry run is insufficient.
    checked_run(repository, shim_run_id, "publish_pypi.yml", completed=False)
    require_job(repository, shim_run_id, "Publish validated shim wheels")
    info = json.loads((directory / "shim-build-info.json").read_text(encoding="utf-8"))
    source = checked_run(repository, str(info["source_run_id"]), "publish_pypi.yml", completed=False)
    require_job(repository, str(info["source_run_id"]), "Validate shim wheel set")
    if info.get("schema") != 1 or info["commit"] != source["head_sha"] or info["version"] != version or info["runtime_version"] != runtime_version:
        raise ValueError("Shim source record differs")
    actual = {entry["name"]: entry for entry in wheel_facts(directory)}
    if [entry for entry in actual.values() if entry["name"].startswith("taichi_forge-")] != info["wheels"]:
        raise ValueError("Shim wheel differs from its validated source record")
    remote = gh("api", f"repos/{repository}/commits/forge-v{version}", "--jq", ".sha").strip()
    if remote != info["commit"]:
        raise ValueError("Version tag must identify the actual shim source commit")
    # This named artifact is metadata only; all runtime wheel bytes come from PyPI.
    name = f"runtime-publication-{runtime_version}-pypi"
    pages = json.loads(gh("api", f"repos/{repository}/actions/artifacts?name={quote(name)}&per_page=100", "--paginate", "--slurp"))
    expected_runtime = [entry for entry in actual.values() if entry["name"].startswith("taichi_forge_runtime-")]
    for artifact in (item for page in pages for item in page["artifacts"]):
        if artifact["expired"] or artifact["name"] != name:
            continue
        checked_run(repository, str(artifact["workflow_run"]["id"]), "publish_runtime_pypi.yml")
        publication = artifact_json(repository, artifact, "runtime-publication.json")
        if (publication.get("schema") != 1 or publication["index"] != "pypi"
                or publication["version"] != runtime_version or publication["wheels"] != expected_runtime
                or str(publication["run_id"]) != str(artifact["workflow_run"]["id"])):
            continue
        build_source = checked_run(repository, str(publication["source_run_id"]), "publish_runtime_pypi.yml", completed=False)
        require_job(repository, str(publication["source_run_id"]), "Validate runtime wheel set")
        if publication["commit"] != build_source["head_sha"]:
            raise ValueError("Runtime compile source differs")
        records = publication["build_inputs"]
        if {r["platform"] for r in records} != {"windows", "linux"} or len(records) != 2:
            raise ValueError("Incomplete runtime build records")
        for record in records:
            if record["commit"] != publication["commit"] or record["version"] != runtime_version:
                raise ValueError("Runtime build source differs")
            write_json(directory / f"build-info-{record['platform']}.json", record)
        write_json(directory / "publication-sources.json", {"shim_run_id": shim_run_id, "shim": info, "runtime": publication})
        return info["commit"]
    raise ValueError("No retained runtime publication record matches the PyPI wheels; restore the original publication artifact")


def find_release(repository: str, tag: str) -> dict | None:
    # The by-tag endpoint only returns published releases. The authenticated
    # list also includes drafts, which must remain discoverable during staging
    # and retries after an interrupted upload.
    pages = json.loads(gh("api", f"repos/{repository}/releases?per_page=100", "--paginate", "--slurp"))
    matches = [release for page in pages for release in page if release["tag_name"] == tag]
    if len(matches) > 1:
        raise ValueError(f"Multiple releases identify tag {tag}; resolve the duplicate drafts")
    return matches[0] if matches else None


def synchronize(directory: Path, repository: str, commit: str, version: str, *, publish: bool) -> None:
    verify_bundle(directory, commit, version)
    tag = f"forge-v{version}"
    remote_commit = gh("api", f"repos/{repository}/commits/{tag}", "--jq", ".sha").strip()
    if remote_commit != commit:
        raise ValueError("Remote tag does not identify the release source")
    release = find_release(repository, tag)
    if release is None:
        if publish:
            raise ValueError("Stage the release and its assets before publishing")
        command = ["release", "create", tag, "--repo", repository, "--verify-tag", "--draft",
                   "--title", f"Taichi Forge {version} / 发布说明", "--notes-file", str(directory / "RELEASE_NOTES.md")]
        if Version(version).is_prerelease:
            command.append("--prerelease")
        gh(*command)
        release = find_release(repository, tag)
        if release is None:
            raise RuntimeError(f"Created release draft is not visible for tag {tag}")
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
    parser.add_argument("command", choices=("record-runtime", "record-shim", "resolve-sources", "stage", "publish"))
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--version")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--runtime-version")
    parser.add_argument("--index", choices=("pypi", "testpypi"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "record-runtime":
        record_runtime(args.directory, args.version, args.index, args.output)
    elif args.command == "record-shim":
        record_shim(args.directory, args.repository, args.run_id, args.version, args.runtime_version)
    elif args.command == "resolve-sources":
        commit = resolve_sources(args.directory, args.repository, args.run_id, args.version, args.runtime_version)
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"commit={commit}\n")
    else:
        synchronize(args.directory, args.repository, args.commit, args.version, publish=args.command == "publish")


if __name__ == "__main__":
    main()
