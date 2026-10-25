"""Reuse validated wheel sets without moving either project's PyPI publisher."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

from packaging.version import Version

from scripts.release_artifacts import write_json
from scripts.release_github import artifact_json, checked_run, gh, require_job, wheel_facts
from scripts.validate_release_wheel_set import validate_release_set


def select(project: str, repository: str, commit: str, version: str,
           runtime_version: str, *, run_id: str = "", reuse: bool = True) -> str:
    workflow = "publish_runtime_pypi.yml" if project == "runtime" else "publish_pypi.yml"
    job = f"Validate {project} wheel set"
    if run_id:
        checked_run(repository, run_id, workflow, completed=False)
        require_job(repository, run_id, job)
        return run_id
    if not reuse:
        return ""
    name = f"validated-{project}-source"
    pages = json.loads(gh("api", f"repos/{repository}/actions/artifacts?name={quote(name)}&per_page=100",
                         "--paginate", "--slurp"))
    for artifact in (a for page in pages for a in page["artifacts"]):
        if artifact["name"] != name or artifact["expired"] or artifact["workflow_run"]["head_sha"] != commit:
            continue
        candidate = str(artifact["workflow_run"]["id"])
        # A failed upload/Release job must not invalidate a successful build.
        try:
            checked_run(repository, candidate, workflow, completed=False)
            require_job(repository, candidate, job)
        except ValueError:
            continue
        info = artifact_json(repository, artifact, f"{project}-build-info.json")
        if (info.get("schema") != 1 or info.get("commit") != commit or info.get("version") != version
                or (project == "shim" and info.get("runtime_version") != runtime_version)):
            continue
        sets = json.loads(gh("api", f"repos/{repository}/actions/runs/{candidate}/artifacts?per_page=100",
                             "--paginate", "--slurp"))
        if any(a["name"] == f"validated-{project}-wheel-set" and not a["expired"]
               for page in sets for a in page["artifacts"]):
            return candidate
    return ""


def record_runtime(directory: Path, version: str, *, reused: bool) -> None:
    validate_release_set(directory, Version(version), project="runtime")
    path = directory / "runtime-build-info.json"
    records = [json.loads((directory / f"build-info-{p}.json").read_text(encoding="utf-8"))
               for p in ("windows", "linux")]
    if reused:
        info = json.loads(path.read_text(encoding="utf-8"))
        if (info.get("schema") != 1 or info["version"] != version
                or info["wheels"] != wheel_facts(directory) or info["build_inputs"] != records):
            raise ValueError("Reused runtime differs from the validated candidate")
    else:
        info = {"schema": 1, "version": version, "commit": os.environ["GITHUB_SHA"],
                "source_run_id": os.environ["GITHUB_RUN_ID"],
                "wheels": wheel_facts(directory), "build_inputs": records}
    for platform, record in zip(("windows", "linux"), records):
        if (record.get("schema") != 1 or record["version"] != version
                or record["commit"] != info["commit"] or record["platform"] != platform):
            raise ValueError("Runtime candidate build records disagree")
    write_json(path, info)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "record-runtime"))
    parser.add_argument("--project", choices=("runtime", "shim"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--runtime-version", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--reuse", choices=("true", "false"), default="true")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.command == "select":
        selected = select(args.project, os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_SHA"],
                          args.version, args.runtime_version, run_id=args.run_id, reuse=args.reuse == "true")
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"run_id={selected}\n")
        print(f"Reuse validated {args.project} wheels from run {selected}" if selected else f"Build {args.project}: no selected reusable candidate")
    else:
        record_runtime(args.directory, args.version, reused=bool(args.run_id))


if __name__ == "__main__":
    main()
