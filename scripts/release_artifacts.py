"""Build and verify a public release bundle from one committed source tree.

Source archives include pinned submodules, not working-tree files. Wheels are
the already validated runtime/shim set; this tool never rebuilds or relabels them.
"""
from __future__ import annotations

import argparse
import configparser
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from urllib.parse import urlsplit
import zipfile

from packaging.version import Version

from scripts.validate_release_wheel_set import validate_release_set


PRIVATE_ROOTS = {"internal", "opt_doc", "temp_outputs", ".codex", ".agents", ".agent"}
REPOSITORY = "https://github.com/fancifulland2718/taichi-forge"


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def public_url(value: str) -> str:
    url = urlsplit(value)
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.hostname in {"localhost", "127.0.0.1", "::1"}):
        raise ValueError(f"Expected a public HTTPS source/download URL: {value!r}")
    return value


def check_version(root: Path, ref: str, *, publishing: bool = False) -> str:
    version = (root / "version.txt").read_text(encoding="utf-8").strip().removeprefix("v")
    parsed = Version(version)
    if str(parsed) != version or parsed.local or parsed.is_devrelease:
        raise ValueError("Release version must be canonical, without local/dev suffix")
    if publishing and ref != f"refs/tags/forge-v{version}":
        raise ValueError("Publication requires the matching forge-v version tag")
    if ref.startswith("refs/tags/") and ref != f"refs/tags/forge-v{version}":
        raise ValueError("Tag and source version differ")
    for relative in ("pyproject.toml", "packaging/runtime/pyproject.toml"):
        metadata = tomllib.loads((root / relative).read_text(encoding="utf-8"))
        defines = metadata["tool"]["scikit-build"]["cmake"]["define"]
        actual = tuple(int(defines[f"TI_VERSION_{part}"]) for part in ("MAJOR", "MINOR", "PATCH"))
        if actual != parsed.release:
            raise ValueError(f"CMake version differs: {relative}")
        if relative == "pyproject.toml" and f"taichi-forge-runtime=={version}" not in metadata["project"]["dependencies"]:
            raise ValueError("Coordinated release requires the matching runtime version")
    header = (root / "taichi/common/version.h").read_text(encoding="utf-8")
    for part, number in zip(("MAJOR", "MINOR", "PATCH"), parsed.release):
        if not re.search(rf"^#define TI_VERSION_{part} {number}$", header, re.M):
            raise ValueError("Native version header differs")
    return version


def is_public(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return bool(parts) and parts[0] not in PRIVATE_ROOTS and ".git" not in parts


def source_archive(root: Path, commit: str, output: Path, version: str) -> dict:
    """Archive commit objects recursively, retaining licenses and executable modes."""
    prefix = f"taichi-forge-{version}"
    records = []
    timestamp = int(git(root, "show", "-s", "--format=%ct", commit))

    def add_bytes(archive, name, data):
        entry = tarfile.TarInfo(f"{prefix}/{name}")
        entry.size, entry.mtime, entry.mode = len(data), timestamp, 0o644
        archive.addfile(entry, io.BytesIO(data))

    def append_repo(archive, repository, revision, relative=""):
        # Stream Git's archive through disk instead of retaining large sources in RAM.
        with tempfile.TemporaryFile() as storage:
            subprocess.run(["git", "-C", str(repository), "archive", "--format=tar", revision], stdout=storage, check=True)
            storage.seek(0)
            with tarfile.open(fileobj=storage, mode="r:") as source:
                for member in source:
                    path = f"{relative}{member.name}"
                    if not is_public(path):
                        continue
                    if member.name.startswith("/") or ".." in PurePosixPath(member.name).parts:
                        raise ValueError("Unsafe source archive member")
                    stream = source.extractfile(member) if member.isfile() else None
                    if stream and member.size < 1024:
                        start = stream.read()
                        if start.startswith(b"version https://git-lfs.github.com/spec/v1"):
                            raise ValueError(f"Unresolved LFS source: {path}")
                        stream = io.BytesIO(start)
                    member.name = f"{prefix}/{path}"
                    member.uid = member.gid = 0
                    member.uname = member.gname = ""
                    member.mtime = timestamp
                    member.pax_headers = {}
                    archive.addfile(member, stream)
        links = []
        for entry in git(repository, "ls-tree", "-rz", revision).split(b"\0"):
            if entry.startswith(b"160000 "):
                metadata, path = entry.split(b"\t", 1)
                links.append((path.decode("utf-8"), metadata.split()[2].decode()))
        if links:
            modules = configparser.ConfigParser(interpolation=None)
            modules.read_string(git(repository, "show", f"{revision}:.gitmodules").decode())
            urls = {modules[s]["path"]: modules[s]["url"] for s in modules.sections() if "path" in modules[s]}
            for path, child_commit in links:
                if not is_public(relative + path):
                    continue
                child = repository / path
                # An absent submodule must not silently fall back to its parent's Git repo.
                if not (child / ".git").exists() or not child.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"Initialize the pinned submodule first: {relative}{path}")
                records.append({"path": relative + path, "commit": child_commit, "url": public_url(urls[path])})
                append_repo(archive, child, child_commit, relative + path + "/")

    with output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            append_repo(archive, root, commit)
            manifest = {"schema": 1, "repository": REPOSITORY, "commit": commit, "version": version, "submodules": records}
            add_bytes(archive, "SOURCE_COMMIT", (commit + "\n").encode())
            add_bytes(archive, "SOURCE_MANIFEST.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def notes(root: Path, version: str, commit: str) -> str:
    sections = []
    for language, heading in (("zh", "中文"), ("en", "English")):
        text = (root / f"docs/forge/release_notes.{language}.md").read_text(encoding="utf-8")
        match = re.search(rf"^## {re.escape(version)}(?=[ \t（(\r\n])[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
        if not match:
            raise ValueError(f"Missing {language} release notes for {version}")
        body = match.group(1)
        # The source index has anchors for the next release immediately before its heading.
        body = re.sub(r'\n<a id="[^\"]+"></a>\s*', "\n", body).strip()
        def link(match):
            target = match.group(1)
            if "://" in target or target.startswith("#"):
                return match.group(0)
            return f"]({REPOSITORY}/blob/{commit}/docs/forge/{target})"
        body = re.sub(r"\]\(([^)]+)\)", link, body)
        sections.append(f"## {heading}\n\n{body}")
    return (f"# Taichi Forge {version}\n\n"
            f"Source / 源码: [{commit[:9]}]({REPOSITORY}/tree/{commit})\n\n"
            "GitHub 与 PyPI 提供相同的 wheel；附件含完整源码、公开文档、构建清单与 SHA256SUMS。\n\n"
            "GitHub and PyPI provide identical wheels. Assets include complete sources, public documentation, build metadata and SHA256SUMS.\n\n"
            + "\n\n".join(sections) + "\n")


def build_bundle(root: Path, wheels: Path, output: Path, ref: str) -> None:
    version = check_version(root, ref)
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    if git(root, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules"):
        raise ValueError("Release packaging requires a clean committed source tree")
    validate_release_set(wheels, Version(version))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Release output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    for wheel in sorted(wheels.glob("*.whl")):
        shutil.copyfile(wheel, output / wheel.name)
    source = output / f"taichi-forge-{version}-source.tar.gz"
    manifest = source_archive(root, commit, source, version)
    # Derive public docs from the committed archive, not ignored local documents.
    with tarfile.open(source) as archive, zipfile.ZipFile(output / f"taichi-forge-{version}-docs.zip", "w", zipfile.ZIP_DEFLATED) as docs:
        for member in archive:
            relative = PurePosixPath(member.name).parts[1:]
            if member.isfile() and (relative[0] == "docs" or relative[0] in {"README.md", "README.zh-CN.md", "LICENSE"}):
                info = zipfile.ZipInfo("/".join(relative))
                info.compress_type = zipfile.ZIP_DEFLATED
                docs.writestr(info, archive.extractfile(member).read())
    (output / "RELEASE_NOTES.md").write_text(notes(root, version, commit), encoding="utf-8")
    build_inputs = []
    for platform in ("windows", "linux"):
        record = json.loads((wheels / f"build-info-{platform}.json").read_text(encoding="utf-8"))
        if (record.get("schema") != 1 or record.get("platform") != platform
                or record.get("commit") != commit or record.get("version") != version):
            raise ValueError("Runtime build inputs differ from release source")
        url, sha = record["llvm"]["url"], record["llvm"]["sha256"]
        public_url(url)
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError(f"Invalid LLVM digest for {platform}")
        if set(record["constraints"]) != {"packaging/constraints/release-build.txt", "packaging/constraints/cudss-build.txt"}:
            raise ValueError("Incomplete runtime build constraints")
        for relative, checksum in record["constraints"].items():
            if relative not in {"packaging/constraints/release-build.txt", "packaging/constraints/cudss-build.txt"} or digest(root / relative) != checksum:
                raise ValueError("Runtime build constraints differ from release source")
        build_inputs.append(record)
    manifest.update({
        "tag": f"forge-v{version}", "workflow": ".github/workflows/release.yml",
        "run_id": os.environ.get("GITHUB_RUN_ID"), "build_inputs": build_inputs,
        "build_constraints": "packaging/constraints/release-build.txt",
        "files": [{"name": p.name, "size": p.stat().st_size, "sha256": digest(p)} for p in sorted(output.iterdir(), key=lambda p: p.name)],
    })
    write_json(output / "release-manifest.json", manifest)
    (output / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(output.iterdir(), key=lambda p: p.name) if p.name != "SHA256SUMS"), encoding="utf-8")
    verify_bundle(output, commit, version)


def verify_bundle(directory: Path, commit: str, version: str) -> dict:
    manifest = json.loads((directory / "release-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("commit") != commit or manifest.get("version") != version or manifest.get("tag") != f"forge-v{version}":
        raise ValueError("Release source identity differs")
    files = manifest["files"]
    names = [entry["name"] for entry in files]
    if len(set(names)) != len(names) or any(Path(name).name != name or "/" in name or "\\" in name for name in names):
        raise ValueError("Invalid release member names")
    required = {f"taichi-forge-{version}-source.tar.gz", f"taichi-forge-{version}-docs.zip", "RELEASE_NOTES.md"}
    if not required.issubset(names):
        raise ValueError("Release is missing required source, documentation or notes")
    expected = set(names) | {"release-manifest.json", "SHA256SUMS"}
    if {path.name for path in directory.iterdir()} != expected:
        raise ValueError("Release file set differs")
    for entry in files:
        path = directory / entry["name"]
        if path.is_symlink() or path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Release file differs: {path.name}")
    expected_sums = "".join(f"{digest(directory / name)}  {name}\n" for name in sorted(expected - {"SHA256SUMS"}))
    if (directory / "SHA256SUMS").read_text(encoding="utf-8") != expected_sums:
        raise ValueError("SHA256SUMS differs")
    validate_release_set(directory, Version(version))
    body = (directory / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    if "## 中文" not in body or "## English" not in body:
        raise ValueError("Release notes must include both languages")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "build", "verify"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--publishing", action="store_true")
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    version = check_version(args.root, args.ref, publishing=args.publishing)
    if args.command == "check":
        notes(args.root, version, git(args.root, "rev-parse", "HEAD").decode().strip())
        print(version)
    elif args.command == "build":
        if args.wheels is None or args.output is None:
            parser.error("build requires --wheels and --output")
        build_bundle(args.root, args.wheels, args.output, args.ref)
    else:
        if args.output is None:
            parser.error("verify requires --output")
        verify_bundle(args.output, git(args.root, "rev-parse", "HEAD").decode().strip(), version)


if __name__ == "__main__":
    main()
