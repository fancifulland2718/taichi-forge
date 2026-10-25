"""Release gates run independently of the native Forge runtime."""
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts import release_artifacts as artifacts
from scripts import release_github as github
from scripts import check_pypi_version_available as index
from scripts import record_release_build


ROOT = Path(__file__).resolve().parents[2]


def run(root, *args):
    return artifacts.git(root, *args).decode().strip()


def initialize(root):
    root.mkdir(parents=True)
    run(root, "init", "-q")
    run(root, "config", "user.email", "release-test@example.invalid")
    run(root, "config", "user.name", "Release fixture")


def commit(root):
    run(root, "add", ".")
    run(root, "commit", "-qm", "Fixture")
    return run(root, "rev-parse", "HEAD")


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    initialize(root)
    for relative in ("version.txt", "pyproject.toml", "packaging/runtime/pyproject.toml", "taichi/common/version.h", "packaging/constraints/release-build.txt", "packaging/constraints/cudss-build.txt"):
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / relative).read_bytes())
    for relative, content in {
        "README.md": "Public build instructions", "LICENSE": "Apache-2.0",
        "docs/guide.md": "Build guide", "internal/secret.md": "excluded",
        "opt_doc/plan.md": "excluded", ".gitignore": "local.bin\n",
        "docs/forge/release_notes.zh.md": "## 0.7.0（待发布）\n\n公开更新 [文档](guide.md)。\n<a id=\"063\"></a>\n## 0.6.3\n旧版本\n",
        "docs/forge/release_notes.en.md": "## 0.7.0 (unreleased)\n\nPublic change [guide](guide.md).\n## 0.6.3\nOld version\n",
    }.items():
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    child = root / "external" / "library"
    initialize(child)
    (child / "LICENSE").write_text("Vendor license")
    (child / "source.cpp").write_text("int public_source = 1;")
    child_commit = commit(child)
    (root / ".gitmodules").write_text('[submodule "library"]\npath = external/library\nurl = https://github.com/example/library.git\n')
    run(root, "add", ".")
    run(root, "update-index", "--add", "--cacheinfo", f"160000,{child_commit},external/library")
    run(root, "commit", "-qm", "Fixture")
    (root / "local.bin").write_bytes(b"not source")
    (root / "untracked.txt").write_text("not source")
    return root


def test_version_tag_and_both_metadata_are_required(source):
    assert artifacts.check_version(source, "refs/heads/master") == "0.7.0"
    assert artifacts.check_version(source, "refs/tags/forge-v0.7.0", publishing=True) == "0.7.0"
    for ref in ("refs/heads/master", "refs/tags/forge-v0.7.1"):
        with pytest.raises(ValueError):
            artifacts.check_version(source, ref, publishing=True)
    path = source / "packaging/runtime/pyproject.toml"
    path.write_text(path.read_text().replace('TI_VERSION_MINOR = "7"', 'TI_VERSION_MINOR = "6"'))
    with pytest.raises(ValueError, match="CMake version"):
        artifacts.check_version(source, "refs/heads/master")


def test_archive_uses_commits_includes_submodule_and_excludes_private_files(source, tmp_path):
    revision = run(source, "rev-parse", "HEAD")
    path = tmp_path / "source.tar.gz"
    manifest = artifacts.source_archive(source, revision, path, "0.7.0")
    assert manifest["submodules"][0]["path"] == "external/library"
    with tarfile.open(path) as archive:
        names = archive.getnames()
        assert "taichi-forge-0.7.0/external/library/source.cpp" in names
        assert "taichi-forge-0.7.0/external/library/LICENSE" in names
        assert archive.extractfile("taichi-forge-0.7.0/SOURCE_COMMIT").read().decode().strip() == revision
        assert not any("/internal/" in n or "/opt_doc/" in n or n.endswith(("local.bin", "untracked.txt")) for n in names)
    # Working-tree and ignored changes must not alter the archive's committed bytes.
    (source / "external/library/source.cpp").write_text("uncommitted")
    other = tmp_path / "again.tar.gz"
    artifacts.source_archive(source, revision, other, "0.7.0")
    assert path.read_bytes() == other.read_bytes()


def test_source_missing_submodule_fails_before_publishing(source, tmp_path):
    (source / "external/library/.git").rename(source / "external/library/git-hidden")
    with pytest.raises(ValueError, match="Initialize the pinned submodule"):
        artifacts.source_archive(source, run(source, "rev-parse", "HEAD"), tmp_path / "bad.tar.gz", "0.7.0")


def test_release_notes_have_both_languages_and_pinned_links(source):
    body = artifacts.notes(source, "0.7.0", "a" * 40)
    assert body.index("## 中文") < body.index("## English")
    assert "公开更新" in body and "Public change" in body
    assert "Old version" not in body and "旧版本" not in body and 'id="063"' not in body
    assert f"/blob/{'a' * 40}/docs/forge/guide.md" in body


@pytest.fixture
def bundle(source, tmp_path, monkeypatch):
    # Content/matrix validation has its own real wheel-fixture suite; this tests
    # the orchestration's provenance, archives and tamper detection independently.
    monkeypatch.setattr(artifacts, "validate_release_set", lambda *args: None)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for platform in ("windows", "linux"):
        artifacts.write_json(wheels / f"build-info-{platform}.json", {
            "schema": 1, "platform": platform, "commit": run(source, "rev-parse", "HEAD"), "version": "0.7.0",
            "llvm": {"url": f"https://example.org/llvm-{platform}.zip", "sha256": "b" * 64},
            "constraints": {name: artifacts.digest(source / name) for name in ("packaging/constraints/release-build.txt", "packaging/constraints/cudss-build.txt")},
        })
    (wheels / "fixture.whl").write_bytes(b"original bytes")
    output = tmp_path / "release"
    artifacts.build_bundle(source, wheels, output, "refs/heads/master")
    return output, run(source, "rev-parse", "HEAD")


def test_bundle_is_complete_and_rejects_mutation(bundle):
    output, revision = bundle
    artifacts.verify_bundle(output, revision, "0.7.0")
    (output / "fixture.whl").write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="differs"):
        artifacts.verify_bundle(output, revision, "0.7.0")


def test_bundle_rejects_cross_commit_reuse_and_unlisted_assets(bundle):
    output, revision = bundle
    with pytest.raises(ValueError, match="identity differs"):
        artifacts.verify_bundle(output, "a" * 40, "0.7.0")
    (output / "unexpected.whl").write_bytes(b"extra")
    with pytest.raises(ValueError, match="file set"):
        artifacts.verify_bundle(output, revision, "0.7.0")


def test_candidate_provenance_must_be_successful_same_source(monkeypatch):
    record = {"head_sha": "a" * 40, "path": ".github/workflows/release.yml", "head_repository": {"full_name": "owner/repo"}, "status": "completed", "conclusion": "success"}
    monkeypatch.setattr(github, "gh", lambda *args: json.dumps(record))
    github.check_run("owner/repo", "123", "a" * 40)
    for key, value in (("head_sha", "b" * 40), ("conclusion", "failure"), ("path", ".github/workflows/other.yml")):
        original = record[key]
        record[key] = value
        with pytest.raises(ValueError, match="successful release.yml"):
            github.check_run("owner/repo", "123", "a" * 40)
        record[key] = original


def test_github_draft_resume_and_finalization_never_replace_assets(bundle, monkeypatch):
    output, revision = bundle
    body = (output / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    release = {"id": 12, "body": body, "name": "Taichi Forge 0.7.0 / 发布说明", "prerelease": False, "draft": True}
    stored = {}
    calls = []
    def fake_gh(*args):
        calls.append(args)
        if args[0] == "api" and "/commits/" in args[1]:
            return revision
        if args[0] == "api" and "/assets?" in args[1]:
            return json.dumps([list(stored.values())])
        if args[:2] == ("release", "upload"):
            path = Path(args[3])
            stored[path.name] = {"name": path.name, "size": path.stat().st_size, "digest": "sha256:" + artifacts.digest(path)}
        if args[:2] == ("release", "edit"):
            release["draft"] = False
        return ""
    monkeypatch.setattr(github, "gh", fake_gh)
    monkeypatch.setattr(github.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, json.dumps(release), ""))
    github.synchronize(output, "owner/repo", revision, "0.7.0", publish=False)
    first_uploads = len([c for c in calls if c[:2] == ("release", "upload")])
    assert first_uploads == len(list(output.iterdir()))
    github.synchronize(output, "owner/repo", revision, "0.7.0", publish=False)
    assert len([c for c in calls if c[:2] == ("release", "upload")]) == first_uploads
    saved = stored.pop("fixture.whl")
    with pytest.raises(ValueError, match="missing asset"):
        github.synchronize(output, "owner/repo", revision, "0.7.0", publish=True)
    assert release["draft"]
    stored["fixture.whl"] = saved
    github.synchronize(output, "owner/repo", revision, "0.7.0", publish=True)
    assert not release["draft"]
    stored["fixture.whl"]["digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="content differs"):
        github.synchronize(output, "owner/repo", revision, "0.7.0", publish=True)
    assert all("--clobber" not in c for c in calls)


def test_complete_index_gate_rejects_missing_files(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["check", "--repository", "pypi", "--wheel-dir", str(tmp_path), "--upload-dir", str(tmp_path / "upload"), "--require-published"])
    monkeypatch.setattr(index, "prepare_upload", lambda *args: 1)
    with pytest.raises(SystemExit, match="not yet published"):
        index.main()


def test_actual_build_record_captures_compiler_and_constraints(source, tmp_path, monkeypatch):
    cache = source / "_skbuild/runtime/CMakeCache.txt"
    cache.parent.mkdir(parents=True)
    cache.write_text("TI_WITH_CUDA:BOOL=ON\nTI_WITH_CUDA_TOOLKIT:BOOL=OFF\n")
    compiler = cache.parent / "CMakeFiles/3.31/CMakeCXXCompiler.cmake"
    compiler.parent.mkdir(parents=True)
    compiler.write_text('set(CMAKE_CXX_COMPILER_ID "MSVC")\nset(CMAKE_CXX_COMPILER_VERSION "19.44")\n')
    monkeypatch.setenv("LLVM20_WIN_URL", "https://example.org/llvm.zip")
    monkeypatch.setenv("LLVM20_WIN_SHA256", "b" * 64)
    monkeypatch.setenv("CUDA_TOOLKIT_VERSION", "12.5.1")
    monkeypatch.setattr(record_release_build.importlib.metadata, "version", lambda name: "1.0")
    output = tmp_path / "build-info-windows.json"
    record_release_build.record(source, "windows", output)
    data = json.loads(output.read_text())
    assert data["compiler"] == {"id": "MSVC", "version": "19.44"}
    assert data["cmake_options"]["TI_WITH_CUDA_TOOLKIT"] == "OFF"
    assert data["constraints"]["packaging/constraints/release-build.txt"] == artifacts.digest(source / "packaging/constraints/release-build.txt")


def test_archive_commit_works_without_git_and_rejects_bad_identity(tmp_path):
    import shutil
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the source archive provenance check")
    script = tmp_path / "check.cmake"
    script.write_text(f'set(CMAKE_SOURCE_DIR "{tmp_path.as_posix()}")\ninclude("{ROOT.as_posix()}/cmake/SourceVersion.cmake")\nmessage(STATUS "identity=${{TI_COMMIT_HASH}}:${{TI_COMMIT_SHORT_HASH}}")\n')
    (tmp_path / "SOURCE_COMMIT").write_text("a" * 40)
    result = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "identity=" + "a" * 40 + ":" + "a" * 8 in result.stdout
    (tmp_path / "SOURCE_COMMIT").write_text("invalid")
    result = subprocess.run([cmake, "-P", str(script)], capture_output=True, text=True)
    assert result.returncode != 0 and "Invalid source archive" in result.stderr


def test_workflow_has_one_publisher_and_orders_complete_release():
    import yaml
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    assert "runtime" in jobs["shim"]["needs"]
    assert {"runtime", "shim"}.issubset(jobs["bundle"]["needs"])
    assert "github_draft" in jobs["publish_runtime"]["needs"]
    assert "publish_runtime" in jobs["publish_shim"]["needs"]
    assert "publish_shim" in jobs["finalize"]["needs"]
    for builder in ("publish_pypi.yml", "publish_runtime_pypi.yml"):
        text = (ROOT / ".github/workflows" / builder).read_text(encoding="utf-8")
        assert "gh-action-pypi-publish" not in text and "contents: write" not in text and "id-token: write" not in text
    assert jobs["preflight"]["steps"][-1]["env"]["PUBLISH"] == "${{ github.event_name == 'push' || inputs.publish }}"
