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
    monkeypatch.setattr(artifacts, "validate_release_set", lambda *args, **kwargs: None)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for platform in ("windows", "linux"):
        artifacts.write_json(wheels / f"build-info-{platform}.json", {
            "schema": 1, "platform": platform, "commit": run(source, "rev-parse", "HEAD"), "version": "0.7.0",
            "llvm": {"url": f"https://example.org/llvm-{platform}.zip", "sha256": "b" * 64},
            "constraints": {name: artifacts.digest(source / name) for name in ("packaging/constraints/release-build.txt", "packaging/constraints/cudss-build.txt")},
        })
    (wheels / "fixture.whl").write_bytes(b"original bytes")
    records = [json.loads((wheels / f"build-info-{platform}.json").read_text()) for platform in ("windows", "linux")]
    artifacts.write_json(wheels / "publication-sources.json", {
        "shim_run_id": "123",
        "shim": {"commit": run(source, "rev-parse", "HEAD"), "version": "0.7.0", "runtime_version": "0.7.0", "wheels": []},
        "runtime": {"index": "pypi", "version": "0.7.0", "commit": records[0]["commit"], "build_inputs": records, "wheels": []},
    })
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


def test_publisher_provenance_requires_original_workflow_and_success(monkeypatch):
    record = {"head_sha": "a" * 40, "event": "workflow_dispatch", "path": ".github/workflows/publish_runtime_pypi.yml", "head_repository": {"full_name": "owner/repo"}, "status": "completed", "conclusion": "success"}
    monkeypatch.setattr(github, "gh", lambda *args: json.dumps(record))
    assert github.checked_run("owner/repo", "123", "publish_runtime_pypi.yml") == record
    for key, value in (("conclusion", "failure"), ("path", ".github/workflows/release.yml"), ("event", "pull_request"), ("head_repository", {"full_name": "fork/repo"})):
        original = record[key]
        record[key] = value
        with pytest.raises(ValueError, match="existing publisher"):
            github.checked_run("owner/repo", "123", "publish_runtime_pypi.yml")
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


def test_existing_publishers_keep_oidc_and_new_workflow_only_publishes_github():
    import yaml
    github_text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(github_text)
    assert set(workflow["jobs"]) == {"bundle", "github_release"}
    assert "id-token" not in github_text and "gh-action-pypi-publish" not in github_text
    assert "environment:" not in github_text
    assert "--require-published" in github_text
    for name, job, project in (("publish_pypi.yml", "publish_shim", "shim"), ("publish_runtime_pypi.yml", "publish_runtime", "runtime")):
        workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))
        publisher = workflow["jobs"][job]
        assert publisher["permissions"]["id-token"] == "write"
        uploads = [step for step in publisher["steps"] if "gh-action-pypi-publish" in step.get("uses", "")]
        assert len(uploads) == 2
        assert all(step["with"]["packages-dir"] == "upload" for step in uploads)
        if project == "shim":
            following = workflow["jobs"]["github_release"]
            assert "publish_shim" in following["needs"]
            assert "publish_shim.result == 'success'" in following["if"]
            assert "target == 'pypi'" in following["if"]
            assert following["uses"] == "./.github/workflows/release.yml"
            assert "id-token" not in following["permissions"]


@pytest.fixture
def published_sources(tmp_path, monkeypatch):
    directory = tmp_path / "wheels"
    directory.mkdir()
    for name in ("taichi_forge-0.7.0-cp313-cp313-win_amd64.whl", "taichi_forge_runtime-0.6.3-py3-none-win_amd64.whl"):
        (directory / name).write_bytes(name.encode())
    facts = github.wheel_facts(directory)
    shim = {"schema": 1, "commit": "a" * 40, "source_run_id": "10", "version": "0.7.0", "runtime_version": "0.6.3", "wheels": facts[:1]}
    artifacts.write_json(directory / "shim-build-info.json", shim)
    runtime = {"schema": 1, "index": "pypi", "version": "0.6.3", "commit": "b" * 40, "run_id": "20", "source_run_id": "20", "wheels": facts[1:],
               "build_inputs": [{"platform": platform, "commit": "b" * 40, "version": "0.6.3"} for platform in ("windows", "linux")]}
    state = {"tag": "a" * 40, "job_conclusion": "success", "artifacts": [{"id": 88, "name": "runtime-publication-0.6.3-pypi", "expired": False, "workflow_run": {"id": 20}}]}
    def fake_gh(*args):
        path = args[1]
        if "/jobs?" in path:
            return json.dumps([{"jobs": [{"name": name, "conclusion": state["job_conclusion"]} for name in ("Publish validated shim wheels", "Validate shim wheel set", "Validate runtime wheel set")]}])
        if "/commits/" in path:
            return state["tag"]
        if "/artifacts?" in path:
            return json.dumps([{"artifacts": state["artifacts"]}])
        run_id = path.rsplit("/", 1)[-1]
        return json.dumps({"path": ".github/workflows/publish_runtime_pypi.yml" if run_id == "20" else ".github/workflows/publish_pypi.yml",
                           "head_repository": {"full_name": "owner/repo"}, "event": "workflow_dispatch",
                           "status": "completed" if run_id == "20" else "in_progress", "conclusion": "success" if run_id == "20" else None,
                           "head_sha": "b" * 40 if run_id == "20" else "a" * 40})
    monkeypatch.setattr(github, "gh", fake_gh)
    monkeypatch.setattr(github, "artifact_json", lambda *args: runtime)
    return directory, shim, runtime, state


def test_github_resolves_independent_runtime_and_active_parent_publication(published_sources):
    directory, shim, runtime, state = published_sources
    assert github.resolve_sources(directory, "owner/repo", "11", "0.7.0", "0.6.3") == shim["commit"]
    sources = json.loads((directory / "publication-sources.json").read_text())
    assert sources["shim"] == shim and sources["runtime"] == runtime
    assert sources["shim_run_id"] == "11"
    assert json.loads((directory / "build-info-windows.json").read_text())["commit"] == "b" * 40


@pytest.mark.parametrize("mutation,reason", [("tag", "actual shim source"), ("job", "completed successfully"), ("wheel", "validated source"), ("runtime", "No retained"), ("expired", "No retained"), ("subset", "validated source")])
def test_github_refuses_incomplete_or_mismatched_publication(published_sources, mutation, reason):
    directory, shim, runtime, state = published_sources
    if mutation == "tag":
        state["tag"] = "c" * 40
    elif mutation == "job":
        state["job_conclusion"] = "skipped"
    elif mutation == "wheel":
        (directory / shim["wheels"][0]["name"]).write_bytes(b"different")
    elif mutation == "runtime":
        runtime["wheels"][0]["sha256"] = "0" * 64
    elif mutation == "expired":
        state["artifacts"][0]["expired"] = True
    else:
        shim["wheels"] = []
        artifacts.write_json(directory / "shim-build-info.json", shim)
    with pytest.raises(ValueError, match=reason):
        github.resolve_sources(directory, "owner/repo", "11", "0.7.0", "0.6.3")


def test_reused_shim_keeps_compile_identity_across_republication(published_sources, monkeypatch, tmp_path):
    directory, shim, runtime, state = published_sources
    for path in directory.glob("taichi_forge_runtime-*.whl"):
        path.unlink()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts").mkdir()
    artifacts.write_json(tmp_path / "artifacts/shim-build-info.json", shim)
    monkeypatch.setenv("GITHUB_RUN_ID", "12")
    monkeypatch.setattr(github, "validate_release_set", lambda *a, **kw: None)
    github.record_shim(directory, "owner/repo", "11", "0.7.0", "0.6.3")
    assert json.loads((directory / "shim-build-info.json").read_text()) == shim
    (tmp_path / "artifacts/shim-build-info.json").unlink()
    with pytest.raises(ValueError, match="original shim-build-info"):
        github.record_shim(directory, "owner/repo", "11", "0.7.0", "0.6.3")


def test_pypi_staging_excludes_all_github_sidecars(tmp_path, monkeypatch):
    import io
    directory = tmp_path / "assets"
    directory.mkdir()
    wheel = directory / "taichi_forge-0.7.0-cp313-cp313-win_amd64.whl"
    wheel.write_bytes(b"wheel")
    for name in ("source.tar.gz", "docs.zip", "release-manifest.json", "shim-build-info.json", "build-info-linux.json", "SHA256SUMS", "RELEASE_NOTES.md"):
        (directory / name).write_bytes(b"github only")
    monkeypatch.setattr(index, "urlopen", lambda *a, **kw: io.BytesIO(b'{"urls": []}'))
    output = tmp_path / "upload"
    assert index.prepare_upload(directory, output, "https://pypi.org/pypi") == 1
    assert [p.name for p in output.iterdir()] == [wheel.name]


def test_bundle_retains_runtime_producer_separately(source, bundle, tmp_path):
    wheels = tmp_path / "wheels"
    records = []
    for platform in ("windows", "linux"):
        path = wheels / f"build-info-{platform}.json"
        record = json.loads(path.read_text())
        record["commit"], record["version"] = "b" * 40, "0.6.3"
        record["constraints"] = {name: "c" * 64 for name in record["constraints"]}
        artifacts.write_json(path, record)
        records.append(record)
    path = wheels / "publication-sources.json"
    sources = json.loads(path.read_text())
    sources["shim"]["runtime_version"] = "0.6.3"
    sources["runtime"].update({"commit": "b" * 40, "version": "0.6.3", "build_inputs": records})
    artifacts.write_json(path, sources)
    output = tmp_path / "independent"
    artifacts.build_bundle(source, wheels, output, "refs/heads/master", runtime_version="0.6.3")
    manifest = artifacts.verify_bundle(output, run(source, "rev-parse", "HEAD"), "0.7.0")
    assert manifest["runtime_version"] == "0.6.3"
    assert manifest["runtime_source"].endswith("/" + "b" * 40)
    assert manifest["commit"] != "b" * 40


def test_runtime_record_requires_complete_index_and_correct_build(published_sources, monkeypatch, tmp_path):
    directory, shim, runtime, state = published_sources
    for path in directory.glob("taichi_forge-*.whl"):
        path.unlink()
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "20")
    monkeypatch.setattr(github, "validate_release_set", lambda *a, **kw: None)
    monkeypatch.setattr(github, "prepare_upload", lambda *a: 0)
    for record in runtime["build_inputs"]:
        artifacts.write_json(directory / f"build-info-{record['platform']}.json", record)
    artifacts.write_json(directory / "runtime-build-info.json", {
        "schema": 1, "commit": "b" * 40, "source_run_id": "20", "version": "0.6.3",
        "wheels": github.wheel_facts(directory), "build_inputs": runtime["build_inputs"],
    })
    output = tmp_path / "runtime-publication.json"
    github.record_runtime(directory, "0.6.3", "pypi", output)
    assert json.loads(output.read_text()) == runtime
    monkeypatch.setattr(github, "prepare_upload", lambda *a: 1)
    with pytest.raises(ValueError, match="not complete"):
        github.record_runtime(directory, "0.6.3", "pypi", output)


def test_candidate_discovery_reuses_only_matching_live_validated_sets(monkeypatch):
    from scripts import release_candidates as candidates
    source = {"schema": 1, "commit": "a" * 40, "version": "0.7.0", "runtime_version": "0.7.0"}
    candidate = {"id": 42, "name": "validated-shim-source", "expired": False,
                 "workflow_run": {"id": 10, "head_sha": "a" * 40}}
    wheels = {"name": "validated-shim-wheel-set", "expired": False}
    reads = []
    def api(*args):
        reads.append(args)
        return json.dumps([{"artifacts": [wheels] if "/runs/" in args[1] else [candidate]}])
    monkeypatch.setattr(candidates, "gh", api)
    monkeypatch.setattr(candidates, "checked_run", lambda *a, **kw: {})
    monkeypatch.setattr(candidates, "require_job", lambda *a: None)
    monkeypatch.setattr(candidates, "artifact_json", lambda *a: source)
    assert candidates.select("shim", "owner/repo", "a" * 40, "0.7.0", "0.7.0") == "10"
    assert candidates.select("shim", "owner/repo", "b" * 40, "0.7.0", "0.7.0") == ""
    assert candidates.select("shim", "owner/repo", "a" * 40, "0.7.1", "0.7.0") == ""
    assert candidates.select("shim", "owner/repo", "a" * 40, "0.7.0", "0.6.3") == ""
    wheels["expired"] = True
    assert candidates.select("shim", "owner/repo", "a" * 40, "0.7.0", "0.7.0") == ""
    reads.clear()
    assert candidates.select("shim", "owner/repo", "a" * 40, "0.7.0", "0.7.0", reuse=False) == ""
    assert reads == []
    # Explicit selection is checked even when automatic reuse is disabled.
    assert candidates.select("shim", "owner/repo", "a" * 40, "0.7.0", "0.7.0", run_id="12", reuse=False) == "12"


def test_runtime_candidate_keeps_original_build_identity(published_sources, monkeypatch):
    from scripts import release_candidates as candidates
    directory, shim, runtime, state = published_sources
    for path in directory.glob("taichi_forge-*.whl"):
        path.unlink()
    for record in runtime["build_inputs"]:
        record["schema"] = 1
        artifacts.write_json(directory / f"build-info-{record['platform']}.json", record)
    monkeypatch.setattr(candidates, "validate_release_set", lambda *a, **kw: None)
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "20")
    candidates.record_runtime(directory, "0.6.3", reused=False)
    before = (directory / "runtime-build-info.json").read_bytes()
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "21")
    candidates.record_runtime(directory, "0.6.3", reused=True)
    assert (directory / "runtime-build-info.json").read_bytes() == before
    path = next(directory.glob("*.whl"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="validated candidate"):
        candidates.record_runtime(directory, "0.6.3", reused=True)


def test_runtime_reuse_skips_builders_but_not_validation_or_bound_publisher():
    import yaml
    workflow = yaml.safe_load((ROOT / ".github/workflows/publish_runtime_pypi.yml").read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    for name in ("build_linux_runtime", "build_windows_runtime"):
        assert "!needs.resolve_version.outputs.wheel_run_id" in jobs[name]["if"]
    assert "needs.resolve_version.outputs.wheel_run_id ||" in jobs["validate_wheel_set"]["if"]
    assert jobs["publish_runtime"]["needs"] == ["resolve_version", "validate_wheel_set"]
    assert "needs.validate_wheel_set.result == 'success'" in jobs["publish_runtime"]["if"]
    assert jobs["publish_runtime"]["permissions"]["id-token"] == "write"
