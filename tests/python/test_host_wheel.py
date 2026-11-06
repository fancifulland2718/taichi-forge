"""Release-boundary checks; real extension installation is in test_host_build."""

import hashlib
import json
from pathlib import Path
import sys
import sysconfig
import subprocess

import pytest

from taichi_forge.tools import _host_wheel as wheel


def inputs(tmp_path):
    layout = tmp_path / "layout"
    (layout / "pkg").mkdir(parents=True)
    name = "pkg/__init__" + sysconfig.get_config_var("EXT_SUFFIX")
    data = b"artifact boundary fixture; never loaded"
    (layout / name).write_bytes(data)
    manifest = dict(
        version=1,
        kind="forge_compiled_host",
        platform=sysconfig.get_platform(),
        cache_tag=sys.implementation.cache_tag,
        modules=[
            dict(
                name="pkg",
                file=name,
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        ],
        aliases={},
    )
    (layout / "host-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "api.pyi").write_text("def run() -> None: ...\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools==82.0.1"]\nbuild-backend="setuptools.build_meta"\n'
        '[project]\nname="closed-host"\nversion="1.0"\n',
        encoding="utf-8",
    )
    spec = dict(
        version=1,
        host_layout="layout",
        project="pyproject.toml",
        files={"pkg/__init__.pyi": "api.pyi"},
    )
    path = tmp_path / "release.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec, manifest


def test_host_release_rejects_modified_artifacts_and_unowned_files(tmp_path):
    path, spec, manifest = inputs(tmp_path)
    wheel.release_inputs(path)
    layout = tmp_path / "layout"
    artifact = layout / manifest["modules"][0]["file"]
    original = artifact.read_bytes()
    artifact.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="artifact changed"):
        wheel.release_inputs(path)
    artifact.write_bytes(original)
    (layout / "accidental.py").write_text("private = True", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected host layout"):
        wheel.release_inputs(path)


@pytest.mark.parametrize(
    "target",
    [
        "pkg/code.py",
        "pkg/native.dll",
        "pkg/libruntime.so.1",
        "../api.pyi",
        "pkg/../api.pyi",
        "another/api.pyi",
    ],
)
def test_public_payload_cannot_escape_or_add_private_code(tmp_path, target):
    path, spec, _ = inputs(tmp_path)
    spec["files"] = {target: "api.pyi"}
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError):
        wheel.release_inputs(path)


def test_host_release_keeps_native_abi_and_metadata_owner_explicit(tmp_path):
    path, _, manifest = inputs(tmp_path)
    manifest["cache_tag"] = "foreign-python-abi"
    (tmp_path / "layout/host-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="matching Python ABI"):
        wheel.release_inputs(path)
    manifest["cache_tag"] = sys.implementation.cache_tag
    (tmp_path / "layout/host-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (tmp_path / "pyproject.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[tool.setuptools]\npackages=["private"]\n')
    with pytest.raises(ValueError, match="source-discovery"):
        wheel.release_inputs(path)


def test_readable_examples_are_explicit_and_preserved_in_wheel(tmp_path):
    from zipfile import ZipFile

    path, spec, _ = inputs(tmp_path)
    source = b"# Public scene example, not a numerical definition.\nprint('example')\n"
    (tmp_path / "scene.py").write_bytes(source)
    spec["examples"] = {"pkg/sdk/examples/scene.py": "scene.py"}
    (tmp_path / "guide.md").write_text("[Scene](../examples/scene.py)\n", encoding="utf-8")
    spec["files"]["pkg/sdk/docs/guide.md"] = "guide.md"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = wheel.build(path, tmp_path / "wheels")
    with ZipFile(result["wheel"]) as archive:
        assert archive.read("pkg/sdk/examples/scene.py") == source
        assert archive.read("pkg/sdk/docs/guide.md") == (tmp_path / "guide.md").read_bytes()
        metadata = next(name for name in archive.namelist() if name.endswith("/forge-host.json"))
        assert json.loads(archive.read(metadata))["public_examples"] == {
            "pkg/sdk/examples/scene.py": hashlib.sha256(source).hexdigest()
        }
    # The same source still cannot enter through the ordinary resource channel.
    spec["files"]["pkg/private.py"] = "scene.py"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot introduce source"):
        wheel.release_inputs(path)


@pytest.mark.parametrize(
    "target",
    [
        "pkg/core.py",
        "other/examples/run.py",
        "pkg/examples/core.pyc",
        "pkg/examples/__init__.py",
        "pkg/examples/../core.py",
    ],
)
def test_public_examples_cannot_replace_host_sources(tmp_path, target):
    path, spec, _ = inputs(tmp_path)
    spec["examples"] = {target: "api.pyi"}
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError):
        wheel.release_inputs(path)


def test_public_example_cannot_shadow_compiled_alias(tmp_path):
    path, spec, manifest = inputs(tmp_path)
    manifest["aliases"] = {"pkg.examples.scene": {"target": "pkg", "registered_by": "pkg"}}
    (tmp_path / "layout/host-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    spec["examples"] = {"pkg/examples/scene.py": "api.pyi"}
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with a compiled host module"):
        wheel.release_inputs(path)


def python_inputs(tmp_path):
    path, _, _ = inputs(tmp_path)
    source = tmp_path / "source" / "source_host"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text(
        "import sys\nfrom . import model\nsys.modules['source_host_alias'] = model\n",
        encoding="utf-8",
    )
    (source / "model.py").write_bytes(
        b"# coding: latin-1\nLABEL = 'caf\xe9'\ndef prepare(n):\n    return list(range(n))\n"
    )
    host = dict(
        version=1,
        source_root="source",
        modules={
            "source_host": "source_host/__init__.py",
            "source_host.model": "source_host/model.py",
        },
        aliases={
            "source_host_alias": {
                "target": "source_host.model",
                "registered_by": "source_host",
            }
        },
        toolchain={"not_a_compiler": "source mode must not query this"},
    )
    (tmp_path / "host.json").write_text(json.dumps(host), encoding="utf-8")
    (tmp_path / "compute.tcm").write_bytes(b"opaque target artifact; the wheel builder does not execute it")
    release = dict(
        version=2,
        host_sources="host.json",
        platform=sysconfig.get_platform(),
        project="pyproject.toml",
        files={"source_host/aot/compute.tcm": "compute.tcm"},
    )
    path.write_text(json.dumps(release), encoding="utf-8")
    return path, release, host


def test_python_host_wheel_preserves_sources_aliases_and_platform_without_extensions(
    tmp_path,
):
    from zipfile import ZipFile

    path, _, host = python_inputs(tmp_path)
    result = wheel.build(path, tmp_path / "python-wheels")
    assert result["modules"] == 2
    assert "-py3-none-" in Path(result["wheel"]).name
    with ZipFile(result["wheel"]) as archive:
        assert not any(name.endswith((".pyd", ".so", ".pyc")) for name in archive.namelist())
        for relative in host["modules"].values():
            assert archive.read(relative) == (tmp_path / "source" / relative).read_bytes()
        assert archive.read("source_host/aot/compute.tcm") == (tmp_path / "compute.tcm").read_bytes()
        manifest = json.loads(archive.read(next(n for n in archive.namelist() if n.endswith("/forge-host.json"))))
        assert manifest["kind"] == "forge_python_host" and manifest["aliases"] == host["aliases"]
        assert "cache_tag" not in manifest and "toolchain" not in manifest
    # Import only the packaged Host in an isolated consumer, without the source
    # checkout, Forge frontend, Cython, or a native host extension.
    program = (
        "import sys; sys.path.insert(0, sys.argv[1]); import source_host, source_host_alias; "
        "assert source_host.model is source_host_alias; "
        "assert source_host_alias.LABEL == 'caf\\u00e9'; "
        "assert source_host_alias.prepare(3) == [0, 1, 2]; "
        "assert 'taichi_forge' not in sys.modules; assert 'Cython' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-B", "-c", program, result["wheel"]], check=True)


@pytest.mark.parametrize(
    "declaration",
    [
        "import taichi_forge as ti\n@ti.kernel\ndef compute():\n    pass\n",
        "from taichi import func as numerical\n@numerical\ndef helper():\n    return 1\n",
    ],
)
def test_python_host_rejects_numerical_definitions(tmp_path, declaration):
    path, _, _ = python_inputs(tmp_path)
    (tmp_path / "source/source_host/model.py").write_text(declaration, encoding="utf-8")
    with pytest.raises(ValueError, match="numerical DSL templates"):
        wheel.release_inputs(path)


def test_python_host_keeps_explicit_closure_and_external_runtime_boundary(tmp_path):
    path, release, host = python_inputs(tmp_path)
    del host["modules"]["source_host"]
    (tmp_path / "host.json").write_text(json.dumps(host), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing package initializer"):
        wheel.release_inputs(path)
    host["modules"]["source_host"] = "source_host/__init__.py"
    (tmp_path / "host.json").write_text(json.dumps(host), encoding="utf-8")
    release["files"]["source_host/runtime.dll"] = "compute.tcm"
    path.write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot introduce source or unverified extensions"):
        wheel.release_inputs(path)
    del release["files"]["source_host/runtime.dll"]
    release["platform"] = "foreign-platform"
    path.write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(ValueError, match="matching target platform"):
        wheel.release_inputs(path)
