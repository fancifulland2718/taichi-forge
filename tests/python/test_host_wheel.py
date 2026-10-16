"""Release-boundary checks; real extension installation is in test_host_build."""

import hashlib
import json
from pathlib import Path
import sys
import sysconfig

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
        modules=[dict(name="pkg", file=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())],
        aliases={},
    )
    (layout / "host-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "api.pyi").write_text("def run() -> None: ...\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires=["setuptools==82.0.1"]\nbuild-backend="setuptools.build_meta"\n'
        '[project]\nname="closed-host"\nversion="1.0"\n',
        encoding="utf-8",
    )
    spec = dict(version=1, host_layout="layout", project="pyproject.toml", files={"pkg/__init__.pyi": "api.pyi"})
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
    ["pkg/code.py", "pkg/native.dll", "pkg/libruntime.so.1", "../api.pyi", "pkg/../api.pyi", "another/api.pyi"],
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
