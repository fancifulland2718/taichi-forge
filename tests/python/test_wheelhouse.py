"""Sealed wheelhouse integrity and relocation; no package resolution or device."""

import hashlib
import pytest

from taichi_forge.tools import _wheelhouse

tomli_w = pytest.importorskip("tomli_w")
pytest.importorskip("packaging.pylock")


def inputs(tmp_path):
    name = "fixture-1.0-py3-none-any.whl"
    directory = tmp_path / "downloads"
    directory.mkdir()
    data = b"not installed: artifact integrity fixture"
    (directory / name).write_bytes(data)
    artifact = dict(
        name=name,
        url="https://example.invalid/" + name,
        size=len(data),
        hashes=dict(sha256=hashlib.sha256(data).hexdigest()),
    )
    document = {
        "lock-version": "1.0",
        "created-by": "fixture",
        "environments": ["sys_platform == 'win32'"],
        "packages": [dict(name="fixture", version="1.0", wheels=[artifact])],
    }
    lock = tmp_path / "pylock.toml"
    lock.write_text(tomli_w.dumps(document), encoding="utf-8")
    return lock, directory, document


def test_seal_preserves_standard_metadata_and_owns_exact_bytes(tmp_path):
    import tomli

    lock, directory, document = inputs(tmp_path)
    (directory / "unused.whl").write_bytes(b"not in the selected closure")
    original = lock.read_bytes()
    output = tmp_path / "bundle"
    report = _wheelhouse.seal(lock, directory, output)
    saved = tomli.loads((output / "pylock.toml").read_text())
    assert saved["environments"] == document["environments"]
    wheel = saved["packages"][0]["wheels"][0]
    assert "url" not in wheel and wheel["path"] == "wheels/fixture-1.0-py3-none-any.whl"
    assert report["packages"] == report["wheels"] == 1
    assert report["wheel_bytes"] == wheel["size"]
    moved = tmp_path / "relocated"
    output.rename(moved)
    assert (moved / wheel["path"]).read_bytes() == (directory / wheel["name"]).read_bytes()
    assert {p.name for p in (moved / "wheels").iterdir()} == {wheel["name"]}
    assert lock.read_bytes() == original
    with pytest.raises(ValueError, match="must be new"):
        _wheelhouse.seal(lock, directory, moved)


@pytest.mark.parametrize("corrupt", ("hash", "size", "name", "version", "missing", "sdist"))
def test_seal_rejects_invalid_or_incomplete_artifacts_without_publication(tmp_path, corrupt):
    lock, directory, document = inputs(tmp_path)
    package = document["packages"][0]
    wheel = package["wheels"][0]
    if corrupt == "hash":
        (directory / wheel["name"]).write_bytes(b"changed after resolution")
    elif corrupt == "size":
        wheel["size"] += 1
    elif corrupt == "name":
        wheel["name"] = "../" + wheel["name"]
    elif corrupt == "version":
        package["version"] = "2.0"
    elif corrupt == "missing":
        (directory / wheel["name"]).unlink()
    elif corrupt == "sdist":
        package["sdist"] = dict(wheel, name="fixture-1.0.tar.gz")
    lock.write_text(tomli_w.dumps(document), encoding="utf-8")
    output = tmp_path / "bundle"
    with pytest.raises(ValueError):
        _wheelhouse.seal(lock, directory, output)
    assert not output.exists()
    assert not list(tmp_path.glob("bundle.pending-*"))
