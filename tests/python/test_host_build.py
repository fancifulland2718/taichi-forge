"""Producer contracts plus an opt-in real source-hidden host build."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from zipfile import ZipFile

import pytest

from taichi_forge.tools import _host_build as host
from taichi_forge.tools import _host_wheel as host_wheel


def test_host_manifest_keeps_package_initializers_and_alias_identity(tmp_path):
    source = tmp_path / "private"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg/__init__.py").write_text("", encoding="utf-8")
    (source / "pkg/part.py").write_text("value = 3", encoding="utf-8")
    spec = {"version": 1, "source_root": "private", "modules": {"pkg.part": "pkg/part.py"}, "toolchain": {}}
    filename = tmp_path / "build.json"

    def read():
        filename.write_text(json.dumps(spec), encoding="utf-8")
        return host.read_spec(filename)

    with pytest.raises(ValueError, match="Missing package initializer"):
        read()
    spec["modules"]["pkg"] = "pkg/__init__.py"
    spec["modules"]["public_alias"] = "pkg/part.py"
    with pytest.raises(ValueError, match="Duplicate source"):
        read()
    del spec["modules"]["public_alias"]
    spec["aliases"] = {"public_alias": {"target": "pkg.part", "registered_by": "pkg"}}
    assert read()[2] == spec["aliases"]
    spec["modules"]["escape"] = "../outside.py"
    with pytest.raises(ValueError, match="relative POSIX"):
        read()


@pytest.mark.skipif(os.environ.get("TI_TEST_HOST_BUILD") != "1", reason="requires the locked host producer toolchain")
def test_real_host_build_semantics_cache_and_source_boundary(tmp_path):
    source = tmp_path / "private"
    package = source / "closed_host"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(
        (
            "# coding: latin-1\nimport sys\nfrom . import ordinary\nsys.modules['closed_host_alias'] = ordinary\n"
            'LABEL = "caf\u00e9"\nCONTINUED = 1 + \\\n    2\n'
        )
        .replace("\n", "\r\n")
        .encode("latin-1")
    )
    (package / "ordinary.py").write_text(
        textwrap.dedent("""
        from __future__ import annotations
        def choose(flag, visit):
            return (visit(1), visit(2)) if flag else (visit(3),)
        def power(value):
            result = max(float(value), 0.0)
            return max(result ** 0.5, 1e-8)
        def ordered(value):
            return tuple(sorted(getattr(value, "items", ()) or ()))
        def missing():
            return unresolved_host_name()
        def python_names(include, *, cdef):
            DEF = include + cdef
            cpdef = {"include": include, "cdef": cdef}
            return DEF, cpdef, locals()["include"]
    """),
        encoding="utf-8",
    )
    ordinary_source = package / "ordinary.py"
    ordinary_source.write_bytes(
        (
            "# coding: latin-1\n"
            + ordinary_source.read_text(encoding="utf-8")
            + '\ndef label():\n    return "caf\u00e9"\n'
        ).encode("latin-1")
    )
    (package / "kernels.py").write_text(
        textwrap.dedent("""
        import taichi_forge as ti
        @ti.data_oriented
        class Model:
            def __init__(self, width):
                self.width = width
                self.x = ti.field(ti.f32, shape=8, needs_grad=True)
                self.loss = ti.field(ti.f32, shape=(), needs_grad=True)
                self.x.fill(0.5)
            @ti.kernel
            def run(self):
                for i in self.x:
                    for j in ti.static(range(self.width)):
                        self.loss[None] += self.x[i] * (j + 1)
    """),
        encoding="utf-8",
    )
    spec = {
        "version": 1,
        "source_root": "private",
        "modules": {
            "closed_host": "closed_host/__init__.py",
            "closed_host.ordinary": "closed_host/ordinary.py",
            "closed_host.kernels": "closed_host/kernels.py",
        },
        "aliases": {"closed_host_alias": {"target": "closed_host.ordinary", "registered_by": "closed_host"}},
        "toolchain": {"python": ".".join(map(str, sys.version_info[:3])), "cython": "3.2.9", "setuptools": "82.0.1"},
    }
    filename = tmp_path / "build.json"
    filename.write_text(json.dumps(spec), encoding="utf-8")
    work, output = tmp_path / "cache", tmp_path / "layout"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (os.environ.get("FORGE_HOST_BUILD_TOOLS"), *sys.path)))

    def build(destination=output, error=None):
        result = subprocess.run(
            [
                sys.executable,
                host.__file__,
                "--spec",
                str(filename),
                "--work",
                str(work),
                "--output",
                str(destination),
                "--jobs",
                "2",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        if error:
            assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
            return None
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout.splitlines()[-1])

    assert build()["cached"] == 0
    assert build()["cached"] == 3
    leak = output / "accidental_source.py"
    leak.write_text("private = True", encoding="utf-8")
    build(error="Unexpected output contents")
    leak.unlink()
    manifest = json.loads((output / "host-manifest.json").read_text(encoding="utf-8"))
    assert str(source) not in json.dumps(manifest)
    assert not list(output.rglob("*.py")) and not list(output.rglob("*.pyx")) and not list(output.rglob("*.cpp"))
    ordinary = next(item for item in manifest["modules"] if item["name"] == "closed_host.ordinary")
    receipt = json.loads((work / ordinary["recipe"] / "receipt.json").read_text(encoding="utf-8"))
    cached_artifact = work / ordinary["recipe"] / receipt["file"]
    # A file with the right name and length is insufficient evidence of a hit.
    content = cached_artifact.read_bytes()
    cached_artifact.write_bytes(bytes((content[0] ^ 1,)) + content[1:])
    assert build(tmp_path / "repaired")["cached"] == 2
    # Package already-built extensions with independent public resources and
    # standard metadata. No Cython invocation or module rebuild is needed.
    release = tmp_path / "release"
    release.mkdir()
    (release / "pyproject.toml").write_text(
        textwrap.dedent("""
        [build-system]
        requires = ["setuptools==82.0.1"]
        build-backend = "setuptools.build_meta"
        [project]
        name = "forge-closed-host-test"
        version = "1.2.3"
        requires-python = ">=3.10"
        readme = "README.md"
        license = "MIT"
        license-files = ["LICENSE.txt"]
        dependencies = ["taichi-forge>=0.6.4", "typing-extensions>=4.9; python_version<'3.11'"]
        [project.optional-dependencies]
        render = ["Pillow>=11"]
        [project.scripts]
        closed-host-label = "closed_host.ordinary:label"
        """),
        encoding="utf-8",
    )
    (release / "README.md").write_text("Public SDK documentation.\n", encoding="utf-8")
    (release / "LICENSE.txt").write_text("MIT fixture license\n", encoding="utf-8")
    (release / "api.pyi").write_text("def power(value: float) -> float: ...\n", encoding="utf-8")
    (release / "py.typed").write_text("", encoding="utf-8")
    (release / "settings.json").write_text('{"widths":[2,4]}', encoding="utf-8")
    release_spec = dict(
        version=1,
        host_layout="layout",
        project="release/pyproject.toml",
        files={
            "closed_host/ordinary.pyi": "release/api.pyi",
            "closed_host/py.typed": "release/py.typed",
            "closed_host/settings.json": "release/settings.json",
            "closed_host/docs/README.md": "release/README.md",
        },
    )
    release_file = tmp_path / "release.json"
    release_file.write_text(json.dumps(release_spec), encoding="utf-8")

    def package(destination=None, error=None):
        result = subprocess.run(
            [
                sys.executable,
                host_wheel.__file__,
                "--spec",
                str(release_file),
                "--output",
                str(destination or tmp_path / "wheels"),
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        if error:
            assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
            return None
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout.splitlines()[-1])

    packaged = package()
    assert packaged["modules"] == 3 and packaged["payload_files"] == 7
    # Reproducible re-publication checks the existing wheel without replacing it.
    assert package()["wheel"] == packaged["wheel"]
    (release / "README.md").write_text("Updated public SDK documentation.\n", encoding="utf-8")
    package(error="Published wheel is immutable")
    updated = package(tmp_path / "updated-wheels")
    with ZipFile(updated["wheel"]) as archive:
        for item in manifest["modules"]:
            assert archive.read(item["file"]) == (output / item["file"]).read_bytes()
    with ZipFile(packaged["wheel"]) as archive:
        names = archive.namelist()
        assert not any(n.endswith((".py", ".pyx", ".cpp", ".dll")) for n in names)
        metadata = next(n for n in names if n.endswith(".dist-info/METADATA"))
        content = archive.read(metadata).decode()
        assert "Requires-Dist: taichi-forge>=0.6.4" in content
        assert "Requires-Dist: typing-extensions>=4.9; python_version <" in content
        assert 'Requires-Dist: Pillow>=11; extra == "render"' in content
        assert any(n.endswith(".dist-info/forge-host.json") for n in names)
    installed = tmp_path / "installed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--no-compile",
            "--target",
            str(installed),
            packaged["wheel"],
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (tmp_path / "host-wheel-costs.json").write_text(json.dumps(packaged, indent=2), encoding="utf-8")
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        textwrap.dedent("""
        import builtins
        from pathlib import Path
        import sys
        layout, private = map(Path, sys.argv[1:3])
        sys.path.insert(0, str(layout))
        def guard(event, values):
            if event == "open" and values and isinstance(values[0], str):
                assert not Path(values[0]).resolve().is_relative_to(private), values[0]
        sys.addaudithook(guard)
        original = builtins.__import__
        def checked(name, *args, **kwargs):
            assert name.split('.')[0] not in ('Cython', 'cython', 'setuptools'), name
            return original(name, *args, **kwargs)
        builtins.__import__ = checked
        import closed_host
        from importlib import resources, metadata
        import json
        assert json.loads(resources.files(closed_host).joinpath('settings.json').read_text()) == {'widths': [2, 4]}
        assert 'power' in resources.files(closed_host).joinpath('ordinary.pyi').read_text()
        assert resources.files(closed_host).joinpath('py.typed').is_file()
        distribution = metadata.distribution('forge-closed-host-test')
        entry = next(e for e in distribution.entry_points if e.name == 'closed-host-label')
        assert entry.load()() == 'caf\u00e9'
        assert closed_host.LABEL == 'caf\u00e9'
        assert closed_host.CONTINUED == 3
        import closed_host_alias
        assert closed_host_alias is closed_host.ordinary
        host = closed_host.ordinary
        assert host.label() == 'caf\u00e9'
        events = []
        def visit(value):
            events.append(value)
            return value
        assert host.choose(True, visit) == (1, 2) and events == [1, 2]
        events.clear()
        assert host.choose(False, visit) == (3,) and events == [3]
        assert host.power(9) == 3 and host.power(-1) == 1e-8
        assert host.ordered(object()) == ()
        assert host.ordered(type('Data', (), {'items': [3, 1, 2]})()) == (1, 2, 3)
        assert host.python_names(include=2, cdef=3) == (5, {'include': 2, 'cdef': 3}, 2)
        try:
            host.missing()
        except NameError:
            pass
        else:
            raise AssertionError('unresolved Python globals must still raise')
        import taichi_forge as ti
        from closed_host.kernels import Model
        ti.init(arch=getattr(ti, sys.argv[3]), offline_cache=False, enable_fallback=False)
        for width in (2, 4):
            model = Model(width)
            with ti.ad.Tape(model.loss):
                model.run()
            total = width * (width + 1) / 2
            assert model.loss[None] == 4 * total
            assert all(value == total for value in model.x.grad.to_numpy())
        assert not {'Cython', 'cython', 'setuptools'}.intersection(sys.modules)
    """),
        encoding="utf-8",
    )
    for arch in os.environ.get("FORGE_HOST_TEST_ARCHS", "cpu").split(","):
        result = subprocess.run(
            [sys.executable, "-S", str(consumer), str(installed), str(source), arch],
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, arch + "\n" + result.stdout + result.stderr
