"""Static audit contracts without executing private application imports."""

import json
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def audit(monkeypatch):
    tools = Path(__file__).resolve().parents[2] / "python/taichi_forge/tools"
    monkeypatch.syspath_prepend(str(tools))
    return runpy.run_path(str(tools / "_host_inventory.py"))


def fixture_spec(tmp_path, files, selected, aliases=None):
    source = tmp_path / "src"
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    spec = tmp_path / "host.json"
    spec.write_text(
        json.dumps(dict(version=1, source_root="src", modules=selected, aliases=aliases or {}, toolchain={}))
    )
    return spec


def test_audit_follows_lazy_imports_through_selected_modules_without_execution(tmp_path, audit):
    spec = fixture_spec(
        tmp_path,
        {
            "app/__init__.py": "raise RuntimeError('must not import')",
            "app/entry.py": "from . import selected\n",
            "app/selected.py": "def build():\n    from .lazy import value\n",
            "app/lazy.py": "from . import leaf\nvalue=3\n",
            "app/leaf.py": "import numpy\n",
        },
        {"app": "app/__init__.py", "app.entry": "app/entry.py", "app.selected": "app/selected.py"},
    )
    before = spec.read_bytes()
    report = audit["audit_imports"](spec, roots=["app.entry"])
    assert set(report["missing_modules"]) == {"app.lazy", "app.leaf"}
    assert report["external_imports"] == ["numpy"]
    assert any(
        e["module"] == "app.lazy" and e["importer"] == "app.selected" and e["line"] == 2 for e in report["import_edges"]
    )
    assert spec.read_bytes() == before


def test_audit_uses_frozen_selected_bytes_and_exact_member_case(tmp_path, audit):
    spec = fixture_spec(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/entry.py": "from . import Coupler\nfrom . import coupler\n",
        },
        {"app": "app/__init__.py", "app.entry": "app/entry.py"},
    )
    candidates = tmp_path / "live"
    (candidates / "app").mkdir(parents=True)
    (candidates / "app/entry.py").write_text("import accidentally_new_version\n")
    (candidates / "app/coupler.py").write_text("class Coupler: pass\n")
    report = audit["audit_imports"](spec, source_root=candidates, roots=["app.entry"])
    assert set(report["missing_modules"]) == {"app.coupler"}
    assert not report["external_imports"]
    assert not report["unresolved_local_imports"]


def test_audit_alias_initializers_optional_and_dynamic_sites(tmp_path, audit):
    spec = fixture_spec(
        tmp_path,
        {
            "app/__init__.py": "from . import registration\n",
            "app/registration.py": "",
            "app/target.py": "if False:\n    import app.optional\nimport importlib\nimportlib.import_module(name)\nfrom .optional import *\n",
            "app/optional.py": "",
        },
        {"app": "app/__init__.py", "app.target": "app/target.py"},
        {"public": dict(target="app.target", registered_by="app")},
    )
    report = audit["audit_imports"](spec, roots=["public"])
    assert set(report["missing_modules"]) == {"app.registration", "app.optional"}
    assert report["dynamic_import_sites"] == [dict(module="app.target", line=4)]
    assert report["wildcard_import_sites"] == [dict(module="app.target", line=5, imported="app.optional")]


def test_audit_namespaces_and_unresolved_local_imports(tmp_path, audit):
    spec = fixture_spec(
        tmp_path, {"ns/entry.py": "import ns.absent\nimport ns.child\n", "ns/child.py": ""}, {"ns.entry": "ns/entry.py"}
    )
    report = audit["audit_imports"](spec)
    assert set(report["missing_modules"]) == {"ns.child"}
    assert report["unresolved_local_imports"] == [dict(module="ns.absent", importer="ns.entry", line=1)]
    with pytest.raises(ValueError, match="Audit root"):
        audit["audit_imports"](spec, roots=["unknown"])


def test_audit_cli_is_advisory_unless_explicitly_strict(tmp_path, audit, capsys):
    spec = fixture_spec(tmp_path, {"entry.py": "import missing\n", "missing.py": ""}, {"entry": "entry.py"})
    assert audit["main"](["--spec", str(spec)]) == 0
    assert "missing" in json.loads(capsys.readouterr().out)["missing_modules"]
    assert audit["main"](["--spec", str(spec), "--fail-on-missing"]) == 1
    assert "missing" in json.loads(capsys.readouterr().out)["missing_modules"]


def test_audit_rejects_ambiguous_module_and_package_candidates(tmp_path, audit):
    spec = fixture_spec(tmp_path, {"entry.py": "", "other.py": "", "other/__init__.py": ""}, {"entry": "entry.py"})
    with pytest.raises(ValueError, match="Ambiguous module/package candidate: other"):
        audit["audit_imports"](spec)


@pytest.mark.parametrize('import_line,guard', [
    ('from typing import TYPE_CHECKING', 'TYPE_CHECKING'),
    ('from typing import TYPE_CHECKING as TC', 'TC'),
    ('import typing as t', 't.TYPE_CHECKING'),
])
def test_runtime_imports_skip_only_proven_type_checking_arms(tmp_path, audit, import_line, guard):
    fixture_spec(tmp_path, {'app.py': f'''{import_line}
if {guard}:
    import dead
else:
    import live
if not {guard}:
    def lazy():
        import late
if unknown:
    import optional
''', **{name + '.py': "raise RuntimeError('must not execute')" for name in ('dead', 'live', 'late', 'optional')}},
                 {'app': 'app.py'})
    sources = audit['source_inventory'](tmp_path/'src')
    broad = audit['audit_source_imports'](sources, roots=['app'])
    runtime = audit['audit_source_imports'](sources, roots=['app'], include_type_checking=False)
    assert set(broad['visited']) == {'app', 'dead', 'live', 'late', 'optional'}
    assert set(runtime['visited']) == {'app', 'live', 'late', 'optional'}


@pytest.mark.parametrize('shadow', [
    'TC = True', 'def caller(TC):\n    pass', 'from custom import value as TC',
    'match {}:\n    case {**TC}:\n        pass',
])
def test_runtime_imports_keep_rebound_type_checking_names(tmp_path, audit, shadow):
    fixture_spec(tmp_path, {'app.py': 'from typing import TYPE_CHECKING as TC\n' + shadow +
                 '\nif TC:\n    import needed\n', 'needed.py': ''}, {'app': 'app.py'})
    report = audit['audit_source_imports'](audit['source_inventory'](tmp_path/'src'),
                                           roots=['app'], include_type_checking=False)
    assert set(report['visited']) == {'app', 'needed'}


def test_runtime_imports_keep_local_typing_module(tmp_path, audit):
    fixture_spec(tmp_path, {'app.py': 'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import needed\n',
                            'typing.py': 'TYPE_CHECKING = True', 'needed.py': ''}, {'app': 'app.py'})
    report = audit['audit_source_imports'](audit['source_inventory'](tmp_path/'src'),
                                           roots=['app'], include_type_checking=False)
    assert set(report['visited']) == {'app', 'typing', 'needed'}


def test_runtime_imports_keep_mutated_typing_attribute(tmp_path, audit):
    fixture_spec(tmp_path, {'app.py': 'import typing\ntyping.TYPE_CHECKING = True\n'
                            'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import needed\n',
                            'needed.py': ''}, {'app': 'app.py'})
    report = audit['audit_source_imports'](audit['source_inventory'](tmp_path/'src'),
                                           roots=['app'], include_type_checking=False)
    assert set(report['visited']) == {'app', 'needed'}
