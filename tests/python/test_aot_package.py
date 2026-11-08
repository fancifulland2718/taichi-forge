"""Definition coverage and source-free consumption of the initial package slice."""

import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import zipfile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import utils
from tests import test_utils
from taichi_forge_runtime.aot import Runtime, load_package

OWNER_INPUT_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Storage:
    def __init__(self, n: int, first: bool):
        global constructors
        constructors += 1
        self.n, self.first = n, first
        fields = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=fields)
        scope.static('first', choices=(False, True))
        scope.values = scope.field(ti.f32, shape=scope.dim('n'))
        self.tree = fields.finalize()
        scope.reader = Reader(self)
    @ti.kernel
    def clear(self):
        for i in self.values:
            self.values[i] = 0
    @ti.func
    def bias(self):
        result = 2
        if ti.static(self.first):
            result = 7
        return result

@ti.data_oriented
class Reader:
    def __init__(self, source):
        scope = ti.experimental.ResourceScope(self)
        scope.owner_reference('source', source)
        scope.borrow('alias', self.source.values, source_scope=True)
    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.alias:
            self.alias[i] = gain * i + self.source.bias()
    def graph(self):
        graph = ti.graph.GraphBuilder()
        graph.dispatch(self.step, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32))
        return graph.compile()

class Forward:
    def __init__(self, source):
        scope = ti.experimental.ResourceScope(self)
        scope.owner_reference('source', source)
        scope.reader = Reader(source)
    def step(self, gain):
        self.reader.step(gain)

class Model:
    def __init__(self, n: int, first: bool):
        scope = ti.experimental.ResourceScope(self)
        scope.storage = Storage(n, first)
        scope.reader = Forward(self.storage)
    def step(self, gain):
        self.reader.step(gain)
"""


FIELD_INPUT_SOURCE = (
    OWNER_INPUT_SOURCE.replace("scope.reader = Reader(self)", "scope.reader = Reader(self.values, n)")
    .replace("def __init__(self, source):", "def __init__(self, values, n):")
    .replace(
        "scope.owner_reference('source', source)",
        "self.n = n\n        extent = scope.dim('n')",
    )
    .replace(
        "scope.borrow('alias', self.source.values, source_scope=True)",
        "scope.borrow('alias', values, dimensions={'n': extent}, source_scope=True)",
    )
    .replace("gain * i + self.source.bias()", "gain * i")
    .replace(
        "scope.reader = Reader(source)",
        "scope.borrow('alias', values, dimensions={'n': extent}, source_scope=True)\n"
        "        scope.reader = Reader(values, n)",
    )
    .replace("Forward(self.storage)", "Forward(self.storage.values, n)")
    .replace("self.n, self.first = n, first", "self.n, self.first = n, first\n        self.m = n + 2")
    .replace(
        "self.tree = fields.finalize()",
        "scope.spare = scope.field(ti.f32, shape=scope.dim('m'))\n        self.tree = fields.finalize()",
    )
)


FIELD_HELPER_SOURCE = """
def declare_source(scope, name, field, *, dimensions=None):
    import inspect
    borrow = getattr(type(scope), 'borrow', None)
    if field is None:
        scope.record(name, None, optional=True)
        return
    if isinstance(field, ti.StructField):
        raise ValueError('member required')
    if (borrow is not None and 'source_scope' in inspect.signature(borrow).parameters
            and getattr(field, '_logical_resource_declaration', None) is not None):
        borrow(scope, name, field, dimensions=dimensions, source_scope=True)
"""


OPTIONAL_FIELD_INPUT_SOURCE = (
    FIELD_INPUT_SOURCE.replace("def __init__(self, values, n):", "def __init__(self, values=None, n=0):")
    .replace(
        "scope.borrow('alias', values, dimensions={'n': extent}, source_scope=True)",
        "if values is None:\n            scope.record('alias', None, optional=True)\n"
        "        else:\n            scope.borrow('alias', values, dimensions={'n': extent}, source_scope=True)",
    )
    .replace("Forward(self.storage.values, n)", "Forward(self.storage.values if first else None, n)")
    .replace(
        "def step(self, gain: ti.f32):\n        for i in self.alias:\n            self.alias[i] = gain * i",
        "def step(self, gain: ti.f32) -> ti.i32:\n        result = 0\n"
        "        if ti.static(self.alias is not None):\n            for i in self.alias:\n"
        "                self.alias[i] = gain * i\n            result = self.alias.shape[0]\n        return result",
    )
)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("helper", [False, True])
def test_package_optional_field_inputs_keep_only_used_presence_variants(tmp_path, monkeypatch, helper):
    source = OPTIONAL_FIELD_INPUT_SOURCE
    if helper:
        source = (
            source.replace(
                "if values is None:\n            scope.record('alias', None, optional=True)\n"
                "        else:\n            scope.borrow('alias', values, dimensions={'n': extent}, source_scope=True)",
                "declare_source(scope, 'alias', values, dimensions={'n': extent})",
            )
            + FIELD_HELPER_SOURCE
        )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 24
    assert len(manifest["entries"]) == 4
    assert len(manifest["field_groups"]) == 1
    assert len(manifest["entry_families"]) == 1
    assert len(next(iter(manifest["entry_families"].values()))["domain"]) == 1
    assert sys.modules["definition_fixture.model"].constructors == 0
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, True), code.host.Model(13, False)
        present, absent = a.reader.reader, b.reader.reader
        assert present.step(3.0) == 7 and absent.step(3.0) == 0
        np.testing.assert_array_equal(a.storage.values.to_numpy(), 3 * np.arange(7))
        assert absent.alias is None and absent._forge_package_field_groups == {}
        # Absence does not allocate or require even a live source owner.
        b.storage.tree.destroy()
        assert absent.step(4.0) == 0
        assert type(absent)().step(4.0) == 0
        prepared = absent.graph().bind(dict(gain=9.0))
        absent.alias = a.storage.values
        with pytest.raises(ValueError, match="presence changed"):
            absent.step(1.0)
        prepared.run()
        np.testing.assert_array_equal(a.storage.values.to_numpy(), 3 * np.arange(7))
        absent.alias = None
        present.alias = None
        with pytest.raises(ValueError, match="presence changed"):
            present.step(1.0)
        prepared.close()
    for entry in manifest["entries"].values():
        if any(not item["present"] for item in entry.get("admission", {}).get("input_presence", {}).values()):
            assert not entry["captures"]
    probe = tmp_path / "consume_optional_input.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc, sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'taichi', 'taichi_forge', 'definition_fixture'}:
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3, True), code.host.Model(13, False)
            assert a.reader.reader.step(4.0) == 3
            assert b.reader.reader.step(4.0) == 0
            np.testing.assert_array_equal(a.storage.values.to_numpy(), 4 * np.arange(3))
            b.storage.tree.destroy()
            assert b.reader.reader.step(1.0) == 0
        print('OPTIONAL_INPUT_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OPTIONAL_INPUT_SOURCE_FREE_OK" in result.stdout


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_independent_optional_inputs_do_not_multiply_each_others_code(tmp_path, monkeypatch):
    source = OPTIONAL_FIELD_INPUT_SOURCE.replace(
        "class Reader:\n    def __init__(self, values=None, n=0):",
        "class Reader:\n    def __init__(self, values=None, n=0, other=None):",
    )
    source = source.replace(
        "scope.borrow('alias', values, dimensions={'n': extent}, source_scope=True)",
        "scope.borrow('alias', values, dimensions={'n': extent}, source_scope=True)\n"
        "        if other is None:\n            scope.record('other', None, optional=True)\n"
        "        else:\n            scope.borrow('other', other, dimensions={'n': extent}, source_scope=True)",
        1,
    )
    source = source.replace("Reader(self.values, n)", "Reader(self.values, n, self.values)")
    source = source.replace("Reader(values, n)", "Reader(values, n, values)")
    source = source.replace(
        "    def graph(self):",
        """    @ti.kernel
    def second(self, value: ti.f32) -> ti.i32:
        result = 0
        if ti.static(self.other is not None):
            for i in self.other:
                self.other[i] = value
            result = self.other.shape[0]
        return result
    def graph(self):""",
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert len(manifest["entries"]) == 6
    assert len(manifest["entry_families"]) == 2
    assert all(len(item["domain"]) == 1 and len(item["variants"]) == 2 for item in manifest["entry_families"].values())
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(7, True)
        reader = type(model.reader.reader)
        for first in (None, model.storage.values):
            for second in (None, model.storage.values):
                value = reader(first, 7, second)
                assert value.step(3.0) == (0 if first is None else 7)
                assert value.second(9.0) == (0 if second is None else 7)


@pytest.mark.parametrize(
    "change, diagnostic",
    [
        ("wrong_guard", "non-None branch"),
        ("mutated_parameter", "unconditional declared boolean"),
        ("unknown_source", "proven source"),
        ("wrong_prior_binding", "replace an existing"),
        ("unproven_absence_helper", "proven absence declaration"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_optional_inputs_reject_unproven_presence_or_source(tmp_path, monkeypatch, change, diagnostic):
    source = OPTIONAL_FIELD_INPUT_SOURCE
    if change == "wrong_guard":
        source = source.replace("if values is None:", "if values is not None:")
    elif change == "mutated_parameter":
        source = source.replace("        if values is None:", "        values = None\n        if values is None:")
    elif change == "unknown_source":
        source = source.replace("self.storage.values if first else None", "self.storage.values if first else n")
    elif change == "wrong_prior_binding":
        source = source.replace("        if values is None:", "        self.alias = None\n        if values is None:")
    else:
        source = source.replace(
            "if values is None:\n            scope.record('alias', None, optional=True)\n"
            "        else:\n            scope.borrow('alias', values, dimensions={'n': extent}, source_scope=True)",
            "declare_source(scope, 'alias', values, dimensions={'n': extent})",
        )
        source += FIELD_HELPER_SOURCE.replace("scope.record(name, None, optional=True)", "application_side_effect()")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk/package.json").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_field_metadata_helpers_require_proven_allocation_inputs(tmp_path, monkeypatch):
    source = FIELD_INPUT_SOURCE.replace("scope.borrow('alias', values,", "declare_source(scope, 'alias', values,")
    source = source.replace(", source_scope=True)", ")") + FIELD_HELPER_SOURCE
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["entries"]) == 3
    facts = manifest["declaration_factories"]
    assert len(facts) == 2 and {item["definition"] for item in facts} == {"definition_fixture.model.declare_source"}
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, True), code.host.Model(13, False)
        a.step(3.0)
        b.step(-2.0)
        np.testing.assert_array_equal(a.storage.values.to_numpy(), 3 * np.arange(7))
        np.testing.assert_array_equal(b.storage.values.to_numpy(), -2 * np.arange(13))


@pytest.mark.parametrize(
    "before, after",
    [
        ("borrow(scope, name, field, dimensions=dimensions, source_scope=True)", "return"),
        ("source_scope=True)", "external=True)"),
        ("borrow(scope, name, field,", "borrow(scope, name, dimensions,"),
        ("import inspect", "import inspect\n    application_side_effect(field)"),
        ("declare_source(scope, 'alias', values,", "result = declare_source(scope, 'alias', values,"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_field_metadata_helpers_cannot_drop_obligations_or_side_effects(tmp_path, monkeypatch, before, after):
    source = FIELD_INPUT_SOURCE.replace("scope.borrow('alias', values,", "declare_source(scope, 'alias', values,")
    source = source.replace(", source_scope=True)", ")") + FIELD_HELPER_SOURCE
    package = _source_package(tmp_path, monkeypatch, source.replace(before, after))
    with pytest.raises(ValueError, match="unproven"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk/package.json").exists()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_field_inputs_retain_source_group_without_host_combination_code(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, FIELD_INPUT_SOURCE)
    original = package.Model(7, True)
    original.step(3.0)
    expected = original.storage.values.to_numpy()
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 23
    assert sys.modules["definition_fixture.model"].constructors == 1
    assert len(manifest["entries"]) == 3
    assert not manifest.get("entry_families")
    assert len(manifest["field_groups"]) == 1
    assert len(next(iter(manifest["field_groups"].values()))["dimensions"]) == 2
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([n for n in archive.namelist() if n.endswith((".ll", ".bc", ".spv"))]) == 3
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, True), code.host.Model(19, False)
        reader = a.reader.reader
        assert reader.alias is a.storage.values
        assert reader._forge_package_field_groups == {}
        a.step(3.0)
        np.testing.assert_array_equal(a.storage.values.to_numpy(), expected)
        b.step(-2.0)
        np.testing.assert_array_equal(b.storage.values.to_numpy(), -2 * np.arange(19))
        with pytest.raises(ValueError, match="constraint"):
            type(reader)(a.storage.values, 6)
        with pytest.raises(ValueError, match="allocation member"):
            type(reader)(a.storage.spare, 9)
        with load_package(runtime, tmp_path / "sdk") as foreign:
            source = foreign.host.Model(7, True).storage
            with pytest.raises(ValueError, match="child owner"):
                type(reader)(source.values, 7)
        prepared = reader.graph().bind(dict(gain=4.0))
        reader.alias = b.storage.values
        with pytest.raises(ValueError, match="Borrowed Field"):
            reader.step(1.0)
        prepared.run()
        np.testing.assert_array_equal(a.storage.values.to_numpy(), 4 * np.arange(7))
        reader.alias = a.storage.values
        a.storage.tree.destroy()
        with pytest.raises((ValueError, RuntimeError)):
            reader.step(1.0)
        b.step(1.0)
        np.testing.assert_array_equal(b.storage.values.to_numpy(), np.arange(19))
        prepared.close()
    probe = tmp_path / "consume_fields.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3, True), code.host.Model(13, False)
            assert a.storage.values is a.reader.reader.alias
            assert a.reader.reader._forge_package_field_groups == {}
            a.step(4.0)
            b.step(-2.0)
            np.testing.assert_array_equal(a.storage.values.to_numpy(), 4 * np.arange(3))
            np.testing.assert_array_equal(b.storage.values.to_numpy(), -2 * np.arange(13))
        print('FIELD_INPUT_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FIELD_INPUT_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("kind", ["vector", "matrix", "struct_member"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_field_input_uses_original_tensor_member_layout(tmp_path, monkeypatch, kind):
    source = FIELD_INPUT_SOURCE
    if kind == "vector":
        source = source.replace("scope.values = scope.field(ti.f32,", "scope.values = scope.vector_field(2, ti.f32,")
    elif kind == "matrix":
        source = source.replace("scope.values = scope.field(ti.f32,", "scope.values = scope.matrix_field(2, 2, ti.f32,")
    else:
        source = (
            source.replace(
                "scope.values = scope.field(ti.f32,",
                "scope.values = scope.struct_field({'x': ti.f32, 'v': ti.types.vector(2, ti.f32)},",
            )
            .replace("self.values", "self.values.x")
            .replace("self.storage.values", "self.storage.values.x")
        )
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(7, False)
    original.step(3.0)
    original_values = original.storage.values.x if kind == "struct_member" else original.storage.values
    expected = original_values.to_numpy()
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert len(manifest["entries"]) == 3
    assert len(manifest["field_groups"]) == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        owner = code.host.Model(7, True)
        owner.step(3.0)
        values = owner.storage.values.x if kind == "struct_member" else owner.storage.values
        assert owner.reader.reader.alias is values
        assert values._group_owner is owner.storage.spare._group_owner
        np.testing.assert_array_equal(values.to_numpy(), expected)


@pytest.mark.parametrize(
    "change, diagnostic",
    [
        (("Reader(self.values, n)", "Reader(None, n)"), "proven source"),
        (
            (
                "scope = ti.experimental.ResourceScope(self)\n        self.n = n",
                "values = None\n        scope = ti.experimental.ResourceScope(self)\n        self.n = n",
            ),
            "unmodified constructor parameter",
        ),
        (
            ("class Model:", "def unproven(values):\n    return Reader(values, 7)\n\nclass Model:"),
            "Unscoped constructor input",
        ),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_field_input_rejects_missing_or_conflicting_source(tmp_path, monkeypatch, change, diagnostic):
    package = _source_package(tmp_path, monkeypatch, FIELD_INPUT_SOURCE.replace(*change))
    with pytest.raises(ValueError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk/package.json").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_field_input_retains_distinct_source_dimension_relations(tmp_path, monkeypatch):
    package = _source_package(
        tmp_path,
        monkeypatch,
        FIELD_INPUT_SOURCE.replace("Reader(self.values, n)", "Reader(self.spare, self.m)").replace(
            "dimensions={'n': extent}, ", ""
        ),
    )
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert len(manifest["input_families"]["definition_fixture.model.Reader"]) == 2
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(7, True)
        model.storage.reader.step(2.0)
        model.reader.step(-3.0)
        np.testing.assert_array_equal(model.storage.spare.to_numpy(), 2 * np.arange(9))
        np.testing.assert_array_equal(model.storage.values.to_numpy(), -3 * np.arange(7))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_typed_field_inputs_preserve_declared_dimension_rejections(tmp_path, monkeypatch):
    package = _source_package(
        tmp_path, monkeypatch, FIELD_INPUT_SOURCE.replace("Reader(self.values, n)", "Reader(self.spare, self.m)")
    )
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        with pytest.raises(ValueError, match="constraint"):
            code.host.Model(7, True)
        assert code._arrays == []


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_constructor_inputs_share_source_allocation_without_combination_code(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, OWNER_INPUT_SOURCE)
    original = package.Model(7, True)
    original.step(3.0)
    expected = original.storage.values.to_numpy()
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 22
    assert sys.modules["definition_fixture.model"].constructors == 1
    assert len(manifest["entries"]) == 4
    assert len(manifest["field_groups"]) == 1
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([n for n in archive.namelist() if n.endswith((".ll", ".bc", ".spv"))]) == 4
    family = next(iter(manifest["entry_families"].values()))
    assert list(family["paths"].values()) == [["source", "first"]]
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, True), code.host.Model(19, False)
        assert a.storage.reader.source is a.reader.source is a.storage
        assert a.reader.reader.source is a.storage
        assert a.storage.reader._forge_package_field_groups == a.reader.reader._forge_package_field_groups == {}
        a.step(3.0)
        np.testing.assert_array_equal(a.storage.values.to_numpy(), expected)
        b.step(-2.0)
        np.testing.assert_array_equal(b.storage.values.to_numpy(), -2 * np.arange(19) + 2)
        inner = a.reader.reader
        prepared = inner.graph().bind(dict(gain=4.0))
        inner.source = b.storage
        with pytest.raises(ValueError, match="Borrowed Field"):
            inner.step(1.0)
        prepared.run()
        np.testing.assert_array_equal(a.storage.values.to_numpy(), 4 * np.arange(7) + 7)
        inner.source = a.storage
        with load_package(runtime, tmp_path / "sdk") as foreign:
            source = foreign.host.Model(7, True).storage
            with pytest.raises(ValueError, match="child owner"):
                type(inner)(source)
        a.storage.tree.destroy()
        with pytest.raises((ValueError, RuntimeError)):
            inner.step(1.0)
        b.step(1.0)
        np.testing.assert_array_equal(b.storage.values.to_numpy(), np.arange(19) + 2)
        prepared.close()
    probe = tmp_path / "consume_inputs.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3, True), code.host.Model(13, False)
            assert a.storage.reader.source is a.reader.reader.source is a.storage
            assert a.reader.reader._forge_package_field_groups == {}
            a.step(4.0)
            b.step(-2.0)
            np.testing.assert_array_equal(a.storage.values.to_numpy(), 4 * np.arange(3) + 7)
            np.testing.assert_array_equal(b.storage.values.to_numpy(), -2 * np.arange(13) + 2)
        print('OWNER_INPUT_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OWNER_INPUT_SOURCE_FREE_OK" in result.stdout


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_owner_dimensions_preserve_source_relations_before_allocation(tmp_path, monkeypatch):
    source = (
        OWNER_INPUT_SOURCE.replace("def __init__(self, source):", "def __init__(self, source, offset: int = 0):")
        .replace(
            "scope.borrow('alias', self.source.values, source_scope=True)",
            "scope.borrow('alias', self.source.values, source_scope=True)\n"
            "        self.count = source.n * 2 + offset\n"
            "        scope.derived_dim('count', scope.owner_dim('source', 'n') * 2)\n"
            "        scope.scratch = scope.field(ti.f32, shape=scope.dim('count'), initialization='zero')",
        )
        .replace(
            "        for i in self.alias:",
            "        for j in range(self.count):\n            self.scratch[j] = gain * j\n        for i in self.alias:",
        )
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 42
    assert sys.modules["definition_fixture.model"].constructors == 0
    original = package.Model(7, True)
    original.step(3.0)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, True), code.host.Model(11, True)
        a.step(3.0)
        b.step(-2.0)
        np.testing.assert_array_equal(a.reader.reader.scratch.to_numpy(), original.reader.reader.scratch.to_numpy())
        np.testing.assert_array_equal(b.reader.reader.scratch.to_numpy(), -2 * np.arange(22))
        prepared = a.reader.reader.graph().bind(dict(gain=4.0))
        a.reader.reader.count += 1
        with pytest.raises(ValueError, match="dimension|Dimension"):
            a.step(1.0)
        prepared.run()
        np.testing.assert_array_equal(a.reader.reader.scratch.to_numpy(), 4 * np.arange(14))
        prepared.close()
        assert b.reader.reader.count == 22
        with pytest.raises(ValueError, match="Derived dimension"):
            type(b.reader.reader)(b.storage, 1)
        b.step(2.0)
        np.testing.assert_array_equal(b.reader.reader.scratch.to_numpy(), 2 * np.arange(22))
    probe = tmp_path / "consume_owner_dimensions.py"
    probe.write_text(
        """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class RejectSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError('Numerical source is unavailable: ' + fullname)
sys.meta_path.insert(0, RejectSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    a, b = code.host.Model(5, True), code.host.Model(13, True)
    a.step(3.0)
    b.step(-2.0)
    np.testing.assert_array_equal(a.reader.reader.scratch.to_numpy(), 3 * np.arange(10))
    np.testing.assert_array_equal(b.reader.reader.scratch.to_numpy(), -2 * np.arange(26))
print('OWNER_DIMENSION_SOURCE_FREE_OK')
""",
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OWNER_DIMENSION_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("change", ["missing", "value", "unbound", "computed"])
@test_utils.test(arch=ti.cpu)
def test_package_owner_dimension_requires_declared_source(tmp_path, monkeypatch, change):
    source = OWNER_INPUT_SOURCE.replace(
        "scope.borrow('alias', self.source.values, source_scope=True)",
        "scope.borrow('alias', self.source.values, source_scope=True)\n" "        scope.owner_dim('source', 'n')",
    )
    if change == "missing":
        source = source.replace("owner_dim('source', 'n')", "owner_dim('source', 'missing')")
    elif change == "value":
        source = source.replace("shape=scope.dim('n')", "shape=3")
    elif change == "unbound":
        source = source.replace("owner_dim('source', 'n')", "owner_dim('alias', 'n')")
    else:
        source = source.replace("owner_dim('source', 'n')", "owner_dim('source', str('n'))")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match="owner_dim|Referenced dimension"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_owner_inputs_allow_same_or_distinct_resource_instances(tmp_path, monkeypatch):
    source = (
        OWNER_INPUT_SOURCE.replace("Reader(self)", "Reader(self, self)")
        .replace("Reader(source)", "Reader(source, source)")
        .replace(
            "class Reader:\n    def __init__(self, source):", "class Reader:\n    def __init__(self, source, other):"
        )
        .replace("scope.borrow('alias'", "scope.owner_reference('other', other)\n        scope.borrow('alias'")
        .replace(
            "self.alias[i] = gain * i + self.source.bias()",
            "self.alias[i] = gain * i + self.source.bias()\n            self.other.values[i] += 1",
        )
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert len(manifest["entries"]) == 4
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, True), code.host.Model(7, False)
        a.step(3.0)
        np.testing.assert_array_equal(a.storage.values.to_numpy(), 3 * np.arange(7) + 8)
        b.storage.clear()
        distinct = type(a.storage.reader)(a.storage, b.storage)
        distinct.step(2.0)
        np.testing.assert_array_equal(a.storage.values.to_numpy(), 2 * np.arange(7) + 7)
        np.testing.assert_array_equal(b.storage.values.to_numpy(), 1)


@pytest.mark.parametrize("change", ["mutable", "model", "shadowed", "path", "rebind"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_fixed_alias_mapping_rejects_unproven_inputs(tmp_path, monkeypatch, change):
    source = _borrowed_alias_source(BORROWED_FIELDS_SOURCE, False, "mapping")
    if change == "mutable":
        source = source.replace("MappingProxyType({'alias': ('values',)})", "{'alias': ('values',)}")
    elif change == "model":
        source = source.replace("for name, (member,) in ALIASES.items():", "for name, (member,) in self.aliases.items():")
    elif change == "shadowed":
        source = source.replace("constructors = 0", "getattr = lambda *a: None\nconstructors = 0")
    elif change == "path":
        source = source.replace("('values',)", "['values']")
    else:
        source = source.replace("for name, (member,)", "ALIASES = self.aliases\n        for name, (member,)")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="Alias"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("Reader(self)", "Reader(None)", "every declared constructor call"),
        ("Forward(self.storage)", "Forward(self)", "Borrowed source member is absent"),
        (
            "scope.owner_reference('source', source)",
            "scope.owner_reference('source', source, optional=True)",
            "named constructor parameter or declared owner path",
        ),
        ("scope.reader = Reader(source)", "scope.reader = Reader(*[source])", "explicit arguments"),
        (
            "class Forward:",
            "def make_reader(source):\n    return Reader(source)\n\nclass Forward:",
            "Unscoped constructor input",
        ),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_owner_inputs_reject_unproven_source_types(tmp_path, monkeypatch, before, after, diagnostic):
    package = _source_package(tmp_path, monkeypatch, OWNER_INPUT_SOURCE.replace(before, after))
    with pytest.raises(ValueError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_owner_input_keeps_struct_member_allocation_scope(tmp_path, monkeypatch):
    source = (
        OWNER_INPUT_SOURCE.replace(
            "scope.values = scope.field(ti.f32, shape=scope.dim('n'))",
            "scope.values = scope.struct_field({'x': ti.f32, 'v': ti.types.vector(2, ti.f32)}, shape=scope.dim('n'))",
        )
        .replace("self.source.values, source_scope", "self.source.values.x, source_scope")
        .replace("self.values[i]", "self.values.x[i]")
        .replace("in self.values:", "in self.values.x:")
    )
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(7, True)
    original.step(3.0)
    expected = original.storage.values.x.to_numpy()
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert len(manifest["entries"]) == 4 and len(manifest["field_groups"]) == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a = code.host.Model(7, True)
        a.step(3.0)
        assert a.reader.reader.alias is a.storage.values.x
        assert a.reader.reader.alias._group_owner is a.storage.values.v._group_owner
        np.testing.assert_array_equal(a.storage.values.x.to_numpy(), expected)


@pytest.mark.parametrize("annotation", ["Storage", '"Storage"', "'\"Storage\"'"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_typed_owner_input_needs_no_constructor_call_sample(tmp_path, monkeypatch, annotation):
    source = OWNER_INPUT_SOURCE[: OWNER_INPUT_SOURCE.index("class Forward:")]
    source = source.replace("        scope.reader = Reader(self)\n", "").replace(
        "def __init__(self, source):", f"def __init__(self, source: {annotation}):"
    )
    package = _source_package(
        tmp_path, monkeypatch, source, initializer="from .model import Reader as Model, Storage\n"
    )
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["entries"]) == 4
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        storage = code.host.Storage(7, True)
        reader = code.host.Model(storage)
        reader.step(3.0)
        assert reader.alias is storage.values
        np.testing.assert_array_equal(storage.values.to_numpy(), 3 * np.arange(7) + 7)


def _default_owner_source(annotation=""):
    source = OWNER_INPUT_SOURCE[: OWNER_INPUT_SOURCE.index("class Forward:")]
    source = source.replace("        scope.reader = Reader(self)\n", "").replace(
        "def __init__(self, source):",
        f"def __init__(self, n, first, source{annotation}=None):\n"
        "        if source is None:\n"
        "            source = Storage(n, first)",
    )
    return source + """
class Model:
    def __init__(self, n, first):
        scope = ti.experimental.ResourceScope(self)
        scope.reader = Reader(n, first)
"""


@pytest.mark.parametrize("annotation", ["", ": Storage"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_owner_input_default_reuses_the_same_numerical_definition(tmp_path, monkeypatch, annotation):
    source = _default_owner_source(annotation)
    package = _source_package(tmp_path, monkeypatch, source, initializer="from .model import Model, Reader, Storage\n")
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["entries"]) == 4 and len(manifest["field_groups"]) == 1
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([name for name in archive.namelist() if name.endswith((".ll", ".bc"))]) == 4
    original = package.Model(7, True)
    original.reader.step(3.0)
    expected = original.reader.source.values.to_numpy()
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, True), code.host.Reader(19, False)
        shared = code.host.Reader(7, True, a.reader.source)
        assert shared.source is a.reader.source
        assert shared.alias is a.reader.alias
        a.reader.step(3.0)
        np.testing.assert_array_equal(shared.alias.to_numpy(), expected)
        b.step(-2.0)
        np.testing.assert_array_equal(b.alias.to_numpy(), -2 * np.arange(19) + 2)
        shared.step(4.0)
        np.testing.assert_array_equal(a.reader.alias.to_numpy(), 4 * np.arange(7) + 7)
        np.testing.assert_array_equal(b.alias.to_numpy(), -2 * np.arange(19) + 2)
        # Loaded Host constructors and the numerical dependency use only the
        # projected package, including the default constructor's allocation.
        assert type(a.reader.source).__module__ != type(original.reader.source).__module__
        assert "@ti.kernel" not in (tmp_path / "sdk/host/model.py").read_text(encoding="utf8")
    if not annotation:
        probe = tmp_path / "consume_default_owner.py"
        probe.write_text(
            textwrap.dedent("""
            import importlib.abc
            import sys
            import numpy as np
            class NoFrontend(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                        raise AssertionError(fullname)
            sys.meta_path.insert(0, NoFrontend())
            from taichi_forge_runtime.aot import Runtime, load_package
            with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
                a, b = code.host.Model(3, True), code.host.Reader(13, False)
                alias = code.host.Reader(3, True, a.reader.source)
                alias.step(2.0)
                b.step(4.0)
                np.testing.assert_array_equal(a.reader.alias.to_numpy(), 2 * np.arange(3) + 7)
                np.testing.assert_array_equal(b.alias.to_numpy(), 4 * np.arange(13) + 2)
                assert alias.source is a.reader.source
            print('DEFAULT_OWNER_SOURCE_FREE_OK')
        """),
            encoding="utf8",
        )
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        )
        result = subprocess.run(
            [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "DEFAULT_OWNER_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("condition", "named constructor parameter or declared owner path"),
        ("rebind", "named constructor parameter or declared owner path"),
        ("delete", "named constructor parameter or declared owner path"),
        ("shadow", "unshadowed source name"),
        ("factory", "declared owner constructor"),
        ("type", "contradicts its declared source type"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_owner_input_default_rejects_unproven_factories(tmp_path, monkeypatch, change, diagnostic):
    source = _default_owner_source(": Storage")
    if change == "condition":
        source = source.replace("if source is None:", "if first:")
    elif change == "rebind":
        source = source.replace("source = Storage(n, first)", "source = Storage(n, first)\n        source = None")
    elif change == "delete":
        source = source.replace(
            "scope.owner_reference('source', source)", "del source\n        scope.owner_reference('source', source)"
        )
    elif change == "shadow":
        source = source.replace("if source is None:", "Storage = type(self)\n        if source is None:")
    elif change == "factory":
        source = source.replace("constructors = 0", "constructors = 0\ndef create(n, first): return Storage(n, first)")
        source = source.replace("source = Storage(n, first)", "source = create(n, first)")
    else:
        source = source.replace("source: Storage", "source: 'Reader'")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


DECLARED_CHILD_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Storage:
    def __init__(self, n: int, first: bool):
        global constructors
        constructors += 1
        self.n, self.first = n, first
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.static('first', choices=(False, True))
        rows = scope.dim('n')
        scope.values = scope.field(ti.f32, shape=rows)
        self.tree = builder.finalize()

    @ti.kernel
    def fill(self, gain: ti.f32):
        for i in self.values:
            self.values[i] = gain * i

    @ti.func
    def bias(self):
        result = 2
        if ti.static(self.first):
            result = 7
        return result

class Group:
    def __init__(self, n: int, m: int, first: bool, second: bool, fail=False):
        scope = ti.experimental.ResourceScope(self)
        scope.left = Storage(n, first)
        scope.right = Storage(m, second)
        if fail:
            raise ValueError('requested constructor failure')

    def fill(self, gain):
        self.left.fill(gain)
        self.right.fill(-gain)

@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int, first: bool, second: bool, fail=False):
        self.n = n
        scope = ti.experimental.ResourceScope(self)
        rows = scope.dim('n')
        scope.child = Group(n, m, first, second, fail)
        scope.borrow('alias', self.child.left.values, source_scope=True, dimensions={'n': rows})

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.alias:
            self.alias[i] = gain * i + self.child.left.bias()

    def graph(self):
        builder = ti.graph.GraphBuilder()
        gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32)
        builder.dispatch(self.step, gain)
        return builder.compile()
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_declared_owner_graph_shares_children_and_keeps_instance_bindings(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, DECLARED_CHILD_SOURCE)
    original = package.Model(7, 11, True, False)
    original.step(3.0)
    expected = original.alias.to_numpy()
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 21
    assert sys.modules["definition_fixture.model"].constructors == 2
    assert len(manifest["field_groups"]) == 1
    assert len(manifest["entries"]) == 4
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([n for n in archive.namelist() if n.endswith((".ll", ".bc", ".spv"))]) == 4
    family = next(iter(manifest["entry_families"].values()))
    assert list(family["paths"].values()) == [["child", "left", "first"]]
    assert len(family["variants"]) == 2
    host = (tmp_path / "sdk/host/model.py").read_text(encoding="utf8")
    assert "self.left.fill(gain)" in host and "scope.left = Storage(n, first)" in host
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, 11, True, False), code.host.Model(19, 3, False, True)
        a.step(3.0)
        np.testing.assert_array_equal(a.alias.to_numpy(), expected)
        b.step(-2.0)
        np.testing.assert_array_equal(b.alias.to_numpy(), -2 * np.arange(19) + 2)
        assert a.alias is a.child.left.values
        assert a.alias._group_owner is not a.child.right.values._group_owner
        assert a.alias._group_owner is not b.alias._group_owner
        a.n = 6
        with pytest.raises(ValueError, match="constraint|Constraint"):
            a.step(3.0)
        a.n = 7
        a.child.fill(5.0)
        np.testing.assert_array_equal(a.child.right.values.to_numpy(), -5 * np.arange(11))
        prepared = a.graph().bind(dict(gain=4.0))
        saved = a.child
        a.child = b.child
        with pytest.raises(ValueError, match="Borrowed Field"):
            a.step(1.0)
        prepared.run()
        np.testing.assert_array_equal(saved.left.values.to_numpy(), 4 * np.arange(7) + 7)
        a.child = saved
        registered = set(runtime._resources)
        with pytest.raises(ValueError, match="requested constructor failure"):
            code.host.Model(5, 6, False, False, True)
        assert runtime._resources == registered
        with load_package(runtime, tmp_path / "sdk") as foreign:
            a.child = foreign.host.Model(7, 11, True, False).child
            with pytest.raises(ValueError, match="child owner"):
                a.step(1.0)
        a.child = saved
        a.child.left.tree.destroy()
        with pytest.raises((ValueError, RuntimeError)):
            a.step(1.0)
        b.step(1.0)
        np.testing.assert_array_equal(b.alias.to_numpy(), np.arange(19) + 2)
        prepared.close()
    probe = tmp_path / "consume_children.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class BlockSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, BlockSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3, 5, True, False), code.host.Model(13, 7, False, True)
            a.step(4.0)
            b.step(-2.0)
            np.testing.assert_array_equal(a.alias.to_numpy(), 4 * np.arange(3) + 7)
            np.testing.assert_array_equal(b.alias.to_numpy(), -2 * np.arange(13) + 2)
            assert a.alias._group_owner is not b.alias._group_owner
            a.child.fill(9.0)
            np.testing.assert_array_equal(a.child.right.values.to_numpy(), -9 * np.arange(5))
        print('CHILD_DEFINITIONS_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHILD_DEFINITIONS_SOURCE_FREE_OK" in result.stdout


TYPED_FIELD_INPUT_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Storage:
    def __init__(self, n: int, width: int):
        global constructors
        constructors += 1
        self.n, self.width = n, width
        scope = ti.experimental.ResourceScope(self)
        columns = scope.dim('width')
        scope.scalar = scope.field(ti.f32, shape=scope.dim('n'), initialization='zero')
        scope.record = scope.struct_field({'x': ti.f32, 'v': ti.types.vector(self.width, ti.f32), 'id': ti.i32}, shape=scope.dim('n'), initialization='zero')
@ti.data_oriented
class Checkpoint:
    def __init__(self, values):
        scope = ti.experimental.ResourceScope(self)
        scope.borrow('source', values, source_scope=True)
        scope.backup = scope.field_like(self.source, layout=ti.Layout.SOA, initialization='zero')
    @ti.kernel
    def save(self):
        for i in ti.grouped(self.source):
            self.backup[i] = self.source[i]
    @ti.kernel
    def restore(self):
        for i in ti.grouped(self.source):
            self.source[i] = self.backup[i]
    @ti.kernel
    def common(self) -> ti.i32:
        return 7
@ti.data_oriented
class Model:
    def __init__(self, n: int, width: int):
        scope = ti.experimental.ResourceScope(self)
        scope.storage = Storage(n, width)
        scope.first = Checkpoint(self.storage.scalar)
        scope.second = Checkpoint(self.storage.record)
    def graph(self):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.first.save)
        builder.dispatch(self.second.save)
        return builder.compile()
"""


TYPED_OWNER_INPUT_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class ScalarStorage:
    def __init__(self, n: int):
        global constructors
        constructors += 1
        self.n = n
        scope = ti.experimental.ResourceScope(self)
        scope.values = scope.field(ti.f32, shape=scope.dim('n'), initialization='zero')
    @ti.func
    def read(self, index):
        return self.values[index] + 3.0
@ti.data_oriented
class VectorStorage:
    def __init__(self, n: int):
        global constructors
        constructors += 1
        self.n = n
        scope = ti.experimental.ResourceScope(self)
        scope.values = scope.vector_field(3, ti.f32, shape=scope.dim('n'), initialization='zero')
    @ti.func
    def read(self, index):
        return self.values[index].sum() - 5.0
@ti.data_oriented
class Reader:
    def __init__(self, source):
        scope = ti.experimental.ResourceScope(self)
        scope.owner_reference('source', source)
        self.n = source.n
        scope.derived_dim('n', scope.owner_dim('source', 'n'))
        scope.output = scope.field(ti.f32, shape=scope.dim('n'), initialization='zero')
    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.output:
            self.output[i] = gain * self.source.read(i)
    @ti.kernel
    def common(self) -> ti.i32:
        return 7
@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int):
        scope = ti.experimental.ResourceScope(self)
        scope.scalar = ScalarStorage(n)
        scope.vector = VectorStorage(m)
        scope.first = Reader(self.scalar)
        scope.second = Reader(self.vector)
    def graph(self):
        builder = ti.graph.GraphBuilder()
        gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32)
        builder.dispatch(self.first.step, gain)
        builder.dispatch(self.second.step, gain)
        return builder.compile()
"""


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_write_inspection_uses_compiler_facts_without_execution(tmp_path, monkeypatch):
    source = TYPED_OWNER_INPUT_SOURCE.replace(
        "    def graph(self):\n        builder = ti.graph.GraphBuilder()",
        "    def record(self, builder):",
    ).replace(
        "        return builder.compile()",
        "    def graph(self):\n        builder = ti.graph.GraphBuilder()\n"
        "        self.record(builder)\n        return builder.compile()\n"
        "    def writes(self):\n        return ti.aot.written_resources(self, self.record)",
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 48
    assert sys.modules["definition_fixture.model"].constructors == 0
    original = package.Model(7, 5)
    assert {id(field) for field in original.writes()} == {id(original.first.output), id(original.second.output)}
    np.testing.assert_array_equal(original.first.output.to_numpy(), 0)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(13, 9), code.host.Model(3, 4)
        graph = a.graph()
        expected = {id(a.first.output), id(a.second.output)}
        allocated = set(runtime._resources)
        assert {id(field) for field in graph.written_resources()} == expected
        assert {id(field) for field in a.writes()} == expected
        assert {id(field) for field in b.writes()} == {id(b.first.output), id(b.second.output)}
        assert set(runtime._resources) == allocated
        np.testing.assert_array_equal(a.first.output.to_numpy(), 0)
        # Incomplete/opaque evidence must keep possible writers, not drop them.
        selected = [entry for entry in code.manifest["entries"].values() if entry["definition"].endswith(".step")]
        for entry in selected:
            assert entry["resource_effects"]["precision"] == "compiler_proven"
            entry["resource_effects"]["precision"] = "conservative"
        assert {id(field) for field in graph.written_resources()} == expected | {
            id(a.scalar.values),
            id(a.vector.values),
        }
        for entry in selected:
            del entry["resource_effects"]
        assert {id(field) for field in graph.written_resources()} == expected | {
            id(a.scalar.values),
            id(a.vector.values),
        }
        selected[0]["resource_effects"] = dict(precision="compiler_proven", arguments={})
        with pytest.raises(ValueError, match="complete compiler effect"):
            graph.written_resources()
        with pytest.raises(ValueError, match="bound recording"):
            from taichi_forge_runtime.aot._package import _ModuleBinding

            _ModuleBinding(code).written_resources(a, b.record)
    probe = tmp_path / "consume_write_facts.py"
    probe.write_text(
        """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class RejectSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError('Numerical source unavailable: ' + fullname)
sys.meta_path.insert(0, RejectSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    model = code.host.Model(17, 11)
    assert {id(x) for x in model.writes()} == {id(model.first.output), id(model.second.output)}
    np.testing.assert_array_equal(model.first.output.to_numpy(), 0)
    model.graph().run(dict(gain=2.0))
    np.testing.assert_array_equal(model.first.output.to_numpy(), 6.0)
    np.testing.assert_array_equal(model.second.output.to_numpy(), -10.0)
print('SOURCE_FREE_WRITE_FACTS_OK')
""",
        encoding="utf8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SOURCE_FREE_WRITE_FACTS_OK" in result.stdout


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_write_inspection_rejects_unbound_external_writes(tmp_path, monkeypatch):
    source = """
import taichi_forge as ti
from taichi_forge.aot import written_resources as writers
@ti.data_oriented
class Model:
    def __init__(self, n: int):
        self.n = n
        scope = ti.experimental.ResourceScope(self)
        scope.output = scope.field(ti.f32, shape=scope.dim('n'), initialization='zero')
    @ti.kernel
    def read(self, data: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in self.output:
            self.output[i] = data[i]
    @ti.kernel
    def write(self, data: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in self.output:
            data[i] = self.output[i]
    def record_read(self, builder):
        builder.dispatch(self.read, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'data', ti.f32, ndim=1))
    def record_write(self, builder):
        builder.dispatch(self.write, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'data', ti.f32, ndim=1))
    def writes(self, external=False):
        return writers(self, self.record_write if external else self.record_read)
"""
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    original = package.Model(7)
    assert original.writes() == (original.output,)
    with pytest.raises(ValueError, match="explicit writable"):
        original.writes(True)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(11)
        assert model.writes() == (model.output,)
        with pytest.raises(ValueError, match="explicit writable"):
            model.writes(True)


@pytest.mark.parametrize("maximum", [0, 3])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_write_inspection_private_control_and_zero_loop(tmp_path, monkeypatch, maximum):
    source = (
        PACKAGE_WHILE_SOURCE.replace("        for i in range(self.n):\n            self.values[i] = 0\n", "")
        .replace("def record(self, builder, maximum):", "def record(self, builder):")
        .replace("max_iterations=maximum", f"max_iterations={maximum}")
        .replace("self.record(builder, maximum)", "self.record(builder)")
    )
    source += """
    def writes(self):
        return ti.aot.written_resources(self, self.record)
"""
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    original = package.Model(7, 2, 3)
    assert {id(x) for x in original.writes()} == ({id(original.values)} if maximum else set())
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(9, 2, 3)
        assert {id(x) for x in model.writes()} == ({id(model.values)} if maximum else set())


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_write_inspection_returns_model_members_not_abi_views(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, MODEL_TENSOR_SOURCE, initializer="from .model import Model, Rows\n")
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        for dof in (3, 65):
            rows = code.host.Rows(2, dof)
            fields = rows.graph().written_resources()
            assert {id(field) for field in fields} == {id(rows.rows.J), id(rows.rows.fixed), id(rows.rows.result)}
            assert rows.rows.J._forge_package_model_scalar_view not in fields


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_typed_owner_inputs_share_host_and_keep_method_dependencies(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, TYPED_OWNER_INPUT_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["input_families"]["definition_fixture.model.Reader"]) == 2
    assert len([item for item in manifest["entries"].values() if item["definition"].endswith(".common")]) == 1
    assert (tmp_path / "sdk/host/model.py").read_text().count("class Reader:") == 1
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        # Three Reader definitions and the fixture's unimported public entry.
        assert len([name for name in archive.namelist() if name.endswith((".ll", ".bc", ".spv"))]) == 4
    original = package.Model(7, 5)
    original.scalar.values.from_numpy(np.arange(7, dtype=np.float32))
    original.vector.values.from_numpy(np.arange(15, dtype=np.float32).reshape(5, 3))
    original.first.step(2.0)
    original.second.step(2.0)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, 5), code.host.Model(13, 11)
        assert type(a.first) is type(a.second) is type(b.first)
        a.scalar.values.from_numpy(np.arange(7, dtype=np.float32))
        a.vector.values.from_numpy(np.arange(15, dtype=np.float32).reshape(5, 3))
        graph = a.graph()
        prepared = graph.bind(dict(gain=2.0))
        prepared.run()
        for current, expected in ((a.first, original.first), (a.second, original.second)):
            np.testing.assert_array_equal(current.output.to_numpy(), expected.output.to_numpy())
        b.first.step(-2.0)
        np.testing.assert_array_equal(b.first.output.to_numpy(), -6.0)
        original_source = a.first.source
        a.first.source = a.vector
        with pytest.raises(ValueError, match="owner|Owner"):
            a.first.step(1.0)
        with pytest.raises(ValueError, match="owner|Owner"):
            graph.bind(dict(gain=1.0))
        prepared.run()
        a.first.source = original_source
        prepared.close()
        assert a.first.common() == a.second.common() == 7
        with pytest.raises(ValueError, match="unique declared"):
            type(a.first)(a.scalar.values)
    probe = tmp_path / "consume_typed_owners.py"
    probe.write_text(
        """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class RejectSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError('Numerical source unavailable: ' + fullname)
sys.meta_path.insert(0, RejectSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    a, b = code.host.Model(19, 17), code.host.Model(3, 5)
    a.vector.values.from_numpy(np.ones((17, 3), dtype=np.float32))
    a.graph().run(dict(gain=3.0))
    b.graph().run(dict(gain=-2.0))
    np.testing.assert_array_equal(a.first.output.to_numpy(), 9.0)
    np.testing.assert_array_equal(a.second.output.to_numpy(), -6.0)
    np.testing.assert_array_equal(b.first.output.to_numpy(), -6.0)
    np.testing.assert_array_equal(b.second.output.to_numpy(), 10.0)
    assert a.first.output._group_owner is not b.first.output._group_owner
print('TYPED_OWNER_SOURCE_FREE_OK')
""",
        encoding="utf8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TYPED_OWNER_SOURCE_FREE_OK" in result.stdout


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_typed_owner_tuples_do_not_copy_code_for_unused_inputs(tmp_path, monkeypatch):
    source = (
        TYPED_OWNER_INPUT_SOURCE.replace("def __init__(self, source):", "def __init__(self, source, peer):")
        .replace(
            "scope.owner_reference('source', source)",
            "scope.owner_reference('source', source)\n        scope.owner_reference('peer', peer)",
        )
        .replace("Reader(self.scalar)", "Reader(self.scalar, self.vector)")
        .replace(
            "Reader(self.vector)",
            "Reader(self.vector, self.scalar)\n        scope.third = Reader(self.scalar, self.scalar)",
        )
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["input_families"]["definition_fixture.model.Reader"]) == 3
    assert len([item for item in manifest["entries"].values() if item["definition"].endswith(".step")]) == 2
    assert len([item for item in manifest["entries"].values() if item["definition"].endswith(".common")]) == 1
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([name for name in archive.namelist() if name.endswith((".ll", ".bc", ".spv"))]) == 4
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(13, 19)
        model.first.step(2.0)
        model.third.step(2.0)
        np.testing.assert_array_equal(model.first.output.to_numpy(), model.third.output.to_numpy())
        with pytest.raises(ValueError, match="unique declared"):
            type(model.first)(model.vector, model.vector)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_typed_owner_inputs_propagate_through_declared_forwarders(tmp_path, monkeypatch):
    source = (
        TYPED_OWNER_INPUT_SOURCE.replace(
            "class Model:",
            """class Forward:
    def __init__(self, source):
        scope = ti.experimental.ResourceScope(self)
        scope.owner_reference('source', source)
        scope.reader = Reader(source)
@ti.data_oriented
class Model:""",
        )
        .replace("Reader(self.scalar)", "Forward(self.scalar)")
        .replace("Reader(self.vector)", "Forward(self.vector)")
    )
    source = source.replace("builder.dispatch(self.first.step, gain)", "builder.dispatch(self.first.reader.step, gain)")
    source = source.replace(
        "builder.dispatch(self.second.step, gain)", "builder.dispatch(self.second.reader.step, gain)"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert len(manifest["input_families"]["definition_fixture.model.Forward"]) == 2
    assert len(manifest["input_families"]["definition_fixture.model.Reader"]) == 2
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([name for name in archive.namelist() if name.endswith((".ll", ".bc", ".spv"))]) == 4
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(13, 19)
        model.graph().run(dict(gain=2.0))
        np.testing.assert_array_equal(model.first.reader.output.to_numpy(), 6.0)
        np.testing.assert_array_equal(model.second.reader.output.to_numpy(), -10.0)


@pytest.mark.parametrize("static", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_typed_owner_inputs_separate_type_admission_from_identical_code(tmp_path, monkeypatch, static):
    source = TYPED_OWNER_INPUT_SOURCE
    start, end = source.index("@ti.data_oriented\nclass VectorStorage"), source.index("@ti.data_oriented\nclass Reader")
    source = source[:start] + "@ti.data_oriented\nclass VectorStorage(ScalarStorage):\n    pass\n" + source[end:]
    source = source.replace("self.source.read(i)", "self.source.values[i]")
    if static:
        source = source.replace("def __init__(self, source):", "def __init__(self, source, enabled: bool = True):")
        source = source.replace(
            "scope.owner_reference('source', source)",
            "scope.owner_reference('source', source)\n        self.enabled = enabled\n"
            "        scope.static('enabled', choices=(False, True))",
        )
        source = source.replace(
            "self.output[i] = gain * self.source.values[i]",
            "if ti.static(self.enabled):\n                self.output[i] = gain * self.source.values[i]\n"
            "            else:\n                self.output[i] = -1.0",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 47
    assert len(manifest["input_families"]["definition_fixture.model.Reader"]) == 2
    assert manifest["entry_bindings"]
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([name for name in archive.namelist() if name.endswith((".ll", ".bc", ".spv"))]) == 3 + static
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(7, 11)
        model.scalar.values.from_numpy(np.arange(7, dtype=np.float32))
        model.vector.values.from_numpy(np.arange(11, dtype=np.float32))
        graph = model.graph()
        prepared = graph.bind(dict(gain=2.0))
        prepared.run()
        np.testing.assert_array_equal(model.first.output.to_numpy(), 2 * np.arange(7))
        np.testing.assert_array_equal(model.second.output.to_numpy(), 2 * np.arange(11))
        model.first.source = model.vector
        with pytest.raises(ValueError, match="owner|Owner"):
            model.first.step(1.0)
        with pytest.raises(ValueError, match="owner|Owner"):
            graph.bind(dict(gain=1.0))
        prepared.run()
        prepared.close()
        np.testing.assert_array_equal(model.first.output.to_numpy(), 2 * np.arange(7))
        if static:
            model.second.enabled = False
            model.second.step(1.0)
            np.testing.assert_array_equal(model.second.output.to_numpy(), -1.0)
    # No binding may target another binding, silently drop a required format,
    # or shadow native code. Reject before importing any retained Host modules.
    if not static:
        binding = next(iter(manifest["entry_bindings"]))
        for change in ("missing", "recursive", "collision", "version", "null"):
            changed = json.loads(json.dumps(manifest))
            if change == "version":
                changed["version"] = 46
            elif change == "null":
                changed["entry_bindings"][binding] = None
            elif change == "collision":
                changed["entry_bindings"][next(iter(changed["entries"]))] = changed["entry_bindings"].pop(binding)
            else:
                changed["entry_bindings"][binding]["code"] = binding if change == "recursive" else "missing"
            (tmp_path / "sdk/package.json").write_text(json.dumps(changed), encoding="utf8")
            with Runtime(manifest["target"]) as runtime, pytest.raises(ValueError, match="binding"):
                load_package(runtime, tmp_path / "sdk")
        (tmp_path / "sdk/package.json").write_text(json.dumps(manifest), encoding="utf8")
        probe = tmp_path / "consume_shared_owner_code.py"
        probe.write_text(
            """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class RejectSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError('Numerical source unavailable: ' + fullname)
sys.meta_path.insert(0, RejectSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    model = code.host.Model(19, 17)
    model.scalar.values.from_numpy(np.arange(19, dtype=np.float32))
    model.vector.values.from_numpy(np.arange(17, dtype=np.float32))
    prepared = model.graph().bind(dict(gain=3.0))
    prepared.run()
    np.testing.assert_array_equal(model.first.output.to_numpy(), 3 * np.arange(19))
    np.testing.assert_array_equal(model.second.output.to_numpy(), 3 * np.arange(17))
    prepared.close()
print('SHARED_OWNER_CODE_SOURCE_FREE_OK')
""",
            encoding="utf8",
        )
        result = subprocess.run(
            [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
            cwd=tmp_path,
            env=dict(
                os.environ,
                PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
                TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
                TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
            ),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SHARED_OWNER_CODE_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("annotation", "contradicts its declared source type"),
        ("model_selection", "one proven source"),
        ("wrong_kind", "resource kind"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_typed_owner_inputs_reject_unproven_domains(tmp_path, monkeypatch, change, diagnostic):
    source = TYPED_OWNER_INPUT_SOURCE
    if change == "annotation":
        source = source.replace("def __init__(self, source):", "def __init__(self, source: ScalarStorage):")
    elif change == "model_selection":
        source = source.replace("Reader(self.scalar)", "Reader(self.scalar if n > 3 else self.vector)")
    else:
        source = source.replace("Reader(self.scalar)", "Reader(self.scalar.values)")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("collection", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_typed_field_inputs_share_host_and_keep_dynamic_models(tmp_path, monkeypatch, collection):
    source = TYPED_FIELD_INPUT_SOURCE
    if collection:
        source = source.replace(
            "scope.first = Checkpoint(self.storage.scalar)\n        scope.second = Checkpoint(self.storage.record)",
            "sources = (self.storage.scalar, self.storage.record)\n"
            "        self.groups = tuple(Checkpoint(value) for value in sources)\n"
            "        scope.owner_sequence('groups', self.groups)\n"
            "        self.first, self.second = self.groups",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert manifest["version"] == (44 if collection else 43)
    assert len(manifest["input_families"]["definition_fixture.model.Checkpoint"]) == 2
    common = [item for item in manifest["entries"].values() if item["definition"].endswith(".common")]
    assert len(common) == 1
    host = (tmp_path / "sdk/host/model.py").read_text()
    assert host.count("class Checkpoint:") == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(5, 3), code.host.Model(11, 7)
        assert type(a.first) is type(a.second) is type(b.second)
        assert a.first.source is a.storage.scalar
        assert a.second.source is a.storage.record
        for model, n, width in ((a, 5, 3), (b, 11, 7)):
            scalar = np.arange(n, dtype=np.float32)
            record = dict(
                x=scalar * 3,
                v=np.arange(n * width, dtype=np.float32).reshape(n, width),
                id=np.arange(n, dtype=np.int32),
            )
            model.storage.scalar.from_numpy(scalar)
            model.storage.record.from_numpy(record)
            model.first.save()
            model.second.save()
            model.storage.scalar.from_numpy(scalar * 0)
            model.storage.record.x.from_numpy(scalar * 0)
            model.first.restore()
            model.second.restore()
            np.testing.assert_array_equal(model.storage.scalar.to_numpy(), scalar)
            for key, expected in record.items():
                np.testing.assert_array_equal(model.storage.record.to_numpy()[key], expected)
            assert model.first.common() == model.second.common() == 7
        graph = a.graph()
        prepared = graph.bind({})
        original = a.first.source
        a.first.source = b.storage.record
        with pytest.raises(ValueError, match="Borrowed Field"):
            a.first.restore()
        with pytest.raises(ValueError, match="Borrowed Field"):
            graph.bind({})
        prepared.run()
        prepared.close()
        a.first.source = original
        with pytest.raises(ValueError, match="unique declared"):
            type(a.first)(a.storage.record.v)
    probe = tmp_path / "consume_typed_inputs.py"
    probe.write_text(
        """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class RejectSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError('Numerical source is not available: ' + fullname)
sys.meta_path.insert(0, RejectSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    a, b = code.host.Model(7, 5), code.host.Model(13, 9)
    for model, n, width in ((a, 7, 5), (b, 13, 9)):
        values = np.arange(n, dtype=np.float32)
        vectors = np.arange(n * width, dtype=np.float32).reshape(n, width)
        model.storage.scalar.from_numpy(values)
        model.storage.record.v.from_numpy(vectors)
        model.graph().run({})
        model.storage.scalar.from_numpy(values * 0)
        model.storage.record.v.from_numpy(vectors * 0)
        model.first.restore()
        model.second.restore()
        np.testing.assert_array_equal(model.storage.scalar.to_numpy(), values)
        np.testing.assert_array_equal(model.storage.record.v.to_numpy(), vectors)
    np.testing.assert_array_equal(a.storage.scalar.to_numpy(), np.arange(7))
print('TYPED_INPUT_SOURCE_FREE_OK')
""",
        encoding="utf8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TYPED_INPUT_SOURCE_FREE_OK" in result.stdout


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_backend_query_retains_typed_input_format_requirement(tmp_path, monkeypatch):
    source = TYPED_FIELD_INPUT_SOURCE + "\ndef selected_backend():\n    return ti.lang.impl.current_cfg().arch.name\n"
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cpu)
    assert manifest["version"] >= 43
    assert len(manifest["input_families"]["definition_fixture.model.Checkpoint"]) == 2
    assert sys.modules["definition_fixture.model"].constructors == 0
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(5, 3)
        values = np.arange(5, dtype=np.float32)
        model.storage.scalar.from_numpy(values)
        model.first.save()
        model.storage.scalar.from_numpy(values * 0)
        model.first.restore()
        np.testing.assert_array_equal(model.storage.scalar.to_numpy(), values)
        assert code.host.model.selected_backend() == ti.cpu.name


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_typed_inputs_use_declared_tuples_without_cartesian_expansion(tmp_path, monkeypatch):
    source = (
        TYPED_FIELD_INPUT_SOURCE.replace("def __init__(self, values):", "def __init__(self, values, peer):")
        .replace(
            "scope.borrow('source', values, source_scope=True)",
            "scope.borrow('source', values, source_scope=True)\n        scope.borrow('peer', peer, source_scope=True)",
        )
        .replace("Checkpoint(self.storage.scalar)", "Checkpoint(self.storage.scalar, self.storage.record)")
        .replace("Checkpoint(self.storage.record)", "Checkpoint(self.storage.record, self.storage.scalar)")
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert len(manifest["input_families"]["definition_fixture.model.Checkpoint"]) == 2
    assert len([item for item in manifest["entries"].values() if item["definition"].endswith(".common")]) == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(7, 5)
        model.first.save()
        model.second.save()
        with pytest.raises(ValueError, match="unique declared"):
            type(model.first)(model.storage.scalar, model.storage.scalar)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_typed_inputs_reject_unproven_host_selection(tmp_path, monkeypatch):
    source = TYPED_FIELD_INPUT_SOURCE.replace(
        "Checkpoint(self.storage.scalar)", "Checkpoint(self.storage.scalar if n > 3 else self.storage.record)"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match="one proven source"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("reverse", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_host_owner_sequences_reuse_independent_typed_entries(tmp_path, monkeypatch, reverse):
    source = TYPED_FIELD_INPUT_SOURCE.replace(
        "scope.first = Checkpoint(self.storage.scalar)\n        scope.second = Checkpoint(self.storage.record)",
        "sources = (self.storage.scalar, self.storage.record)\n"
        "        self.groups = tuple(Checkpoint(value) for value in sources)\n"
        "        scope.owner_sequence('groups', self.groups)\n"
        "        self.first, self.second = self.groups",
    )
    if reverse:
        source = source.replace(
            "(self.storage.scalar, self.storage.record)", "(self.storage.record, self.storage.scalar)"
        )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 44
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["input_families"]["definition_fixture.model.Checkpoint"]) == 2
    assert len([item for item in manifest["entries"].values() if item["definition"].endswith(".common")]) == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(5, 3), code.host.Model(13, 7)
        for model, n in ((a, 5), (b, 13)):
            expected = np.arange(n, dtype=np.float32)
            model.storage.scalar.from_numpy(expected)
            model.graph().run({})
            model.storage.scalar.from_numpy(expected * 0)
            for group in reversed(model.groups):
                group.restore()
            np.testing.assert_array_equal(model.storage.scalar.to_numpy(), expected)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_host_owner_sequence_presence_does_not_specialize_code(tmp_path, monkeypatch):
    source = TYPED_FIELD_INPUT_SOURCE[: TYPED_FIELD_INPUT_SOURCE.index("@ti.data_oriented\nclass Model:")]
    source += """
@ti.data_oriented
class Model:
    def __init__(self, n: int, width: int):
        scope = ti.experimental.ResourceScope(self)
        scope.storage = Storage(n, width)
        self.groups = (Checkpoint(self.storage.scalar), Checkpoint(self.storage.record)) if n > width else ()
        scope.owner_sequence('groups', self.groups)
"""
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert len(manifest["input_families"]["definition_fixture.model.Checkpoint"]) == 2
    assert not any("domain" in item for item in manifest["entry_families"].values())
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        empty, full = code.host.Model(3, 7), code.host.Model(9, 5)
        assert empty.groups == ()
        assert len(full.groups) == 2
        assert len(empty._forge_package_field_groups) == 0
        for group in full.groups:
            assert group.common() == 7


@pytest.mark.parametrize(
    "change, diagnostic",
    [
        (("sources = (self.storage.scalar, self.storage.record)", "sources = unknown(self.storage)"), "fixed tuple"),
        (
            ("scope.owner_sequence('groups', self.groups)", "scope.owner_sequence('elsewhere', self.groups)"),
            "one named",
        ),
        (
            (
                "scope.owner_sequence('groups', self.groups)",
                "self.groups = ()\n        scope.owner_sequence('groups', self.groups)",
            ),
            "one named",
        ),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_host_owner_sequences_reject_unproven_sources(tmp_path, monkeypatch, change, diagnostic):
    source = TYPED_FIELD_INPUT_SOURCE.replace(
        "scope.first = Checkpoint(self.storage.scalar)\n        scope.second = Checkpoint(self.storage.record)",
        "sources = (self.storage.scalar, self.storage.record)\n"
        "        self.groups = tuple(Checkpoint(value) for value in sources)\n"
        "        scope.owner_sequence('groups', self.groups)\n"
        "        self.first, self.second = self.groups",
    )
    package = _source_package(tmp_path, monkeypatch, source.replace(*change))
    with pytest.raises(ValueError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


INVENTORY_FIELD_INPUT_SOURCE = (
    TYPED_FIELD_INPUT_SOURCE[: TYPED_FIELD_INPUT_SOURCE.index("@ti.data_oriented\nclass Model:")] + """
from operator import attrgetter
def source_names(mode):
    names = ['scalar']
    if mode > 0:
        names.append('record')
    return tuple(names)
def source_fields(storage, mode):
    return tuple((name, attrgetter(name)(storage)) for name in source_names(mode))
@ti.data_oriented
class Model:
    def __init__(self, n: int, width: int, mode: int):
        scope = ti.experimental.ResourceScope(self)
        scope.storage = Storage(n, width)
        self.names, sources = zip(*source_fields(self.storage, mode))
        self.groups = tuple(Checkpoint(value) for value in sources)
        scope.owner_sequence('groups', self.groups)
    def save(self):
        for group in self.groups:
            group.save()
    def restore(self):
        for group in self.groups:
            group.restore()
"""
)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_owner_sequence_reuses_normal_source_inventory(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, INVENTORY_FIELD_INPUT_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["input_families"]["definition_fixture.model.Checkpoint"]) == 2
    assert not any("domain" in item for item in manifest["entry_families"].values())
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(5, 3, 0), code.host.Model(11, 7, 1)
        assert len(a.groups) == 1 and len(b.groups) == 2
        for model, n in ((a, 5), (b, 11)):
            expected = np.arange(n, dtype=np.float32)
            model.storage.scalar.from_numpy(expected)
            model.save()
            model.storage.scalar.from_numpy(expected * 0)
            model.restore()
            np.testing.assert_array_equal(model.storage.scalar.to_numpy(), expected)
    probe = tmp_path / "consume_inventory.py"
    probe.write_text(
        """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class RejectSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError(fullname)
sys.meta_path.insert(0, RejectSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    a, b = code.host.Model(7, 3, 0), code.host.Model(17, 5, 1)
    assert len(a.groups) == 1 and len(b.groups) == 2
    for model, n in ((a, 7), (b, 17)):
        values = np.arange(n, dtype=np.float32)
        model.storage.scalar.from_numpy(values)
        model.save()
        model.storage.scalar.from_numpy(values * 0)
        model.restore()
        np.testing.assert_array_equal(model.storage.scalar.to_numpy(), values)
print('INVENTORY_SOURCE_FREE_OK')
""",
        encoding="utf8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INVENTORY_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize(
    "before, after",
    [
        ("names = ['scalar']", "effect()\n    names = ['scalar']"),
        ("names.append('record')", "names.append(dynamic_name(mode))"),
        ("names.append('record')", "names.append(mode)"),
        ("return tuple(names)", "return names"),
        ("return tuple(names)", "return tuple([names])"),
        ("def source_names(mode):", "def source_names(mode):\n    return source_names(mode)"),
        (
            "self.names, sources = zip(*source_fields(self.storage, mode))",
            "self.names, sources = zip(*source_fields(self.storage, mode))\n        sources = unknown()",
        ),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_owner_inventory_rejects_unproven_sources(tmp_path, monkeypatch, before, after):
    package = _source_package(tmp_path, monkeypatch, INVENTORY_FIELD_INPUT_SOURCE.replace(before, after))
    with pytest.raises(ValueError, match="fixed tuple|unproven|Recursive"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_present_field_inputs_do_not_specialize_on_source_presence(tmp_path, monkeypatch):
    source = (
        INVENTORY_FIELD_INPUT_SOURCE.replace(
            "self.n, self.width = n, width",
            "self.n, self.width = n, width\n        self.has_record = width > 0\n        self.record = None",
        )
        .replace("columns = scope.dim('width')", "scope.static('has_record', choices=(False, True))")
        .replace(
            "scope.record = scope.struct_field({'x': ti.f32, 'v': ti.types.vector(self.width, ti.f32), 'id': ti.i32}, shape=scope.dim('n'), initialization='zero')",
            "if self.has_record:\n            scope.record = scope.struct_field({'x': ti.f32, 'v': ti.types.vector(3, ti.f32), 'id': ti.i32}, shape=scope.dim('n'), initialization='zero')",
        )
    )
    source += """
    def restore_graph(self):
        builder = ti.graph.GraphBuilder()
        for group in self.groups:
            builder.dispatch(group.restore)
        return builder.compile()
"""
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 45
    assert not any("domain" in item for item in manifest["entry_families"].values())
    assert sys.modules["definition_fixture.model"].constructors == 0
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(5, 0, 0), code.host.Model(13, 3, 1)
        assert a.storage.record is None
        expected = np.arange(13, dtype=np.float32)
        b.storage.record.x.from_numpy(expected)
        b.save()
        graph = b.restore_graph()
        prepared = graph.bind({})
        b.storage.record.x.from_numpy(expected * 0)
        b.storage.has_record = False
        assert b.groups[1].common() == 7
        with pytest.raises(ValueError, match="presence"):
            b.groups[1].restore()
        with pytest.raises(ValueError, match="presence"):
            type(b.groups[1])(b.storage.record)
        prepared.run()
        prepared.close()
        b.storage.has_record = True
        np.testing.assert_array_equal(b.storage.record.x.to_numpy(), expected)
        b.groups[1].source = b.groups[1].backup
        with pytest.raises(ValueError, match="declared allocation"):
            b.groups[1].restore()
        b.groups[1].source = b.storage.record
        assert len(a.groups) == 1
    probe = tmp_path / "consume_present.py"
    probe.write_text(
        """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class RejectSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError(fullname)
sys.meta_path.insert(0, RejectSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    a, b = code.host.Model(7, 0, 0), code.host.Model(17, 3, 1)
    assert a.storage.record is None and len(a.groups) == 1
    expected = np.arange(17, dtype=np.float32)
    b.storage.record.x.from_numpy(expected)
    b.save()
    b.storage.record.x.from_numpy(expected * 0)
    b.restore_graph().run({})
    np.testing.assert_array_equal(b.storage.record.x.to_numpy(), expected)
print('PRESENT_INPUT_SOURCE_FREE_OK')
""",
        encoding="utf8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PRESENT_INPUT_SOURCE_FREE_OK" in result.stdout


FIELD_LIKE_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Storage:
    def __init__(self, n: int):
        global constructors
        constructors += 1
        self.n = n
        scope = ti.experimental.ResourceScope(self)
        scope.values = scope.vector_field(self.n, ti.f32, shape=scope.dim('n'), initialization='zero')

@ti.data_oriented
class Model:
    def __init__(self, n: int):
        scope = ti.experimental.ResourceScope(self)
        scope.storage = Storage(n)
        scope.backup = scope.field_like(self.storage.values, layout=ti.Layout.AOS, initialization='zero')

    @ti.kernel
    def fill(self, gain: ti.f32):
        for i in self.storage.values:
            for j in range(self.storage.n):
                self.storage.values[i][j] = gain * i + j

    @ti.kernel
    def save(self):
        for i in ti.grouped(self.storage.values):
            self.backup[i] = self.storage.values[i]

    @ti.kernel
    def restore(self):
        for i in ti.grouped(self.storage.values):
            self.storage.values[i] = self.backup[i]
"""


@pytest.mark.parametrize(
    "kind", ["scalar", "vector", "matrix", "struct", "soa_vector", "soa_matrix"]
)
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_field_like_keeps_dynamic_axes_and_independent_storage(
    tmp_path, monkeypatch, kind
):
    source = FIELD_LIKE_SOURCE
    if kind.startswith("soa_"):
        source = source.replace("layout=ti.Layout.AOS", "layout=ti.Layout.SOA")
        kind = kind.removeprefix("soa_")
    if kind == "scalar":
        source = source.replace(
            "scope.vector_field(self.n, ti.f32,", "scope.field(ti.f32,"
        )
        source = source.replace(
            "            for j in range(self.storage.n):\n                self.storage.values[i][j] = gain * i + j",
            "            self.storage.values[i] = gain * i",
        )
    elif kind == "matrix":
        source = source.replace(
            "scope.vector_field(self.n, ti.f32,",
            "scope.matrix_field(self.n, 2, ti.f32,",
        )
        source = source.replace(
            "self.storage.values[i][j] = gain * i + j",
            "for k in range(2):\n                    self.storage.values[i][j, k] = gain * i + j + k",
        )
    elif kind == "struct":
        source = source.replace("layout=ti.Layout.AOS", "layout=ti.Layout.SOA")
        source = source.replace(
            "scope.vector_field(self.n, ti.f32,",
            "scope.struct_field(dict(value=ti.f32, tag=ti.i32),",
        )
        source = source.replace(
            "            for j in range(self.storage.n):\n                self.storage.values[i][j] = gain * i + j",
            "            self.storage.values[i].value = gain * i\n            self.storage.values[i].tag = i + 1",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(3)
    original.fill(2.0)
    original.save()
    original.fill(9.0)
    original.restore()
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 39
    assert sys.modules["definition_fixture.model"].constructors == 1
    with (
        Runtime(manifest["target"]) as runtime,
        load_package(runtime, tmp_path / "sdk") as code,
    ):
        a, b = code.host.Model(3), code.host.Model(7)
        for model, gain, n in ((a, 2.0, 3), (b, -3.0, 7)):
            model.fill(gain)
            model.save()
            model.fill(9.0)
            model.restore()
            actual = model.storage.values.to_numpy()
            expected = gain * np.arange(n)
            if kind in ("vector", "matrix"):
                expected = expected[:, None] + np.arange(n)
            if kind == "matrix":
                expected = expected[:, :, None] + np.arange(2)
            if kind == "struct":
                np.testing.assert_array_equal(actual["tag"], np.arange(n) + 1)
                actual = actual["value"]
                assert (
                    model.backup.value._group_owner
                    is not model.storage.values.value._group_owner
                )
            else:
                assert (
                    model.backup._group_owner is not model.storage.values._group_owner
                )
            np.testing.assert_array_equal(actual, expected)
        a.fill(13.0)
        b.restore()
        actual = b.storage.values.to_numpy()
        np.testing.assert_array_equal(
            actual["value"] if kind == "struct" else actual, expected
        )
    if kind == "vector":
        probe = tmp_path / "field_like_consumer.py"
        probe.write_text(
            textwrap.dedent("""
            import importlib.abc
            import sys
            import numpy as np
            class NoSource(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                        raise AssertionError(fullname)
            sys.meta_path.insert(0, NoSource())
            from taichi_forge_runtime.aot import Runtime, load_package
            with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
                a, b = code.host.Model(11), code.host.Model(5)
                for model, n in ((a, 11), (b, 5)):
                    model.fill(3.0)
                    model.save()
                    model.fill(-2.0)
                    model.restore()
                    np.testing.assert_array_equal(model.storage.values.to_numpy(), 3 * np.arange(n)[:, None] + np.arange(n))
            print('FIELD_LIKE_SOURCE_FREE_OK')
        """),
            encoding="utf8",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(probe),
                str(tmp_path / "sdk"),
                manifest["target"],
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=90,
            env=dict(
                os.environ,
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=str(
                    Path(__file__).resolve().parents[2] / "packaging/runtime/src"
                ),
                TAICHI_NATIVE_RUNTIME_DIR=str(
                    Path(utils._loaded_native_runtime_path).parent
                ),
                TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
            ),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "FIELD_LIKE_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("failure", ["missing", "forward"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_field_like_rejects_unproved_layouts(tmp_path, monkeypatch, failure):
    source = FIELD_LIKE_SOURCE
    if failure == "missing":
        source = source.replace(
            "scope.field_like(self.storage.values", "scope.field_like(self.unknown"
        )
        message = "declared complete Field"
    elif failure == "forward":
        source = source.replace(
            "scope.field_like(self.storage.values", "scope.field_like(self.backup"
        )
        message = "Cyclic field_like"
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match=message):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize(
    "kind", ["vector", "struct", "struct_helper", "struct_rejected_helper", "invalid_type", "invalid_value"]
)
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_field_like_borrowed_input_preserves_source_dimensions(tmp_path, monkeypatch, kind):
    source = FIELD_LIKE_SOURCE.replace(
        "scope.backup = scope.field_like(self.storage.values, layout=ti.Layout.AOS, initialization='zero')",
        "scope.checkpoint = Checkpoint(self.storage.values)",
    )
    source = source[: source.index("    @ti.kernel\n    def save(self):")]
    source += """
    def save(self):
        self.checkpoint.save()
    def restore(self):
        self.checkpoint.restore()

@ti.data_oriented
class Checkpoint:
    def __init__(self, values):
        scope = ti.experimental.ResourceScope(self)
        scope.borrow('source', values, source_scope=True)
        scope.backup = scope.field_like(self.source, initialization='zero')
    @ti.kernel
    def save(self):
        for i in range(self.source.shape[0]):
            self.backup[i] = self.source[i]
    @ti.kernel
    def restore(self):
        for i in range(self.source.shape[0]):
            self.source[i] = self.backup[i]
    @ti.func
    def finite(self, field: ti.template(), index):
        valid = True
        if ti.static(isinstance(field, ti.MatrixField)):
            if ti.static(field.ndim == 1):
                for j in ti.static(range(field.n)):
                    valid = valid and (-3.4028234663852886e38 <= field[index][j] <= 3.4028234663852886e38)
            else:
                for j, k in ti.static(ti.ndrange(field.n, field.m)):
                    valid = valid and (-3.4028234663852886e38 <= field[index][j, k] <= 3.4028234663852886e38)
        else:
            valid = -3.4028234663852886e38 <= field[index] <= 3.4028234663852886e38
        return valid
    @ti.kernel
    def invalid_count(self) -> ti.i32:
        invalid = 0
        for index in ti.grouped(self.source):
            valid = True
            if ti.static(isinstance(self.source, ti.StructField)):
                for name in ti.static(self.source.keys):
                    field = ti.static(getattr(self.source, name))
                    if ti.static(field.dtype == ti.f32):
                        valid = valid and self.finite(field, index)
            elif ti.static(self.source.dtype == ti.f32):
                valid = self.finite(self.source, index)
            invalid += int(not valid)
        return invalid
"""
    if kind.startswith("struct"):
        source = source.replace(
            "scope.vector_field(self.n, ti.f32,",
            "scope.struct_field({'x': ti.f32, 'v': ti.types.vector(self.n, ti.f32), 'id': ti.i32},",
        )
        source = source.replace("shape=scope.dim('n')", "shape=(scope.dim('n'), 2)")
        source = source.replace("for i in self.storage.values:", "for i in ti.grouped(self.storage.values):")
        source = source.replace(
            "self.storage.values[i][j] = gain * i + j",
            "self.storage.values.v[i][j] = gain * i[0] + i[1] + j\n            self.storage.values.x[i] = gain * i[0] + i[1]",
        )
        source = source.replace(
            "scope.field_like(self.source, initialization='zero')",
            "scope.field_like(self.source, layout=ti.Layout.SOA, initialization='zero')",
        )
        source = source.replace("range(self.source.shape[0])", "ti.grouped(self.source)")
        if kind.endswith("helper"):
            helper = FIELD_HELPER_SOURCE
            if kind == "struct_helper":
                helper = helper.replace(
                    "    if isinstance(field, ti.StructField):",
                    "    if getattr(type(scope), 'whole_struct_borrows', False):\n        borrow(scope, name, field, dimensions=dimensions, source_scope=True)\n        return\n    if isinstance(field, ti.StructField):",
                )
            source += helper
            source = source.replace(
                "scope.borrow('source', values, source_scope=True)", "declare_source(scope, 'source', values)"
            )
    if kind.startswith("invalid_"):
        query = "isinstance(self.source, int)" if kind == "invalid_type" else "isinstance(3, ti.StructField)"
        source = source.replace("isinstance(self.source, ti.StructField)", query)
    package = _source_package(tmp_path, monkeypatch, source, initializer="from .model import Model, Checkpoint\n")
    if kind.startswith("invalid_"):
        with pytest.raises(ValueError, match="Component isinstance"):
            ti.aot.export_package(package, tmp_path / "sdk")
        assert sys.modules["definition_fixture.model"].constructors == 0
        assert not (tmp_path / "sdk").exists()
        return
    if kind == "struct_rejected_helper":
        with pytest.raises(ValueError, match="helper excludes whole StructField"):
            ti.aot.export_package(package, tmp_path / "sdk")
        assert sys.modules["definition_fixture.model"].constructors == 0
        assert not (tmp_path / "sdk").exists()
        return
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    if kind.startswith("struct"):
        check = next(entry for entry in manifest["entries"].values() if entry["definition"].endswith(".invalid_count"))
        assert not any(capture["path"][-1] == "id" for capture in check["captures"])
    assert sys.modules["definition_fixture.model"].constructors == 0
    original = package.Model(6)
    original.fill(4.0)
    original.save()
    original.fill(-2.0)
    original.restore()
    assert original.checkpoint.invalid_count() == 0
    with (
        Runtime(manifest["target"]) as runtime,
        load_package(runtime, tmp_path / "sdk") as code,
    ):
        model = code.host.Model(6)
        model.fill(4.0)
        model.save()
        model.fill(-2.0)
        model.restore()
        actual = model.storage.values.to_numpy()
        reference = original.storage.values.to_numpy()
        assert model.checkpoint.invalid_count() == 0
        if kind.startswith("struct"):
            assert manifest["version"] == 41
            for name in actual:
                np.testing.assert_array_equal(actual[name], reference[name])
            other = code.host.Model(9)
            other.fill(-3.0)
            other.save()
            other.fill(7.0)
            other.restore()
            expected = -3 * np.arange(9)[:, None] + np.arange(2)
            np.testing.assert_array_equal(other.storage.values.x.to_numpy(), expected)
            np.testing.assert_array_equal(other.storage.values.v.to_numpy(), expected[:, :, None] + np.arange(9))
            np.testing.assert_array_equal(model.storage.values.x.to_numpy(), actual["x"])
            invalid = other.storage.values.v.to_numpy()
            invalid[-1, -1, -1] = np.nan
            other.storage.values.v.from_numpy(invalid)
            assert other.checkpoint.invalid_count() == 1
            other.restore()
            assert other.checkpoint.invalid_count() == 0
            with pytest.raises(ValueError, match="declared allocation member"):
                code.host.Checkpoint(model.storage.values.v)
        else:
            np.testing.assert_array_equal(actual, reference)
            invalid = model.storage.values.to_numpy()
            invalid[-1, -1] = np.inf
            model.storage.values.from_numpy(invalid)
            assert model.checkpoint.invalid_count() == 1
            model.restore()
            assert model.checkpoint.invalid_count() == 0
    if kind == "struct":
        probe = tmp_path / "consume_struct_checkpoint.py"
        probe.write_text(
            """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class RejectSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError('Numerical source is not available: ' + fullname)
sys.meta_path.insert(0, RejectSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    a, b = code.host.Model(5), code.host.Model(11)
    for model, n, gain in ((a, 5, 2), (b, 11, -3)):
        model.fill(gain)
        model.save()
        model.fill(9)
        model.restore()
        expected = gain * np.arange(n)[:, None] + np.arange(2)
        np.testing.assert_array_equal(model.storage.values.x.to_numpy(), expected)
        np.testing.assert_array_equal(model.storage.values.v.to_numpy(), expected[:, :, None] + np.arange(n))
        assert model.checkpoint.invalid_count() == 0
        invalid = model.storage.values.v.to_numpy()
        invalid[-1, -1, -1] = np.nan
        model.storage.values.v.from_numpy(invalid)
        assert model.checkpoint.invalid_count() == 1
        model.restore()
    np.testing.assert_array_equal(a.storage.values.x.to_numpy(), 2 * np.arange(5)[:, None] + np.arange(2))
print('STRUCT_CHECKPOINT_SOURCE_FREE_OK')
""",
            encoding="utf8",
        )
        env = dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        )
        result = subprocess.run(
            [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "STRUCT_CHECKPOINT_SOURCE_FREE_OK" in result.stdout


CONDITIONAL_CHILD_SOURCE = (
    DECLARED_CHILD_SOURCE[
        : DECLARED_CHILD_SOURCE.index("@ti.data_oriented\nclass Model:")
    ]
    + """
@ti.data_oriented
class Model:
    def __init__(self, n: int, enabled: bool, first: bool = False):
        self.n, self.enabled = n, enabled
        scope = ti.experimental.ResourceScope(self)
        scope.static('enabled', choices=(False, True))
        scope.values = scope.field(ti.f32, shape=scope.dim('n'), initialization='zero')
        self.child = None
        if self.enabled:
            scope.child = Group(n, n + 1, first, False)

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.values:
            if ti.static(self.enabled):
                self.values[i] = self.child.left.values[i] + gain * self.child.left.bias()
            else:
                self.values[i] = -gain * i

    @ti.kernel
    def common(self):
        for i in self.values:
            self.values[i] = 2 * i

    def graph(self):
        builder = ti.graph.GraphBuilder()
        gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32)
        builder.dispatch(self.step, gain)
        return builder.compile()
"""
)


@pytest.mark.parametrize("absence", ["initial", "record"])
@pytest.mark.parametrize("reference", [False, True, "field", "model_field"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_conditional_children_keep_local_state_and_code_domains(tmp_path, monkeypatch, absence, reference):
    source = CONDITIONAL_CHILD_SOURCE
    if absence == "record":
        source = source.replace("        self.child = None\n", "").replace(
            "            scope.child = Group(n, n + 1, first, False)",
            "            scope.child = Group(n, n + 1, first, False)\n"
            "        else:\n            scope.record('child', None, optional=True)",
        )
    if reference is True:
        source = source.replace(
            "self.child.left.values[i] + gain * self.child.left.bias()",
            "self.selected.values[i] + gain * self.selected.bias()",
        )
        source = source.replace(
            "            scope.child = Group(n, n + 1, first, False)",
            "            scope.child = Group(n, n + 1, first, False)\n"
            "            scope.owner_reference('selected', self.child.left)",
        )
    elif reference in ("field", "model_field"):
        source = source.replace("self.child.left.values[i] + gain", "self.alias[i] + gain")
        source = source.replace(
            "            scope.child = Group(n, n + 1, first, False)",
            "            scope.child = Group(n, n + 1, first, False)\n"
            "            scope.borrow('alias', self.child.left.values, source_scope=True, dimensions={'n': scope.dim('n')})",
        )
        if reference == "model_field":
            source = source.replace(
                "scope.values = scope.field(ti.f32, shape=rows)",
                "scope.values = scope.vector_field(self.n, ti.f32, shape=rows)",
            )
            source = source.replace(
                "for i in self.values:\n            self.values[i] = gain * i",
                "for i, j in ti.ndrange(self.n, self.n):\n            self.values[i][j] = gain * i + j",
            )
            source = source.replace("self.alias[i] + gain", "self.alias[i][0] + gain")
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(7, True, True)
    original.child.fill(3.0)
    original.step(2.0)
    expected = original.values.to_numpy()
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == (38 if reference is True else 37)
    assert sys.modules["definition_fixture.model"].constructors == 2
    assert len(manifest["entries"]) == 6  # Three step paths, fill, common, unimported entry.
    (family,) = manifest["entry_families"].values()
    assert len(family["variants"]) == 3
    assert list(family["conditions"].values()) == [{"enabled": True}]
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b, c = code.host.Model(7, True, True), code.host.Model(19, True), code.host.Model(5, False)
        assert c.child is None
        a.child.fill(3.0)
        b.child.fill(-2.0)
        a.step(2.0)
        b.step(4.0)
        c.step(2.0)
        np.testing.assert_array_equal(a.values.to_numpy(), expected)
        np.testing.assert_array_equal(b.values.to_numpy(), -2 * np.arange(19) + 8)
        np.testing.assert_array_equal(c.values.to_numpy(), -2 * np.arange(5))
        assert a.child.left.values._group_owner is not b.child.left.values._group_owner
        if reference is True:
            saved_reference = a.selected
            a.selected = b.selected
            with pytest.raises(ValueError, match="Owner reference"):
                a.step(2.0)
            a.selected = saved_reference
        elif reference in ("field", "model_field"):
            saved_alias = a.alias
            a.alias = b.alias
            with pytest.raises(ValueError, match="Borrowed Field"):
                a.step(2.0)
            a.alias = saved_alias
        prepared = a.graph().bind(dict(gain=2.0))
        saved = a.child
        a.child, a.enabled = None, False
        prepared.run()
        np.testing.assert_array_equal(a.values.to_numpy(), expected)
        a.step(2.0)
        np.testing.assert_array_equal(a.values.to_numpy(), -2 * np.arange(7))
        a.enabled = True
        with pytest.raises((ValueError, AttributeError)):
            a.step(2.0)
        a.common()
        np.testing.assert_array_equal(a.values.to_numpy(), 2 * np.arange(7))
        a.child = saved
        prepared.close()
    probe = tmp_path / "consume_conditional_child.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(17, True, True), code.host.Model(33, False)
            a.child.fill(3.0)
            a.step(2.0)
            b.step(2.0)
            np.testing.assert_array_equal(a.values.to_numpy(), 3 * np.arange(17) + 14)
            np.testing.assert_array_equal(b.values.to_numpy(), -2 * np.arange(33))
            assert b.child is None
        print('CONDITIONAL_CHILD_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONDITIONAL_CHILD_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("expression", ["self.child.left.values[i]", "self.child.left.bias()"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_conditional_child_capture_requires_parent_guard(tmp_path, monkeypatch, expression):
    source = CONDITIONAL_CHILD_SOURCE.replace("self.values[i] = 2 * i", f"self.values[i] = {expression}")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="requires its declared static guard"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0


@pytest.mark.parametrize(
    "reference,diagnostic",
    [
        ("self.child.missing", "resolved declared owner path"),
        ("self.child.left.values", "object source"),
        ("self.unknown", "named constructor parameter or declared owner path"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_owner_alias_rejects_unproven_sources(tmp_path, monkeypatch, reference, diagnostic):
    source = CONDITIONAL_CHILD_SOURCE.replace(
        "            scope.child = Group(n, n + 1, first, False)",
        "            scope.child = Group(n, n + 1, first, False)\n"
        f"            scope.owner_reference('selected', {reference})",
    )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0


@pytest.mark.parametrize("branch", ["outside", "opposite"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_optional_owner_borrow_requires_source_presence(tmp_path, monkeypatch, branch):
    call = "scope.borrow('alias', self.child.left.values, source_scope=True)"
    suffix = f"\n        {call}" if branch == "outside" else f"\n        else:\n            {call}"
    source = CONDITIONAL_CHILD_SOURCE.replace(
        "            scope.child = Group(n, n + 1, first, False)",
        "            scope.child = Group(n, n + 1, first, False)" + suffix,
    )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match="original allocation presence guards"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("scope.left = Storage(n, first)", "scope.left = Group(n, n, first, first)", "cyclic"),
        ("scope.left = Storage(n, first)", "scope.left = list()", "supported resource declaration"),
        ("second: bool, fail=False):", "second: bool, fail=False, Storage=None):", "unshadowed"),
        ("scope.left = Storage(n, first)", "if first:\n            scope.left = Storage(n, first)", "unconditional"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_declared_children_reject_unproven_construction(tmp_path, monkeypatch, before, after, diagnostic):
    package = _source_package(tmp_path, monkeypatch, DECLARED_CHILD_SOURCE.replace(before, after))
    with pytest.raises(ValueError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_child_capture_checks_original_conditional_layout(tmp_path, monkeypatch):
    source = DECLARED_CHILD_SOURCE.replace(
        "self.tree = builder.finalize()",
        "if self.first:\n            scope.extra = scope.field(ti.f32, shape=rows)\n        self.tree = builder.finalize()",
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a = code.host.Model(7, 11, True, False)
        a.step(3.0)
        a.child.left.first = False
        with pytest.raises(ValueError, match="Conditional layout configuration changed"):
            a.step(3.0)


BORROWED_FIELDS_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int):
        global constructors
        constructors += 1
        self.n, self.m = n, m
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        rows, cols = scope.dim('n'), scope.dim('m')
        scope.values = scope.field(ti.f32, shape=rows)
        scope.sibling = scope.vector_field(3, ti.f32, shape=cols + 1)
        self.tree = builder.finalize()
        scope.borrow('alias', self.values, source_scope=True, dimensions={'n': cols})

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.values:
            self.values[i] = gain * i
            self.alias[i] += 2

    @ti.kernel
    def clear(self):
        for i in self.values:
            self.values[i] = 0

    def graph(self):
        builder = ti.graph.GraphBuilder()
        gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32)
        builder.dispatch(self.step, gain)
        return builder.compile()
"""


def _borrowed_alias_source(source, struct_member, alias_form):
    path = "('values', 'x')" if struct_member else "('values',)"
    source = source.replace(
        "constructors = 0",
        f"from types import MappingProxyType\nALIASES = MappingProxyType({{'alias': {path}}})\nconstructors = 0",
    )
    original_borrow = next(line.strip() for line in source.splitlines() if "scope.borrow(" in line)
    access = "getattr(getattr(self, group), member)" if struct_member else "getattr(self, member)"
    target = "name, (group, member)" if struct_member else "name, (member,)"
    relation = ", dimensions={'n': cols}" if alias_form == "mapping" else ""
    loop = f"for {target} in ALIASES.items():\n            scope.borrow(name, {access}, source_scope=True{relation})"
    source = source.replace(original_borrow, loop if alias_form == "mapping" else "self._bind_aliases()")
    if alias_form == "method":
        start = source.index("        builder = ti.FieldsBuilder()")
        end = source.index("        self._bind_aliases()", start)
        allocation = source[start:end]
        source = source[:start] + "        self._allocate()\n" + source[end:]
        source = source.replace(
            "    @ti.kernel\n    def step",
            "    def _bind_aliases(self):\n        scope = ti.experimental.ResourceScope(self)\n        "
            + loop
            + "\n\n    def _allocate(self):\n" + allocation + "\n    @ti.kernel\n    def step",
        )
    return source


@pytest.mark.parametrize("struct_member", [False, True])
@pytest.mark.parametrize("alias_form", ["direct", "mapping", "method", "relations"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_borrowed_fields_preserve_source_layout_and_instance_binding(
    tmp_path, monkeypatch, struct_member, alias_form
):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    source = BORROWED_FIELDS_SOURCE
    if struct_member:
        source = source.replace(
            "scope.values = scope.field(ti.f32, shape=rows)",
            "scope.values = scope.struct_field({'x': ti.f32, 'v': ti.types.vector(2, ti.f32)}, shape=rows)",
        )
        source = source.replace("self.values, source_scope", "self.values.x, source_scope")
        source = source.replace("self.values[i]", "self.values.x[i]")
        source = source.replace("in self.values:", "in self.values.x:")
    if alias_form == "relations":
        source += "\ndef relations(scope):\n    return {name: scope.dim('m') for name in ('n',)}\n"
        source = source.replace(
            "        scope.borrow('alias', self.values", "        dims = relations(scope)\n        scope.borrow('alias', self.values"
        ).replace("dimensions={'n': cols}", "dimensions=dims")
        # Use an ordinary helper-returned mapping, retaining the actual local
        # dictionary and Runtime's per-binding relation check.
    elif alias_form != "direct":
        source = _borrowed_alias_source(source, struct_member, alias_form)
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(7, 7)
    original.step(3.0)
    original_field = original.values.x if struct_member else original.values
    expected = original_field.to_numpy()
    root = ti.lang.impl.get_runtime().prog.get_snode_root(original_field.snode.ptr.get_snode_tree_id())
    layout = _compiled_nodes(root)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == (20 if alias_form in ("direct", "relations") else 36)
    assert sys.modules["definition_fixture.model"].constructors == 1
    assert len(manifest["field_groups"]) == 1
    assert len(manifest["entries"]) == 3  # step, clear and the unimported public definition
    clear = next(item for item in manifest["entries"].values() if item["definition"].endswith("Model.clear"))
    assert "admission" not in clear  # Unused aliases cannot add per-entry guards or code.
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, 7), code.host.Model(19, 19)
        assert a.alias is (a.values.x if struct_member else a.values)
        assert a.alias._group_owner.layout["nodes"] == layout
        assert a.alias._group_owner is a.sibling._group_owner
        assert a.alias._group_owner is not b.alias._group_owner
        a.step(3.0)
        np.testing.assert_array_equal(a.alias.to_numpy(), expected)
        b.step(-2.0)
        np.testing.assert_array_equal(b.alias.to_numpy(), -2 * np.arange(19) + 2)
        registered = set(runtime._resources)
        if alias_form != "method":
            with pytest.raises(ValueError, match="constraint|Constraint"):
                code.host.Model(5, 6)
        assert runtime._resources == registered
        a.m = 6
        if alias_form != "method":
            with pytest.raises(ValueError, match="constraint|Constraint"):
                a.step(3.0)
        a.clear()
        a.m = 7
        plan = a.graph()
        prepared = plan.bind(dict(gain=4.0))
        saved = a.alias
        a.alias = b.alias
        with pytest.raises(ValueError, match="Borrowed Field"):
            a.step(1.0)
        with pytest.raises(ValueError, match="Borrowed Field"):
            plan.bind(dict(gain=1.0))
        prepared.run()
        np.testing.assert_array_equal(saved.to_numpy(), 4 * np.arange(7) + 2)
        np.testing.assert_array_equal(b.alias.to_numpy(), -2 * np.arange(19) + 2)
        a.alias = saved
        a.tree.destroy()
        with pytest.raises((ValueError, RuntimeError)):
            a.step(1.0)
        b.step(1.0)
        np.testing.assert_array_equal(b.alias.to_numpy(), np.arange(19) + 2)
        prepared.close()
        if alias_form in ("mapping", "method"):
            from types import MappingProxyType

            original_mapping = (
                code.host.Model._bind_aliases.__globals__
                if alias_form == "method"
                else code.host.Model.__init__.__wrapped__.__globals__
            )
            # Host-side edits cannot silently change or remove declared aliases.
            saved_mapping = original_mapping["ALIASES"]
            original_mapping["ALIASES"] = MappingProxyType({})
            resources = set(runtime._resources)
            with pytest.raises(ValueError, match="Alias mapping differs"):
                code.host.Model(3, 3)
            assert runtime._resources == resources
            original_mapping["ALIASES"] = saved_mapping
    probe = tmp_path / "consume_borrows.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3, 3), code.host.Model(13, 13)
            a.step(7.0)
            b.step(-1.0)
            np.testing.assert_array_equal(a.alias.to_numpy(), 7 * np.arange(3) + 2)
            np.testing.assert_array_equal(b.alias.to_numpy(), -np.arange(13) + 2)
            assert a.alias._group_owner is not b.alias._group_owner
        print('BORROW_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BORROW_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("mutation", ["dims['n'] = rows", "dims.clear()"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_borrowed_relation_alias_checks_actual_host_mapping(tmp_path, monkeypatch, mutation):
    source = BORROWED_FIELDS_SOURCE.replace("n: int, m: int):", "n: int, m: int, corrupt=False):")
    source = source.replace(
        "        scope.borrow('alias', self.values",
        "        dims = {'n': cols}\n        if corrupt:\n            " + mutation
        + "\n        scope.borrow('alias', self.values",
    ).replace("dimensions={'n': cols}", "dimensions=dims")
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(7, 7)
        model.step(3.0)
        np.testing.assert_array_equal(model.alias.to_numpy(), 3 * np.arange(7) + 2)
        resources = set(runtime._resources)
        with pytest.raises(ValueError, match="Borrowed (dimensions|target relation)"):
            code.host.Model(7, 7, True)
        assert runtime._resources == resources
        model.clear()


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("self.values, source_scope", "unknown, source_scope", "unmodified constructor parameter"),
        ("source_scope=True", "source_scope=False", "source_scope=True"),
        ("source_scope=True", "source_scope=True, external=True", "source_scope=True"),
        ("{'n': cols}", "{'unknown': cols}", "allocation namespace"),
        ("{'n': cols}", "{'n': m}", "declaration namespace"),
        ("scope.borrow('alias'", "scope.borrow('values'", "replace an existing"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_borrowed_fields_reject_unproven_sources(tmp_path, monkeypatch, before, after, diagnostic):
    package = _source_package(tmp_path, monkeypatch, BORROWED_FIELDS_SOURCE.replace(before, after))
    with pytest.raises((ti.aot.PackageExportError, ValueError), match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


RECORD_RESOURCES_SOURCE = """
from dataclasses import dataclass
import taichi_forge as ti
constructors = 0
@dataclass(frozen=True)
class Layout:
    n: int
    gain: float
    mode: bool = False
    ignored: bool = False
@ti.data_oriented
class Model:
    def __init__(self, capacity: int, layout):
        global constructors
        constructors += 1
        self.capacity = capacity
        scope = ti.experimental.ResourceScope(self)
        parent = scope.dim('capacity')
        record = scope.record('layout', layout)
        count = record.dim('n')
        outer = record.parent_dim('capacity')
        record.require_le('fits', count, outer, when='n')
        record.value('gain', ti.f32)
        record.static('mode', choices=(False, True))
        record.static('ignored', choices=(False, True))
        scope.values = scope.field(ti.f32, shape=parent)

    @ti.kernel
    def step(self):
        for i in range(self.layout.n):
            gain = self.layout.gain
            if ti.static(self.layout.mode):
                gain *= 2
            self.values[i] = i + gain

    @ti.kernel
    def clear(self):
        for i in self.values:
            self.values[i] = -1

    def graph(self):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.step)
        return builder.compile()
"""


RECORD_HELPER_SOURCE = """
effects = []
def declare_layout(scope, layout, capacity, dtype, *, bounded=True):
    scope.capacity = capacity
    count = scope.dim('capacity')
    record = scope.record('layout', layout)
    for member, parent_name in (('n', 'capacity'),):
        active = record.dim(member)
        parent = record.parent_dim(parent_name)
        if bounded:
            record.require_le(member + '_fits', active, parent, when=member)
    record.value('gain', dtype)
    for flag in ('mode', 'ignored'):
        record.static(flag, choices=(False, True))
    return count
"""


def _record_helper_fixture(tmp_path, monkeypatch, helper=RECORD_HELPER_SOURCE):
    (tmp_path / "declaration_helpers.py").write_text(textwrap.dedent(helper), encoding="utf8")
    monkeypatch.delitem(sys.modules, "declaration_helpers", raising=False)
    source = RECORD_RESOURCES_SOURCE.replace(
        "import taichi_forge as ti\n",
        "import taichi_forge as ti\nfrom declaration_helpers import declare_layout\nScalar = ti.f32\n",
    )
    start, end = source.index("        parent = scope.dim("), source.index("        scope.values =")
    return source[:start] + "        parent = declare_layout(scope, layout, capacity, Scalar)\n" + source[end:]


@pytest.mark.parametrize("forwarded", [False, True, "attribute", "nested"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_records_preserve_paths_static_projection_and_parent_constraints(tmp_path, monkeypatch, forwarded):
    source = _record_helper_fixture(tmp_path, monkeypatch) if forwarded else RECORD_RESOURCES_SOURCE
    if forwarded == "nested":
        helper = RECORD_HELPER_SOURCE.replace(
            "def declare_layout(scope, layout, capacity, dtype, *, bounded=True):",
            "def declare_record(scope, layout, capacity, dtype, *, bounded=True):",
        )
        helper += """
def declare_layout(scope, layout, capacity, dtype, *, bounded=True):
    return declare_record(scope, layout, capacity, dtype, bounded=bounded)
"""
        helper = helper.replace(
            "    return count", "    sizes = {name: scope.dim(name) for name in ('capacity',)}\n    return count"
        )
        source = _record_helper_fixture(tmp_path, monkeypatch, helper)
    if forwarded == "attribute":
        source = source.replace(
            "        self.capacity = capacity", "        self.input_layout = layout\n        self.capacity = capacity"
        )
        source = source.replace(
            "declare_layout(scope, layout, capacity, Scalar)",
            "declare_layout(scope, self.input_layout, self.capacity, Scalar)",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 19
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["entries"]) == 4  # two step variants, common clear, unimported public entry
    family = next(iter(manifest["entry_families"].values()))
    assert list(family["paths"].values()) == [["layout", "mode"]]
    if forwarded:
        assert sys.modules["declaration_helpers"].effects == []
        assert manifest["declaration_factories"][0]["kind"] == "declaration_helper"
        if forwarded == "nested":
            dependency = manifest["declaration_factories"][0]["dependencies"][0]
            assert dependency["definition"] == "declaration_helpers.declare_record"
            assert len(dependency["sha256"]) == 64
        assert "from declaration_helpers" not in (tmp_path / "sdk/host/model.py").read_text()
    original = package.Model(7, sys.modules["definition_fixture.model"].Layout(5, 2.5, True))
    original.clear()
    original.step()
    np.testing.assert_array_equal(original.values.to_numpy(), [5, 6, 7, 8, 9, -1, -1])
    consumer = tmp_path / "consume_records.py"
    consumer.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture', 'declaration_helpers'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as library:
            Model = library.host.Model
            Layout = sys.modules[Model.__module__].Layout
            layout = Layout(5, 2.5, True)
            a = Model(7, layout)
            b = Model(23, Layout(23, 1.5, False, True))
            assert a.layout is layout
            a.clear(); b.clear()
            a.step(); b.step()
            np.testing.assert_array_equal(a.values.to_numpy(), [5,6,7,8,9,-1,-1])
            np.testing.assert_array_equal(b.values.to_numpy(), np.arange(23) + 1.5)
            graph = a.graph()
            prepared = graph.bind({})
            a.layout = Layout(8, 99., True)
            for operation in (a.step, lambda: graph.bind({})):
                try:
                    operation()
                except ValueError as error:
                    assert 'Dimension constraint' in str(error)
                else:
                    raise AssertionError('Out of bounds record dimension accepted')
            a.clear()  # This entry does not use layout.n or its constraint.
            prepared.run()
            np.testing.assert_array_equal(a.values.to_numpy(), [5,6,7,8,9,-1,-1])
            prepared.close()
            existing = set(runtime._resources)
            try:
                Model(3, Layout(4, 1.))
            except ValueError as error:
                assert 'Dimension constraint' in str(error)
            else:
                raise AssertionError('Invalid constructor relation accepted')
            assert set(runtime._resources) == existing
            a.layout = Layout(2, 4., False, True)
            a.step()
            np.testing.assert_array_equal(a.values.to_numpy()[:2], [4, 5])
            np.testing.assert_array_equal(b.values.to_numpy(), np.arange(23) + 1.5)
        print('PACKAGE_RECORDS_OK')
    """),
        encoding="utf8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(consumer), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PACKAGE_RECORDS_OK" in result.stdout


@pytest.mark.parametrize("change", ["effect", "condition", "loop", "call", "shadow"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_declaration_helpers_reject_unproven_execution(tmp_path, monkeypatch, change):
    helper = RECORD_HELPER_SOURCE
    if change == "effect":
        helper = helper.replace("    count =", "    effects.append(1)\n    count =")
    elif change == "condition":
        helper = helper.replace("if bounded:", "if layout.n:")
    elif change == "loop":
        helper = helper.replace("('mode', 'ignored')", "layout.names")
    elif change == "call":
        helper = helper.replace("return count", "return layout.compute(count)")
    source = _record_helper_fixture(tmp_path, monkeypatch, helper)
    if change == "shadow":
        source = source.replace(
            "        parent = declare_layout",
            "        declare_layout = layout.compute\n        parent = declare_layout",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="factory forwarding is unproven|binding is shadowed"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["declaration_helpers"].effects == []
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize(
    "change", ["recursive", "effect", "local_shadow", "argument_shadow", "argument_order", "model_dictionary"]
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_nested_declaration_helpers_reject_unproven_dependencies(tmp_path, monkeypatch, change):
    helper = RECORD_HELPER_SOURCE.replace("def declare_layout(", "def declare_record(")
    helper += """
def declare_layout(scope, layout, capacity, dtype, *, bounded=True):
    return declare_record(scope, layout, capacity, dtype, bounded=bounded)
"""
    if change == "recursive":
        helper = helper.replace(
            "    count = scope.dim('capacity')", "    count = declare_layout(scope, layout, capacity, dtype)"
        )
    elif change == "effect":
        helper = helper.replace("    count =", "    effects.append(1)\n    count =")
    elif change == "local_shadow":
        helper += "    declare_record = layout\n"
    elif change == "argument_shadow":
        helper = helper.replace(
            "def declare_layout(scope, layout, capacity, dtype, *, bounded=True):",
            "def declare_layout(scope, layout, capacity, dtype, *, bounded=True, declare_record=None):",
        )
    elif change == "argument_order":
        helper = helper.replace(
            "return declare_record(scope, layout, capacity, dtype, bounded=bounded)",
            "return scope.require_le('bound', layout.n, declare_record(scope, layout, capacity, dtype), when='capacity')",
        )
    else:
        helper = helper.replace(
            "    count =", "    sizes = {name: scope.dim(name) for name in layout.names}\n    count ="
        )
    source = _record_helper_fixture(tmp_path, monkeypatch, helper)
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="factory forwarding is unproven"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["declaration_helpers"].effects == []
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("attribute", [False, True])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_declaration_helper_preserves_unused_argument_evaluation(tmp_path, monkeypatch, attribute):
    helper = RECORD_HELPER_SOURCE.replace("bounded=True", "bounded=True, unused=None")
    missing = "layout.missing_input" if attribute else "missing_input"
    error = AttributeError if attribute else NameError
    source = _record_helper_fixture(tmp_path, monkeypatch, helper).replace(
        "scope, layout, capacity, Scalar)", f"scope, layout, capacity, Scalar, unused={missing})"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    Layout = sys.modules["definition_fixture.model"].Layout
    with pytest.raises(error, match="missing_input"):
        package.Model(7, Layout(5, 2.5))
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        registered = set(runtime._resources)
        with pytest.raises(error, match="missing_input"):
            library.host.Model(7, Layout(5, 2.5))
        assert set(runtime._resources) == registered


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_declaration_helper_reads_attribute_arguments_once_in_order(tmp_path, monkeypatch):
    helper = RECORD_HELPER_SOURCE.replace("bounded=True", "bounded=True, first=None, second=None")
    source = _record_helper_fixture(tmp_path, monkeypatch, helper).replace(
        "capacity: int, layout):", "capacity: int, layout, inputs):"
    ).replace("scope, layout, capacity, Scalar)",
              "scope, layout, capacity, Scalar, first=inputs.first, second=inputs.second)")
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    Layout = sys.modules["definition_fixture.model"].Layout

    class Inputs:
        def __init__(self):
            self.reads = []

        @property
        def first(self):
            self.reads.append("first")
            return 1

        @property
        def second(self):
            self.reads.append("second")
            return 2

    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        for constructor in (package.Model, library.host.Model):
            inputs = Inputs()
            constructor(7, Layout(5, 2.5), inputs)
            assert inputs.reads == ["first", "second"]


@pytest.mark.parametrize(
    "change, message",
    [
        ("alias", "Record declaration requires"),
        ("allocation", "cannot allocate Fields"),
        ("unknown", "requires declared dimensions|declared dimension algebra"),
        ("parent", "parent_dim requires"),
        ("scope", "cannot escape their ResourceScope"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_records_reject_unsafe_declaration_paths(tmp_path, monkeypatch, change, message):
    source = RECORD_RESOURCES_SOURCE
    if change == "alias":
        source = source.replace("        count =", "        record = layout\n        count =")
    elif change == "allocation":
        source = source.replace(
            "        count =", "        record.extra = record.field(ti.i32, shape=())\n        count ="
        )
    elif change == "unknown":
        source = source.replace("count, outer, when=", "count, layout.n, when=")
    elif change == "scope":
        source = source.replace("shape=parent", "shape=outer")
    else:
        source = source.replace("scope.dim('capacity')", "scope.parent_dim('capacity')")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match=message):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    assert sys.modules["definition_fixture.model"].constructors == 0


STRUCT_FIELDS_SOURCE = """
import taichi_forge as ti
constructors = 0
Particle = ti.types.struct(mass=ti.f32, position=ti.types.vector(3, ti.f32), tag=ti.i32)
@ti.data_oriented
class Model:
    def __init__(self, n: int):
        global constructors
        constructors += 1
        self.n = n
        scope = ti.experimental.ResourceScope(self)
        count = scope.dim('n')
        scope.particles = scope.struct_field(Particle.members, shape=count, layout=ti.Layout.LAYOUT)
        scope.total = scope.field(ti.f32, shape=())

    @ti.kernel
    def initialize(self, gain: ti.f32):
        for i in self.particles:
            self.particles[i] = Particle(gain * (i + 1), ti.Vector([i, i+1, i+2]), i)

    @ti.kernel
    def step(self, delta: ti.f32):
        for i in self.particles:
            self.particles.mass[i] += delta
            self.particles.position[i] += ti.Vector([delta, 0.0, 0.0])
        self.total[None] = self.particles.mass[0]

    def graph(self):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.step, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'delta', ti.f32))
        return builder.compile()
"""


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_struct_constructor_type_is_a_compiler_dependency(tmp_path, monkeypatch):
    from taichi_forge.aot._component import Component
    from taichi_forge_runtime.aot import Scalar

    _source_package(
        tmp_path,
        monkeypatch,
        """
import taichi_forge as ti
Packet = ti.types.struct(x=ti.i32)
class Model:
    pass
@ti.kernel
def convert(value: ti.f32) -> ti.f32:
    return Packet(value).x
""",
    )
    source = sys.modules["definition_fixture.model"]
    environment = Component()
    first = environment.compile(source.convert, 1.75)
    source.Packet = ti.types.struct(x=ti.f32)
    second = environment.compile(source.convert, 1.75)
    assert first.semantic_key != second.semantic_key
    assert environment.compile(source.convert, 9.25) is second
    module = ti.aot.Module()
    first.export(module, "integer")
    second.export(module, "floating")
    path = tmp_path / "types.tcm"
    module.archive(str(path))
    with Runtime("cpu") as runtime, runtime.load_module(path) as loaded:
        assert loaded.kernel("integer").run(Scalar("f32", 1.75)) == 1.0
        assert loaded.kernel("floating").run(Scalar("f32", 1.75)) == 1.75


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("nested", "flat scalar/vector/matrix"),
        ("dynamic_layout", "original AOS/SOA"),
        ("explicit_soa", "original AOS"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_struct_fields_reject_unproven_layouts(tmp_path, monkeypatch, change, diagnostic):
    source = STRUCT_FIELDS_SOURCE.replace("LAYOUT", "AOS")
    if change == "nested":
        source = source.replace("tag=ti.i32", "tag=ti.types.struct(code=ti.i32)")
    elif change == "dynamic_layout":
        source = source.replace("layout=ti.Layout.AOS", "layout=self.layout")
    else:
        source = source.replace(
            "scope = ti.experimental.ResourceScope(self)",
            "fields = ti.FieldsBuilder()\n        scope = ti.experimental.ResourceScope(self, builder=fields)",
        )
        source = source.replace("layout=ti.Layout.AOS", "layout=ti.Layout.SOA")
        source = source.replace(
            "scope.total = scope.field(ti.f32, shape=())",
            "scope.total = scope.field(ti.f32, shape=())\n        self.tree = fields.finalize()",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises((ValueError, ti.aot.PackageExportError), match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    assert sys.modules["definition_fixture.model"].constructors == 0


@pytest.mark.parametrize("layout", ["AOS", "SOA"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_struct_fields_keep_original_layout_and_member_paths(tmp_path, monkeypatch, layout):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    source = STRUCT_FIELDS_SOURCE.replace("LAYOUT", layout)
    if layout == "SOA":
        source = source.replace(
            "Particle = ti.types.struct(mass=ti.f32, position=ti.types.vector(3, ti.f32), tag=ti.i32)",
            "@ti.dataclass\nclass Particle:\n    mass: ti.f32\n    position: ti.types.vector(3, ti.f32)\n    tag: ti.i32",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 14
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["field_groups"]) == 1
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        # Vulkan emits the loop and trailing serial task of step separately.
        assert len([item for item in archive.infolist() if item.filename.endswith((".ll", ".bc", ".spv"))]) == (
            4 if ti.cfg.arch == ti.vulkan else 3
        )
    original = package.Model(7)
    original.initialize(2.0)
    original.step(3.0)
    expected = original.particles.to_numpy()
    root = ti.lang.impl.get_runtime().prog.get_snode_root(original.total.snode.ptr.get_snode_tree_id())
    nodes = _compiled_nodes(root)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as loaded:
        a, b = loaded.host.Model(7), loaded.host.Model(19)
        assert a.total._group_owner.layout["nodes"] == nodes
        assert a.particles.mass._group_owner is a.particles.position._group_owner is a.total._group_owner
        assert a.total._group_owner is not b.total._group_owner
        a.initialize(2.0)
        b.initialize(-1.0)
        prepared = a.graph().bind({"delta": 3.0})
        prepared.run()
        for name, array in a.particles.to_numpy().items():
            np.testing.assert_array_equal(array, expected[name])
        before = b.particles.to_numpy()
        a.particles.from_numpy({name: np.zeros_like(array) for name, array in expected.items()})
        np.testing.assert_array_equal(a.particles.mass.to_numpy(), 0)
        for name, array in b.particles.to_numpy().items():
            np.testing.assert_array_equal(array, before[name])
        a.total._group_owner.close()
        with pytest.raises((RuntimeError, ValueError)):
            a.step(1.0)
        b.step(2.0)
        np.testing.assert_array_equal(b.particles.mass.to_numpy(), 1 - np.arange(19))
    probe = tmp_path / "consume_struct_fields.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as loaded:
            a, b = loaded.host.Model(3), loaded.host.Model(19)
            a.initialize(2.0)
            b.initialize(-1.0)
            a.graph().bind({'delta': 4.0}).run()
            np.testing.assert_array_equal(a.particles.mass.to_numpy(), [6, 8, 10])
            np.testing.assert_array_equal(b.particles.mass.to_numpy(), -np.arange(1,20))
        print('STRUCT_FIELDS_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STRUCT_FIELDS_OK" in result.stdout


SOURCE = """
import taichi_forge as ti
from .device_helpers import scale
constructors = 0

@ti.data_oriented
class Model:
    def __init__(self, n: int, gain: float):
        global constructors
        constructors += 1
        self.n = n
        self.gain = gain
        self.values = ti.ndarray(ti.f32, shape=n)
        self.initialize(self.values, 0.5)

    @ti.func
    def affine(self, x):
        return scale(x, self.gain)

    @ti.kernel
    def initialize(self, values: ti.types.ndarray(dtype=ti.f32, ndim=1), offset: ti.f32):
        for i in range(self.n):
            values[i] = self.affine(i) + offset

    @ti.kernel
    def add(self, values: ti.types.ndarray(dtype=ti.f32, ndim=1), amount: ti.f32):
        for i in values:
            values[i] += amount

    @ti.kernel
    def subtract(self, values: ti.types.ndarray(dtype=ti.f32, ndim=1), amount: ti.f32):
        for i in values:
            values[i] -= amount

    def step(self, positive, amount):
        if positive:
            self.add(self.values, amount)
        else:
            self.subtract(self.values, amount)

    def description(self):
        # Same name as a numerical method, but this is ordinary local Host code.
        def initialize():
            return f"Model with {self.n} values"
        return initialize()

@ti.kernel
def unused_public_function(x: ti.i32) -> ti.i32:
    return x * 3 + 7
"""


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_host_preparation_arrays_are_not_owner_declaration_methods(tmp_path, monkeypatch):
    source = SOURCE.replace("        self.initialize(self.values, 0.5)",
        "        self.initialize(self.values, 0.5)\n        self.reference = self.prepare_reference(n, 0.75)")
    source = source.replace(
        "    @ti.func\n    def affine",
        """    def prepare_reference(self, n, offset):
        temporary = ti.ndarray(ti.f32, shape=n)
        self.initialize(temporary, offset)
        return temporary.to_numpy()

    @ti.func
    def affine""",
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not any(
        item.get("definition", "").endswith(".prepare_reference") for item in manifest.get("declaration_factories", ())
    )
    assert len(manifest["entries"]) == 5
    original = package.Model(7, 2.0)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, 2.0), code.host.Model(19, -3.0)
        np.testing.assert_array_equal(a.reference, original.reference)
        np.testing.assert_array_equal(b.reference, -3 * np.arange(19) + 0.75)
        a.step(True, 1.0)
        np.testing.assert_array_equal(a.reference, original.reference)
        np.testing.assert_array_equal(a.values.to_numpy(), 2 * np.arange(7) + 1.5)


def _source_package(tmp_path, monkeypatch, source=SOURCE, *, initializer="from .model import Model\n"):
    root = tmp_path / "definition_fixture"
    root.mkdir()
    (root / "__init__.py").write_text(initializer, encoding="utf8")
    (root / "model.py").write_text(textwrap.dedent(source), encoding="utf8")
    (root / "device_helpers.py").write_text(
        textwrap.dedent("""
        from taichi_forge import func as device
        @device
        def scale(x, gain):
            return x * gain
    """),
        encoding="utf8",
    )
    (root / "unimported.py").write_text(
        textwrap.dedent("""
        from taichi_forge import kernel as device, i32 as integer
        @device
        def available_without_a_test(x: integer) -> integer:
            return x * 5
    """),
        encoding="utf8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for key in (
        "definition_fixture",
        "definition_fixture.model",
        "definition_fixture.unimported",
        "definition_fixture.device_helpers",
    ):
        monkeypatch.delitem(sys.modules, key, raising=False)
    return importlib.import_module("definition_fixture")


CACHED_FACTORY_SOURCE = """
import taichi_forge as ti
from functools import lru_cache, cache

@lru_cache(maxsize=8)
def make_update(field, stage, tier):
    if stage == 'positive':
        @ti.kernel(opt_level=tier)
        def positive(gain: ti.f32):
            for i in field:
                field[i] = gain + i
        return positive
    @ti.kernel(opt_level=tier)
    def negative(gain: ti.f32):
        for i in field:
            field[i] = -gain - i
    return negative

class Model:
    def __init__(self, n: int):
        self.n = n
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.values = scope.field(ti.f32, shape=scope.dim('n'))
        self.tree = builder.finalize()
        self.fast = make_update(self.values, 'positive', 'fast')
        self.full = make_update(self.values, 'positive', 'full')
        self.negative = make_update(self.values, 'negative', 'fast')
        self.same_negative = make_update(self.values, 'another_label', 'fast')
"""


@pytest.mark.parametrize("cache", ["lru_cache(maxsize=8)", "cache"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_cached_conditional_factory_shares_definitions_and_checks_options(tmp_path, monkeypatch, cache):
    package = _source_package(tmp_path, monkeypatch, CACHED_FACTORY_SOURCE.replace("lru_cache(maxsize=8)", cache))
    original = sys.modules["definition_fixture.model"]
    forbidden = {original.Model.__init__.__code__, original.make_update.__wrapped__.__code__}
    previous = sys.getprofile()

    def no_sample(frame, event, arg):
        if event == "call" and frame.f_code in forbidden:
            raise AssertionError("export executed a cached factory or model")

    sys.setprofile(no_sample)
    try:
        manifest = ti.aot.export_package(package, tmp_path / "sdk")
    finally:
        sys.setprofile(previous)
    assert original.make_update.cache_info().currsize == 0
    assert manifest["version"] == 31
    assert len(manifest["entries"]) == 4  # 3 numerical definitions/options plus the fixture's unrelated kernel.
    assert len(manifest["closure_entries"]) == 2
    negative = manifest["closure_entries"]["definition_fixture.model.make_update.negative"]
    assert len({entry["symbol"] for entry in negative}) == 1
    positive = manifest["closure_entries"]["definition_fixture.model.make_update.positive"]
    assert len({entry["symbol"] for entry in positive}) == 2
    assert all(set(entry["bindings"]) == {"field"} and set(entry["options"]) == {"opt_level"} for entry in positive)
    host = (tmp_path / "sdk/host/model.py").read_text(encoding="utf8")
    assert "if stage == 'positive'" in host and "field[i] =" not in host
    reference = original.Model(5)
    reference.full(7.0)
    np.testing.assert_array_equal(reference.values.to_numpy(), np.arange(5) + 7)
    probe = tmp_path / "consume_cached_factory.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3), code.host.Model(7)
            factory = code.host.model.make_update
            assert a.fast is factory(a.values, 'positive', 'fast')
            a.fast(2.0)
            b.full(5.0)
            np.testing.assert_array_equal(a.values.to_numpy(), np.arange(3) + 2)
            np.testing.assert_array_equal(b.values.to_numpy(), np.arange(7) + 5)
            a.same_negative(9.0)
            np.testing.assert_array_equal(a.values.to_numpy(), -np.arange(3) - 9)
            try:
                factory(a.values, 'positive', 'balanced')
            except ValueError as error:
                assert 'unambiguous declared source/type binding' in str(error)
            else:
                raise AssertionError('unbuilt compile option accepted')
            old = a.values
            a.values = b.values
            a.full(11.0)
            np.testing.assert_array_equal(old.to_numpy(), np.arange(3) + 11)
            np.testing.assert_array_equal(b.values.to_numpy(), np.arange(7) + 5)
            a.tree.destroy()
            try:
                a.fast(0.0)
            except (ValueError, RuntimeError) as error:
                assert 'closed' in str(error) or 'retired' in str(error)
            else:
                raise AssertionError('cached retired resource accepted')
            b.negative(4.0)
            np.testing.assert_array_equal(b.values.to_numpy(), -np.arange(7) - 4)
        print('CACHED_FACTORY_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CACHED_FACTORY_OK" in result.stdout


TEMPLATE_FACTORY_SOURCE = """
import taichi_forge as ti
from functools import lru_cache

@lru_cache(maxsize=8)
def make_update(stage, tier):
    if stage == 'positive':
        @ti.kernel(opt_level=tier)
        def positive(self: ti.template(), gain: ti.f32):
            self._write(gain)
        return positive
    @ti.kernel(opt_level=tier)
    def negative(self: ti.template(), gain: ti.f32):
        self._write(-gain)
    return negative

@ti.data_oriented
class Model:
    def __init__(self, n: int, offset: float):
        self.n, self.offset = n, offset
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.value('offset', ti.f32)
        scope.values = scope.field(ti.f32, shape=scope.dim('n'))
        self.tree = builder.finalize()
        self.fast = make_update('positive', 'fast')
        self.full = make_update('positive', 'full')
        self.negative = make_update('negative', 'fast')
    @ti.func
    def _write(self, gain):
        for i in self.values:
            self.values[i] = gain + i + self.offset
    def graph(self):
        graph = ti.graph.GraphBuilder()
        graph.dispatch(self.fast, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32), template_args={'self': self})
        return graph.compile()
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_template_factory_keeps_shared_code_and_instance_bindings(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, TEMPLATE_FACTORY_SOURCE)
    original = sys.modules["definition_fixture.model"]
    forbidden = {original.make_update.__wrapped__.__code__, original.Model.__init__.__code__}
    previous = sys.getprofile()

    def no_sample(frame, event, arg):
        if event == "call" and frame.f_code in forbidden:
            raise AssertionError("export executed an application constructor/factory")

    sys.setprofile(no_sample)
    try:
        manifest = ti.aot.export_package(package, tmp_path / "sdk")
    finally:
        sys.setprofile(previous)
    assert original.make_update.cache_info().currsize == 0
    assert manifest["version"] == 32 and len(manifest["entries"]) == 4
    for candidates in manifest["closure_entries"].values():
        assert all(candidate["template_owner"] and not candidate["bindings"] for candidate in candidates)
    a = original.Model(3, 2.0)
    a.fast(a, 4.0)
    np.testing.assert_array_equal(a.values.to_numpy(), np.arange(3) + 6)
    probe = tmp_path / "consume_template_factory.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3, 2.0), code.host.Model(9, 7.0)
            assert a.fast is b.fast
            a.fast(self=a, gain=4.0)
            b.full.__get__(b, type(b))(5.0)
            np.testing.assert_array_equal(a.values.to_numpy(), np.arange(3) + 6)
            np.testing.assert_array_equal(b.values.to_numpy(), np.arange(9) + 12)
            for operation in (lambda: a.fast(None, 1.0), lambda: a.fast(a, self=b, gain=1.0)):
                try:
                    operation()
                except (ValueError, TypeError):
                    pass
                else:
                    raise AssertionError('invalid receiver accepted')
            ga, gb = a.graph(), b.graph()
            with ga.bind({'gain': 11.0}) as pa, gb.bind({'gain': 17.0}) as pb:
                a.offset, b.offset = -99.0, -101.0
                pb.run()
                pa.run()
                pb.run()
                np.testing.assert_array_equal(a.values.to_numpy(), np.arange(3) + 13)
                np.testing.assert_array_equal(b.values.to_numpy(), np.arange(9) + 24)
                a.negative(a, 1.0)
                np.testing.assert_array_equal(a.values.to_numpy(), np.arange(3) - 100)
                pa.run()
                np.testing.assert_array_equal(a.values.to_numpy(), np.arange(3) + 13)
            with load_package(runtime, sys.argv[1]) as other:
                foreign = other.host.Model(2, 1.0)
                try:
                    a.fast(foreign, 1.0)
                except ValueError as error:
                    assert 'same package' in str(error)
                else:
                    raise AssertionError('foreign package receiver accepted')
            a.tree.destroy()
            try:
                a.fast(a, 0.0)
            except (ValueError, RuntimeError) as error:
                assert 'closed' in str(error) or 'retired' in str(error)
            else:
                raise AssertionError('retired receiver accepted')
            b.fast(b, 2.0)
            np.testing.assert_array_equal(b.values.to_numpy(), np.arange(9) - 99)
        print('TEMPLATE_FACTORY_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TEMPLATE_FACTORY_OK" in result.stdout


SEQUENCE_FACTORY_SOURCE = (
    TEMPLATE_FACTORY_SOURCE.replace(
        "def __init__(self, n: int, offset: float):",
        "def __init__(self, n: int, offset: float, first='fast', second='full', biased=False):",
    )
    .replace(
        "        self.fast = make_update('positive', 'fast')\n"
        "        self.full = make_update('positive', 'full')\n"
        "        self.negative = make_update('negative', 'fast')",
        "        self.first, self.second, self.biased = first, second, biased\n"
        "        scope.static('first', choices=(None, 'fast', 'full'))\n"
        "        scope.static('second', choices=(None, 'fast', 'full'))\n"
        "        scope.static('biased', choices=(False, True))\n"
        "        tiers = (self.first, self.second)\n"
        "        self.selected = tuple(None if tier is None else make_update(stage, tier)\n"
        "                              for stage, tier in zip(('positive', 'negative'), tiers))",
    )
    .replace(
        "            self.values[i] = gain + i + self.offset",
        "            self.values[i] = gain + i + self.offset\n"
        "            if ti.static(self.biased):\n"
        "                self.values[i] += 10",
    )
    .replace(
        "        graph.dispatch(self.fast, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32), template_args={'self': self})",
        "        for kernel in self.selected:\n"
        "            if kernel is not None:\n"
        "                graph.dispatch(kernel, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32), template_args={'self': self})",
    )
)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_factory_sequence_projects_each_stage_domain(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, SEQUENCE_FACTORY_SOURCE)
    original = sys.modules["definition_fixture.model"]
    forbidden = {original.Model.__init__.__code__, original.make_update.__wrapped__.__code__}
    previous = sys.getprofile()

    def no_sample(frame, event, arg):
        if event == "call" and frame.f_code in forbidden:
            raise AssertionError("export executed model/factory code")

    sys.setprofile(no_sample)
    try:
        manifest = ti.aot.export_package(package, tmp_path / "sdk")
    finally:
        sys.setprofile(previous)
    # 18 legal Host configurations: 2 stages x 2 non-None options x 2 actual
    # numerical specializations, plus the fixture's unrelated entry. No product
    # of the two stage option domains appears in numerical code or families.
    assert len(manifest["entries"]) == 9 and len(manifest["entry_families"]) == 4
    for family in manifest["entry_families"].values():
        assert set(family["domain"]) == {"biased"}
        assert len(family["variants"]) == 2
    for candidates in manifest["closure_entries"].values():
        assert len(candidates) == 2
        assert len({candidate["symbol"] for candidate in candidates}) == 2
        for candidate in candidates:
            option = candidate["options"]["opt_level"][1]
            for variant in manifest["entry_families"][candidate["symbol"]]["variants"]:
                assert manifest["entries"][variant["symbol"]]["compile_options"]["opt_level"] == option
    assert original.make_update.cache_info().currsize == 0
    reference = original.Model(3, 2.0, "fast", "full", True)
    reference.selected[1](reference, 5.0)
    np.testing.assert_array_equal(reference.values.to_numpy(), np.arange(3) + 7)
    probe = tmp_path / "consume_factory_sequence.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a = code.host.Model(3, 2., 'fast', 'full', True)
            b = code.host.Model(17, 7., 'full', 'fast', False)
            c = code.host.Model(5, 3., 'full', None, False)
            assert c.selected[1] is None
            ga, gb, gc = a.graph(), b.graph(), c.graph()
            with ga.bind({'gain': 5.}) as pa, gb.bind({'gain': 11.}) as pb:
                a.offset, a.biased = -100., False
                b.offset, b.biased = 100., True
                pb.run()
                pa.run()
                pb.run()
                np.testing.assert_array_equal(a.values.to_numpy(), np.arange(3) + 7)
                np.testing.assert_array_equal(b.values.to_numpy(), np.arange(17) - 4)
            gc.run({'gain': 4.})
            np.testing.assert_array_equal(c.values.to_numpy(), np.arange(5) + 7)
            try:
                code.host.Model(1, 0., 'balanced', 'fast')
            except ValueError:
                pass
            else:
                raise AssertionError('undeclared option accepted')
        print('FACTORY_SEQUENCE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FACTORY_SEQUENCE_OK" in result.stdout


@pytest.mark.parametrize(
    "change",
    [
        "runtime_sequence",
        "rebound",
        "mutation",
        "mutating_call",
        "alias_mutation",
        "shadowed_zip",
        "shadowed_factory",
        "filtered",
        "kernel_only",
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_factory_sequence_rejects_unproven_input_domains(tmp_path, monkeypatch, change):
    source = SEQUENCE_FACTORY_SOURCE
    if change == "runtime_sequence":
        source = source.replace("tiers = (self.first, self.second)", "tiers = range(n)")
    elif change == "rebound":
        source = source.replace(
            "tiers = (self.first, self.second)", "tiers = (self.first, self.second)\n        tiers = ('fast', 'fast')"
        )
    elif change == "mutation":
        source = source.replace(
            "tiers = (self.first, self.second)", "tiers = [self.first, self.second]\n        tiers[0] = 'full'"
        )
    elif change == "mutating_call":
        source = source.replace(
            "tiers = (self.first, self.second)", "tiers = [self.first, self.second]\n        tiers.reverse()"
        )
    elif change == "alias_mutation":
        source = source.replace(
            "tiers = (self.first, self.second)",
            "tiers = [self.first, self.second]\n        alias = tiers\n        alias[0] = 'full'",
        )
    elif change == "shadowed_zip":
        source = source.replace(
            "import taichi_forge as ti", "import taichi_forge as ti\ndef zip(*args):\n    raise AssertionError()"
        )
    elif change == "shadowed_factory":
        source = source.replace("biased=False):", "biased=False, make_update=None):")
    elif change == "filtered":
        source = source.replace("'negative'), tiers))", "'negative'), tiers) if tier is not None)")
    else:
        source = source.replace("choices=(None, 'fast', 'full')", "choices=(None, 'fast', 'full'), kernel_only=True")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("change", ["extra_template", "mixed_lexical"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_template_factory_rejects_ambiguous_receiver_sources(tmp_path, monkeypatch, change):
    source = TEMPLATE_FACTORY_SOURCE
    if change == "extra_template":
        source = source.replace("self: ti.template(), gain", "self: ti.template(), other: ti.template(), gain")
    else:
        source = CACHED_FACTORY_SOURCE.replace("(gain: ti.f32)", "(self: ti.template(), gain: ti.f32)")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="template receiver"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("change", ["wrapper", "cache_argument", "condition", "side_effect"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_cached_factory_rejects_unproven_control(tmp_path, monkeypatch, change):
    source = CACHED_FACTORY_SOURCE
    if change == "wrapper":
        source = source.replace("@lru_cache(maxsize=8)", "def identity(f):\n    return f\n\n@identity")
    elif change == "cache_argument":
        source = source.replace("@lru_cache(maxsize=8)", "limit = 8\n@lru_cache(maxsize=limit)")
    elif change == "condition":
        source = source.replace("stage == 'positive'", "len(field) > 1")
    else:
        source = source.replace("    if stage ==", "    print('factory side effect')\n    if stage ==")
    package = _source_package(tmp_path, monkeypatch, source)
    original = sys.modules["definition_fixture.model"]
    factory = getattr(original.make_update, "__wrapped__", original.make_update)
    previous = sys.getprofile()

    def no_sample(frame, event, arg):
        if event == "call" and frame.f_code in (factory.__code__, original.Model.__init__.__code__):
            raise AssertionError("export executed an unproven factory")

    sys.setprofile(no_sample)
    try:
        with pytest.raises(ti.aot.PackageExportError, match="factory|Cold"):
            ti.aot.export_package(package, tmp_path / "sdk")
    finally:
        sys.setprofile(previous)
    assert not (tmp_path / "sdk").exists()


COLD_FACTORY_SOURCE = """
import taichi_forge as ti
def make_upload(fields):
    pose_fields = tuple(fields)
    field_count = len(pose_fields)
    if field_count <= 0:
        raise ValueError('empty fields')
    pose_type = ti.types.vector(field_count * 3, ti.f32)
    @ti.kernel
    def upload(pose: pose_type):
        for index in ti.static(range(field_count)):
            base = index * 3
            pose_fields[index][None] = ti.Vector([pose[base], pose[base + 1], pose[base + 2]])
    return upload

class Model:
    def __init__(self):
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.position = scope.vector_field(3, dtype=ti.f32, shape=())
        scope.direction = scope.vector_field(3, dtype=ti.f32, shape=())
        self.tree = builder.finalize()
        self.upload = make_upload((self.position, self.direction))
        self.same_upload = make_upload((self.position, self.direction))
    def update(self, values):
        self.upload(ti.Vector(values))
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_cold_factory_uses_definitions_and_retains_lexical_instance_snapshot(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, "\ufeff" + COLD_FACTORY_SOURCE)
    original = sys.modules["definition_fixture.model"]
    forbidden = {original.Model.__init__.__code__, original.make_upload.__code__}
    previous = sys.getprofile()

    def no_sample(frame, event, arg):
        if event == "call" and frame.f_code in forbidden:
            raise AssertionError("export executed an application constructor/factory")

    sys.setprofile(no_sample)
    try:
        manifest = ti.aot.export_package(package, tmp_path / "sdk")
    finally:
        sys.setprofile(previous)
    assert manifest["version"] == 25
    assert len(manifest["entries"]) == 2 and len(manifest["field_groups"]) == 1
    assert len(manifest["closure_entries"]) == 1
    assert len({item["symbol"] for item in next(iter(manifest["closure_entries"].values()))}) == 1
    host = (tmp_path / "sdk/host/model.py").read_text()
    assert "def make_upload" in host and "field_count = len(pose_fields)" in host
    assert "pose_fields[index][None]" not in host and "@ti.kernel" not in host
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(), code.host.Model()
        a.update([1, 2, 3, 4, 5, 6])
        a.same_upload([1, 2, 3, 4, 5, 6])
        b.update([-1, -2, -3, -4, -5, -6])
        old = a.position
        a.position = b.position
        a.update([7, 8, 9, 10, 11, 12])
        np.testing.assert_array_equal(old.to_numpy(), [7, 8, 9])
        np.testing.assert_array_equal(b.position.to_numpy(), [-1, -2, -3])
        with pytest.raises(ValueError, match="declared source/type binding"):
            code.host.model.make_upload((b.position, a.direction))
        a.tree.destroy()
        with pytest.raises((RuntimeError, ValueError), match="closed|retired"):
            a.update([0] * 6)
        b.update([1] * 6)
    reference = original.Model()
    reference.update([1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(reference.position.to_numpy(), [1, 2, 3])


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("pose_fields = tuple(fields)", "print('side effect')\n    pose_fields = tuple(fields)", "side effect"),
        ("make_upload((self.position, self.direction))", "make_upload(())", "input guard"),
        ("make_upload((self.position, self.direction))", "make_upload(self.unknown)", "declared Field"),
        ("pose_fields = tuple(fields)", "pose_fields = sorted(fields)", "Unproven cold factory"),
        ("return upload", "upload(0)\n    return upload", "must return its numerical definition"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_cold_factory_rejects_unproven_inputs_without_execution(
    tmp_path, monkeypatch, before, after, diagnostic
):
    package = _source_package(tmp_path, monkeypatch, COLD_FACTORY_SOURCE.replace(before, after))
    original = sys.modules["definition_fixture.model"]
    forbidden = {original.Model.__init__.__code__, original.make_upload.__code__}
    previous = sys.getprofile()

    def no_sample(frame, event, arg):
        if event == "call" and frame.f_code in forbidden:
            raise AssertionError("export executed an unproven constructor/factory")

    sys.setprofile(no_sample)
    try:
        with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
            ti.aot.export_package(package, tmp_path / "sdk")
    finally:
        sys.setprofile(previous)
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_host_ndarray_checks_distinguish_field_views(tmp_path, monkeypatch):
    source = """
import taichi_forge as ti
from taichi_forge.lang._ndarray import Ndarray as Array

@ti.data_oriented
class Model:
    def __init__(self, n: int):
        self.n = n
        self.values = ti.ndarray(ti.f32, shape=n)
        self.vectors = ti.ndarray(ti.types.vector(2, ti.f32), shape=n)
        scope = ti.experimental.ResourceScope(self)
        count = scope.dim('n')
        scope.field_values = scope.field(ti.f32, shape=count)

    @ti.kernel
    def fill(self, values: ti.types.ndarray(dtype=ti.f32, ndim=1), gain: ti.f32):
        for i in values:
            values[i] = gain * i

    def classify(self, value: Array):
        from taichi_forge.lang._ndarray import Ndarray
        return isinstance(value, Array), isinstance(value, Ndarray), Array is Ndarray

    def bind_arrays(self, graph, arguments):
        from taichi_forge.lang._ndarray import Ndarray
        if not all(isinstance(value, Ndarray) or type(value) in (bool, int, float)
                   for value in arguments.values()):
            return None
        return graph.bind(arguments)
"""
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 17
    original = package.Model(3)
    for value, expected in (
        (original.values, True),
        (original.vectors, True),
        (original.field_values, False),
        (3, False),
    ):
        assert original.classify(value) == (expected, expected, True)
    original.fill(original.values, 2.0)
    probe = tmp_path / "consume_resource_types.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        class Graph:
            def bind(self, values):
                return values
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as loaded:
            a, b = loaded.host.Model(3), loaded.host.Model(7)
            for model in (a, b):
                for value, expected in ((model.values, True), (model.vectors, True),
                                        (model.field_values, False), (3, False), (np.ones(3), False)):
                    assert model.classify(value) == (expected, expected, True)
                arguments = {'array': model.values, 'vector': model.vectors, 'n': 2}
                assert model.bind_arrays(Graph(), arguments) is arguments
                assert model.bind_arrays(Graph(), {'field': model.field_values}) is None
                assert model.bind_arrays(Graph(), {'array': np.ones(3)}) is None
            external = runtime.ndarray('f32', (3,))
            assert a.classify(external) == (True, True, True)
            array_type = loaded.module('definition_fixture.model').Array
            for unsupported in (lambda: array_type(), lambda: issubclass(type(external), array_type),
                                lambda: type('Derived', (array_type,), {})):
                try:
                    unsupported()
                except TypeError:
                    pass
                else:
                    raise AssertionError('Unqualified resource implementation API was accepted')
            a.fill(a.values, 2.0)
            b.fill(b.values, -1.0)
            np.testing.assert_array_equal(a.values.to_numpy(), [0, 2, 4])
            np.testing.assert_array_equal(b.values.to_numpy(), -np.arange(7))
        print('HOST_RESOURCE_TYPES_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HOST_RESOURCE_TYPES_OK" in result.stdout


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_host_resource_import_keeps_unprojected_types_rejected(tmp_path, monkeypatch):
    source = """
import taichi_forge as ti
from taichi_forge.lang._ndarray import Ndarray, ScalarNdarray
@ti.data_oriented
class Model:
    pass
"""
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="frontend submodule imports"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_keeps_cold_producer_paths_explicit_and_scalar_field_io_native(tmp_path, monkeypatch):
    source = STRUCT_FIELDS_SOURCE.replace("LAYOUT", "AOS") + """
class Legacy:
    def allocate(self):
        return ti.field(ti.i32, shape=3)
    def place(self):
        return ti.root.dense(None, 3)
    def allocate_tensor(self):
        return ti.Vector.field(3, ti.f32, shape=3)
    def allocate_record(self):
        return ti.types.struct(**{'x': ti.f32}).field(shape=3)

def classify(value):
    return (isinstance(value, ti.Field), isinstance(value, ti.ScalarField),
            isinstance(value, ti.MatrixField), isinstance(value, ti.StructField))

def scalar_roundtrip(model, value):
    model.total[None] = value
    return model.total[None]

def layout_names():
    return ti.Layout.AOS == ti.Layout.AOS, ti.Layout.AOS != ti.Layout.SOA

def legacy_capability_fallback():
    return getattr(ti, 'experimental', None) is None
"""
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 18
    assert {item["operation"] for item in manifest["producer_only"]} == {
        "field",
        "root",
        "Vector.field",
        "types.struct.field",
    }
    original = package.Model(3)
    original.initialize(2.0)
    expected = original.particles.to_numpy()
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as loaded:
        host = loaded.module("definition_fixture.model")
        a, b = loaded.host.Model(3), loaded.host.Model(7)
        a.initialize(2.0)
        b.initialize(-1.0)
        assert host.classify(a.total) == (True, True, False, False)
        assert host.classify(a.particles) == (True, False, False, True)
        assert host.classify(a.particles.position) == (True, False, True, False)
        assert host.classify(runtime.ndarray("f32", (3,))) == (False, False, False, False)
        assert host.layout_names() == (True, True)
        with pytest.raises(RuntimeError, match="frontend capability ti.experimental"):
            host.legacy_capability_fallback()
        assert host.scalar_roundtrip(a, 3.5) == 3.5
        assert b.total[None] == 0.0
        for name, array in a.particles.to_numpy().items():
            np.testing.assert_array_equal(array, expected[name])
        builder = loaded.graph_builder()
        builder.dispatch(a.step, builder.private_scalar("delta", "f32", 2.0))
        builder.compile().run({})
        assert a.total[None] == 4.0
        resources = set(runtime._resources)
        for method in ("allocate", "place", "allocate_tensor", "allocate_record"):
            with pytest.raises(RuntimeError, match="producer resource declaration lowering"):
                getattr(host.Legacy(), method)()
            assert set(runtime._resources) == resources
        with pytest.raises(TypeError, match="zero-dimensional Field"):
            a.particles.mass[None]
        a.total._group_owner.close()
        with pytest.raises((ValueError, RuntimeError)):
            host.scalar_roundtrip(a, 1.0)
        assert host.scalar_roundtrip(b, -4.0) == -4.0


@pytest.mark.parametrize("eager", ["value = ti.field(ti.i32, shape=3)", "def setup(value=ti.FieldsBuilder()): pass"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_does_not_defer_eager_unprojected_allocations(tmp_path, monkeypatch, eager):
    source = "import taichi_forge as ti\nclass Model: pass\n" + eager
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="eager Host code"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("array_annotation", ["ti.types.ndarray(ndim=1)", "ti.types.ndarray(dtype=ti.f32)"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_reports_input_signature_gaps_before_numerical_compilation(tmp_path, monkeypatch, array_annotation):
    source = SOURCE + f"""
@ti.kernel
def use_template(data: ti.template()):
    data[0] = 4
@ti.kernel
def use_array(data: {array_annotation}):
    data[0] = 5
"""
    package = _source_package(tmp_path, monkeypatch, source)
    module = sys.modules["definition_fixture.model"]

    def reject_compile(*args, **kwargs):
        raise AssertionError("Signature errors must be collected before numerical compilation")

    monkeypatch.setattr(ti.aot.Component, "compile", reject_compile)
    with pytest.raises(ti.aot.PackageExportError, match="entry signature preflight") as failure:
        ti.aot.export_package(package, tmp_path / "sdk")
    message = str(failure.value)
    assert "use_template argument 'data': template input requires a definition-proven" in message
    assert "use_array argument 'data': ndarray arguments require explicit element type/rank" in message
    assert "object at 0x" not in message
    assert module.constructors == 0
    assert not (tmp_path / "sdk").exists()
    # JIT still uses the actual caller's types; package rejection is not a JIT restriction.
    field = ti.field(ti.i32, shape=3)
    module.use_template(field)
    assert field[0] == 4
    data = ti.ndarray(ti.f32, shape=3)
    module.use_array(data)
    assert data[0] == 5


@pytest.mark.parametrize("helper_kind", ["func", "real_func", "pyfunc"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_compiles_definitions_without_constructor_or_test_trace(tmp_path, monkeypatch, helper_kind):
    source = SOURCE.replace(
        "@ti.func\n    def affine(self, x):", f"@ti.{helper_kind}\n    def affine(self, x: ti.f32) -> ti.f32:"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    assert "definition_fixture.unimported" not in sys.modules
    model_module = sys.modules["definition_fixture.model"]
    result = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch)
    assert model_module.constructors == 0
    expected_entries = {
        "initialize",
        "add",
        "subtract",
        "unused_public_function",
        "available_without_a_test",
    }
    if helper_kind == "pyfunc":
        expected_entries.add("affine")
    assert {item["definition"].rsplit(".", 1)[-1] for item in result["entries"].values()} == expected_entries
    host = (tmp_path / "sdk/host/model.py").read_text(encoding="utf8")
    assert "scale" not in host
    assert "return x * gain" not in (tmp_path / "sdk/host/device_helpers.py").read_text(encoding="utf8")
    assert not any(
        isinstance(n, ast.FunctionDef) and n.name in {"affine", "initialize", "add", "subtract"}
        for cls in ast.parse(host).body
        if isinstance(cls, ast.ClassDef)
        for n in cls.body
    )
    assert "def step" in host and "def __init__" in host and "def description" in host
    assert not list((tmp_path / "sdk").rglob("*.pyd"))

    # Ordinary JIT continues using the unmodified source class and decorators.
    jit = package.Model(7, 2.0)
    jit.step(False, 1.25)
    np.testing.assert_array_equal(jit.values.to_numpy(), np.arange(7, dtype=np.float32) * 2 - 0.75)

    probe = tmp_path / "consumer.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib
        import importlib.abc
        import json
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError('Private frontend/source import: ' + fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime:
            with load_package(runtime, sys.argv[1]) as code:
                a, b = code.host.Model(7, 2.0), code.host.Model(19, -3.0)
                a.step(False, 1.25)
                b.step(True, 4.0)
                read = lambda model: np.frombuffer(model.values.read(), np.float32)
                np.testing.assert_array_equal(read(a), np.arange(7, dtype=np.float32) * 2 - .75)
                np.testing.assert_array_equal(read(b), np.arange(19, dtype=np.float32) * -3 + 4.5)
                assert a.description() == 'Model with 7 values'
                if sys.argv[3] == 'pyfunc':
                    assert a.affine(3.0) == 6.0 and b.affine(3.0) == -9.0
                module = importlib.import_module(code.host.__name__ + '.model')
                assert module.unused_public_function(5) == 22
                with load_package(runtime, sys.argv[1]) as other:
                    c = other.host.Model(3, 4.0)
                    late = importlib.import_module(code.host.__name__ + '.unimported')
                    assert late.available_without_a_test(4) == 20
                    a.gain = 5.0
                    a.initialize(a.values, offset=2.0)  # bound to a's package, not the active other package
                    np.testing.assert_array_equal(read(a), np.arange(7, dtype=np.float32) * 5 + 2)
                    np.testing.assert_array_equal(read(c), np.arange(3, dtype=np.float32) * 4 + .5)
                assert late.available_without_a_test(7) == 35
                b.step(False, 1.0)
                np.testing.assert_array_equal(read(b), np.arange(19, dtype=np.float32) * -3 + 3.5)
            try:
                a.add(a.values, 1.0)
            except RuntimeError as error:
                assert 'closed' in str(error)
            else:
                raise AssertionError('Retired package accepted invocation')
        assert 'taichi_forge' not in sys.modules and 'taichi' not in sys.modules
        print(json.dumps({'source_free': True, 'instances': 3, 'both_host_branches': True}))
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), result["target"], helper_kind],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"source_free": true' in result.stdout


@pytest.mark.parametrize(
    "source, diagnostic",
    [
        (
            SOURCE.replace(
                "self.values = ti.ndarray(ti.f32, shape=n)",
                "if n > 0:\n            self.values = ti.ndarray(ti.f32, shape=n)",
            ),
            "conditional",
        ),
        (SOURCE.replace("ti.ndarray(ti.f32, shape=n)", "ti.field(ti.f32, shape=n, order='j')"), "Invalid axis"),
        (SOURCE.replace("ti.ndarray(ti.f32, shape=n)", "ti.field(ti.f32, shape=(n,n), order='ii')"), "axes in order"),
        (SOURCE.replace("ti.ndarray(ti.f32, shape=n)", "ti.field(ti.f32, shape=n, order='ij')"), "dimensionality"),
        (
            SOURCE.replace("ti.ndarray(ti.f32, shape=n)", "ti.field(ti.f32, shape=n, order='i' if n else None)"),
            "literal axis",
        ),
        (SOURCE.replace("range(self.n)", "ti.static(range(self.n))"), "runtime capture"),
        (
            SOURCE.replace('return f"Model with {self.n} values"', "return self.affine(3)"),
            "Host references a device-only helper",
        ),
        (
            SOURCE.replace('return f"Model with {self.n} values"', "return scale(3, self.gain)"),
            "Host references a device-only helper",
        ),
        (
            SOURCE.replace('return f"Model with {self.n} values"', "return ti.f32(self.gain)"),
            "Host scalar casts",
        ),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproven_definitions_without_publishing(tmp_path, monkeypatch, source, diagnostic):
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    assert sys.modules["definition_fixture.model"].constructors == 0


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_backend_selection_keeps_live_jit_and_initializes_cold_backend(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch)
    program = ti.lang.impl.get_runtime().prog
    with pytest.raises(ValueError, match="live JIT"):
        ti.aot.export_package(package, tmp_path / "wrong", arch="cuda")
    assert ti.lang.impl.get_runtime().prog is program
    with pytest.raises(ValueError, match="arch must"):
        ti.aot.export_package(package, tmp_path / "unknown", arch="automatic")
    ti.reset()
    result = ti.aot.export_package(package, tmp_path / "cold", arch="cpu")
    assert result["target"] == ti.cpu.name
    assert ti.lang.impl.current_cfg().arch == ti.cpu


COMPOUND_SOURCE = """
import taichi_forge as ti
Vec = ti.types.vector(3, ti.f32)
Mat = ti.types.matrix(2, 2, ti.f32)
@ti.dataclass
class Record:
    value: ti.f32
    vector: Vec
    matrix: Mat

    @ti.func
    def total(self):
        return self.value + self.vector[2] + self.matrix[1, 1]

@ti.data_oriented
class Model:
    def __init__(self, n: int):
        self.records = ti.ndarray(Record, shape=n)
        self.vectors = ti.ndarray(Vec, shape=n)
        self.matrices = ti.ndarray(Mat, shape=n)
        self.output = ti.ndarray(ti.f32, shape=n)

    @ti.kernel
    def advance(self, records: ti.types.ndarray(Record, 1), vectors: ti.types.ndarray(Vec, 1),
                matrices: ti.types.ndarray(Mat, 1), output: ti.types.ndarray(ti.f32, 1),
                vector: Vec, matrix: Mat, scalar_record: Record):
        for i in output:
            records[i].value = i + scalar_record.value
            records[i].vector = vector
            records[i].matrix = matrix
            vectors[i] = vector * 2
            matrices[i] = matrix * 3
            output[i] = records[i].total() + scalar_record.vector[0]

    def step(self):
        self.advance(self.records, self.vectors, self.matrices, self.output,
                     Vec(2, 3, 4), Mat([[1, 2], [3, 5]]),
                     Record(value=7, vector=Vec(11, 13, 17), matrix=Mat(0)))
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_uses_native_struct_layout_and_compound_values(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, COMPOUND_SOURCE)
    result = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch.name)
    ordinary = package.Model(7)
    ordinary.step()
    expected = ordinary.output.to_numpy()
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        a, b = library.host.Model(7), library.host.Model(19)
        a.step()
        b.step()
        np.testing.assert_array_equal(np.frombuffer(a.output.read(), np.float32), expected)
        np.testing.assert_array_equal(np.frombuffer(b.output.read(), np.float32), np.arange(19) + 27)
        assert a.records.dtype.element_stride == b.records.dtype.element_stride
        np.testing.assert_array_equal(np.frombuffer(b.vectors.read(), np.float32).reshape(19, 3), [[4, 6, 8]] * 19)
        np.testing.assert_array_equal(
            np.frombuffer(b.matrices.read(), np.float32).reshape(19, 2, 2), [[[3, 6], [9, 15]]] * 19
        )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_preserves_host_ndarray_annotations_without_frontend(tmp_path, monkeypatch):
    source = COMPOUND_SOURCE
    aliases = {}
    for element in ("Record", "Vec", "Mat", "ti.f32"):
        alias = element.replace(".", "_") + "Array"
        annotation = f"ti.types.ndarray({element}, 1)"
        source = source.replace(annotation, alias)
        aliases[alias] = annotation
    source = source.replace(
        "@ti.data_oriented", "\n".join(f"{k} = {v}" for k, v in aliases.items()) + "\n@ti.data_oriented"
    )
    source += "\ndef accept_array(value: VecArray) -> VecArray:\n    return value\n"
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 9  # Host tensor type factories need only logical Runtime metadata.
    original = package.Model(7)
    original.step()
    np.testing.assert_array_equal(original.output.to_numpy(), np.arange(7) + 27)
    script = tmp_path / "consume_annotations.py"
    script.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            module = code.host.model
            assert module.VecArray.ndim == 1
            assert module.VecArray.dtype.spec['element_shape'] == [3]
            assert module.ti_f32Array.dtype == 'f32'
            assert module.VecArray.boundary == 'UNSAFE'
            assert module.accept_array.__annotations__['value'] is module.VecArray
            for n in (7, 19):
                model = code.host.Model(n)
                model.step()
                assert module.accept_array(model.vectors) is model.vectors
                np.testing.assert_array_equal(np.frombuffer(model.output.read(), np.float32), np.arange(n) + 27)
                np.testing.assert_array_equal(np.frombuffer(model.vectors.read(), np.float32).reshape(n, 3), [[4, 6, 8]] * n)
            assert not any(name.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture') for name in sys.modules)
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(script), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_incomplete_host_ndarray_annotations(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, SOURCE + "\nUnknown = ti.types.ndarray(ti.f32)\n")
    with pytest.raises(ti.aot.PackageExportError, match="fixed rank"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


REAL_SOURCE = """
import taichi_forge as ti

@ti.real_func
def sum_to(n: ti.i32) -> ti.i32:
    result = 0
    if n > 0:
        result = n + sum_to(n - 1)
    return result

@ti.kernel
def sum_kernel(n: ti.i32) -> ti.i32:
    return sum_to(n)

@ti.data_oriented
class Model:
    def __init__(self, n: int, gain: float):
        self.gain = gain
        self.values = ti.ndarray(ti.f32, shape=n)

    @ti.real_func
    def write(self, i: ti.i32):
        self.values[i] = sum_to(i) * self.gain

    @ti.real_func
    def nested(self, i: ti.i32):
        self.write(i)

    @ti.kernel
    def run(self):
        for i in range(self.values.shape[0]):
            self.nested(i)
"""


@pytest.mark.parametrize("recursive", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_real_functions_keep_nested_array_captures_and_explicit_recursion(tmp_path, monkeypatch, recursive):
    source = (
        REAL_SOURCE if recursive else REAL_SOURCE.replace("result = n + sum_to(n - 1)", "result = n * (n + 1) // 2")
    )
    # Keep device recursion shallow: standalone Runtime currently has no CUDA
    # stack-limit setting. Sizes still vary independently of the call depth.
    source = source.replace("sum_to(i) * self.gain", "sum_to(i % 3) * self.gain")
    package = _source_package(tmp_path, monkeypatch, source)
    result = ti.aot.export_package(package, tmp_path / "sdk")
    # Ordinary JIT requires ndarrays to be explicit arguments. Deployment
    # capture supplies those hidden arguments; it does not alter that JIT rule.
    ordinary = sys.modules["definition_fixture.model"].sum_kernel
    assert ordinary(6) == 21
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        a, b = library.host.Model(7, 2.0), library.host.Model(19, -3.0)
        a.run()
        b.run()
        small, large = np.arange(7) % 3, np.arange(19) % 3
        np.testing.assert_array_equal(np.frombuffer(a.values.read(), np.float32), small * (small + 1))
        np.testing.assert_array_equal(np.frombuffer(b.values.read(), np.float32), large * (large + 1) * -1.5)
        a.gain = 4.0
        a.run()
        np.testing.assert_array_equal(np.frombuffer(a.values.read(), np.float32), small * (small + 1) * 2)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_recursive_real_function_hidden_captures(tmp_path, monkeypatch):
    source = REAL_SOURCE.replace(
        "self.values[i] = sum_to(i) * self.gain",
        "if i > 0:\n            self.write(i - 1)\n        self.values[i] = self.gain",
    )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="Recursive real_func captures"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_package_diagnoses_real_function_backend_before_publication(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, REAL_SOURCE)
    with pytest.raises(ti.aot.PackageExportError, match="real_func AOT requires an LLVM backend"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


PYFUNC_SOURCE = """
import taichi_forge as ti

@ti.pyfunc
def twice(x: ti.i32) -> ti.i32:
    return x * 2

@ti.data_oriented
class Model:
    def __init__(self, bias: int):
        self.bias = bias

    @ti.pyfunc
    def adjust(self, x: ti.i32) -> ti.i32:
        return twice(x) + self.bias

    @ti.kernel
    def run(self, x: ti.i32) -> ti.i32:
        return self.adjust(x)

    def host(self, x):
        return self.adjust(x) + twice(x + 1)
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_typed_pyfunc_is_available_from_host_and_device(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, PYFUNC_SOURCE)
    original = package.Model(3)
    expected = original.host(5)
    assert original.run(5) == 13
    result = ti.aot.export_package(package, tmp_path / "sdk")
    projected = (tmp_path / "sdk/host/model.py").read_text(encoding="utf8")
    assert "return x * 2" not in projected and "return twice(x) + self.bias" not in projected
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        a, b = library.host.Model(3), library.host.Model(-7)
        assert a.host(5) == expected and a.run(5) == 13
        assert b.host(5) == expected - 10 and b.run(5) == 3
        a.bias = 9
        assert a.host(5) == expected + 6


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_pyfunc_without_declared_domain_does_not_sample_calls(tmp_path, monkeypatch):
    source = PYFUNC_SOURCE.replace("def twice(x: ti.i32)", "def twice(x)")
    package = _source_package(tmp_path, monkeypatch, source)
    # Calling it before export does not establish a production signature.
    assert package.Model(3).host(5) == 25
    with pytest.raises(ti.aot.PackageExportError, match="pyfunc export requires explicit parameter types"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_scope_dependent_pyfunc(tmp_path, monkeypatch):
    source = PYFUNC_SOURCE.replace("return x * 2", "return x * 2 if ti.lang.impl.inside_kernel() else x * 3")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="scope-dependent pyfunc"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_submodule_device_imports_and_host_graph_share_code(tmp_path, monkeypatch):
    package = _source_package(
        tmp_path,
        monkeypatch,
        SOURCE + """

def record(builder):
    builder.dispatch(unused_public_function, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'x', ti.i32))
""",
    )
    module = sys.modules["definition_fixture.model"]
    # Select one source module, not the parent package: its relative helper is
    # compiler-only and must not leak a dependency on the private parent.
    result = ti.aot.export_package(module, tmp_path / "sdk")
    assert len(result["entries"]) == 4
    assert ".device_helpers" not in (tmp_path / "sdk/host/__init__.py").read_text(encoding="utf8")
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        builder = library.host.ti.graph.GraphBuilder()
        library.host.record(builder)
        graph = builder.compile()
        with graph.bind(dict(x=4)) as binding:
            binding.run()
        with pytest.raises(ValueError, match="differs from the entry's type"):
            builder.dispatch(
                library.host.unused_public_function,
                library.host.ti.graph.Arg(library.host.ti.graph.ArgKind.SCALAR, "wrong", "f32"),
            )
        # Host recording references the same compiled definition; it never
        # creates another artifact for this sequence or argument value.
        assert len(library.manifest["entries"]) == 4


OWNER_GRAPH_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, gain: float):
        global constructors
        constructors += 1
        self.n = n
        self.gain = gain
        self.bias = ti.Vector([1.0, 2.0], ti.f32)
        scope = ti.experimental.ResourceScope(self)
        scope.value('bias', ti.types.vector(2, ti.f32))
        scope.values = scope.field(ti.f32, shape=scope.dim('n'))

    @ti.kernel
    def reset(self):
        for i in self.values:
            self.values[i] = 0

    @ti.kernel
    def step(self, increment: ti.i32, extra: ti.types.vector(2, ti.f32)):
        for i in range(self.n):
            self.values[i] += self.gain * (i + increment) + self.bias[0] + extra[1]

    def record(self, builder):
        def vector(name, dtype):
            return ti.graph.Arg(ti.graph.ArgKind.MATRIX, name, ti.types.vector(2, dtype))
        extra = vector('extra', ti.f32)
        one = builder.private_scalar('one', ti.i32, 1)
        builder.dispatch(self.step, one, extra, template_args={'self': self})
        child = builder.create_sequential()
        child.dispatch(self.step, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'increment', ti.i32), extra,
                       template_args={'self': self})
        builder.append(child)

    def graph(self):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.reset, template_args={'self': self})
        self.record(builder)
        return builder.compile()

    def compile_component(self):
        return ti.aot.Component.from_owner(self).compile_graph(self.record)

    def wait(self):
        import taichi_forge as ti
        assert ti.lang.impl.get_runtime().prog is not None
        ti.sync()
"""


PACKAGE_WHILE_SOURCE = """
import taichi_forge as ti
@ti.data_oriented
class Model:
    def __init__(self, n: int, limit: int, gain: int):
        self.n, self.limit, self.gain = n, limit, gain
        scope = ti.experimental.ResourceScope(self)
        scope.values = scope.field(ti.i32, shape=scope.dim('n'))
        scope.value('limit', ti.i32)
        scope.value('gain', ti.i32)
    @ti.kernel
    def reset(self, counter: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        counter[0] = 0
        for i in range(self.n):
            self.values[i] = 0
    @ti.kernel
    def condition(self, predicate: ti.types.ndarray(dtype=ti.i32, ndim=1),
                  counter: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        predicate[0] = counter[0] < self.limit
    @ti.kernel
    def advance(self, counter: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(self.n):
            self.values[i] += self.gain * (i + 1)
        counter[0] += 1
    def record(self, builder, maximum):
        predicate = builder.private_ndarray('predicate', ti.i32, (1,))
        counter = builder.private_ndarray('counter', ti.i32, (1,))
        builder.dispatch(self.reset, counter)
        condition, body = builder.create_sequential(), builder.create_sequential()
        condition.dispatch(self.condition, predicate, counter)
        body.dispatch(self.advance, counter)
        builder.while_loop(condition, body, predicate=predicate, counter=counter,
                           max_iterations=maximum, masked_execution=True, lowering_mode='auto')
    def graph(self, maximum):
        builder = ti.graph.GraphBuilder()
        self.record(builder, maximum)
        return builder.compile()
    def control_info(self):
        return (ti.graph.structured_control_capabilities(),
                callable(getattr(ti.graph.GraphBuilder, 'private_ndarray', None)))
"""


@pytest.mark.parametrize("plan_value", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_while_private_arrays_are_prepared_per_invocation(tmp_path, monkeypatch, plan_value):
    source = PACKAGE_WHILE_SOURCE
    if plan_value:
        source = source.replace("scope.value('limit', ti.i32)", "scope.value('limit', ti.i32, affects_plan=True)")
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert len(manifest["entries"]) == 4 and not manifest["entry_families"]  # three kernels plus fixture entry
    reference = package.Model(3, 5, 2)
    counter = ti.ndarray(ti.i32, shape=1)
    reference.reset(counter)
    for _ in range(5):
        reference.advance(counter)
    np.testing.assert_array_equal(reference.values.to_numpy(), np.arange(1, 4) * 10)
    if plan_value:
        assert manifest["version"] == 33
        compiled = ti.aot.Component.from_owner(reference).compile_graph(reference.record, 5)
        assert compiled.contract["configuration"]["limit"]["selected"] == ["int", 5]
        assert all("limit" not in dict(entry["static"]) for entry in compiled.contract["entries"])
    probe = tmp_path / "consume_package_while.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import json
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3, 5, 2), code.host.Model(17, 2, -3)
            control, private_arrays = a.control_info()
            assert private_arrays
            assert control['scope'] == 'definition_package_flat_kernel_while'
            expected = 'cuda_conditional_while' if sys.argv[2] == 'cuda' else 'cpu_exact_while'
            assert runtime.execution_control_capabilities() == {expected}
            assert control['device_control']['while'] == (sys.argv[2] == 'cuda')
            assert control['device_control']['per_iteration_host_observation'] == (sys.argv[2] != 'cuda')
            if sys.argv[2] != 'cuda':
                assert control['portable']['while'] == 'cpu_exact_host_loop'
            # Queries are cold Host decisions, never implicit replay work.
            def unexpected_query():
                raise AssertionError('control capability queried during recording/preparation/replay')
            runtime.execution_control_capabilities = unexpected_query
            ga, gb = a.graph(5), b.graph(9)
            document = json.loads(ga.plan.to_json())
            assert len(document['operations']) == 2
            assert document['operations'][1]['kind'] == 'while'
            assert 'lowering' not in document['operations'][1]  # exact native route
            assert len(document['resources']) == 2
            assert not ga.arguments and not gb.arguments
            pa, pb = ga.bind({}), gb.bind({})
            a.limit, a.gain = 1, 7
            if sys.argv[3] == 'True':
                try:
                    ga.bind({})
                except ValueError as error:
                    assert 'Plan parameter' in str(error)
                else:
                    raise AssertionError('old plan accepted a changed iteration parameter')
                ga = a.graph(5)
                changing = code.graph_builder()
                a.record(changing, 5)
                a.limit = 3
                try:
                    changing.compile()
                except ValueError as error:
                    assert 'Plan parameter' in str(error)
                else:
                    raise AssertionError('changed recording published a plan')
                a.limit = 1
            with ga.bind({}) as fresh:
                fresh.run()
                np.testing.assert_array_equal(a.values.to_numpy(), np.arange(1, 4) * 7)
                pb.run()
                pa.run()
                np.testing.assert_array_equal(a.values.to_numpy(), np.arange(1, 4) * 10)
                np.testing.assert_array_equal(b.values.to_numpy(), np.arange(1, 18) * -6)
            pa.run()
            pb.run()
            pa.close()
            pb.close()
            with a.graph(0).bind({}) as zero:
                zero.run()
                np.testing.assert_array_equal(a.values.to_numpy(), 0)
            # Reject unrelated recording scopes before adding a partial loop.
            root = code.graph_builder()
            other = code.graph_builder()
            pred = root.private_ndarray('predicate', 'i32', (1,))
            cnt = root.private_ndarray('counter', 'i32', (1,))
            condition, body = root.create_sequential(), root.create_sequential()
            condition.dispatch(a.condition, pred, cnt)
            body.dispatch(a.advance, cnt)
            invalid = [dict(condition=other, body=body, predicate=pred, max_iterations=1),
                       dict(condition=condition, body=body, predicate=pred, max_iterations=-1),
                       dict(condition=condition, body=body, predicate=cnt, counter=cnt, max_iterations=1)]
            for kwargs in invalid:
                try:
                    root.while_loop(**kwargs)
                except ValueError:
                    pass
                else:
                    raise AssertionError('invalid loop accepted')
                assert not root.operations
            try:
                body.private_ndarray('counter', 'i32', (1,))
            except ValueError:
                pass
            else:
                raise AssertionError('private scope collision accepted')
        try:
            Runtime.execution_control_capabilities(runtime)
        except RuntimeError:
            pass
        else:
            raise AssertionError('closed runtime accepted a capability query')
        print('PACKAGE_WHILE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"], str(plan_value)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PACKAGE_WHILE_OK" in result.stdout


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_graph_captures_are_instance_snapshots_without_new_code(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, OWNER_GRAPH_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 9
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert [item["operation"] for item in manifest["producer_only"]] == ["aot.Component"]
    reference = package.Model(7, 2.0)
    reference.reset()
    for increment in (1, 3):
        reference.step(increment, ti.Vector([0.0, 0.0]))
    np.testing.assert_array_equal(reference.values.to_numpy(), np.arange(7) * 4 + 10)
    probe = tmp_path / "consume_owner_graph.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import hashlib
        import sys
        from pathlib import Path
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        root = Path(sys.argv[1])
        before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, root) as code:
            a, b = code.host.Model(7, 2.0), code.host.Model(19, -1.0)
            ga, gb = a.graph(), b.graph()
            assert len(ga.plan.argument_names) > len(ga.arguments)  # private captures, not public arguments
            assert ga.arguments.keys() == {'increment', 'extra'}
            pa = ga.bind(dict(increment=3, extra=(0.0, 0.0)))
            pb = gb.bind(dict(increment=5, extra=(0.0, 0.0)))
            a.gain = 9.0
            a.bias.x = 7.0
            pb.run()
            pa.run()
            pb.run()
            a.wait()
            np.testing.assert_array_equal(np.frombuffer(a.values.read(), np.float32), np.arange(7) * 4 + 10)
            np.testing.assert_array_equal(np.frombuffer(b.values.read(), np.float32), np.arange(19) * -2 - 4)
            with ga.bind(dict(increment=3, extra=(0.0, 0.0))) as fresh:
                fresh.run()
            np.testing.assert_array_equal(np.frombuffer(a.values.read(), np.float32), np.arange(7) * 18 + 50)
            pa.run()  # the earlier snapshot remains independent
            np.testing.assert_array_equal(np.frombuffer(a.values.read(), np.float32), np.arange(7) * 4 + 10)
            pa.close()
            pb.close()
            # Transfers use the native member view, including strided Host input.
            source = np.arange(14, dtype=np.float64)[::2]
            a.values.from_numpy(source)
            copy = a.values.to_numpy()
            np.testing.assert_array_equal(copy, source)
            copy[:] = -1
            np.testing.assert_array_equal(a.values.to_numpy(), source)
            try:
                a.values.from_numpy(np.zeros((7, 1), np.float32))
            except ValueError as error:
                assert 'shape' in str(error)
            else:
                raise AssertionError('Wrong transfer shape accepted')
            try:
                a.compile_component()
            except RuntimeError as error:
                assert 'producer-only' in str(error)
            else:
                raise AssertionError('Installed package attempted producer compilation')
            builder = code.host.model.ti.graph.GraphBuilder()
            builder.private_scalar('one', 'i32', 1)
            try:
                builder.private_scalar('one', 'i32', 2)
            except ValueError:
                pass
            else:
                raise AssertionError('Repeated private scalar accepted')
            with load_package(runtime, root) as other:
                foreign = other.host.Model(3, 1.0)
                try:
                    builder.dispatch(foreign.reset)
                except ValueError as error:
                    assert 'same loaded package' in str(error)
                else:
                    raise AssertionError('Foreign code owner accepted')
        after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
        assert before == after
        print('OWNER_GRAPH_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OWNER_GRAPH_SOURCE_FREE_OK" in result.stdout


MODEL_TENSOR_SOURCE = """
import taichi_forge as ti
from types import SimpleNamespace
constructors = 0

@ti.data_oriented
class Model:
    def __init__(self, n: int, dof: int):
        global constructors
        constructors += 1
        self.dof = dof
        self.v = ti.Vector.field(self.dof, ti.f32, shape=n)
        self.matrix = ti.Matrix.field(dof, 2, ti.f32, shape=n)
        self.fixed = ti.Vector.field(3, ti.f32, shape=n)

    @ti.func
    def touch(self, i, j, gain):
        self.v[i][j] += gain * (i + j)
        self.matrix[i][j, 1] = self.v[i][j] + 2

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.v:
            for j in range(self.v.n):
                self.touch(i, j, gain)
            self.fixed[i] = ti.Vector([gain, i, 3])

@ti.data_oriented
class Rows:
    def __init__(self, n: int, dof: int):
        global constructors
        constructors += 1
        self.n = n
        self.layout = SimpleNamespace(dof=dof)
        scope = ti.experimental.ResourceScope(self)
        layout = scope.record('layout', self.layout)
        layout.static('dof', choices=(self.layout.dof,))
        scope.rows = scope.struct_field({
            'J': ti.types.vector(self.layout.dof, ti.f32),
            'fixed': ti.types.vector(3, ti.f32),
            'result': ti.f32,
        }, shape=scope.dim('n'))

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.rows.J:
            total = 0.0
            for j in range(self.layout.dof):
                self.rows.J[i][j] += gain * (i + j)
                total += self.rows.J[i][j]
            self.rows.result[i] = total
            self.rows.fixed[i] = ti.Vector([gain, i, 3])

    def graph(self):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.step, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32))
        return builder.compile()

@ti.data_oriented
class Reader:
    def __init__(self, n: int, dof: int):
        scope = ti.experimental.ResourceScope(self)
        scope.source = Model(n, dof)
        scope.borrow('alias', self.source.v, source_scope=True)

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.alias:
            for j in range(self.alias.n):
                self.alias[i][j] = gain * (i + j)
"""


DEFERRED_DIMENSION_SOURCE = """
import taichi_forge as ti
from types import SimpleNamespace
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, worlds: int, dof: int, mutate=False):
        global constructors
        constructors += 1
        self.layout = SimpleNamespace(dof=dof)
        self.worlds = worlds
        w, d = self.worlds, self.layout.dof
        scope = ti.experimental.ResourceScope(self)
        scope.dim('worlds')
        record = scope.record('layout', self.layout)
        record.static('dof', choices=(self.layout.dof,))
        if mutate:
            self.layout.dof += 1
        scope.rows = scope.struct_field({'H': ti.types.vector(d, ti.f32)}, shape=(w, d))

    @ti.kernel
    def step(self, gain: ti.f32):
        for world, row in self.rows.H:
            for col in range(self.layout.dof):
                self.rows.H[world, row][col] = gain * (world + row + col)
"""


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_resolves_early_dimension_aliases_and_checks_snapshots(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, DEFERRED_DIMENSION_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch)
    assert manifest["version"] == 34
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len([entry for entry in manifest["entries"].values() if entry["definition"].endswith("Model.step")]) == 1
    assert not manifest["entry_families"]
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        original = package.Model(2, 3)
        original.step(2.0)
        first = library.host.Model(2, 3)
        second = library.host.Model(3, 17)
        first.step(2.0)
        second.step(-1.0)
        np.testing.assert_array_equal(first.rows.H.to_numpy(), original.rows.H.to_numpy())
        np.testing.assert_array_equal(second.rows.H.to_numpy(), -np.indices((3, 17, 17)).sum(axis=0))
        before = set(runtime._resources)
        with pytest.raises(ValueError, match="Dimension alias 'd'"):
            library.host.Model(2, 3, mutate=True)
        assert set(runtime._resources) == before
    probe = tmp_path / "consume_dimensions.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc, sys, numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as library:
            owner = library.host.Model(2, 65)
            owner.step(3.0)
            np.testing.assert_array_equal(owner.rows.H.to_numpy(), 3 * np.indices((2, 65, 65)).sum(axis=0))
        print('DIMENSION_ALIAS_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DIMENSION_ALIAS_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("change", ["d = dof", "del d", "d = self.layout.dof + 1"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_rebound_early_dimension_aliases(tmp_path, monkeypatch, change):
    source = DEFERRED_DIMENSION_SOURCE.replace("        scope.rows =", "        " + change + "\n        scope.rows =")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="declared dimensions"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


MODEL_STATIC_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, dof: int):
        global constructors
        constructors += 1
        self.n = n
        self.dof = dof
        self.vector = ti.Vector.field(self.dof, ti.i32, shape=self.n)
        self.result = ti.field(ti.i32, shape=self.n)

    @ti.func
    def evaluate(self):
        count = ti.static(self.dof * (self.dof + 1) // 2)
        extra = ti.static(5 if self.dof == 1 else 7)
        extra_check = ti.static((self.dof == 1 and 5) or 7)
        result = extra
        if extra != extra_check:
            result += 10000
        for j in ti.static(range(count)):
            result += 1
        for j, k in ti.static(ti.ndrange(self.dof, 2)):
            result += j + k
        for j in ti.static(range(3)):
            result += (1, 3, 5)[j]
        if ti.static(self.dof > 32 and self.dof < 128):
            result += 2
        else:
            result += 3
        if ti.static(self.dof and False and self.missing):
            result += 999
        return result

    @ti.kernel
    def step(self):
        for i in range(self.n):
            self.result[i] = self.evaluate()

    @ti.kernel
    def ordered(self):
        self.result[0] = 0
        for j in ti.static(range(self.dof)):
            self.result[0] = (self.result[0] * 5 + j) % 97
        for j in ti.static(range(self.dof, self.dof - 1)):
            self.result[0] = -1

    def graph(self):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.step)
        return builder.compile()
"""


@pytest.mark.parametrize("grouped", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_model_static_control_uses_one_implementation(tmp_path, monkeypatch, grouped):
    source = MODEL_STATIC_SOURCE
    if grouped:
        source = source.replace("for j, k in ti.static(ti.ndrange(self.dof, 2)):\n            result += j + k",
                                "for I in ti.static(ti.grouped(ti.ndrange(self.vector.n, 2))):\n"
                                "            result += I[0] + I[1]")
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch)
    assert sys.modules["definition_fixture.model"].constructors == 0
    for name in ("step", "ordered"):
        entries = [entry for entry in manifest["entries"].values()
                   if entry["definition"].endswith("Model." + name)]
        assert len(entries) == 1
    expected = package.Model(2, 17)
    expected.step()
    np.testing.assert_array_equal(expected.result.to_numpy(), 461)
    probe = tmp_path / "consume_control.py"
    probe.write_text(textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            models = [code.host.Model(2, dof) for dof in (1, 17, 65, 129)]
            graphs = [model.graph() for model in models]
            calls = [graph.bind({}) for graph in graphs]
            for model in reversed(models):
                model.step()
                n = model.dof
                expected = n*(n+1)//2 + n*n + 9 + (5 if n == 1 else 7) + (2 if 32 < n < 128 else 3)
                np.testing.assert_array_equal(model.result.to_numpy(), expected)
                model.ordered()
                ordered = 0
                for j in range(n):
                    ordered = (ordered * 5 + j) % 97
                assert model.result.to_numpy()[0] == ordered
            for call in reversed(calls):
                call.run()
            calls[0].close()
            for model, call in zip(models[1:], calls[1:]):
                call.run()
                n = model.dof
                expected = n*(n+1)//2 + n*n + 16 + (2 if 32 < n < 128 else 3)
                np.testing.assert_array_equal(model.result.to_numpy(), expected)
                call.close()
            # The final triangular count fits i32 at 50000, but its multiply
            # does not. Check the exact submission snapshot before any work.
            from taichi_forge_runtime.aot import _package, _package_graph
            for bindings, action in ((_package, models[1].step),
                                     (_package_graph, lambda: graphs[1].bind({}))):
                original = bindings.capture_value
                def changed(owner, spec):
                    value = original(owner, spec)
                    return 50000 if spec['name'] == 'dof' else value
                bindings.capture_value = changed
                try:
                    action()
                except ValueError as error:
                    assert 'i32 intermediate range' in str(error), str(error)
                else:
                    raise AssertionError('overflowing static loop expression submitted')
                finally:
                    bindings.capture_value = original
            models[1].step()
            np.testing.assert_array_equal(models[1].result.to_numpy(), 461)
        print('MODEL_STATIC_CONTROL_OK')
    """), encoding="utf8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
               TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
               TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir())
    backend = {ti.cpu: "cpu", ti.cuda: "cuda", ti.vulkan: "vulkan"}[ti.cfg.arch]
    result = subprocess.run([sys.executable, "-B", str(probe), str(tmp_path / "sdk"), backend],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MODEL_STATIC_CONTROL_OK" in result.stdout


@pytest.mark.parametrize("condition", ["self.result[0] > 0", "self.dof + self.result[0] > 0", "mutable > 0"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_model_static_control_rejects_device_or_mutable_values(tmp_path, monkeypatch, condition):
    source = MODEL_STATIC_SOURCE.replace("        count = ti.static", "        mutable = self.dof\n        count = ti.static")
    source = source.replace("self.dof > 32 and self.dof < 128", condition)
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(Exception, match="Input to ti.static must be compile-time"):
        ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cpu)


MODEL_SHARED_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, dof: int):
        global constructors
        constructors += 1
        self.n = n
        self.dof = dof
        self.vector = ti.Vector.field(self.dof, ti.f32, shape=self.n)
        self.result = ti.field(ti.f32, shape=self.n)

    @ti.func
    def shared_sum(self, matrix, extra, fixed):
        return matrix[self.dof - 1, 1] + extra[self.dof] + fixed[0] + extra[0]

    @ti.kernel
    def step(self, gain: ti.f32):
        ti.loop_config(block_dim=32)
        for i in range(self.n * 32):
            lane, world = i % 32, i // 32
            matrix = ti.simt.block.SharedArray((self.dof, 2), ti.f32)
            extra = ti.simt.block.SharedArray((self.dof + 1,), ti.f32)
            fixed = ti.simt.block.SharedArray((1,), ti.f32)
            if lane == 0:
                fixed[0] = gain
                extra[0] = 0.0
            ti.simt.block.sync()
            j = lane
            while j < self.dof:
                matrix[j, 1] = gain * (world + j + 1)
                extra[j + 1] = gain * (j + 1)
                ti.atomic_add(extra[0], gain)
                j += 32
            ti.simt.block.sync()
            if lane == 0:
                self.result[world] = self.shared_sum(matrix, extra, fixed)

    def graph(self):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.step, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32))
        return builder.compile()
"""


@pytest.mark.parametrize("square", [False, True])
@pytest.mark.parametrize("model_block", [False, True])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_package_model_shared_arrays_bind_native_storage(tmp_path, monkeypatch, square, model_block):
    import ctypes as C
    driver = C.WinDLL("nvcuda.dll") if sys.platform == "win32" else C.CDLL("libcuda.so.1")
    device, capacity = C.c_int(), C.c_int()
    assert driver.cuDeviceGet(C.byref(device), 0) == 0
    if driver.cuDeviceGetAttribute(C.byref(capacity), 97, device) != 0:
        assert driver.cuDeviceGetAttribute(C.byref(capacity), 8, device) == 0
    source = MODEL_SHARED_SOURCE
    if model_block:
        source = source.replace("ti.loop_config(block_dim=32)",
                                "workers = ti.static(128 if self.dof > 32 else 32)\n"
                                "        ti.loop_config(block_dim=workers)")
        source = source.replace("self.n * 32", "self.n * workers").replace(
            "i % 32, i // 32", "i % workers, i // workers").replace("j += 32", "j += workers")
        # This allocation ties prepare geometry to the same selector used by
        # device indexing; no per-DOF compilation or fixed shared-array proxy.
        source = source.replace("(self.dof + 1,)", "(self.dof + workers,)")
        source = source.replace("            matrix = ti.simt.block.SharedArray",
                                "            local = ti.Vector.zero(ti.f32, (self.dof + workers - 1) // workers)\n"
                                "            local[0] = gain\n"
                                "            matrix = ti.simt.block.SharedArray")
        source = source.replace("self.shared_sum(matrix, extra, fixed)",
                                "self.shared_sum(matrix, extra, fixed) + local[0] - gain")
    if square:
        source = source.replace("(self.dof, 2)", "(self.dof, self.dof + 1)")
        if model_block:
            source = source.replace("128 if self.dof > 32", "64 if self.dof > 32")
            source = source.replace("    @ti.kernel\n    def step(self, gain: ti.f32):",
                                    "    @ti.func\n    def shared_step(self, gain: ti.f32):")
            source = source.replace("    def graph(self):", "    @ti.kernel\n"
                                    "    def step(self, gain: ti.f32):\n"
                                    "        self.shared_step(gain)\n\n    def graph(self):")
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cuda)
    assert sys.modules["definition_fixture.model"].constructors == 0
    entries = [entry for entry in manifest["entries"].values() if entry["definition"].endswith("Model.step")]
    assert len(entries) == 1
    if model_block:
        assert manifest["version"] >= 29
        assert entries[0]["model_blocks"]
    expected = package.Model(3, 17)
    expected.step(2.0)
    np.testing.assert_array_equal(expected.result.to_numpy(), 2 * (np.arange(3) + 52))
    probe = tmp_path / "consume_shared.py"
    probe.write_text(textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime('cuda') as runtime, load_package(runtime, sys.argv[1]) as code:
            capacity = int(sys.argv[3])
            cases = ((1, 1), (3, 17), (2, 32), (4, 33), (7, 65))
            if sys.argv[2] == 'False':
                cases += ((5, 129),)
                if capacity >= 60528:
                    cases += ((2, 5000),)  # exceeds the historical 48 KiB cap
            elif capacity >= 68128:
                cases += ((5, 129),)
            elif capacity >= 49304:
                cases += ((5, 110),)
            models = [code.host.Model(n, dof) for n, dof in cases]
            for model in reversed(models):
                model.step(3.0)
            for model in models:
                np.testing.assert_array_equal(model.result.to_numpy(), 3 * (np.arange(model.n) + 3 * model.dof + 1))
            calls = [model.graph().bind({'gain': 5.0}) for model in models]
            for call in reversed(calls):
                call.run()
            for model in models:
                np.testing.assert_array_equal(model.result.to_numpy(), 5 * (np.arange(model.n) + 3 * model.dof + 1))
            calls[0].close()
            calls[1].run()
            for call in calls[1:]:
                call.close()
            oversized = code.host.Model(1, capacity // 4 + 1)
            for action in (lambda: oversized.step(1.0), lambda: oversized.graph().bind({'gain': 1.0})):
                try:
                    action()
                except (RuntimeError, ValueError) as error:
                    assert 'shared memory' in str(error), str(error)
                else:
                    raise AssertionError('oversized shared allocation was submitted')
            models[1].step(7.0)
            np.testing.assert_array_equal(models[1].result.to_numpy(), 7 * (np.arange(3) + 52))
        print('MODEL_SHARED_SOURCE_FREE_OK')
    """), encoding="utf8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
               TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
               TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir())
    result = subprocess.run([sys.executable, "-B", str(probe), str(tmp_path / "sdk"), str(square), str(capacity.value)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MODEL_SHARED_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("invalid", ["device_extent", "serial_scope", "device_block", "nested_block", "serial_block"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_package_model_shared_arrays_reject_unproven_storage(tmp_path, monkeypatch, invalid):
    source = MODEL_SHARED_SOURCE
    if invalid == "device_extent":
        source = source.replace("(self.dof, 2)", "(ti.cast(self.result[0], ti.i32), 2)")
        message = "immutable declared dimensions"
    elif invalid == "serial_scope":
        source = source.replace("        ti.loop_config(block_dim=32)",
                                "        unused = ti.simt.block.SharedArray((self.dof,), ti.f32)\n"
                                "        ti.loop_config(block_dim=32)")
        message = "parallel kernel range"
    elif invalid == "device_block":
        source = source.replace("block_dim=32", "block_dim=ti.cast(self.result[0], ti.i32)")
        message = "immutable AOT model expression"
    elif invalid == "nested_block":
        source = source.replace("            lane, world =", "            ti.loop_config(block_dim=self.dof)\n"
                                "            lane, world =")
        message = "outer kernel range"
    else:
        source = source.replace("block_dim=32", "serialize=True, block_dim=self.dof")
        message = "parallel range"
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(Exception, match=message):
        ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cuda)
    assert not (tmp_path / "sdk" / "package.json").exists()


MODEL_LOCAL_TENSOR_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, dof: int):
        global constructors
        constructors += 1
        self.n = n
        self.dof = dof
        self.vector = ti.Vector.field(self.dof, ti.f32, shape=self.n)
        self.result = ti.field(ti.f32, shape=self.n)

    @ti.func
    def solve(self, i, gain):
        forward = ti.Vector.zero(ti.f32, self.dof)
        matrix = ti.Matrix.zero(ti.f32, self.dof, 2)
        for j in range(self.dof):
            value = gain * (i + 1)
            if j > 0:
                value += forward[j - 1]
            forward[j] = value
            matrix[j, 1] = forward[j] + matrix[j, 0]
        copied = forward
        copied[0] = 123.0
        return matrix[self.dof - 1, 1] + forward[0]

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in range(1, self.n):
            total = 0.0
            for repeat in range(2):
                if i % 2 == 1:
                    total += self.solve(i, gain + repeat)
            self.result[i] = total

    @ti.kernel
    def accumulate(self, gain: ti.f32):
        for i in range(1, self.n):
            values = ti.Vector.zero(ti.f32, self.dof + 1)
            for j in range(self.dof + 1):
                values[j] = gain * (i + 1)
                self.result[i] += values[j]

    def graph(self, mixed=False):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.step, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32))
        if mixed:
            builder.dispatch(self.accumulate, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32))
        return builder.compile()
"""


MODEL_TENSOR_COPY_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, dof: int):
        global constructors
        constructors += 1
        self.n = n
        self.dof = dof
        self.values = ti.Vector.field(self.dof, ti.f32, shape=self.n)
        self.output = ti.Vector.field(self.dof, ti.f32, shape=self.n)
        self.result = ti.field(ti.f32, shape=self.n)

    @ti.func
    def copy(self, i, gain):
        self.output[i] = ti.Vector.zero(ti.f32, self.dof)
        self.values[i] = gain
        for j in range(self.dof):
            self.values[i][j] += i + 2 * j
        left = right = self.values[i]
        left[0] += 2.0
        self.values[i] = 0.0
        self.output[i] = right
        # Both RHS elements must be read before either destination changes.
        self.values[i], self.output[i] = self.output[i], self.values[i]
        right = self.output[i]
        return left[0] + right[0]

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in range(self.n):
            self.result[i] = self.copy(i, gain)

    def graph(self):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.step, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32))
        return builder.compile()
"""


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("helper_kind", ["func", "real_func"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_model_tensor_value_copies(tmp_path, monkeypatch, matrix, helper_kind):
    if helper_kind == "real_func" and ti.cfg.arch == ti.vulkan:
        pytest.skip("Vulkan does not support real functions")
    source = MODEL_TENSOR_COPY_SOURCE
    if matrix:
        source = source.replace("ti.Vector.field(self.dof,", "ti.Matrix.field(self.dof, 2,")
        source = source.replace("ti.Vector.zero(ti.f32, self.dof)", "ti.Matrix.zero(ti.f32, self.dof, 2)")
        source = source.replace("for j in range(self.dof):\n            self.values[i][j] += i + 2 * j",
                                "for j, k in ti.ndrange(self.dof, 2):\n            self.values[i][j, k] += i + 2 * j + 3 * k")
        source = source.replace("left[0]", "left[0, 1]").replace("right[0]", "right[0, 1]")
    if helper_kind == "real_func":
        source = source.replace("@ti.func\n    def copy(self, i, gain):",
                                "@ti.real_func\n    def copy(self, i: ti.i32, gain: ti.f32) -> ti.f32:")
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch)
    assert sys.modules["definition_fixture.model"].constructors == 0
    entries = [entry for entry in manifest["entries"].values() if entry["definition"].endswith("Model.step")]
    assert len(entries) == 1
    # One constructor, chained local copies, two swap snapshots, reassignment.
    assert sum(item["kind"] == "local_tensor_workspace" for item in entries[0]["captures"]) == 6
    expected = package.Model(7, 3)
    expected.step(4.0)
    values = 4.0 + np.arange(7)[:, None] + 2 * np.arange(3)[None, :]
    if matrix:
        values = values[:, :, None] + 3 * np.arange(2)
    np.testing.assert_array_equal(expected.values.to_numpy(), values)
    np.testing.assert_array_equal(expected.output.to_numpy(), 0.0)
    np.testing.assert_array_equal(expected.result.to_numpy(), 6.0 + np.arange(7) + (3 if matrix else 0))
    probe = tmp_path / "consume_copies.py"
    probe.write_text(textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        def check(model, gain):
            values = gain + np.arange(model.n)[:, None] + 2 * np.arange(model.dof)[None, :]
            if sys.argv[3] == '2':
                values = values[:, :, None] + 3 * np.arange(2)
            np.testing.assert_array_equal(model.values.to_numpy(), values)
            np.testing.assert_array_equal(model.output.to_numpy(), 0.)
            np.testing.assert_array_equal(model.result.to_numpy(),
                gain + 2. + np.arange(model.n) + (3 if sys.argv[3] == '2' else 0))
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            models = [code.host.Model(7, 3), code.host.Model(19, 65)]
            calls = []
            for model, gain in zip(models, (4., 7.)):
                model.step(gain)
                check(model, gain)
                call = model.graph().bind({'gain': gain + 1.})
                assert call.info['resource_bytes'] == model.n * model.dof * int(sys.argv[3]) * 4 * 6
                calls.append(call)
            original_array, original_wait = runtime.ndarray, runtime.wait
            def forbidden(*args, **kwargs):
                raise AssertionError('prepared copy allocated or waited on the Host')
            runtime.ndarray = runtime.wait = forbidden
            try:
                for unused in range(2):
                    for call in reversed(calls):
                        call.run()
            finally:
                runtime.ndarray, runtime.wait = original_array, original_wait
            calls[0].close()
            calls[1].run()
            for model, gain in zip(models, (5., 8.)):
                check(model, gain)
            calls[1].close()
        print('MODEL_TENSOR_COPIES_SOURCE_FREE_OK')
    """), encoding="utf8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
               TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
               TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir())
    result = subprocess.run([sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"],
                             "2" if matrix else "1"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MODEL_TENSOR_COPIES_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("mismatch", ["extent", "dtype", "rank"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_model_tensor_copy_rejects_incompatible_values(tmp_path, monkeypatch, mismatch):
    source = MODEL_TENSOR_COPY_SOURCE
    replacement = {
        "extent": "ti.Vector.field(self.n, ti.f32, shape=self.n)",
        "dtype": "ti.Vector.field(self.dof, ti.i32, shape=self.n)",
        "rank": "ti.Matrix.field(self.dof, 2, ti.f32, shape=self.n)",
    }[mismatch]
    source = source.replace("self.output = ti.Vector.field(self.dof, ti.f32, shape=self.n)",
                            "self.output = " + replacement)
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="matching element type and extent relations"):
        ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch)
    assert not (tmp_path / "sdk" / "package.json").exists()


@pytest.mark.parametrize("ndrange", [False, True])
@pytest.mark.parametrize("helper_kind", ["func", "real_func", "nested_real_func"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_model_local_tensors_use_invocation_workspace(tmp_path, monkeypatch, ndrange, helper_kind):
    source = MODEL_LOCAL_TENSOR_SOURCE
    if helper_kind != "func":
        if ti.cfg.arch == ti.vulkan:
            pytest.skip("Vulkan does not support real functions")
        source = source.replace("@ti.func\n    def solve(self, i, gain):",
                                "@ti.real_func\n    def solve(self, i: ti.i32, gain: ti.f32) -> ti.f32:")
        if helper_kind == "nested_real_func":
            source = source.replace("def solve(self,", "def inner(self,")
            source += """
    @ti.real_func
    def solve(self, i: ti.i32, gain: ti.f32) -> ti.f32:
        return self.inner(i, gain)
"""
    if ndrange:
        source = source.replace("for i in range(1, self.n):", "for i, unused in ti.ndrange((1, self.n), 1):")
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch)
    assert manifest["version"] == 27
    assert sys.modules["definition_fixture.model"].constructors == 0
    entries = [entry for entry in manifest["entries"].values() if entry["definition"].endswith("Model.step")]
    assert len(entries) == 1
    captures = entries[0]["captures"]
    workspaces = [item for item in captures if item["kind"] == "local_tensor_workspace"]
    assert len(workspaces) == 3  # vector, matrix, independent value copy
    expected = package.Model(7, 3)
    expected.step(2.0)
    answer = np.where(np.arange(7) % 2 == 1, (np.arange(7) + 1) * 5 * 4, 0)
    np.testing.assert_array_equal(expected.result.to_numpy(), answer)
    probe = tmp_path / "consume_locals.py"
    probe.write_text(textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            models = [code.host.Model(7, 3), code.host.Model(19, 65), code.host.Model(11, 129)]
            gains = [2.0, 4.0, 7.0]
            original_wait, waits = runtime.wait, []
            def observed_wait():
                waits.append(True)
                return original_wait()
            runtime.wait = observed_wait
            for model, gain in zip(models, gains):
                model.step(gain)
            runtime.wait = original_wait
            assert not waits, 'void local workspace call added a Runtime.wait'
            for model, gain in zip(models, gains):
                indices = np.arange(model.n)
                expected = np.where(indices % 2 == 1, (indices + 1) * (2 * gain + 1) * (model.dof + 1), 0)
                np.testing.assert_array_equal(np.frombuffer(model.result.read(), np.float32), expected)
            runtime.wait()
            resources = set(runtime._resources)
            for model in models:
                model.step(3.0)
            runtime.wait()
            assert runtime._resources == resources, 'workspace resources leaked'
            from taichi_forge_runtime.aot import _package
            original_capture = _package.capture_value
            def changed_after_snapshot(owner, spec):
                value = original_capture(owner, spec)
                if owner is models[0] and spec['name'] == 'dof':
                    owner.dof = 2**30
                return value
            _package.capture_value = changed_after_snapshot
            try:
                models[0].step(2.0)
            finally:
                _package.capture_value = original_capture
                models[0].dof = 3
            indices = np.arange(7)
            np.testing.assert_array_equal(models[0].result.to_numpy(),
                np.where(indices % 2 == 1, (indices + 1) * 5 * 4, 0))
            runtime.wait()
            assert runtime._resources == resources
            model = models[0]
            original_dof = model.dof
            for invalid in (0, 2**30):
                model.dof = invalid
                try:
                    model.step(1.0)
                except ValueError as error:
                    assert 'tensor axes' in str(error) or 'indexing range' in str(error)
                else:
                    raise AssertionError('invalid workspace extent accepted')
                assert runtime._resources == resources
            model.dof = original_dof
            model.n = 1
            model.step(1.0)  # empty range, no model-sized scratch allocation
            runtime.wait()
            assert runtime._resources == resources
            model.n = 7
            graphs = [model.graph(mixed=True) for model in models]
            prepared = [graph.bind({'gain': gain}) for graph, gain in zip(graphs, gains)]
            for call, model in zip(prepared, models):
                assert call.info['resource_bytes'] == (model.n - 1) * (5 * model.dof + 1) * 4
            # A second bind of the same Graph also owns independent scratch.
            extra = graphs[0].bind({'gain': 9.0})
            from taichi_forge_runtime.aot import _package_graph
            original_graph_capture = _package_graph.capture_value
            _package_graph.capture_value = changed_after_snapshot
            try:
                frozen = graphs[0].bind({'gain': 2.0})
            finally:
                _package_graph.capture_value = original_graph_capture
                models[0].dof = 3
            original_array = runtime.ndarray
            def no_allocation(*args, **kwargs):
                raise AssertionError('prepared replay allocated a Python workspace')
            runtime.ndarray = no_allocation
            waits.clear()
            runtime.wait = observed_wait
            try:
                for _ in range(2):
                    extra.run()
                    frozen.run()
                    for call in reversed(prepared):
                        call.run()
            finally:
                runtime.ndarray = original_array
                runtime.wait = original_wait
            assert not waits, 'prepared replay added a Runtime.wait'
            # Closing an in-flight instance must leave the others usable.
            extra.close()
            frozen.close()
            prepared[0].close()
            prepared[1].run()
            for model, gain in zip(models, gains):
                indices = np.arange(model.n)
                expected = np.where(indices % 2 == 1, (indices + 1) * (2 * gain + 1) * (model.dof + 1), 0)
                expected += np.where(indices >= 1, (indices + 1) * gain * (model.dof + 1), 0)
                np.testing.assert_array_equal(model.result.to_numpy(), expected)
            for call in prepared[1:]:
                call.close()
            runtime.wait()
            assert runtime._resources == resources, 'prepared workspaces leaked'
            for invalid in (0, 2**30):
                models[0].dof = invalid
                try:
                    graphs[0].bind({'gain': 1.0})
                except ValueError as error:
                    assert 'tensor axes' in str(error) or 'indexing range' in str(error)
                else:
                    raise AssertionError('invalid Graph workspace extent accepted')
                assert runtime._resources == resources
            models[0].dof = 3
        print('MODEL_LOCAL_TENSOR_SOURCE_FREE_OK')
    """), encoding="utf8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
               TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
               TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir())
    result = subprocess.run([sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MODEL_LOCAL_TENSOR_SOURCE_FREE_OK" in result.stdout


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_real_local_workspaces_share_code_across_call_ranges(tmp_path, monkeypatch):
    source = MODEL_LOCAL_TENSOR_SOURCE.replace(
        "@ti.func\n    def solve(self, i, gain):",
        "@ti.real_func\n    def solve(self, i: ti.i32, gain: ti.f32) -> ti.f32:",
    )
    begin = source.index("    @ti.kernel\n    def step")
    end = source.index("    @ti.kernel\n    def accumulate")
    source = source[:begin] + """
    @ti.kernel
    def step(self, gain: ti.f32):
        for i in range(self.n):
            self.result[i] = self.solve(i, gain)
        for i in range(1, self.n):
            self.result[i] += self.solve(i, gain)
        self.result[0] += self.solve(0, gain)

""" + source[end:]
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch)
    entry = next(item for item in manifest["entries"].values() if item["definition"].endswith("Model.step"))
    assert len([item for item in entry["captures"] if item["kind"] == "local_tensor_workspace"]) == 3
    expected = package.Model(7, 3)
    expected.step(2.0)
    np.testing.assert_array_equal(expected.result.to_numpy(), (np.arange(7) + 1) * 16)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        for n, dof in ((1, 3), (19, 65), (11, 129)):
            model = code.host.Model(n, dof)
            with model.graph().bind({"gain": 3.0}) as prepared:
                assert prepared.info["resource_bytes"] == n * dof * 4 * 4
                prepared.run()
                np.testing.assert_array_equal(model.result.to_numpy(), (np.arange(n) + 1) * 6 * (dof + 1))


@pytest.mark.parametrize("failure", ["device_extent", "mutable_alias", "device_range", "real_device_range", "recursive_real_func"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproven_model_local_workspaces(tmp_path, monkeypatch, failure):
    source = MODEL_LOCAL_TENSOR_SOURCE
    if failure == "device_extent":
        source = source.replace("ti.Vector.zero(ti.f32, self.dof)", "ti.Vector.zero(ti.f32, ti.cast(self.result[0], ti.i32))")
        message = "immutable declared dimensions"
    elif failure == "mutable_alias":
        source = source.replace("forward = ti.Vector.zero(ti.f32, self.dof)",
                                "size = self.dof\n        forward = ti.Vector.zero(ti.f32, size)")
        message = "immutable declared dimensions"
    elif failure in ("device_range", "real_device_range"):
        source = source.replace("range(1, self.n)", "range(1, ti.cast(self.result[0], ti.i32))", 1)
        if failure == "real_device_range":
            source = source.replace("@ti.func\n    def solve(self, i, gain):",
                                    "@ti.real_func\n    def solve(self, i: ti.i32, gain: ti.f32) -> ti.f32:")
        message = "declared range/ndrange iteration domain"
    else:
        source = source.replace("@ti.func\n    def solve(self, i, gain):",
                                "@ti.real_func\n    def solve(self, i: ti.i32, gain: ti.f32) -> ti.f32:")
        source = source.replace("forward = ti.Vector.zero(ti.f32, self.dof)",
                                "if i > 0:\n            return self.solve(i - 1, gain)\n        forward = ti.Vector.zero(ti.f32, self.dof)")
        message = "Recursive real_func captures"
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match=message):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


def _local_struct_source():
    return MODEL_TENSOR_SOURCE.replace(
        "        scope.rows = scope.struct_field({\n            'J': ti.types.vector(self.layout.dof, ti.f32),\n"
        "            'fixed': ti.types.vector(3, ti.f32),\n            'result': ti.f32,\n"
        "        }, shape=scope.dim('n'))",
        "        Row = ti.types.struct(result=ti.f32, J=ti.types.vector(self.layout.dof, ti.f32),\n"
        "                              fixed=ti.types.vector(3, ti.f32))\n"
        "        scope.rows = scope.struct_field(Row.members, shape=scope.dim('n'),\n"
        "            member_initialization={name: 'zero' for name in Row.members if name not in ('fixed',)})",
    )


@pytest.mark.parametrize("change", ["unknown", "policy", "model", "rebound"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_local_struct_rejects_unproven_member_policies(tmp_path, monkeypatch, change):
    source = _local_struct_source()
    policy = "{name: 'zero' for name in Row.members if name not in ('fixed',)}"
    replacement = {
        "unknown": "{'missing': 'zero'}",
        "policy": "{'J': 'snapshot'}",
        "model": "{name: 'zero' for name in range(n)}",
    }.get(change)
    if change == "rebound":
        source = source.replace("        scope.rows =", "        Row = ti.types.struct(J=ti.i32)\n        scope.rows =")
    else:
        source = source.replace(policy, replacement)
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises((ValueError, KeyError), match="[Mm]ember|Row|type|model tensor axis"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("change", ["extent", "member", "order"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_local_struct_checks_actual_type_before_allocation(tmp_path, monkeypatch, change):
    statement = {"extent": "self.layout.dof += 1", "member": "Row.members['result'] = ti.i32",
                 "order": "Row.members['result'] = Row.members.pop('result')"}[change]
    source = _local_struct_source().replace("        scope.rows =", f"        {statement}\n        scope.rows =")
    package = _source_package(tmp_path, monkeypatch, source, initializer="from .model import Model, Rows, Reader\n")
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        before = set(runtime._resources)
        with pytest.raises(ValueError, match="Local element type"):
            code.host.Rows(2, 3)
        assert set(runtime._resources) == before


@pytest.mark.parametrize("real_helper,local_struct", [(False, False), (True, False), (False, True)])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_infers_model_tensor_axes_without_constructor_samples(tmp_path, monkeypatch, real_helper, local_struct):
    if real_helper and ti.cfg.arch == ti.vulkan:
        pytest.skip("Vulkan does not support real functions")
    source = _local_struct_source() if local_struct else MODEL_TENSOR_SOURCE
    if real_helper:
        source = source.replace(
            "@ti.func\n    def touch(self, i, j, gain):",
            "@ti.real_func\n    def touch(self, i: ti.i32, j: ti.i32, gain: ti.f32):",
        )
    package = _source_package(tmp_path, monkeypatch, source, initializer="from .model import Model, Rows, Reader\n")
    manifest = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch)
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert manifest["version"] == (35 if local_struct else 26)
    if local_struct:
        assert manifest["owners"]["definition_fixture.model.Rows"]["rows"]["member_initialization"] == {
            "J": "zero",
            "result": "zero",
        }
    model_entries = [
        entry for entry in manifest["entries"].values() if entry["definition"].endswith(("Model.step", "Rows.step"))
    ]
    assert len(model_entries) == 2 and not manifest["entry_families"]
    assert any(
        item["kind"] == "model_tensor_field" for entry in manifest["entries"].values() for item in entry["captures"]
    )
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        for name in ("Model", "Rows"):
            a = getattr(library.host, name)(2, 3)
            b = getattr(library.host, name)(3, 65)
            original = getattr(package, name)(2, 3)
            original.step(2.0)
            a.step(2.0)
            b.step(-1.0)
            arrays = ("v", "matrix", "fixed") if name == "Model" else ("J", "fixed", "result")
            left = a if name == "Model" else a.rows
            right = b if name == "Model" else b.rows
            reference = original if name == "Model" else original.rows
            for attribute in arrays:
                expected = getattr(reference, attribute).to_numpy()
                np.testing.assert_array_equal(
                    np.frombuffer(getattr(left, attribute).read(), np.float32).reshape(expected.shape), expected
                )
            field = right.v if name == "Model" else right.J
            assert field.shape == (3,) and field.element_shape == (65,) and field.n == 65
            np.testing.assert_array_equal(
                np.frombuffer(field.read(), np.float32).reshape(3, 65), -(np.arange(3)[:, None] + np.arange(65))
            )
            assert field._group_owner is not (left.v if name == "Model" else left.J)._group_owner
            if name == "Rows":
                graph = b.graph()
                with graph.bind({"gain": 0.5}) as prepared:
                    b.layout.dof = 66
                    for operation in (lambda: b.step(1.0), lambda: graph.bind({"gain": 1.0})):
                        with pytest.raises(ValueError, match="Derived dimension"):
                            operation()
                    prepared.run()
                np.testing.assert_array_equal(
                    np.frombuffer(field.read(), np.float32).reshape(3, 65),
                    -0.5 * (np.arange(3)[:, None] + np.arange(65)),
                )
        registered = set(runtime._resources)
        with pytest.raises((ValueError, RuntimeError)):
            library.host.Model(2, 0)
        assert runtime._resources == registered
        reader = library.host.Reader(2, 17)
        reader.step(3.0)
        assert reader.alias is reader.source.v
        assert reader._forge_package_field_groups == {}
        np.testing.assert_array_equal(reader.source.v.to_numpy(), 3 * (np.arange(2)[:, None] + np.arange(17)))
    probe = tmp_path / "consume_model_tensor.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as library:
            for name in ('Model', 'Rows'):
                model = getattr(library.host, name)(2, 129)
                model.step(3.0)
                field = model.v if name == 'Model' else model.rows.J
                np.testing.assert_array_equal(np.frombuffer(field.read(), np.float32).reshape(2, 129), 3 * (np.arange(2)[:, None] + np.arange(129)))
            reader = library.host.Reader(2, 129)
            reader.step(-2.0)
            assert reader.alias is reader.source.v and reader._forge_package_field_groups == {}
            np.testing.assert_array_equal(reader.alias.to_numpy(), -2 * (np.arange(2)[:, None] + np.arange(129)))
        print('MODEL_TENSOR_PACKAGE_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    consumer = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert consumer.returncode == 0, consumer.stdout + consumer.stderr
    assert "MODEL_TENSOR_PACKAGE_SOURCE_FREE_OK" in consumer.stdout


@pytest.mark.parametrize("replacement", ["choices=(self.layout.dof + 1,)", "choices=(self.n,)"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproven_model_tensor_static_domains(tmp_path, monkeypatch, replacement):
    source = MODEL_TENSOR_SOURCE.replace("choices=(self.layout.dof,)", replacement)
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="Static domain"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_keeps_computed_statics_without_tensor_axis_use_rejected(tmp_path, monkeypatch):
    source = MODEL_TENSOR_SOURCE.replace("ti.types.vector(self.layout.dof, ti.f32)", "ti.types.vector(3, ti.f32)")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="proven model tensor axis"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("rebind", [None, "local", "owner", "delete"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_record_tensor_axis_requires_single_binding_provenance(tmp_path, monkeypatch, rebind):
    source = MODEL_TENSOR_SOURCE.replace(
        "self.layout = SimpleNamespace(dof=dof)",
        "model_layout = SimpleNamespace(dof=dof)\n        self.layout = model_layout",
    ).replace("'J': ti.types.vector(self.layout.dof, ti.f32)",
              "'J': ti.types.vector(model_layout.dof, ti.f32)")
    extra = {"local": "model_layout = SimpleNamespace(dof=dof + 1)",
             "owner": "self.layout = SimpleNamespace(dof=dof + 1)",
             "delete": "del model_layout"}.get(rebind)
    if extra:
        source = source.replace("        scope.rows = scope.struct_field({", f"        {extra}\n        scope.rows = scope.struct_field({{")
    package = _source_package(tmp_path, monkeypatch, source, initializer="from .model import Model, Rows\n")
    if rebind:
        with pytest.raises(ti.aot.PackageExportError, match="proven model tensor axis"):
            ti.aot.export_package(package, tmp_path / "sdk")
        assert not (tmp_path / "sdk").exists()
        return
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert not manifest["entry_families"]
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        for dof in (17, 65):
            model = code.host.Rows(2, dof)
            model.step(2.0)
            np.testing.assert_array_equal(model.rows.J.to_numpy(), 2 * (np.arange(2)[:, None] + np.arange(dof)))


FIELD_SOURCE = """
import taichi_forge as ti
constructors = 0

@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int):
        global constructors
        constructors += 1
        self.n = n
        self.a = ti.field(ti.f32, shape=(self.n, m))
        self.v = ti.Vector.field(3, ti.f32, shape=n)
        self.mat = ti.Matrix.field(2, 2, ti.f32, shape=m)
        self.total = ti.field(ti.f32, shape=())

    @ti.kernel
    def step(self, gain: ti.f32):
        for i, j in self.a:
            self.a[i, j] += gain * (i * 10 + j)
        for i in self.v:
            self.v[i] += ti.Vector([gain, i, self.a.shape[1]])
        for j in self.mat:
            self.mat[j] += ti.Matrix([[gain, j], [2, 3]])
        self.total[None] += gain * self.n
"""


@pytest.mark.parametrize("real_helper", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_dense_fields_use_native_group_layout_and_dynamic_instances(tmp_path, monkeypatch, real_helper):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    if real_helper and ti.cfg.arch == ti.vulkan:
        pytest.skip("Vulkan does not support real functions")
    source = FIELD_SOURCE
    if real_helper:
        source = source.replace("self.a[i, j] += gain * (i * 10 + j)", "self.touch(i, j, gain)")
        source = source.replace(
            "    @ti.kernel\n    def step",
            """    @ti.real_func
    def touch(self, i: ti.i32, j: ti.i32, gain: ti.f32):
        self.a[i, j] += gain * (i * 10 + j)

    @ti.kernel
    def step""",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(7, 3)
    original.step(2.0)
    expected = {name: getattr(original, name).to_numpy() for name in ("a", "v", "mat", "total")}
    root = ti.lang.impl.get_runtime().prog.get_snode_root(original.a.snode.ptr.get_snode_tree_id())
    expected_layout = _compiled_nodes(root)
    result = ti.aot.export_package(package, tmp_path / "sdk")
    original.step(0.0)  # Existing JIT remains callable after structural export.
    np.testing.assert_array_equal(original.a.to_numpy(), expected["a"])
    assert sys.modules["definition_fixture.model"].constructors == 1  # no export-time model
    assert len(result["field_groups"]) == 1
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        a, b = library.host.Model(7, 3), library.host.Model(19, 5)
        group, dimensions = next(iter(a._forge_package_field_groups.values()))
        assert group.layout["nodes"] == expected_layout
        assert dimensions == {
            "_forge_package_extent_arg_n": 7,
            "_forge_package_extent_arg_m": 3,
            "_forge_package_extent_self_n": 7,
        }
        assert a.a._group_owner is a.v._group_owner is a.mat._group_owner is a.total._group_owner
        assert a.a._group_owner is not b.a._group_owner
        assert not np.frombuffer(a.a.read(), np.float32).any()
        a.step(2.0)
        b.step(-3.0)
        for name, data in expected.items():
            actual = np.frombuffer(getattr(a, name).read(), np.float32).reshape(data.shape)
            np.testing.assert_array_equal(actual, data)
        np.testing.assert_array_equal(
            np.frombuffer(b.a.read(), np.float32).reshape(19, 5), -3 * (np.arange(19)[:, None] * 10 + np.arange(5))
        )
        assert np.frombuffer(b.total.read(), np.float32).item() == -57
        b.step(1.0)
        np.testing.assert_array_equal(np.frombuffer(a.a.read(), np.float32).reshape(7, 3), expected["a"])
        registered = set(runtime._resources)
        with pytest.raises((ValueError, RuntimeError)):
            library.host.Model(0, 3)
        assert runtime._resources == registered
    if not real_helper:
        probe = tmp_path / "consume_fields.py"
        probe.write_text(
            textwrap.dedent("""
            import importlib.abc
            import sys
            import numpy as np
            class NoFrontend(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                        raise AssertionError('Private frontend/source import: ' + fullname)
            sys.meta_path.insert(0, NoFrontend())
            from taichi_forge_runtime.aot import Runtime, load_package
            with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as library:
                model = library.host.Model(3, 2)
                model.step(2.0)
                np.testing.assert_array_equal(np.frombuffer(model.a.read(), np.float32), [0, 2, 20, 22, 40, 42])
                assert model.a._group_owner is model.v._group_owner is model.mat._group_owner
                assert np.frombuffer(model.total.read(), np.float32).item() == 6
            print('FIELD_PACKAGE_SOURCE_FREE_OK')
        """),
            encoding="utf8",
        )
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        )
        consumer = subprocess.run(
            [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), result["target"]],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert consumer.returncode == 0, consumer.stdout + consumer.stderr
        assert "FIELD_PACKAGE_SOURCE_FREE_OK" in consumer.stdout


@pytest.mark.parametrize(
    "replacement",
    [
        "self.n = n\n        self.n += 1",
        "if n > 0:\n            self.n = n",
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproved_field_extent_aliases(tmp_path, monkeypatch, replacement):
    package = _source_package(tmp_path, monkeypatch, FIELD_SOURCE.replace("self.n = n", replacement))
    with pytest.raises(ti.aot.PackageExportError, match="Field extents require"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_keeps_actual_host_field_extent_expressions(tmp_path, monkeypatch):
    source = FIELD_SOURCE.replace("self.n = n", "self.n = n\n        self.bump()")
    source += "\n    def bump(self):\n        self.n += 1\n"
    package = _source_package(tmp_path, monkeypatch, source)
    result = ti.aot.export_package(package, tmp_path / "sdk")
    original = package.Model(7, 3)
    original.step(1.0)
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        model = library.host.Model(7, 3)
        assert model.a.shape == (8, 3) and model.v.shape == (7,)
        model.step(1.0)
        np.testing.assert_array_equal(np.frombuffer(model.a.read(), np.float32).reshape(8, 3), original.a.to_numpy())


@pytest.mark.parametrize(
    "allocation,diagnostic",
    [
        ("ti.field(ti.f32, shape=2147483648)", "positive i32 dimension"),
        ("ti.field(ti.types.struct(x=ti.f32), shape=())", "scalar/vector/matrix elements"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_field_preflight_rejection_preserves_normal_jit(tmp_path, monkeypatch, allocation, diagnostic):
    source = FIELD_SOURCE.replace("ti.field(ti.f32, shape=())", allocation)
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    # Earlier fields in the rejected owner must not leave a pending builder or
    # unplaced fields that break later, otherwise unrelated JIT execution.
    output = ti.field(ti.i32, shape=3)

    @ti.kernel
    def fill():
        for i in output:
            output[i] = i + 7

    fill()
    np.testing.assert_array_equal(output.to_numpy(), [7, 8, 9])


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_absolute_and_lazy_imports_keep_their_code_owner(tmp_path, monkeypatch):
    source = SOURCE.replace("from .device_helpers import scale", "from definition_fixture.device_helpers import scale")
    source += """

def imports(x):
    import definition_fixture.unimported
    import definition_fixture.unimported as alias
    from definition_fixture import unimported as from_root
    from definition_fixture.unimported import available_without_a_test as call
    return (definition_fixture.unimported.available_without_a_test(x)
            + alias.available_without_a_test(x) + from_root.available_without_a_test(x) + call(x))
"""
    package = _source_package(tmp_path, monkeypatch, source)
    ti.aot.export_package(package, tmp_path / "sdk")
    probe = tmp_path / "consume_imports.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError('Private frontend/source import: ' + fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime('cpu') as runtime, load_package(runtime, sys.argv[1]) as a:
            with load_package(runtime, sys.argv[1]) as b:
                assert a.host.model.imports(2) == 40
                assert b.host.model.imports(3) == 60
                assert a.host.unimported is not b.host.unimported
                assert a.host.unimported.available_without_a_test.package is a
                assert b.host.unimported.available_without_a_test.package is b
                assert 'definition_fixture' not in sys.modules
            assert a.host.model.imports(4) == 80
        print('PACKAGE_IMPORT_ISOLATION_OK')
        """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PACKAGE_IMPORT_ISOLATION_OK" in result.stdout


HOST_BUFFER_SOURCE = """
import taichi_forge as ti

@ti.kernel
def update(source: ti.types.ndarray(ti.f32, 1), output: ti.types.ndarray(ti.f32, 1), offset: ti.f32):
    for i in source:
        output[i] = source[i] + offset
        source[i] += 1

@ti.kernel
def transform(values: ti.types.ndarray(ti.types.vector(3, ti.f32), 1)):
    for i in values:
        values[i] += ti.Vector([1, 2, 3])
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_host_buffers_preserve_types_and_reject_unproved_aliases(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, HOST_BUFFER_SOURCE, initializer="from . import model\n")
    original = sys.modules["definition_fixture.model"]
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        resources = set(runtime._resources)
        for count in (7, 19):
            expected = np.arange(count, dtype=np.float32)
            expected_output = np.zeros(count, dtype=np.float32)
            original.update(expected, expected_output, 2.0)
            actual = np.arange(count, dtype=np.float32)
            output = np.zeros(count, dtype=np.float32)
            library.host.model.update(actual, output, 2.0)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(output, expected_output)
            separate = np.zeros(count, dtype=np.float32)
            library.host.model.update(actual, separate, -4.0)
            np.testing.assert_array_equal(separate, expected - 4)
            np.testing.assert_array_equal(actual, expected + 1)
            vectors = np.zeros((count, 3), dtype=np.float32)
            library.host.model.transform(vectors)
            np.testing.assert_array_equal(vectors, [[1, 2, 3]] * count)
            assert runtime._resources == resources

        original_data = np.arange(9, dtype=np.float32)
        for a, b in ((original_data, original_data.view()), (original_data[:-1], original_data[1:])):
            with pytest.raises(ValueError, match="Aliased or overlapping Host arrays"):
                library.host.model.update(a, b, 3.0)
        np.testing.assert_array_equal(original_data, np.arange(9, dtype=np.float32))
        with pytest.raises(ValueError, match="dtype"):
            library.host.model.update(np.zeros(7, dtype=np.float64), np.zeros(7, dtype=np.float32), 1.0)
        with pytest.raises(ValueError, match="C-contiguous"):
            library.host.model.update(original_data[::2], original_data[::2], 1.0)
        with pytest.raises(ValueError, match="element shape"):
            library.host.model.transform(np.zeros((7, 2), dtype=np.float32))
        assert runtime._resources == resources


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_collects_local_dependencies_without_tracing_and_isolates_dynamic_imports(tmp_path, monkeypatch):
    source_root = tmp_path / "src"
    source_root.mkdir()
    dependencies = {
        "sibling_fixture/__init__.py": "initialized = 31\n",
        "sibling_fixture/ops.py": """import taichi_forge as ti
@ti.kernel
def extra(x: ti.i32) -> ti.i32:
    return x * 11
@ti.kernel
def never_called(x: ti.i32) -> ti.i32:
    return x * 13
""",
        "sibling_fixture/lazy.py": """import taichi_forge as ti
@ti.kernel
def calc(x: ti.i32) -> ti.i32:
    return x * 17
""",
        "sibling_fixture/not_selected.py": "raise AssertionError('must not import unspecified dynamic target')\n",
        "ns_fixture/helper.py": "def label(value):\n    return 'item-' + str(value)\n",
    }
    for name, text in dependencies.items():
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf8")
    source = SOURCE + """

def use_dependencies(x):
    import sibling_fixture.ops
    from ns_fixture import helper
    return sibling_fixture.ops.extra(x), sibling_fixture.initialized, helper.label(x)

from importlib import import_module as late
import importlib as imports
import importlib.util
def literal_late(x):
    return late('sibling_fixture.lazy').calc(x)
def dynamic(name):
    return imports.import_module(name)
def relative_late(x):
    return importlib.import_module('.lazy', 'sibling_fixture').calc(x)
"""
    package = _source_package(source_root, monkeypatch, source)
    with pytest.raises(ValueError, match="outside its application source root"):
        ti.aot.export_package(package, source_root / "sdk", source_root=source_root)
    manifest = ti.aot.export_package(package, tmp_path / "sdk", source_root=source_root)
    assert manifest["version"] == 2
    assert "sibling_fixture.not_selected" not in manifest["application_modules"]
    assert "ns_fixture" in manifest["application_modules"]
    assert {"sibling_fixture.ops.never_called", "sibling_fixture.lazy.calc"} <= {
        entry["definition"] for entry in manifest["entries"].values()
    }
    assert any(edge["kind"] == "literal_dynamic" for edge in manifest["dependencies"]["import_edges"])
    assert manifest["dependencies"]["dynamic_import_sites"]
    probe = tmp_path / "consume_dependencies.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture', 'sibling_fixture', 'ns_fixture'):
                    raise AssertionError('Private source import: ' + fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime('cpu') as runtime, load_package(runtime, sys.argv[1]) as a:
            model = a.host.model
            assert model.use_dependencies(3) == (33, 31, 'item-3')
            assert model.literal_late(2) == 34
            assert model.relative_late(3) == 51
            with load_package(runtime, sys.argv[1]) as b:
                left = model.dynamic('sibling_fixture.ops')
                right = b.host.model.dynamic('sibling_fixture.ops')
                assert left is not right and left.extra.package is a and right.extra.package is b
                assert model.dynamic('math').sqrt(9) == 3
            assert left.never_called(4) == 52
            try:
                model.dynamic('sibling_fixture.not_selected')
            except ImportError as error:
                assert 'was not exported' in str(error)
            else:
                raise AssertionError('Unresolved local dependency fell back to source')
        print('LOCAL_DEPENDENCIES_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LOCAL_DEPENDENCIES_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("storage", ["field", "ndarray"])
@pytest.mark.parametrize("helper_kind", ["func", "real_func", "pyfunc"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_inheritance_uses_actual_method_dependencies_and_shared_code(
    tmp_path, monkeypatch, storage, helper_kind
):
    if helper_kind == "real_func" and ti.cfg.arch == ti.vulkan:
        pytest.skip("Vulkan does not support real functions")
    source = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, gain: float):
        global constructors
        constructors += 1
        self.n = n
        self.gain = gain
        self.values = ti.field(ti.f32, shape=n)
        self.step(1.0)

    @ti.func
    def affine(self, x: ti.f32) -> ti.f32:
        return x * self.gain

    @ti.kernel
    def step(self, amount: ti.f32):
        for i in self.values:
            self.values[i] = self.affine(i) + amount

    @ti.kernel
    def size(self) -> ti.i32:
        return self.n
"""
    if storage == "ndarray":
        source = source.replace("ti.field(ti.f32, shape=n)", "ti.ndarray(ti.f32, shape=n)")
        source = source.replace("self.step(1.0)", "self.step(self.values, 1.0)")
        source = source.replace(
            "def step(self, amount:", "def step(self, values: ti.types.ndarray(dtype=ti.f32, ndim=1), amount:"
        )
        source = source.replace("for i in self.values:", "for i in values:").replace("self.values[i] =", "values[i] =")
    methods = source[source.index("    @ti.kernel\n    def step") :]
    source = source[: source.index("    @ti.kernel\n    def step")]
    constructor = source[source.index("    def __init__") : source.index("    @ti.func")]
    source = source.replace(constructor, "")
    source = source.replace(
        "@ti.data_oriented\nclass Model:",
        "class Storage:\n"
        + constructor
        + "\nclass KernelMethods:\n"
        + methods
        + "\n@ti.data_oriented\nclass Model(Storage, KernelMethods):",
    )
    source = source.replace("@ti.func", "@ti." + helper_kind)
    package = _source_package(tmp_path, monkeypatch, source)
    derived = tmp_path / "definition_fixture/derived.py"
    derived.write_text(
        textwrap.dedent("""
        import taichi_forge as ti
        from .model import Model

        class Unchanged(Model):
            pass

        class Offset:
            @ti.func
            def affine(self, x: ti.f32) -> ti.f32:
                return x * self.gain + 10.0

        class Changed(Offset, Model):
            def __init__(self, n: int, gain: float):
                super().__init__(n, gain)
                self.host_only = 23

        class Fails(Model):
            def __init__(self, n: int, gain: float):
                super().__init__(n, gain)
                raise ValueError('after base allocation')
        """).replace("@ti.func", "@ti." + helper_kind),
        encoding="utf8",
    )
    monkeypatch.delitem(sys.modules, "definition_fixture.derived", raising=False)
    assert "definition_fixture.derived" not in sys.modules
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    base = sys.modules["definition_fixture.model"]
    assert manifest["version"] == 3
    classes = sys.modules["definition_fixture.derived"]
    assert base.constructors == 0
    # A new class/Host field does not create another compiled numerical unit.
    by_name = {name: entry for entry in manifest["entries"].values() for name in entry["definitions"]}
    original = by_name["definition_fixture.model.Model.step"]
    assert by_name["definition_fixture.derived.Unchanged.step"] is original
    assert by_name["definition_fixture.derived.Fails.step"] is original
    assert by_name["definition_fixture.derived.Changed.step"] is not original
    assert by_name["definition_fixture.derived.Changed.size"] is by_name["definition_fixture.model.Model.size"]
    if ti.cfg.arch == ti.cpu and helper_kind == "func" and storage == "field":
        base_manifest = ti.aot.export_package(base, tmp_path / "base_sdk")
        base_symbols = {item["definition"]: item["symbol"] for item in base_manifest["entries"].values()}
        assert base_symbols["definition_fixture.model.Model.step"] == original["symbol"]
        assert base.constructors == 0
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        code = [item for item in archive.infolist() if item.filename.endswith((".ll", ".bc", ".spv"))]
        code_count = len(code)
        assert sum(item.file_size for item in code) > 0
    expected = classes.Changed(7, 2.0).values.to_numpy()
    np.testing.assert_array_equal(expected, np.arange(7, dtype=np.float32) * 2 + 11)
    probe = tmp_path / "consume_inheritance.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib
        import importlib.abc
        import sys
        import numpy as np
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            derived = importlib.import_module(code.host.__name__ + '.derived')
            a, b = derived.Changed(7, 2.0), derived.Unchanged(19, -3.0)
            read = lambda owner: np.frombuffer(owner.values.read(), np.float32)
            np.testing.assert_array_equal(read(a), np.arange(7, dtype=np.float32) * 2 + 11)
            np.testing.assert_array_equal(read(b), np.arange(19, dtype=np.float32) * -3 + 1)
            assert a.size() == 7 and b.size() == 19 and a.host_only == 23
            if sys.argv[3] == 'field':
                assert a.values._group_owner is not b.values._group_owner
                a.step(4.0)
            else:
                a.step(a.values, 4.0)
            np.testing.assert_array_equal(read(a), np.arange(7, dtype=np.float32) * 2 + 14)
            np.testing.assert_array_equal(read(b), np.arange(19, dtype=np.float32) * -3 + 1)
            registered = set(runtime._resources)
            try:
                derived.Fails(3, 2.0)
            except ValueError as error:
                assert str(error) == 'after base allocation'
            else:
                raise AssertionError('Original constructor failure disappeared')
            assert runtime._resources == registered
        print('INHERITANCE_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"], storage],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INHERITANCE_SOURCE_FREE_OK" in result.stdout
    assert code_count == (6 if helper_kind == "pyfunc" else 4)  # changed helper affects only step and public pyfunc


@pytest.mark.parametrize(
    "constructor,diagnostic",
    [
        ("if n > 0:\n            super().__init__(n, m)", "one unconditional"),
        ("super().__init__(n, m)\n        self.other = ti.field(ti.f32, shape=n)", "multiple constructors"),
        ("super().__init__(n, m)\n        self.a = ti.ndarray(ti.f32, shape=n)", "inherited resource/type replacement"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproven_inherited_resource_construction(tmp_path, monkeypatch, constructor, diagnostic):
    source = (
        FIELD_SOURCE + "\nclass Derived(Model):\n    def __init__(self, n: int, m: int):\n        " + constructor + "\n"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    assert sys.modules["definition_fixture.model"].constructors == 0


@pytest.mark.parametrize(
    "host,diagnostic",
    [
        ("super().step(1.0)", "explicit super numerical calls"),
        ("Model.step(self, 1.0)", "unbound numerical references"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unbound_inherited_numerical_calls(tmp_path, monkeypatch, host, diagnostic):
    source = FIELD_SOURCE + "\nclass Derived(Model):\n    def drive(self):\n        " + host + "\n"
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_numerical_mixin_without_concrete_receiver(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, FIELD_SOURCE.replace("@ti.data_oriented\n", ""))
    with pytest.raises(ti.aot.PackageExportError, match="no exported concrete"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_preserves_nonreturning_interfaces_without_numerical_payload(tmp_path, monkeypatch):
    source = """
from __future__ import annotations
import taichi_forge as ti
from typing import NoReturn as NeverReturns
calls = 0

def unsupported(message: str) -> NeverReturns:
    global calls
    calls += 1
    raise NotImplementedError(message)

def broken_contract() -> 'NeverReturns':
    return None

@ti.data_oriented
class Model:
    @ti.kernel
    def clear(self):
        unsupported('implement clear')

    @ti.func
    def missing(self, x):
        unsupported('implement missing')

    @ti.pyfunc
    def alternative(self, x):
        unsupported('implement alternative')

    def drive(self):
        self.missing(3)

class Inherited(Model):
    pass

class Concrete(Model):
    @ti.kernel
    def clear(self) -> ti.i32:
        return 17

@ti.func
def unavailable():
    unsupported('unavailable')

@ti.kernel
def invalid_contract():
    broken_contract()
"""
    # Forge annotations on typed numerical entries are evaluated normally;
    # postpone only the ordinary Host helpers' NoReturn annotations.
    source = source.replace("from __future__ import annotations\n", "").replace(
        "-> NeverReturns:", "-> 'NeverReturns':"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    (tmp_path / "definition_fixture/caller.py").write_text(
        "from .model import unavailable\ndef call():\n    unavailable()\n", encoding="utf8"
    )
    monkeypatch.delitem(sys.modules, "definition_fixture.caller", raising=False)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    original = sys.modules["definition_fixture.model"]
    assert original.calls == 0
    assert original.Concrete().clear() == 17
    assert len(manifest["entries"]) == 2  # concrete implementation and unimported typed kernel
    assert "definition_fixture.model.Model.clear" in manifest["error_interfaces"]
    assert "definition_fixture.model.Inherited.clear" in manifest["error_interfaces"]
    host = (tmp_path / "sdk/host/model.py").read_text(encoding="utf8")
    assert "implement clear" in host and "return 17" not in host
    probe = tmp_path / "consume_errors.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib
        import importlib.abc
        import sys
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime('cpu') as runtime, load_package(runtime, sys.argv[1]) as code:
            model = importlib.import_module(code.host.__name__ + '.model')
            caller = importlib.import_module(code.host.__name__ + '.caller')
            for operation, message in [
                (model.Model().clear, 'implement clear'),
                (model.Inherited().clear, 'implement clear'),
                (model.Inherited().drive, 'implement missing'),
                (lambda: model.Concrete().alternative(1), 'implement alternative'),
                (caller.call, 'unavailable'),
            ]:
                try:
                    operation()
                except NotImplementedError as error:
                    assert str(error) == message
                else:
                    raise AssertionError('Error interface silently succeeded')
            assert model.calls == 5 and model.Concrete().clear() == 17
            try:
                model.invalid_contract()
            except RuntimeError as error:
                assert 'returned unexpectedly' in str(error)
            else:
                raise AssertionError('NoReturn contract silently returned')
        print('ERROR_INTERFACES_SOURCE_FREE_OK')
        """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ERROR_INTERFACES_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize(
    "body,annotation",
    [
        ("unsupported(1 + 2)", "NoReturn"),
        ("value = 3\n        unsupported('missing')", "NoReturn"),
        ("unsupported('missing')", "None"),
        ("unsupported('missing')", "invented_NoReturn"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_does_not_misclassify_computation_or_unproven_error_names(tmp_path, monkeypatch, body, annotation):
    source = f"""
import taichi_forge as ti
from typing import NoReturn
invented_NoReturn = object()
def unsupported(value) -> '{annotation}':
    raise NotImplementedError(value)
@ti.data_oriented
class Model:
    @ti.kernel
    def clear(self):
        {body}
"""
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="undeclared global/closure capture 'unsupported'"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


SCOPE_VALUES_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, gain, shift, transform):
        global constructors
        constructors += 1
        self.gain, self.shift, self.transform = gain, shift, transform
        self.unused = 11
        scope = ti.experimental.ResourceScope(self)
        scope.value('gain', ti.f32)
        scope.value('shift', ti.types.vector(2, ti.f32))
        scope.value('transform', ti.types.matrix(2, 2, ti.f32))
        scope.value('unused', ti.i32)

    @ti.kernel
    def compute(self, x: ti.f32) -> ti.f32:
        value = self.transform @ ti.Vector([x, self.gain]) + self.shift
        return value[0] + value[1]

    @ti.kernel
    def independent(self, x: ti.i32) -> ti.i32:
        return x + 5
"""


@pytest.mark.parametrize("host_tensor", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_resource_scope_values_preserve_compound_captures(tmp_path, monkeypatch, host_tensor):
    source = SCOPE_VALUES_SOURCE
    if host_tensor:
        source = "from taichi_forge import Matrix as MatrixValue\n" + source.replace(
            "self.gain, self.shift, self.transform = gain, shift, transform",
            "self.gain = gain\n        self.shift = ti.Vector(shift, dt=ti.f32)\n        self.transform = MatrixValue(transform, ti.f32)",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    original = sys.modules["definition_fixture.model"]
    assert manifest["version"] == (8 if host_tensor else 5) and original.constructors == 0
    entries = {entry["definition"].rsplit(".", 1)[-1]: entry for entry in manifest["entries"].values()}
    captures = {item["name"]: item for item in entries["compute"]["captures"]}
    assert set(captures) == {"gain", "shift", "transform"}
    assert captures["shift"]["element_shape"] == [2]
    assert captures["transform"]["element_shape"] == [2, 2]
    assert not entries["independent"]["captures"]
    shift, transform = [1.0, 3.0], [[2.0, 1.0], [0.0, 4.0]]
    reference = original.Model(
        2.0, shift if host_tensor else ti.Vector(shift), transform if host_tensor else ti.Matrix(transform)
    )
    assert reference.compute(3.0) == 20.0
    probe = tmp_path / "consume_scope_values.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a = code.host.Model(2.0, [1.0, 3.0], [[2.0, 1.0], [0.0, 4.0]])
            b = code.host.Model(-1.0, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
            assert a.compute(3.0) == 20.0 and b.compute(3.0) == 2.0
            if sys.argv[3] == 'True':
                assert a.shift.get_shape() == (2,) and a.transform.get_shape() == (2, 2)
                assert a.shift.x == 1 and a.shift.y == 3
                assert a.transform[0, 1] == 1
                assert a.shift.to_numpy().dtype.name == 'float32'
                assert a.transform.to_list() == [[2, 1], [0, 4]]
                a.shift.x = 2
                assert a.compute(3.0) == 21.0 and b.compute(3.0) == 2.0
                try:
                    a.shift + 1
                except TypeError:
                    pass
                else:
                    raise AssertionError('Unimplemented Host algebra must not silently change semantics')
            a.shift = [10.0, -3.0]
            assert a.compute(3.0) == 23.0 and b.compute(3.0) == 2.0
            assert a.independent(7) == 12
        print('SCOPE_VALUES_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"], str(host_tensor)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCOPE_VALUES_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize(
    "replacement,diagnostic",
    [
        ("if gain > 0:\n            scope.value('gain', ti.f32)", "unconditional declared boolean owner static"),
        ("alias = scope\n        scope.value('gain', ti.f32)", "unconditional supported resource declaration"),
        ("scope = ti.experimental.ResourceScope(self)\n        scope.value('gain', ti.f32)", "must not be reassigned"),
        ("scope.array = scope.field(ti.f32, shape=2, placement=None)", "placement must retain"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_resource_scope_rejects_unproven_roles(tmp_path, monkeypatch, replacement, diagnostic):
    package = _source_package(
        tmp_path, monkeypatch, SCOPE_VALUES_SOURCE.replace("scope.value('gain', ti.f32)", replacement)
    )
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize(
    "binding",
    [
        "scope = ti.experimental.ResourceScope(self) if gain else None",
        "scope: object = ti.experimental.ResourceScope(self)",
        "scope = (ti.experimental.ResourceScope(self),)[0]",
        "scope = identity(ti.experimental.ResourceScope(self))",
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unprojected_scope_construction(tmp_path, monkeypatch, binding):
    source = SCOPE_VALUES_SOURCE.replace("scope = ti.experimental.ResourceScope(self)", binding)
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="ResourceScope construction requires"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize(
    "binding",
    [
        "scope = resource_scope(self) if gain else None",
        "scope = (resource_scope(self),)[0]",
        "scope = identity(resource_scope(self))",
        "resource_scope(self).child = self\n        scope = ti.experimental.ResourceScope(self)",
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unprojected_scope_factories(tmp_path, monkeypatch, binding):
    source = SCOPE_VALUES_SOURCE.replace("scope = ti.experimental.ResourceScope(self)", binding)
    source = source.replace(
        "constructors = 0",
        "constructors = 0\ndef resource_scope(owner):\n    return ti.experimental.ResourceScope(owner)",
    )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="ResourceScope factory construction requires"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


SCOPE_FIELDS_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int):
        global constructors
        constructors += 1
        self.n, self.m = n, m
        unrelated_host_value = 2**48
        unrelated_host_value = 0
        scope = ti.experimental.ResourceScope(self)
        rows, cols = scope.dim('n'), scope.dim('m')
        count = (rows + 1) * 2
        scope.a = scope.field(ti.f32, shape=count, initialization='zero')
        self.total = ti.field(ti.f32, shape=())
        scope.v = scope.vector_field(2, ti.f32, shape=(rows.maximum(1), cols))
        scope.mat = scope.matrix_field(2, 2, ti.f32, shape=())

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.a:
            self.a[i] = gain * (i + self.n)
        for i, j in self.v:
            self.v[i, j] = ti.Vector([gain * i, gain * j])
        self.mat[None] = ti.Matrix([[gain, 2.0], [3.0, gain + 4.0]])
        self.total[None] = gain * (self.n + self.m)
"""


@pytest.mark.parametrize("explicit_builders", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_sequential_scope_aliases_preserve_allocation_boundaries(tmp_path, monkeypatch, explicit_builders):
    source = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int):
        global constructors
        constructors += 1
        self.n, self.m = n, m
        scope = ti.experimental.ResourceScope(self, initialization='zero')
        rows = scope.dim('n')
        scope.a = scope.field(ti.i32, shape=rows)
        scope = ti.experimental.ResourceScope(self, initialization='consumer_explicit')
        scope.b = scope.field(ti.i32, shape=scope.dim('m'), initialization='zero')
        scope = ti.experimental.ResourceScope(self)
        scope.value('gain', ti.i32)
    @ti.kernel
    def step(self):
        for i in self.a:
            self.a[i] += self.gain
        for i in self.b:
            self.b[i] += 2 * self.gain
""".replace("        self.n, self.m = n, m", "        self.n, self.m = n, m\n        self.gain = 3")
    if explicit_builders:
        source = (
            source.replace(
                "scope = ti.experimental.ResourceScope(self, initialization='zero')",
                "first = ti.FieldsBuilder()\n        scope = ti.experimental.ResourceScope(self, builder=first, initialization='zero')",
            )
            .replace(
                "scope = ti.experimental.ResourceScope(self, initialization='consumer_explicit')",
                "self.first_tree = first.finalize()\n        second = ti.FieldsBuilder()\n"
                "        scope = ti.experimental.ResourceScope(self, builder=second, initialization='consumer_explicit')",
            )
            .replace(
                "        scope = ti.experimental.ResourceScope(self)\n",
                "        self.second_tree = second.finalize()\n        scope = ti.experimental.ResourceScope(self)\n",
            )
        )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["field_groups"]) == (2 if explicit_builders else 1)
    assert len(manifest["entries"]) == 2  # The fixture also has an unimported public kernel.
    original = package.Model(7, 5)
    original.step()
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, 5), code.host.Model(3, 19)
        assert (a.a._group_owner is a.b._group_owner) != explicit_builders
        assert a.a._group_owner is not b.a._group_owner
        a.step()
        b.gain = -2
        b.step()
        np.testing.assert_array_equal(a.a.to_numpy(), original.a.to_numpy())
        np.testing.assert_array_equal(a.b.to_numpy(), original.b.to_numpy())
        np.testing.assert_array_equal(b.a.to_numpy(), -2)
        np.testing.assert_array_equal(b.b.to_numpy(), -4)
        if explicit_builders:
            a.first_tree.destroy()
            np.testing.assert_array_equal(a.b.to_numpy(), 6)
    if not explicit_builders:
        probe = tmp_path / "consume_scope_aliases.py"
        probe.write_text(
            """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class RejectSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError('Numerical source unavailable: ' + fullname)
sys.meta_path.insert(0, RejectSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    a, b = code.host.Model(13, 17), code.host.Model(3, 7)
    a.step()
    b.gain = -5
    b.step()
    np.testing.assert_array_equal(a.a.to_numpy(), 3)
    np.testing.assert_array_equal(a.b.to_numpy(), 6)
    np.testing.assert_array_equal(b.b.to_numpy(), -10)
print('SOURCE_FREE_SCOPE_ALIASES_OK')
""",
            encoding="utf8",
        )
        result = subprocess.run(
            [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
            cwd=tmp_path,
            env=dict(
                os.environ,
                PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
                TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
                TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
            ),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "SOURCE_FREE_SCOPE_ALIASES_OK" in result.stdout


@pytest.mark.parametrize("change", ["conditional", "ordinary", "delete", "capture", "global", "import"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_sequential_scope_aliases_reject_ambiguous_lifetimes(tmp_path, monkeypatch, change):
    binding = "        scope = ti.experimental.ResourceScope(self)"
    source = SCOPE_FIELDS_SOURCE.replace(
        "        self.total = ti.field(ti.f32, shape=())",
        "        self.total = ti.field(ti.f32, shape=())\n" + binding,
    )
    if change == "conditional":
        source = source.replace(binding, "        if n:\n    " + binding, 1)
    elif change == "ordinary":
        source = source.replace("        scope.v =", "        scope = object()\n        scope.v =")
    elif change == "delete":
        source = source.replace("        scope.v =", "        del scope\n        scope.v =")
    elif change == "global":
        source = source.replace("        global constructors", "        global constructors, scope")
    elif change == "import":
        source = source.replace("        scope.v =", "        import math as scope\n        scope.v =")
    else:
        source = source.replace("        scope.v =", "        captured = lambda: scope\n        scope.v =")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="ResourceScope"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("construction_method", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_scope_fields_share_dimension_and_native_layout_rules(tmp_path, monkeypatch, construction_method):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    source = SCOPE_FIELDS_SOURCE
    if construction_method:
        source = source.replace(
            "        scope = ti.experimental.ResourceScope(self)",
            "        self._allocate()\n        self.after_allocation = self.a.shape\n"
            "    def _allocate(self):\n        scope = ti.experimental.ResourceScope(self)",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(7, 3)
    original.step(2.0)
    expected = {name: getattr(original, name).to_numpy() for name in ("a", "v", "mat", "total")}
    root = ti.lang.impl.get_runtime().prog.get_snode_root(original.a.snode.ptr.get_snode_tree_id())
    layout = _compiled_nodes(root)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 5
    assert sys.modules["definition_fixture.model"].constructors == 1
    assert len(manifest["field_groups"]) == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        a, b = library.host.Model(7, 3), library.host.Model(0, 5)
        assert a.a._group_owner.layout["nodes"] == layout
        assert a.a._group_owner is a.v._group_owner is a.mat._group_owner is a.total._group_owner
        assert a.a._group_owner is not b.a._group_owner
        assert a.a.shape == (16,) and b.a.shape == (2,) and b.v.shape == (1, 5)
        if construction_method:
            assert a.after_allocation == (16,) and b.after_allocation == (2,)
        a.step(2.0)
        b.step(-3.0)
        for name, values in expected.items():
            np.testing.assert_array_equal(
                np.frombuffer(getattr(a, name).read(), np.float32).reshape(values.shape), values
            )
        np.testing.assert_array_equal(np.frombuffer(b.a.read(), np.float32), [0.0, -3.0])
        assert np.frombuffer(b.total.read(), np.float32).item() == -15.0
        registered = set(runtime._resources)
        with pytest.raises((ValueError, RuntimeError)):
            library.host.Model(-1, 2)
        assert runtime._resources == registered
    probe = tmp_path / "consume_scope_fields.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3, 2), code.host.Model(19, 1)
            a.step(2.0)
            b.step(-1.0)
            np.testing.assert_array_equal(np.frombuffer(a.a.read(), np.float32), 2 * (np.arange(8) + 3))
            np.testing.assert_array_equal(np.frombuffer(b.a.read(), np.float32), -(np.arange(40) + 19))
            assert a.v.shape == (3, 2) and b.v.shape == (19, 1)
            assert a.a._group_owner is not b.a._group_owner
        print('SCOPE_FIELDS_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCOPE_FIELDS_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("declaration", ["direct", "scope", "builder", "selected"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_field_index_order_uses_original_native_layout(tmp_path, monkeypatch, declaration):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    source = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int):
        global constructors
        constructors += 1
        self.n, self.m = n, m
        self.a = ti.field(ti.i32, shape=(n, 3, m), order='kij')
        self.v = ti.Vector.field(3, ti.f32, shape=(n, m), order='ji')
        self.mat = ti.Matrix.field(2, 2, ti.f32, shape=(n, m), order='ij')
    @ti.kernel
    def step(self, gain: ti.f32):
        for i, j, k in self.a:
            self.a[i, j, k] = 100 * i + 10 * j + k
        for i, j in self.v:
            self.v[i, j] = gain * ti.Vector([i, j, i + j])
            self.mat[i, j] = ti.Matrix([[gain, i], [j, gain + i + j]])
"""
    if declaration != "direct":
        setup = "        scope = ti.experimental.ResourceScope(self)\n"
        if declaration in ("builder", "selected"):
            setup = "        builder = ti.FieldsBuilder()\n        scope = ti.experimental.ResourceScope(self, builder=builder)\n"
        if declaration == "selected":
            source = source.replace(
                "n: int, m: int):", "n: int, m: int, island_local: bool=True, alternate: bool=False):"
            )
            source = source.replace("class Model:\n", "class Model:\n    width = 3\n")
            setup = "        self.island_local, self.alternate = island_local, alternate\n" + setup
            setup += "        scope.static('island_local', choices=(False, True))\n        scope.static('alternate', choices=(False, True))\n        scope.constant('width')\n"
            source = source.replace("shape=(n, 3, m)", "shape=(n, self.width, m)")
            source = source.replace("ti.Vector.field(3,", "ti.Vector.field(self.width,")
            for order in ("kij", "ji"):
                source = source.replace("order='" + order + "'", "order='" + order + "' if self.island_local else None")
            source = source.replace("order='ij'", "order=None if self.alternate else 'ij'")
            source += """
    def graph(self):
        graph = ti.graph.GraphBuilder()
        graph.dispatch(self.step, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32), template_args={'self': self})
        return graph.compile()
"""
        setup += "        n, m = scope.dim('n'), scope.dim('m')\n"
        source = source.replace("        self.a =", setup + "        self.a =", 1)
        source = source.replace("ti.field(", "scope.field(").replace("ti.Vector.field(", "scope.vector_field(")
        source = source.replace("ti.Matrix.field(", "scope.matrix_field(")
        if declaration in ("builder", "selected"):
            source = source.replace("    @ti.kernel", "        self.tree = builder.finalize()\n    @ti.kernel", 1)
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(5, 7)
    original.step(2.0)
    root = ti.lang.impl.get_runtime().prog.get_snode_root(original.a.snode.ptr.get_snode_tree_id())
    layout = _compiled_nodes(root)
    expected = {name: getattr(original, name).to_numpy() for name in ("a", "v", "mat")}
    if declaration == "selected":
        other = package.Model(3, 11, False, True)
        other.step(-3.0)
        other_root = ti.lang.impl.get_runtime().prog.get_snode_root(other.a.snode.ptr.get_snode_tree_id())
        other_layout = _compiled_nodes(other_root)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == (2 if declaration == "selected" else 1)
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([name for name in archive.namelist() if name.endswith((".ll", ".bc", ".spv"))]) == 2
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a = code.host.Model(5, 7)
        b = code.host.Model(3, 11, False, True) if declaration == "selected" else code.host.Model(3, 11)
        assert a.a._group_owner.layout["nodes"] == layout
        assert a.a._group_owner is a.v._group_owner is a.mat._group_owner
        assert b.a._group_owner is not a.a._group_owner
        if declaration == "selected":
            assert b.a._group_owner.layout["nodes"] == other_layout
        a.step(2.0)
        b.step(-3.0)
        for name, value in expected.items():
            np.testing.assert_array_equal(getattr(a, name).to_numpy(), value)
        grid = np.indices((3, 3, 11))
        np.testing.assert_array_equal(b.a.to_numpy(), grid[0] * 100 + grid[1] * 10 + grid[2])
        i, j = np.indices((3, 11))
        np.testing.assert_array_equal(b.v.to_numpy(), -3 * np.stack((i, j, i + j), axis=-1))
        if declaration == "selected":
            pa, pb = a.graph().bind({"gain": 5.0}), b.graph().bind({"gain": -2.0})
            a.island_local = False
            with pytest.raises(ValueError, match="layout configuration changed"):
                a.step(3.0)
            pa.run()
            pb.run()
            i, j = np.indices((5, 7))
            np.testing.assert_array_equal(a.v.to_numpy(), 5 * np.stack((i, j, i + j), axis=-1))
            i, j = np.indices((3, 11))
            np.testing.assert_array_equal(b.v.to_numpy(), -2 * np.stack((i, j, i + j), axis=-1))
    if declaration in ("builder", "selected"):
        probe = tmp_path / "consume_ordered.py"
        probe.write_text(
            textwrap.dedent("""
            import importlib.abc
            import sys
            import numpy as np
            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                        raise AssertionError(fullname)
            sys.meta_path.insert(0, Block())
            from taichi_forge_runtime.aot import Runtime, load_package
            with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
                a, b = code.host.Model(19, 2), code.host.Model(2, 13)
                a.step(4.0)
                b.step(-1.0)
                grid = np.indices((19, 3, 2))
                np.testing.assert_array_equal(a.a.to_numpy(), grid[0]*100 + grid[1]*10 + grid[2])
                i, j = np.indices((2, 13))
                np.testing.assert_array_equal(b.v.to_numpy(), -np.stack((i,j,i+j),axis=-1))
            print('ORDERED_FIELDS_SOURCE_FREE_OK')
        """),
            encoding="utf8",
        )
        if declaration == "selected":
            probe.write_text(
                probe.read_text()
                .replace("code.host.Model(19, 2)", "code.host.Model(19, 2, False)")
                .replace("code.host.Model(2, 13)", "code.host.Model(2, 13, True, True)")
            )
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        )
        result = subprocess.run(
            [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ORDERED_FIELDS_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("change", ["kernel_only", "runtime_value", "model_condition", "unproven_branch"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_field_order_rejects_unproven_layout_selection(tmp_path, monkeypatch, change):
    source = SCOPE_FIELDS_SOURCE.replace("        scope =", "        self.tiled = True\n        scope =", 1)
    source = source.replace(
        "        rows, cols =", "        scope.static('tiled', choices=(False, True))\n        rows, cols =", 1
    )
    source = source.replace(
        "shape=(rows.maximum(1), cols))", "shape=(rows.maximum(1), cols), order='ji' if self.tiled else None)"
    )
    if change == "kernel_only":
        source = source.replace("choices=(False, True)", "choices=(False, True), kernel_only=True")
    elif change == "runtime_value":
        source = source.replace("scope.static('tiled', choices=(False, True))", "scope.value('tiled', ti.i32)")
    elif change == "model_condition":
        source = source.replace("if self.tiled else", "if self.n > 4 else")
    else:
        source = source.replace("else None)", "else self.order)")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="Field order"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_order_selection_retains_model_tensor_recipe_members(tmp_path, monkeypatch):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    source = """
import taichi_forge as ti
@ti.data_oriented
class Model:
    def __init__(self, n: int, dofs: int, tiled: bool):
        self.n, self.dofs, self.tiled = n, dofs, tiled
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.static('tiled', choices=(False, True))
        scope.dim('dofs')
        scope.values = scope.vector_field(self.dofs, ti.f32,
            shape=(scope.dim('n'), 2), order='ji' if self.tiled else None)
        self.tree = builder.finalize()
    @ti.kernel
    def step(self):
        for i, j in self.values:
            for k in range(self.dofs):
                self.values[i, j][k] = 100 * i + 10 * j + k
"""
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(4, 5, False)
    original.step()
    root = ti.lang.impl.get_runtime().prog.get_snode_root(original.values.snode.ptr.get_snode_tree_id())
    expected_layout = _compiled_nodes(root)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(4, 5, False), code.host.Model(7, 19, True)
        assert a.values._group_owner.layout["nodes"] == expected_layout
        a.step()
        b.step()
        np.testing.assert_array_equal(a.values.to_numpy(), original.values.to_numpy())
        i, j, k = np.indices((7, 2, 19))
        np.testing.assert_array_equal(b.values.to_numpy(), 100 * i + 10 * j + k)
    assert len(manifest["entries"]) == 2


INHERITED_CONSTRUCTION_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int):
        global constructors
        constructors += 1
        self.n = n
        self._allocate()
    def _allocate(self):
        scope = ti.experimental.ResourceScope(self)
        count = scope.dim('n')
        nd = self.n
        element = ti.types.struct(x=ti.f32, **self._members(nd))
        scope.values = scope.struct_field(element.members, shape=count,
            member_initialization={name: 'zero' for name in element.members if name == 'vector'})
        self._extra()
    def _members(self, nd):
        return {}
    def _extra(self):
        pass
    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.values:
            self.values.x[i] = gain * i
    @ti.kernel
    def size(self) -> ti.i32:
        return self.n
class Extended(Model):
    def _members(self, nd):
        return {'vector': ti.types.vector(nd, ti.f32)}
    def _extra(self):
        scope = ti.experimental.ResourceScope(self)
        count = scope.dim('n')
        scope.total = scope.field(ti.f32, shape=count)
    @ti.kernel
    def extended_step(self):
        for i in range(self.n):
            total = 0.0
            for j in range(self.n):
                self.values.vector[i][j] = i + j
                total += self.values.vector[i][j]
            self.total[i] = total
"""


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_inherited_construction_retains_concrete_layout_and_shared_code(tmp_path, monkeypatch):
    package = _source_package(
        tmp_path, monkeypatch, INHERITED_CONSTRUCTION_SOURCE, initializer="from .model import Model, Extended\n"
    )
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    by_name = {name: entry for entry in manifest["entries"].values() for name in entry["definitions"]}
    assert by_name["definition_fixture.model.Model.size"] is by_name["definition_fixture.model.Extended.size"]
    original = package.Extended(7)
    original.step(3.0)
    original.extended_step()
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        base, a, b = code.host.Model(4), code.host.Extended(7), code.host.Extended(65)
        assert set(base.values.to_numpy()) == {"x"}
        assert not hasattr(base, "total")
        a.step(3.0)
        a.extended_step()
        np.testing.assert_array_equal(a.values.x.to_numpy(), original.values.x.to_numpy())
        np.testing.assert_array_equal(a.total.to_numpy(), original.total.to_numpy())
        b.extended_step()
        np.testing.assert_array_equal(b.total.to_numpy(), 65 * np.arange(65) + sum(range(65)))
        base.step(-2.0)
        np.testing.assert_array_equal(base.values.x.to_numpy(), -2 * np.arange(4))
        np.testing.assert_array_equal(a.total.to_numpy(), original.total.to_numpy())
        assert (base.size(), a.size(), b.size()) == (4, 7, 65)
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([name for name in archive.namelist() if name.endswith((".ll", ".bc"))]) == len(manifest["entries"])
    probe = tmp_path / "consume_construction.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            base, a, b = code.host.Model(3), code.host.Extended(17), code.host.Extended(33)
            assert not hasattr(base, 'total')
            np.testing.assert_array_equal(a.values.vector.to_numpy(), 0)
            a.extended_step()
            b.extended_step()
            base.step(2.0)
            np.testing.assert_array_equal(base.values.x.to_numpy(), 2 * np.arange(3))
            np.testing.assert_array_equal(a.total.to_numpy(), 17 * np.arange(17) + sum(range(17)))
            np.testing.assert_array_equal(b.total.to_numpy(), 33 * np.arange(33) + sum(range(33)))
        print('INHERITED_CONSTRUCTION_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INHERITED_CONSTRUCTION_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("side_effect", "one fixed dictionary"),
        ("model", "cannot read model state"),
        ("replace", "cannot be replaced"),
        ("duplicate", "unique valid member names"),
        ("call", "builtin numerical types"),
        ("dimension", "Repeated construction resource declaration"),
        ("super", "lexical class state"),
        ("private", "lexical class state"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_inherited_type_members_reject_unproven_contracts(tmp_path, monkeypatch, change, diagnostic):
    source = INHERITED_CONSTRUCTION_SOURCE
    if change == "side_effect":
        source = source.replace("return {'vector'", "self.changed = 1\n        return {'vector'")
    elif change == "model":
        source = source.replace("ti.types.vector(nd, ti.f32)", "ti.types.vector(self.n, ti.f32)")
    elif change == "replace":
        source = source.replace("self._allocate()", "self._members = lambda n: {}\n        self._allocate()")
    elif change == "duplicate":
        source = source.replace("{'vector':", "{'x':")
    elif change == "call":
        source = source.replace("ti.types.vector(nd, ti.f32)", "dict()")
    elif change == "super":
        source = source.replace("self._allocate()", "super().__init__()\n        self._allocate()")
    elif change == "private":
        source = source.replace("self._allocate()", "self.__secret = 2\n        self._allocate()")
    else:
        source = source.replace(
            "count = scope.dim('n')\n        scope.total",
            "count = scope.derived_dim('n', scope.dim('m'))\n        scope.total",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize(
    "change", ["conditional", "argument", "repeated", "replace", "return", "override", "recursive"]
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_ambiguous_construction_methods(tmp_path, monkeypatch, change):
    source = SCOPE_FIELDS_SOURCE.replace(
        "        scope = ti.experimental.ResourceScope(self)",
        "        self._allocate()\n    def _allocate(self):\n        scope = ti.experimental.ResourceScope(self)",
    )
    if change == "conditional":
        source = source.replace("        self._allocate()", "        if n:\n            self._allocate()")
    elif change == "argument":
        source = source.replace("self._allocate()", "self._allocate(n)").replace(
            "def _allocate(self)", "def _allocate(self, n)"
        )
    elif change == "repeated":
        source = source.replace("        self._allocate()", "        self._allocate()\n        self._allocate()")
    elif change == "replace":
        source = source.replace(
            "        self._allocate()", "        self._allocate = lambda: None\n        self._allocate()"
        )
    elif change == "return":
        source = source.replace("    @ti.kernel", "        return self.a\n    @ti.kernel")
    elif change == "recursive":
        source = source.replace("    def _allocate(self):", "    def _allocate(self):\n        self._allocate()")
    else:
        source += "\nclass Other(Model):\n    def _allocate(self):\n        pass\n"
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="[Cc]onstruction"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("count = (rows + 1) * 2", "count = n * 2", "shape requires declared dimensions"),
        (
            "count = (rows + 1) * 2",
            "count = (rows + 1) * 2\n        count = rows",
            "shape requires declared dimensions",
        ),
        (
            "self.total = ti.field(ti.f32, shape=())",
            "self.a = 7\n        self.total = ti.field(ti.f32, shape=())",
            "one unconditional owner attribute",
        ),
        (
            "rows, cols = scope.dim('n'), scope.dim('m')",
            "if n > 0:\n            rows, cols = scope.dim('n'), scope.dim('m')",
            "unconditional declared boolean owner static",
        ),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_scope_fields_reject_unproven_dimensions_and_replacements(
    tmp_path, monkeypatch, before, after, diagnostic
):
    package = _source_package(tmp_path, monkeypatch, SCOPE_FIELDS_SOURCE.replace(before, after))
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    assert sys.modules["definition_fixture.model"].constructors == 0


EXPLICIT_GROUP_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int):
        global constructors
        constructors += 1
        self.n, self.m = n, m
        fields = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=fields, initialization='zero')
        other = scope.dense(ti.i, scope.dim('m'))
        cells = scope.dense(ti.i, scope.dim('n'))
        scope.a = scope.field(ti.f32, shape=scope.dim('n'), placement=cells)
        scope.v = scope.vector_field(2, ti.f64, shape=scope.dim('n'), placement=cells)
        scope.extra = scope.field(ti.i32, shape=scope.dim('m'), placement=other)
        self.tree = fields.finalize()
        separate = ti.FieldsBuilder()
        scalar = ti.experimental.ResourceScope(self, builder=separate)
        scalar.total = scalar.field(ti.f32, shape=())
        scalar.independent = scalar.field(ti.f32, shape=scalar.dim('n'))
        self.scalar_tree = separate.finalize()
        self.ordinary = ti.field(ti.i32, shape=1)

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.a:
            self.a[i] = gain * i
            self.v[i] = ti.Vector([i, gain])
        for i in self.extra:
            self.extra[i] = i + 9
        self.total[None] = gain
        self.ordinary[0] = 7

    @ti.kernel
    def summary(self, gain: ti.f32):
        self.total[None] += gain

    def destroy_fields(self):
        self.tree.destroy()
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_explicit_groups_preserve_placement_and_lifetime(tmp_path, monkeypatch):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    # The current Vulkan AOT target does not enable f64; retain f64 coverage
    # on LLVM and exercise the same placement/ownership rules with f32 there.
    source = EXPLICIT_GROUP_SOURCE.replace("ti.f64", "ti.f32") if ti.cfg.arch == ti.vulkan else EXPLICIT_GROUP_SOURCE
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(7, 3)
    original.step(2.0)
    expected = {name: getattr(original, name).to_numpy() for name in ("a", "v", "extra", "total", "ordinary")}
    program = ti.lang.impl.get_runtime().prog
    layouts = {
        name: _compiled_nodes(program.get_snode_root(getattr(original, name).snode.ptr.get_snode_tree_id()))
        for name in ("a", "total", "ordinary")
    }
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 10
    assert len(manifest["field_groups"]) == 3
    assert sys.modules["definition_fixture.model"].constructors == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, 3), code.host.Model(19, 5)
        for name, layout in layouts.items():
            assert getattr(a, name)._group_owner.layout["nodes"] == layout
        assert a.a._group_owner is a.v._group_owner is a.extra._group_owner
        assert a.a._group_owner is not a.total._group_owner
        a.step(2.0)
        b.step(-1.0)
        for name, data in expected.items():
            np.testing.assert_array_equal(getattr(a, name).to_numpy(), data)
        a.destroy_fields()
        a.summary(3.0)
        assert a.total.to_numpy().item() == 5.0
        with pytest.raises(RuntimeError):
            a.a.to_numpy()
        b.step(-2.0)
        np.testing.assert_array_equal(b.a.to_numpy(), -2 * np.arange(19))
        registered = set(runtime._resources)
        with pytest.raises((ValueError, RuntimeError)):
            code.host.Model(0, 3)
        assert set(runtime._resources) == registered
    probe = tmp_path / "consume_explicit_groups.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3, 2), code.host.Model(19, 4)
            a.step(2.)
            b.step(-1.)
            np.testing.assert_array_equal(a.a.to_numpy(), np.arange(3)*2)
            a.destroy_fields()
            a.summary(7.)
            assert a.total.to_numpy().item() == 9.
            np.testing.assert_array_equal(b.a.to_numpy(), -np.arange(19))
        print('EXPLICIT_GROUPS_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPLICIT_GROUPS_OK" in result.stdout


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("placement=cells)", "placement=other)", "group and extent relations"),
        (
            "self.tree = fields.finalize()",
            "leak = fields\n        self.tree = fields.finalize()",
            "escaping/reassigned",
        ),
        ("self.tree = fields.finalize()", "self.tree = fields.finalize()\n        fields.finalize()", "finalize once"),
        (
            "cells = scope.dense(ti.i, scope.dim('n'))",
            "cells = scope.dense(ti.j, scope.dim('n'))",
            "canonical ordered axes",
        ),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_explicit_groups_reject_ambiguous_ownership(tmp_path, monkeypatch, before, after, diagnostic):
    package = _source_package(tmp_path, monkeypatch, EXPLICIT_GROUP_SOURCE.replace(before, after))
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    assert sys.modules["definition_fixture.model"].constructors == 0


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("scope.a =", "self.n += 1\n        scope.a =", "placement changed"),
        ("scalar = ti.experimental", "self.n += 1\n        scalar = ti.experimental", "Shared dimension changed"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_explicit_placement_keeps_its_construction_snapshot(tmp_path, monkeypatch, before, after, diagnostic):
    source = EXPLICIT_GROUP_SOURCE.replace(before, after)
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        registered = set(runtime._resources)
        with pytest.raises(ValueError, match=diagnostic):
            code.host.Model(7, 3)
        assert set(runtime._resources) == registered


CONTEXT_FIELDS_SOURCE = """
import taichi_forge as ti
from resource_contexts import resource_group, read_events
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, fail=0):
        global constructors
        constructors += 1
        self.n = n
        with resource_group(self, 'state', initialization='zero') as scope:
            if fail == 1:
                raise ValueError('before allocation')
            cells = scope.dense(ti.i, scope.dim('n'))
            scope.values = scope.field(ti.i32, shape=scope.dim('n'), placement=cells)
            if fail == 2:
                raise ValueError('after allocation')
            scope.weights = scope.field(ti.f32, shape=scope.dim('n'), placement=cells)
    @ti.kernel
    def step(self):
        for i in self.values:
            self.values[i] += 1
"""

RESOURCE_CONTEXT_SOURCE = """
from contextlib import contextmanager
import taichi_forge as ti
events = []
def read_events():
    return list(events)
@contextmanager
def resource_group(owner, name, *, initialization='consumer_explicit'):
    events.append(('enter', name))
    builder = ti.FieldsBuilder()
    scope = ti.experimental.ResourceScope(owner, builder=builder, initialization=initialization)
    try:
        yield scope
    except BaseException:
        events.append(('failure', name))
        owner.failed_tree = builder.finalize()
        raise
    owner.tree = builder.finalize()
    events.append(('exit', name))
"""


def _resource_context_fixture(tmp_path, monkeypatch, *, helper=RESOURCE_CONTEXT_SOURCE, model=CONTEXT_FIELDS_SOURCE):
    (tmp_path / "resource_contexts.py").write_text(textwrap.dedent(helper), encoding="utf8")
    monkeypatch.delitem(sys.modules, "resource_contexts", raising=False)
    package = _source_package(tmp_path, monkeypatch, model)
    source = Path(sys.modules["definition_fixture.model"].__file__)
    return package, source


@pytest.mark.parametrize("inline,construction_method", [(False, False), (True, False), (False, True)])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_discovers_context_group_without_discarding_host_lifecycle(
    tmp_path, monkeypatch, inline, construction_method
):
    from taichi_forge.aot._package import _analyze

    model = (
        RESOURCE_CONTEXT_SOURCE
        + CONTEXT_FIELDS_SOURCE.replace("from resource_contexts import resource_group, read_events\n", "")
        if inline
        else CONTEXT_FIELDS_SOURCE
    )
    if construction_method:
        model = model.replace(
            "        with resource_group(self,",
            "        self.fail = fail\n        self._initialize()\n"
            "    def _initialize(self):\n        self._allocate()\n"
            "    def _allocate(self):\n        fail = self.fail\n        with resource_group(self,",
        )
    package, source = _resource_context_fixture(tmp_path, monkeypatch, model=model)
    module = sys.modules["definition_fixture.model"]
    helper = module if inline else sys.modules["resource_contexts"]
    tree, _, owners, _ = _analyze(module.__name__, source.read_text(), vars(module), "x64")
    schema = owners["Model"]
    assert schema["n"]["kind"] == "dimension"
    assert schema["values"]["group"] == schema["weights"]["group"]
    assert schema["values"]["placement"] == schema["weights"]["placement"]
    assert len(tree._package_resource_contexts) == 1
    recipe = tree._package_resource_contexts[0]
    assert recipe["lifecycle"] == "original_host_generator_required"
    assert len(recipe["finalization_lines"]) == 2  # Success and failure stay distinct.
    assert module.constructors == 0 and helper.events == []
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 16
    original = package.Model(5)
    original.step()
    np.testing.assert_array_equal(original.values.to_numpy(), 1)
    assert helper.events == [("enter", "state"), ("exit", "state")]
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(3), code.host.Model(19)
        a.step()
        b.step()
        np.testing.assert_array_equal(a.values.to_numpy(), 1)
        np.testing.assert_array_equal(b.values.to_numpy(), 1)
        assert a.values._group_owner is a.weights._group_owner
        registered = set(runtime._resources)
        for fail in (1, 2):
            with pytest.raises(ValueError, match="allocation"):
                code.host.Model(7, fail=fail)
            assert set(runtime._resources) == registered
        assert code.module("definition_fixture.model").read_events() == [
            ("enter", "state"),
            ("exit", "state"),
            ("enter", "state"),
            ("exit", "state"),
            ("enter", "state"),
            ("failure", "state"),
            ("enter", "state"),
            ("failure", "state"),
        ]
        a.tree.destroy()
        b.step()
        np.testing.assert_array_equal(b.values.to_numpy(), 2)


@pytest.mark.parametrize("change", ["conditional", "escape", "hidden", "multiple", "shadow", "dynamic"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_context_rejects_unproven_group_boundaries(tmp_path, monkeypatch, change):
    helper, model = RESOURCE_CONTEXT_SOURCE, CONTEXT_FIELDS_SOURCE
    if change == "conditional":
        helper = helper.replace("        yield scope", "        if owner.n:\n            yield scope")
    elif change == "escape":
        helper = helper.replace("    try:", "    owner.builder = builder\n    try:")
    elif change == "hidden":
        helper = helper.replace("    try:", "    scope.hidden = scope.field(ti.i32, shape=())\n    try:")
    elif change == "multiple":
        helper = helper.replace("    owner.tree =", "    yield scope\n    owner.tree =")
    elif change == "dynamic":
        helper = helper.replace("    try:", "    locals()\n    try:")
    else:
        model = model.replace("n: int, fail=0):", "n: int, fail=0, resource_group=None):")
    package, _ = _resource_context_fixture(tmp_path, monkeypatch, helper=helper, model=model)
    with pytest.raises(ti.aot.PackageExportError, match="Resource context"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert sys.modules["resource_contexts"].events == []
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_context_alias_reuse_preserves_distinct_native_groups(tmp_path, monkeypatch):
    helper = RESOURCE_CONTEXT_SOURCE.replace(
        "    owner.tree = builder.finalize()",
        "    if not hasattr(owner, 'trees'):\n        owner.trees = {}\n" "    owner.trees[name] = builder.finalize()",
    )
    model = CONTEXT_FIELDS_SOURCE.replace(
        "    @ti.kernel",
        "        with resource_group(self, 'scratch', initialization='zero') as scope:\n"
        "            scope.aux = scope.field(ti.i32, shape=scope.dim('n'))\n"
        "    @ti.kernel",
    ).replace("            self.values[i] += 1", "            self.values[i] += 1\n            self.aux[i] += 2")
    package, _ = _resource_context_fixture(tmp_path, monkeypatch, helper=helper, model=model)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 16
    assert len(manifest["field_groups"]) == 2
    original = package.Model(5)
    original.step()
    np.testing.assert_array_equal(original.aux.to_numpy(), 2)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(3), code.host.Model(19)
        assert a.values._group_owner is a.weights._group_owner
        assert a.values._group_owner is not a.aux._group_owner
        assert a.aux._group_owner is not b.aux._group_owner
        a.step()
        b.step()
        np.testing.assert_array_equal(a.values.to_numpy(), 1)
        np.testing.assert_array_equal(a.aux.to_numpy(), 2)
        a.trees["state"].destroy()
        np.testing.assert_array_equal(a.aux.to_numpy(), 2)
        b.step()
        np.testing.assert_array_equal(b.aux.to_numpy(), 4)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_context_alias_can_be_rebound_to_a_later_owner_scope(tmp_path, monkeypatch):
    model = CONTEXT_FIELDS_SOURCE.replace(
        "    @ti.kernel",
        "        self.gain = 3\n        scope = ti.experimental.ResourceScope(self)\n"
        "        scope.value('gain', ti.i32)\n    @ti.kernel",
    ).replace("self.values[i] += 1", "self.values[i] += self.gain")
    package, _ = _resource_context_fixture(tmp_path, monkeypatch, model=model)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["field_groups"]) == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(3), code.host.Model(11)
        a.step()
        b.gain = -2
        b.step()
        np.testing.assert_array_equal(a.values.to_numpy(), 3)
        np.testing.assert_array_equal(b.values.to_numpy(), -2)
        assert a.values._group_owner is not b.values._group_owner


@pytest.mark.parametrize("change", ["reassign", "escape", "after"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_context_alias_rejects_ambiguous_rebinding(tmp_path, monkeypatch, change):
    model = CONTEXT_FIELDS_SOURCE
    if change == "reassign":
        model = model.replace("            cells =", "            scope = None\n            cells =")
    elif change == "escape":
        model = model.replace("    @ti.kernel", "        self.scope = scope\n    @ti.kernel")
    else:
        model = model.replace("    @ti.kernel", "        scope.late = scope.field(ti.i32, shape=())\n    @ti.kernel")
    package, _ = _resource_context_fixture(tmp_path, monkeypatch, model=model)
    with pytest.raises(ti.aot.PackageExportError, match="ResourceScope"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


CONDITIONAL_CONTEXT_SOURCE = """
import taichi_forge as ti
from resource_contexts import resource_group, read_events
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int, first=False, second=False):
        global constructors
        constructors += 1
        self.n, self.m = n, m
        self.first, self.second = first, second
        self.a = self.b = None
        if self.first:
            with resource_group(self, 'a', initialization='zero') as scope:
                n = scope.dim('n')
                scope.a = scope.field(ti.i32, shape=n)
        if self.second:
            with resource_group(self, 'b', initialization='zero') as scope:
                scope.b = scope.field(ti.i32, shape=scope.dim('m'))
        with resource_group(self, 'base', initialization='zero') as scope:
            scope.static('first', choices=(False, True))
            scope.static('second', choices=(False, True))
            scope.total = scope.field(ti.i32, shape=())
    @ti.kernel
    def common(self):
        self.total[None] += 1
    @ti.kernel
    def first_step(self):
        self.total[None] += 2
        if ti.static(self.first):
            for i in self.a:
                self.a[i] += 3
    @ti.kernel
    def second_step(self):
        self.total[None] += 4
        if ti.static(self.second):
            for i in self.b:
                self.b[i] += 5
"""


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_conditional_contexts_prepare_only_enabled_native_groups(tmp_path, monkeypatch):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    package, _ = _resource_context_fixture(tmp_path, monkeypatch, model=CONDITIONAL_CONTEXT_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["field_groups"]) == 3
    assert len(manifest["entries"]) == 6  # 1 common + 2 per used flag + 1 other public function.
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([n for n in archive.namelist() if n.endswith((".ll", ".bc"))]) == 6
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        instances = []
        for first, second in ((False, False), (True, False), (False, True), (True, True)):
            # Inactive group extents are deliberately invalid and never evaluated.
            n, m = (3 if first else -17), (19 if second else -29)
            original = package.Model(n, m, first, second)
            original.common()
            original.first_step()
            original.second_step()
            current = code.host.Model(n, m, first, second)
            instances.append(current)
            assert len(current._forge_package_field_groups) == 1 + first + second
            assert (current.a is not None) is first and (current.b is not None) is second
            current.common()
            current.first_step()
            current.second_step()
            for name in ("total", "a", "b"):
                field = getattr(current, name)
                if field is not None:
                    expected = getattr(original, name)
                    np.testing.assert_array_equal(field.to_numpy(), expected.to_numpy())
                    root = ti.lang.impl.get_runtime().prog.get_snode_root(expected.snode.ptr.get_snode_tree_id())
                    assert field._group_owner.layout["nodes"] == _compiled_nodes(root)
        instances[-1].a._group_owner.close()
        instances[-1].common()
        instances[-1].second_step()
        np.testing.assert_array_equal(instances[-1].b.to_numpy(), 10)
        instances[1].first_step()
        np.testing.assert_array_equal(instances[1].a.to_numpy(), 6)


@pytest.mark.parametrize("change", ["missing", "conditional_domain", "reassign_extent", "loop"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_conditional_contexts_reject_unproven_domains(tmp_path, monkeypatch, change):
    model = CONDITIONAL_CONTEXT_SOURCE
    if change == "missing":
        model = model.replace("            scope.static('first', choices=(False, True))", "            pass")
    elif change == "conditional_domain":
        model = model.replace(
            "            scope.static('first', choices=(False, True))",
            "            if self.first:\n                scope.static('first', choices=(False, True))",
        )
    elif change == "reassign_extent":
        model = model.replace(
            "                n = scope.dim('n')", "                n = scope.dim('n')\n                n = 9"
        )
    else:
        model = model.replace("        if self.first:", "        for ignored in (1,):", 1)
    package, _ = _resource_context_fixture(tmp_path, monkeypatch, model=model)
    with pytest.raises(ti.aot.PackageExportError, match="declared boolean|declared dimensions|conditional branches"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


FIELD_LOOP_SOURCE = """
import taichi_forge as ti
from resource_contexts import resource_group, read_events
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int):
        global constructors
        constructors += 1
        self.n = n
        with resource_group(self, 'state', initialization='zero') as scope:
            cells = scope.dense(ti.i, scope.dim('n'))
            for name in ('values', 'weights'):
                setattr(scope, name, scope.field(ti.i32, shape=scope.dim('n'), placement=cells))
            self.last_member = name
            for name in ('vectors', 'unused_vectors'):
                setattr(scope, name, scope.vector_field(3, ti.f32, shape=scope.dim('n'), placement=cells))
            self.last_vector = name
    @ti.kernel
    def step(self):
        for i in self.values:
            self.values[i] += 1
            self.vectors[i] = ti.Vector([i, 2 * i, 3 * i])
"""


@pytest.mark.parametrize("before", [True, False])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_resource_loop_normalization_starts_at_scope_binding(tmp_path, monkeypatch, before):
    source = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, labels=None):
        global constructors
        constructors += 1
        self.n = n
        for name, value in (labels or {}).items():
            setattr(self, name, value)
        self = ti.experimental.ResourceScope(self, initialization='zero')
        for name in ('values', 'weights'):
            setattr(self, name, self.field(ti.i32, shape=self.dim('n')))
    @ti.kernel
    def step(self):
        for i in self.values:
            self.values[i] += i + 1
"""
    if not before:
        binding = "        self = ti.experimental.ResourceScope(self, initialization='zero')\n"
        source = source.replace(binding, "").replace("        for name, value", binding + "        for name, value")
    package = _source_package(tmp_path, monkeypatch, source)
    if not before:
        with pytest.raises(ti.aot.PackageExportError, match="Field-name loop|ResourceScope"):
            ti.aot.export_package(package, tmp_path / "sdk")
        assert sys.modules["definition_fixture.model"].constructors == 0
        return
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    original = package.Model(7, {"label": "jit"})
    original.step()
    assert original.label == "jit"
    host = (tmp_path / "sdk/host/model.py").read_text()
    assert "setattr(self, name, value)" in host
    assert "labels" in host
    with (
        Runtime(manifest["target"]) as runtime,
        load_package(runtime, tmp_path / "sdk") as code,
    ):
        a = code.host.Model(7, {"label": "aot", "other": 19})
        b = code.host.Model(19, {"label": "independent"})
        assert (a.label, a.other, b.label) == ("aot", 19, "independent")
        a.step()
        np.testing.assert_array_equal(a.values.to_numpy(), original.values.to_numpy())
        np.testing.assert_array_equal(b.values.to_numpy(), 0)
        b.step()
        np.testing.assert_array_equal(b.values.to_numpy(), np.arange(1, 20))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_literal_field_name_loops_preserve_layout_and_host_order(tmp_path, monkeypatch):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    package, _ = _resource_context_fixture(tmp_path, monkeypatch, model=FIELD_LOOP_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["entries"]) == 2
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([n for n in archive.namelist() if n.endswith((".ll", ".bc", ".spv"))]) == 2
    original = package.Model(7)
    original.step()
    root = ti.lang.impl.get_runtime().prog.get_snode_root(original.values.snode.ptr.get_snode_tree_id())
    nodes = _compiled_nodes(root)
    expected = np.arange(7)[:, None] * np.arange(1, 4)[None, :]
    np.testing.assert_array_equal(original.vectors.to_numpy(), expected)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7), code.host.Model(19)
        assert a.values._group_owner.layout["nodes"] == nodes
        assert a.last_member == original.last_member == "weights"
        assert a.last_vector == original.last_vector == "unused_vectors"
        assert a.values._group_owner is a.vectors._group_owner
        a.step()
        b.step()
        np.testing.assert_array_equal(a.vectors.to_numpy(), expected)
        np.testing.assert_array_equal(a.weights.to_numpy(), 0)
        np.testing.assert_array_equal(a.unused_vectors.to_numpy(), 0)
        a.tree.destroy()
        b.step()
        np.testing.assert_array_equal(b.values.to_numpy(), 2)


@pytest.mark.parametrize("change", ["dynamic", "duplicate", "name_use", "mangled", "shadow", "effect"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_field_name_loops_reject_unproven_declarations(tmp_path, monkeypatch, change):
    model = FIELD_LOOP_SOURCE
    if change == "dynamic":
        model = model.replace("('values', 'weights')", "self.names")
    elif change == "duplicate":
        model = model.replace("('values', 'weights')", "('values', 'values')")
    elif change == "name_use":
        model = model.replace("shape=scope.dim('n'), placement=cells", "shape=name, placement=cells")
    elif change == "mangled":
        model = model.replace("('values', 'weights')", "('__values', 'weights')")
    elif change == "shadow":
        model = model.replace("self, n: int):", "self, n: int, setattr=None):")
    else:
        model = model.replace(
            "                setattr(scope, name, scope.field",
            "                print(name)\n                setattr(scope, name, scope.field",
        )
    package, _ = _resource_context_fixture(tmp_path, monkeypatch, model=model)
    with pytest.raises(ti.aot.PackageExportError, match="Field-name loop|ResourceScope"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


DERIVED_FIELDS_SOURCE = """
import taichi_forge as ti
from resource_factories import declare_extent
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, wrong=False):
        global constructors
        constructors += 1
        self.n = n
        self.work = 2 * n + 1 + int(wrong)
        self.capacity = 1 << (self.work - 1).bit_length()
        scope = ti.experimental.ResourceScope(self)
        work = declare_extent(scope, 'work', 2 * scope.dim('n') + 1)
        scope.dim('work')
        capacity = scope.power_of_two_dim('capacity', work)
        scope.values = scope.field(ti.i32, shape=capacity)

    @ti.kernel
    def step(self, bias: ti.i32):
        for i in self.values:
            self.values[i] = i + bias if i < self.work else -1

    def graph(self):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.step, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'bias', ti.i32),
                         template_args={'self': self})
        return builder.compile()
"""

DERIVED_HELPER_SOURCE = """
effects = []
def declare_extent(scope, name, expression):
    method = getattr(type(scope), 'derived_dim', None)
    return method(scope, name, expression) if method is not None else scope.dim(name)
"""


@pytest.mark.parametrize("forwarded", [False, True, "nested"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_derived_dimensions_preserve_relations_without_model_execution(tmp_path, monkeypatch, forwarded):
    helper = DERIVED_HELPER_SOURCE
    if forwarded == "nested":
        helper += """
def declare_layout(scope):
    size = 2 * scope.dim('n') + 1
    return declare_extent(scope, 'work', size.maximum(1))
"""
    (tmp_path / "resource_factories.py").write_text(textwrap.dedent(helper), encoding="utf8")
    monkeypatch.delitem(sys.modules, "resource_factories", raising=False)
    source = DERIVED_FIELDS_SOURCE
    if forwarded == "nested":
        source = source.replace("import declare_extent", "import declare_layout")
        source = source.replace("declare_extent(scope, 'work', 2 * scope.dim('n') + 1)", "declare_layout(scope)")
    if not forwarded:
        source = source.replace("declare_extent(scope, 'work',", "scope.derived_dim('work',").replace(
            "from resource_factories import declare_extent\n", ""
        )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 15
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["entries"]) == 2  # One implementation plus the unimported public entry.
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([item for item in archive.infolist() if item.filename.endswith((".ll", ".bc", ".spv"))]) == 2
    assert len(manifest["declaration_factories"]) == int(bool(forwarded))
    assert "from resource_factories import" not in (tmp_path / "sdk/host/model.py").read_text()
    original = package.Model(7)
    original.step(5)
    np.testing.assert_array_equal(original.values.to_numpy(), [*range(5, 20), -1])
    with pytest.raises(ValueError, match="does not match"):
        package.Model(7, wrong=True)
    probe = tmp_path / "consume_derived.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture', 'resource_factories'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3), code.host.Model(19)
            empty = code.host.Model(0)
            empty.step(9)
            np.testing.assert_array_equal(empty.values.to_numpy(), [9])
            a.step(5)
            b.step(-2)
            np.testing.assert_array_equal(a.values.to_numpy(), [*range(5, 12), -1])
            np.testing.assert_array_equal(b.values.to_numpy(), [*range(-2, 37), *([-1] * 25)])
            registered = set(runtime._resources)
            try:
                code.host.Model(7, wrong=True)
            except ValueError as error:
                assert 'does not match' in str(error)
            else:
                raise AssertionError('Incorrect Host dimension relation accepted')
            assert set(runtime._resources) == registered
            graph = a.graph()
            call = graph.bind({'bias': 11})
            a.work += 1
            for operation in (lambda: a.step(2), lambda: graph.bind({'bias': 2})):
                try:
                    operation()
                except ValueError as error:
                    assert 'does not match' in str(error)
                else:
                    raise AssertionError('Mutated derived count accepted')
            call.run()
            np.testing.assert_array_equal(a.values.to_numpy(), [*range(11, 18), -1])
            call.close()
            b.step(4)
            np.testing.assert_array_equal(b.values.to_numpy()[:39], np.arange(39) + 4)
        print('DERIVED_PACKAGE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DERIVED_PACKAGE_OK" in result.stdout


@pytest.mark.parametrize("change", ["cycle", "effect", "model", "drop", "literal"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_derived_dimension_proof_rejects_unsafe_helpers(tmp_path, monkeypatch, change):
    helper, source = DERIVED_HELPER_SOURCE, DERIVED_FIELDS_SOURCE
    if change == "cycle":
        source = source.replace("2 * scope.dim('n') + 1", "scope.dim('work') + 1")
    elif change == "literal":
        source = source.replace("2 * scope.dim('n') + 1", "3")
    elif change == "effect":
        helper = helper.replace("    method =", "    effects.append(1)\n    method =")
    elif change == "model":
        helper = helper.replace("    method =", "    if scope.n:\n        return scope.dim(name)\n    method =")
    else:
        helper = helper.replace("method(scope, name, expression)", "scope.dim(name)")
    (tmp_path / "resource_factories.py").write_text(textwrap.dedent(helper), encoding="utf8")
    monkeypatch.delitem(sys.modules, "resource_factories", raising=False)
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(
        ti.aot.PackageExportError, match="Cyclic derived|factory forwarding is unproven|declared dimension algebra"
    ):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["resource_factories"].effects == []
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


RESOURCE_FACTORY_SOURCE = """
import taichi_forge as ti
effects = []
def resource_scope(owner, *, builder=None, requires=(), initialization='consumer_explicit'):
    if initialization not in ('consumer_explicit', 'zero'):
        raise ValueError('Unknown resource initialization')
    factory = getattr(getattr(ti, 'experimental', None), 'ResourceScope', None)
    qualified = factory is not None and all(hasattr(factory, name) for name in requires)
    if qualified and initialization != 'consumer_explicit':
        import inspect
        qualified = 'initialization' in inspect.signature(factory).parameters
        if qualified:
            return factory(owner, builder=builder, initialization=initialization)
    return factory(owner, builder=builder) if qualified else object()
"""


CONSTANT_SOURCE = """
import taichi_forge as ti
GROUP_SIZE = 4
constructors = 0
def declare_constant(scope, name):
    method = getattr(type(scope), 'constant', None)
    if method is not None:
        method(scope, name)
    else:
        scope.static(name, choices=(getattr(scope, name),), kernel_only=True)
@ti.data_oriented
class Model:
    group_size = GROUP_SIZE
    unused = 99
    def __init__(self, n: int, mode: bool = False):
        global constructors
        constructors += 1
        self.n = n
        self.mode = mode
        self.scale = 2.0
        scope = ti.experimental.ResourceScope(self)
        declare_constant(scope, 'group_size')
        scope.constant('scale')
        scope.constant('unused')
        scope.static('mode', choices=(False, True))
        scope.values = scope.field(ti.f32, shape=scope.dim('n'))
    @ti.kernel
    def step(self):
        for i in self.values:
            value = self.scale
            for j in ti.static(range(self.group_size)):
                value += j
            if ti.static(self.mode):
                value += 10
            self.values[i] = value + i
    @ti.kernel
    def clear(self):
        for i in self.values:
            self.values[i] = 0
    def graph(self):
        graph = ti.graph.GraphBuilder()
        graph.dispatch(self.step)
        return graph.compile()
class Wider(Model):
    group_size = 6
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_definition_constants_preserve_dependencies_and_snapshots(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, CONSTANT_SOURCE, initializer="from .model import Model, Wider\n")
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 30
    assert sys.modules["definition_fixture.model"].constructors == 0
    entries = [entry for entry in manifest["entries"].values() if ".model." in entry["definition"]]
    assert len(entries) == 5  # Four used static combinations and one shared clear.
    assert all(set(family["domain"]) == {"mode"} for family in manifest["entry_families"].values())
    clear = next(entry for entry in manifest["entries"].values() if entry["definition"].endswith(".clear"))
    assert not clear.get("admission")
    assert len(clear["definitions"]) == 2
    for entry in manifest["entries"].values():
        if entry["definition"].endswith(".step"):
            assert set(entry["admission"]["constants"]) == {"group_size", "scale"}
    original = package.Wider(5, True)
    original.step()
    np.testing.assert_array_equal(original.values.to_numpy(), np.arange(5) + 27)
    probe = tmp_path / "consume_constants.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3), code.host.Wider(5, True)
            graph = a.graph()
            with graph.bind({}) as prepared:
                a.unused = 101
                a.step()
                np.testing.assert_array_equal(a.values.to_numpy(), np.arange(3) + 8)
                b.step()
                np.testing.assert_array_equal(b.values.to_numpy(), np.arange(5) + 27)
                a.group_size = 6
                for operation in (a.step, lambda: graph.bind({})):
                    try:
                        operation()
                    except ValueError as error:
                        assert 'Definition constant' in str(error)
                    else:
                        raise AssertionError('changed constant accepted')
                a.clear()  # Unread constants do not guard unrelated entries.
                prepared.run()
                np.testing.assert_array_equal(a.values.to_numpy(), np.arange(3) + 8)
            code.host.Wider.group_size = 9
            before = set(runtime._resources)
            try:
                code.host.Wider(7)
            except ValueError as error:
                assert 'Definition constant' in str(error)
            else:
                raise AssertionError('changed class constant accepted')
            assert set(runtime._resources) == before
        print('DEFINITION_CONSTANTS_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DEFINITION_CONSTANTS_OK" in result.stdout


@pytest.mark.parametrize("change", ["model", "call", "rebound", "conditional", "descriptor", "setter"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_definition_constants_reject_unproven_sources(tmp_path, monkeypatch, change):
    source = CONSTANT_SOURCE
    if change == "model":
        source = source.replace("self.scale = 2.0", "self.scale = n")
    elif change == "call":
        source = source.replace("self.scale = 2.0", "self.scale = float(2)")
    elif change == "rebound":
        source = source.replace("self.scale = 2.0", "self.scale = 2.0\n        self.scale = 3.0")
    elif change == "conditional":
        source = source.replace("self.scale = 2.0", "if mode:\n            self.scale = 2.0")
    elif change == "setter":
        source = source.replace(
            "    unused = 99",
            """    unused = 99
    @property
    def scale(self):
        raise AssertionError('getter evaluated')
    @scale.setter
    def scale(self, value):
        raise AssertionError('setter evaluated')""",
        )
    else:
        source = source.replace(
            "group_size = GROUP_SIZE",
            "@property\n    def group_size(self):\n        raise AssertionError('getter evaluated')",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="Definition constant"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


def _resource_factory_fixture(
    tmp_path,
    monkeypatch,
    factory_source=RESOURCE_FACTORY_SOURCE,
    *,
    requires="dense",
    extra_argument="",
    model_source=EXPLICIT_GROUP_SOURCE,
):
    (tmp_path / "resource_factories.py").write_text(textwrap.dedent(factory_source), encoding="utf8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "resource_factories", raising=False)
    source = model_source.replace(
        "import taichi_forge as ti", "import taichi_forge as ti\nfrom resource_factories import resource_scope"
    )
    source = source.replace(
        "ti.experimental.ResourceScope(self, builder=fields, initialization='zero')",
        f"resource_scope(self, {extra_argument}builder=fields, requires=('{requires}',), initialization='zero')",
    )
    source = source.replace(
        "ti.experimental.ResourceScope(self, builder=separate)", "resource_scope(self, builder=separate)"
    )
    if ti.cfg.arch == ti.vulkan:
        source = source.replace("ti.f64", "ti.f32")
    return _source_package(tmp_path, monkeypatch, source)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_proves_imported_resource_factory_without_running_it(tmp_path, monkeypatch):
    package = _resource_factory_fixture(tmp_path, monkeypatch)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["declaration_factories"]) == 2
    assert {d["definition"] for d in manifest["declaration_factories"]} == {"resource_factories.resource_scope"}
    host = (tmp_path / "sdk/host/model.py").read_text()
    assert "from resource_factories import" not in host
    original = package.Model(7, 3)
    original.step(2.0)
    probe = tmp_path / "consume_factory.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi','taichi_forge','definition_fixture','resource_factories'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a,b = code.host.Model(7,3),code.host.Model(19,2)
            a.step(2.)
            b.step(-1.)
            np.testing.assert_array_equal(a.a.to_numpy(), 2*np.arange(7))
            np.testing.assert_array_equal(b.a.to_numpy(), -np.arange(19))
        print('RESOURCE_FACTORY_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESOURCE_FACTORY_OK" in result.stdout


@pytest.mark.parametrize("change", ["effect", "model", "global", "missing", "argument", "dropped", "early_return"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproven_resource_factory_paths(tmp_path, monkeypatch, change):
    source = RESOURCE_FACTORY_SOURCE
    if change == "effect":
        source = source.replace("    if initialization", "    effects.append(1)\n    if initialization", 1)
    if change == "model":
        source = source.replace("    factory =", "    if owner.n > 3:\n        effects.append(1)\n    factory =", 1)
    if change == "global":
        source = "runtime_choice = True\n" + source.replace(
            "    factory =", "    if runtime_choice:\n        effects.append(1)\n    factory =", 1
        )
    if change == "early_return":
        source = source.replace(
            "    factory =", "    if initialization == 'zero':\n        return None\n    factory =", 1
        )
    extra_argument = ""
    if change in ("argument", "dropped"):
        source = source.replace("requires=(),", "requires=(), ignored=None,")
        extra_argument = "ignored=self.observe(), " if change == "argument" else "ignored=self, "
    package = _resource_factory_fixture(
        tmp_path,
        monkeypatch,
        source,
        requires="missing" if change == "missing" else "dense",
        extra_argument=extra_argument,
    )
    with pytest.raises(ti.aot.PackageExportError, match="factory forwarding is unproven"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["resource_factories"].effects == []
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("binding", ["parameter", "assignment"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_shadowed_resource_factory(tmp_path, monkeypatch, binding):
    source = EXPLICIT_GROUP_SOURCE
    if binding == "parameter":
        source = source.replace("n: int, m: int):", "n: int, m: int, resource_scope=None):")
    else:
        source = source.replace(
            "fields = ti.FieldsBuilder()", "resource_scope = None\n        fields = ti.FieldsBuilder()"
        )
    package = _resource_factory_fixture(tmp_path, monkeypatch, model_source=source)
    with pytest.raises(ti.aot.PackageExportError, match="factory binding is shadowed"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


LAYOUT_SHARING_SOURCE = """
import taichi_forge as ti
constructors = 0

class Methods:
    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.values:
            self.values[i] = gain * (i + 1)

@ti.data_oriented
class Model(Methods):
    def __init__(self, n: int):
        global constructors
        constructors += 1
        self.n = n
        fields = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=fields)
        cells = scope.dense(ti.i, scope.dim('n'))
        scope.values = scope.field(ti.f32, shape=scope.dim('n'), placement=cells)
        scope.padding = scope.field(ti.i32, shape=scope.dim('n'), placement=cells)
        self.tree = fields.finalize()

@ti.data_oriented
class Expanded(Methods):
    def __init__(self, n: int):
        global constructors
        constructors += 1
        self.n = n
        fields = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=fields)
        cells = scope.dense(ti.i, scope.dim('n'))
        scope.padding = scope.vector_field(3, ti.i32, shape=scope.dim('n'), placement=cells)
        scope.values = scope.field(ti.f32, shape=scope.dim('n'), placement=cells)
        self.tree = fields.finalize()
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_layout_contracts_do_not_duplicate_compatible_numerical_code(tmp_path, monkeypatch):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    package = _source_package(
        tmp_path, monkeypatch, LAYOUT_SHARING_SOURCE, initializer="from .model import Model, Expanded\n"
    )
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    groups = manifest["field_groups"]
    assert len(groups) == 2
    assert (
        groups["definition_fixture.model.Model:builder_fields"]
        != groups["definition_fixture.model.Expanded:builder_fields"]
    )
    # Different complete layout/binding contracts remain explicit. Both access
    # ABIs are affine, so the original native builder stores their code once.
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        code = [item for item in archive.infolist() if item.filename.endswith((".ll", ".bc", ".spv"))]
        assert len(code) == 2  # shared step + unimported public entry
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as loaded:
        a, b = loaded.host.Model(7), loaded.host.Expanded(19)
        assert a.values._group_owner is not b.values._group_owner
        a.step(2.0)
        b.step(-3.0)
        np.testing.assert_array_equal(a.values.to_numpy(), 2 * (np.arange(7) + 1))
        np.testing.assert_array_equal(b.values.to_numpy(), -3 * (np.arange(19) + 1))
        for cls, instance, n in ((package.Model, a, 7), (package.Expanded, b, 19)):
            original = cls(n)
            original.step(1.0)
            root = ti.lang.impl.get_runtime().prog.get_snode_root(original.values.snode.ptr.get_snode_tree_id())
            assert instance.values._group_owner.layout["nodes"] == _compiled_nodes(root)
        a.tree.destroy()
        b.step(4.0)
        np.testing.assert_array_equal(b.values.to_numpy(), 4 * (np.arange(19) + 1))
        with pytest.raises(RuntimeError):
            a.step(1.0)


CONDITIONAL_VALUES_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, enabled: bool, mode: int = 0, gain: float = 2.0):
        global constructors
        constructors += 1
        self.n, self.enabled = n, enabled
        scope = ti.experimental.ResourceScope(self)
        scope.static('enabled', choices=(False, True))
        if self.enabled:
            self.mode, self.gain, self.size = mode, gain, n + 1
            scope.static('mode', choices=(0, 1))
            scope.value('gain', ti.f32)
            scope.derived_dim('size', scope.dim('n') + 1)
        scope.values = scope.field(ti.f32, shape=scope.dim('n'), initialization='zero')
        if self.enabled:
            scope.extra = scope.field(ti.f32, shape=scope.dim('size'), initialization='zero')
        else:
            scope.record('extra', None, optional=True)

    @ti.kernel
    def step(self):
        if ti.static(self.enabled):
            for i in range(self.size):
                self.extra[i] = self.gain * (i + ti.static(self.mode))
        else:
            for i in self.values:
                self.values[i] = -i

    @ti.kernel
    def common(self):
        for i in self.values:
            self.values[i] = i + 1

    def graph(self):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.step)
        return builder.compile()
"""


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_conditional_values_and_domains_follow_actual_guards(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, CONDITIONAL_VALUES_SOURCE)
    original = package.Model(7, True, 1, 3.0)
    original.step()
    expected = original.extra.to_numpy()
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 37
    assert sys.modules["definition_fixture.model"].constructors == 1
    assert len(manifest["entries"]) == 5  # Three step paths, common, unimported public entry.
    (family,) = manifest["entry_families"].values()
    assert len(family["variants"]) == 3
    assert family["conditions"] == {"mode": {"enabled": True}}
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b, c = code.host.Model(7, True, 1, 3.0), code.host.Model(19, True, 0, -2.0), code.host.Model(5, False)
        assert not hasattr(c, "mode") and not hasattr(c, "gain") and not hasattr(c, "size")
        a.step()
        b.step()
        c.step()
        np.testing.assert_array_equal(a.extra.to_numpy(), expected)
        np.testing.assert_array_equal(b.extra.to_numpy(), -2 * np.arange(20))
        np.testing.assert_array_equal(c.values.to_numpy(), -np.arange(5))
        assert c.extra is None
        assert a.extra._group_owner is not b.extra._group_owner
        plan = a.graph()
        prepared = plan.bind({})
        a.gain = 7.0
        prepared.run()
        np.testing.assert_array_equal(a.extra.to_numpy(), expected)
        a.step()
        np.testing.assert_array_equal(a.extra.to_numpy(), 7 * (np.arange(8) + 1))
        a.mode = 9
        with pytest.raises(ValueError, match="outside its declared package domain"):
            a.step()
        a.common()  # Unused optional configuration does not enter this code.
        np.testing.assert_array_equal(a.values.to_numpy(), np.arange(7) + 1)
        a.mode = 1
        a.size = 3
        with pytest.raises(ValueError, match="[Dd]erived dimension"):
            a.step()
        prepared.close()
    probe = tmp_path / "consume_conditional_values.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(17, True, 1, 3.0), code.host.Model(33, False)
            a.step()
            b.step()
            np.testing.assert_array_equal(a.extra.to_numpy(), 3 * (np.arange(18) + 1))
            np.testing.assert_array_equal(b.values.to_numpy(), -np.arange(33))
            assert not hasattr(b, 'mode') and b.extra is None
        print('CONDITIONAL_VALUES_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONDITIONAL_VALUES_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("expression", ["self.gain", "self.size", "0 + ti.static(self.mode)"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_conditional_values_require_their_numerical_guard(tmp_path, monkeypatch, expression):
    begin = CONDITIONAL_VALUES_SOURCE.index("    @ti.kernel\n    def step")
    end = CONDITIONAL_VALUES_SOURCE.index("    @ti.kernel\n    def common")
    source = (
        CONDITIONAL_VALUES_SOURCE[:begin]
        + "    @ti.kernel\n    def step(self):\n"
        + f"        for i in self.values:\n            self.values[i] = {expression}\n\n"
        + CONDITIONAL_VALUES_SOURCE[end:]
    )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="requires its declared static guard"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0


CONDITIONAL_LAYOUT_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int, first: bool, second: bool):
        global constructors
        constructors += 1
        self.n, self.m = n, m
        self.first, self.second = first, second
        self.extra = self.other = None
        fields = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=fields)
        scope.static('first', choices=(False, True))
        scope.static('second', choices=(False, True))
        cells = scope.dense(ti.i, scope.dim('n'))
        scope.values = scope.field(ti.f32, shape=scope.dim('n'), placement=cells)
        if self.first:
            scope.extra = scope.vector_field(3, ti.f32, shape=scope.dim('n'), placement=cells)
        if self.second:
            scope.other = scope.field(ti.f32, shape=scope.dim('m'))
        self.tree = fields.finalize()

    @ti.kernel
    def common(self, gain: ti.f32):
        for i in self.values:
            self.values[i] = gain * (i + 1)

    @ti.kernel
    def first_step(self):
        if ti.static(self.first):
            for i in self.extra:
                self.extra[i] = ti.Vector([i, i + 1, i + 2])
        else:
            for i in self.values:
                self.values[i] = 10.0

    @ti.kernel
    def second_step(self):
        if ti.static(self.second):
            for i in self.other:
                self.other[i] = i * 3.0
        else:
            for i in self.values:
                self.values[i] = 20.0
"""


def _conditional_struct_source(layout):
    return (
        CONDITIONAL_LAYOUT_SOURCE.replace("fields = ti.FieldsBuilder()", "# Original implicit groups")
        .replace("ti.experimental.ResourceScope(self, builder=fields)", "ti.experimental.ResourceScope(self)")
        .replace("        cells = scope.dense(ti.i, scope.dim('n'))\n", "")
        .replace(", placement=cells", "")
        .replace("        self.tree = fields.finalize()\n", "")
        .replace(
            "scope.vector_field(3, ti.f32, shape=scope.dim('n'))",
            "scope.struct_field(dict(x=ti.f32, v=ti.types.vector(3, ti.f32)), "
            f"shape=scope.dim('n'), layout=ti.Layout.{layout}, initialization='zero')",
        )
        .replace(
            "self.extra[i] = ti.Vector([i, i + 1, i + 2])",
            "self.extra.x[i] = 2 * i\n                self.extra.v[i] = ti.Vector([i, i + 1, i + 2])",
        )
    )


@pytest.mark.parametrize("layout", ["AOS", "SOA"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_conditional_struct_fields_preserve_layout_and_independent_code(tmp_path, monkeypatch, layout):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    package = _source_package(tmp_path, monkeypatch, _conditional_struct_source(layout))
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    counts = {}
    for item in manifest["entries"].values():
        counts[item["definition"].rsplit(".", 1)[-1]] = counts.get(item["definition"].rsplit(".", 1)[-1], 0) + 1
    assert counts == dict(common=1, first_step=2, second_step=2, available_without_a_test=1)
    assert len(manifest["entry_families"]) == 2
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        instances = []
        for first, second in ((False, False), (False, True), (True, False), (True, True)):
            n = 7 + len(instances)
            original, current = package.Model(n, 13, first, second), code.host.Model(n, 13, first, second)
            instances.append(current)
            for instance in (original, current):
                instance.common(3.0)
                instance.first_step()
                instance.second_step()
            np.testing.assert_array_equal(current.values.to_numpy(), original.values.to_numpy())
            if first:
                for name in ("x", "v"):
                    np.testing.assert_array_equal(
                        getattr(current.extra, name).to_numpy(), getattr(original.extra, name).to_numpy()
                    )
            else:
                assert current.extra is None
            root = ti.lang.impl.get_runtime().prog.get_snode_root(original.values.snode.ptr.get_snode_tree_id())
            assert current.values._group_owner.layout["nodes"] == _compiled_nodes(root)
        builder = code.graph_builder()
        builder.dispatch(instances[-1].first_step)
        with builder.compile().bind({}) as prepared:
            instances[-1].first = False
            prepared.run()
            with pytest.raises(ValueError, match="layout configuration changed"):
                instances[-1].common(1.0)
        assert len({id(value.values._group_owner) for value in instances}) == 4
    probe = tmp_path / "consume_conditional_struct.py"
    probe.write_text(
        """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    a, b = code.host.Model(5, 3, False, False), code.host.Model(17, 11, True, True)
    a.common(3.0)
    b.first_step()
    b.second_step()
    assert a.extra is None and a.other is None
    np.testing.assert_array_equal(a.values.to_numpy(), 3 * (np.arange(5) + 1))
    np.testing.assert_array_equal(b.extra.x.to_numpy(), 2 * np.arange(17))
    np.testing.assert_array_equal(b.extra.v.to_numpy(), np.arange(17)[:, None] + np.arange(3))
    np.testing.assert_array_equal(b.other.to_numpy(), 3 * np.arange(11))
print('CONDITIONAL_STRUCT_SOURCE_FREE_OK')
""",
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONDITIONAL_STRUCT_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("read", ["self.extra.x[0]", "len(ti.static(self.extra.keys))"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_conditional_struct_requires_valid_presence(tmp_path, monkeypatch, read):
    source = _conditional_struct_source("SOA").replace("self.values[i] = gain * (i + 1)", f"self.values[i] = {read}")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="NoneType|Conditional"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("layout,record", [("AOS", False), ("AOS", True), ("SOA", True), ("placement", False)])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_conditional_model_tensors_preserve_native_layout(tmp_path, monkeypatch, layout, record):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    source = (
        _conditional_struct_source(layout)
        .replace("ti.types.vector(3, ti.f32)", "ti.types.vector(self.m, ti.f32)")
        .replace("        if self.first:", "        scope.dim('m')\n        if self.first:", 1)
        .replace(
            "self.extra.v[i] = ti.Vector([i, i + 1, i + 2])",
            "for j in range(self.m):\n                    self.extra.v[i][j] = i + j",
        )
    )
    if not record:
        source = (
            source.replace(
                "scope.struct_field(dict(x=ti.f32, v=ti.types.vector(self.m, ti.f32)),",
                "scope.vector_field(self.m, ti.f32,",
            )
            .replace(", layout=ti.Layout.AOS", "")
            .replace("                self.extra.x[i] = 2 * i\n", "")
            .replace("self.extra.v[i]", "self.extra[i]")
        )
    if layout == "placement":
        source = (
            CONDITIONAL_LAYOUT_SOURCE.replace("scope.vector_field(3,", "scope.vector_field(self.m,")
            .replace("        if self.first:", "        scope.dim('m')\n        if self.first:", 1)
            .replace(
                "self.extra[i] = ti.Vector([i, i + 1, i + 2])",
                "for j in range(self.m):\n                    self.extra[i][j] = i + j",
            )
        )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 46
    assert sys.modules["definition_fixture.model"].constructors == 0
    counts = {}
    for item in manifest["entries"].values():
        method = item["definition"].rsplit(".", 1)[-1]
        counts[method] = counts.get(method, 0) + 1
    assert counts == dict(common=1, first_step=2, second_step=2, available_without_a_test=1)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        instances = []
        for n, m, first, second in ((5, 0, False, False), (7, 5, True, False), (3, 13, True, True)):
            original, current = package.Model(n, m, first, second), code.host.Model(n, m, first, second)
            instances.append(current)
            for instance in (original, current):
                instance.common(4.0)
                instance.first_step()
                instance.second_step()
            root = ti.lang.impl.get_runtime().prog.get_snode_root(original.values.snode.ptr.get_snode_tree_id())
            assert current.values._group_owner.layout["nodes"] == _compiled_nodes(root)
            np.testing.assert_array_equal(current.values.to_numpy(), original.values.to_numpy())
            if first:
                actual = current.extra.v if record else current.extra
                expected = original.extra.v if record else original.extra
                np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
                np.testing.assert_array_equal(actual.to_numpy(), np.arange(n)[:, None] + np.arange(m))
            else:
                assert current.extra is None
        builder = code.graph_builder()
        builder.dispatch(instances[1].first_step)
        with builder.compile().bind({}) as prepared:
            instances[1].first = False
            prepared.run()
            with pytest.raises(ValueError, match="layout configuration changed"):
                instances[1].common(1.0)
        assert len({id(instance.values._group_owner) for instance in instances}) == 3
    probe = tmp_path / "consume_conditional_model.py"
    probe.write_text(
        """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    absent = code.host.Model(5, 0, False, False)
    current = code.host.Model(3, 129, True, True)
    absent.common(2.0)
    current.first_step()
    current.second_step()
    value = current.extra.v if int(sys.argv[3]) else current.extra
    np.testing.assert_array_equal(value.to_numpy(), np.arange(3)[:, None] + np.arange(129))
    np.testing.assert_array_equal(absent.values.to_numpy(), 2 * (np.arange(5) + 1))
    assert absent.extra is None and absent.other is None
print('CONDITIONAL_MODEL_SOURCE_FREE_OK')
""",
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"], str(int(record))],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONDITIONAL_MODEL_SOURCE_FREE_OK" in result.stdout


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_conditional_model_tensor_rejects_unguarded_reads(tmp_path, monkeypatch):
    source = (
        _conditional_struct_source("SOA")
        .replace("ti.types.vector(3, ti.f32)", "ti.types.vector(self.m, ti.f32)")
        .replace("        if self.first:", "        scope.dim('m')\n        if self.first:", 1)
    )
    source = source.replace("self.values[i] = gain * (i + 1)", "self.values[i] = self.extra.v[i][0]")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="NoneType|Conditional"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_conditional_struct_in_child_keeps_qualified_guards(tmp_path, monkeypatch):
    source = _conditional_struct_source("SOA") + """
@ti.data_oriented
class Wrapper:
    def __init__(self, n: int, enabled: bool):
        scope = ti.experimental.ResourceScope(self)
        scope.child = Model(n, 1, enabled, False)
    @ti.kernel
    def read(self) -> ti.f32:
        value = -1.0
        if ti.static(self.child.first):
            self.child.extra.x[0] = 9.0
            value = self.child.extra.x[0]
        return value
"""
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        cls = code.module("definition_fixture.model").Wrapper
        a, b = cls(7, False), cls(13, True)
        assert a.read() == -1.0 and b.read() == 9.0
        assert a.child.extra is None
        assert a.read() == -1.0


def _allocation_helper_source(method=False):
    source = CONDITIONAL_LAYOUT_SOURCE.replace(
        "        if self.second:\n            scope.other = scope.field(ti.f32, shape=scope.dim('m'))",
        "        def optional_field(dtype, shape):\n"
        "            return scope.field(dtype, shape=shape) if self.second else None\n"
        "        scope.other = optional_field(shape=scope.dim('m'), dtype=ti.f32)",
    )
    if method:
        source = source.replace("@ti.data_oriented\nclass Model:", """class ResourceMixin:
    def allocate(self, field, resources):
        extent = resources.dim('m')
        resources.other = field(ti.f32, extent)
@ti.data_oriented
class Model(ResourceMixin):""").replace(
            "scope.other = optional_field(shape=scope.dim('m'), dtype=ti.f32)",
            "self.allocate(optional_field, scope)",
        )
    return source


@pytest.mark.parametrize("change", ["escape", "rebound", "side_effect", "override", "receiver"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_allocation_helpers_reject_unproven_forwarding(tmp_path, monkeypatch, change):
    source = _allocation_helper_source(method=True)
    if change == "escape":
        source = source.replace("        self.allocate(", "        self.saved = optional_field\n        self.allocate(")
    elif change == "rebound":
        source = source.replace("        self.allocate(", "        optional_field = None\n        self.allocate(")
    elif change == "side_effect":
        source = source.replace("        extent = resources.dim", "        self.changed = 1\n        extent = resources.dim")
    elif change == "override":
        source += "\nclass Derived(Model):\n    def allocate(self, field, resources):\n        pass\n"
    elif change == "receiver":
        source = source.replace("        self.allocate(", "        self.allocate = None\n        self.allocate(")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="[Aa]llocation|reassigned"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("placement", ["implicit", "conditional", "declarations", "expression", "helper", "methodhelper"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_conditional_dense_layouts_share_code_and_native_rules(tmp_path, monkeypatch, placement):
    import itertools
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    source = CONDITIONAL_LAYOUT_SOURCE
    if placement == "conditional":
        source = source.replace(
            "if self.second:", "if self.second:\n            other_cells = scope.dense(ti.i, scope.dim('m'))"
        )
    if placement == "conditional":
        source = source.replace("shape=scope.dim('m'))", "shape=scope.dim('m'), placement=other_cells)")
        source = source.replace(
            "other_cells = scope.dense(ti.i, scope.dim('m'))",
            "other_count, = (scope.dim(name) for name in ('m',))\n            other_cells = scope.dense(ti.i, other_count)",
        )
    if placement == "declarations":
        source = source.replace(
            "        scope.static('first', choices=(False, True))\n        scope.static('second', choices=(False, True))",
            "        for name in ('first', 'second'):\n            scope.static(name, choices=(False, True))\n"
            "        self.last_declared = name",
        )
    if placement == "expression":
        source = source.replace(
            "        if self.first:\n            scope.extra = scope.vector_field(3, ti.f32, shape=scope.dim('n'), placement=cells)",
            "        scope.extra = scope.vector_field(3, ti.f32, shape=scope.dim('n'), placement=cells) if self.first else None",
        ).replace(
            "        if self.second:\n            scope.other = scope.field(ti.f32, shape=scope.dim('m'))",
            "        scope.other = scope.field(ti.f32, shape=scope.dim('m')) if self.second else None",
        )
    if placement in ("helper", "methodhelper"):
        source = _allocation_helper_source(method=placement == "methodhelper")
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == (35 if placement.endswith("helper") else 13 if placement == "expression" else 12)
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["field_groups"]) == 1  # one conditional topology, not four layout products
    assert sorted(len(family["variants"]) for family in manifest["entry_families"].values()) == [2, 2]
    assert all(len(family["domain"]) == 1 for family in manifest["entry_families"].values())
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        code = [item for item in archive.infolist() if item.filename.endswith((".ll", ".bc", ".spv"))]
        assert len(code) == 6  # common + two per actual static dependency + public entry
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as loaded:
        instances = []
        for first, second in itertools.product((False, True), repeat=2):
            # A disabled branch must not evaluate its invalid, unused extent.
            m = 5 if second or placement.endswith("helper") else -3
            original, current = package.Model(7, m, first, second), loaded.host.Model(7, m, first, second)
            if placement == "declarations":
                assert original.last_declared == current.last_declared == "second"
            instances.append(current)
            for instance in (original, current):
                instance.first_step()
                instance.second_step()
                instance.common(2.0)
            for name in ("values", "extra", "other"):
                left, right = getattr(original, name), getattr(current, name)
                if left is None:
                    assert right is None
                else:
                    np.testing.assert_array_equal(right.to_numpy(), left.to_numpy())
            root = ti.lang.impl.get_runtime().prog.get_snode_root(original.values.snode.ptr.get_snode_tree_id())
            assert current.values._group_owner.layout["nodes"] == _compiled_nodes(root)
        assert len({id(instance.values._group_owner) for instance in instances}) == 4
        assert instances[0].values._group_owner.nbytes < instances[-1].values._group_owner.nbytes
        builder = loaded.module("definition_fixture.model").ti.graph.GraphBuilder()
        builder.dispatch(instances[-1].first_step)
        with builder.compile().bind({}) as prepared:
            instances[0].tree.destroy()
            prepared.run()
        instances[-1].first = False
        with pytest.raises(ValueError, match="layout configuration changed"):
            instances[-1].common(1.0)
        with pytest.raises(ValueError, match="layout configuration changed"):
            builder.dispatch(instances[-1].common)
        instances[1].first = True
        with pytest.raises(ValueError, match="layout configuration changed"):
            instances[1].first_step()
        resources = set(runtime._resources)
        with pytest.raises((ValueError, RuntimeError)):
            loaded.host.Model(7, -3, False, True)
        if placement.endswith("helper"):
            # The call's argument is evaluated before entering its disabled
            # helper body, unlike a direct conditional Field expression.
            with pytest.raises((ValueError, RuntimeError)):
                loaded.host.Model(7, -3, False, False)
        assert set(runtime._resources) == resources
    probe = tmp_path / "consume_conditional.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc, sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'definition_fixture','taichi','taichi_forge'}:
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a,b=code.host.Model(3,-1,False,False),code.host.Model(19,11,True,True)
            a.common(2.0)
            b.common(-1.0)
            b.first_step()
            b.second_step()
            np.testing.assert_array_equal(a.values.to_numpy(),2*(np.arange(3)+1))
            np.testing.assert_array_equal(b.values.to_numpy(),-(np.arange(19)+1))
            np.testing.assert_array_equal(b.other.to_numpy(),3*np.arange(11))
            assert a.extra is None and a.other is None
            a.tree.destroy()
            b.common(4.0)
        print('CONDITIONAL_LAYOUT_OK')
    """).replace("Model(3,-1", "Model(3,1" if placement.endswith("helper") else "Model(3,-1"),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONDITIONAL_LAYOUT_OK" in result.stdout


@pytest.mark.parametrize(
    "change, diagnostic",
    [
        ("missing_declaration", "unconditional declared boolean"),
        ("kernel_only", "unconditional declared boolean"),
        ("unguarded_field", "None"),
        ("empty_placement", "requires a member whenever its branch is active"),
        ("escaped_dimension", "Dimension alias must remain within its declaration branch"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_conditional_layouts_reject_unproven_guards(tmp_path, monkeypatch, change, diagnostic):
    source = CONDITIONAL_LAYOUT_SOURCE
    if change == "missing_declaration":
        source = source.replace("scope.static('first', choices=(False, True))", "pass")
    elif change == "kernel_only":
        source = source.replace("'first', choices=(False, True)", "'first', choices=(False, True), kernel_only=True")
    elif change == "empty_placement":
        source = source.replace(
            "if self.second:", "other_cells = scope.dense(ti.i, scope.dim('m'))\n        if self.second:"
        )
        source = source.replace("shape=scope.dim('m'))", "shape=scope.dim('m'), placement=other_cells)")
    elif change == "escaped_dimension":
        source = source.replace("if self.first:", "if self.first:\n            only_here = scope.dim('n')")
        source = source.replace(
            "self.tree = fields.finalize()",
            "scope.invalid = scope.field(ti.f32, shape=only_here)\n        self.tree = fields.finalize()",
        )
    else:
        source = source.replace("if ti.static(self.first):", "if ti.static(True):")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("names", ["('first', 'first')", "tuple(('first', 'second'))", "('first', n)"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_declaration_loops_require_fixed_distinct_names(tmp_path, monkeypatch, names):
    source = CONDITIONAL_LAYOUT_SOURCE.replace(
        "        scope.static('first', choices=(False, True))\n        scope.static('second', choices=(False, True))",
        f"        for name in {names}:\n            scope.static(name, choices=(False, True))",
    )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="fixed literal names"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


def _optional_layout_source():
    source = CONDITIONAL_LAYOUT_SOURCE
    for flag, field in (("first", "extra"), ("second", "other")):
        source = source.replace(f"        scope.static('{flag}', choices=(False, True))\n", "")
        source = source.replace(f"if self.{flag}:", f"if {flag}:")
        source = source.replace(f"if ti.static(self.{flag}):", f"if ti.static(self.{field} is not None):")
    source = source.replace(
        "        if second:",
        "        else:\n            scope.record('extra', None, optional=True)\n        if second:",
    )
    source = source.replace(
        "        self.tree = fields.finalize()",
        "        else:\n            scope.record('other', None, optional=True)\n        self.tree = fields.finalize()",
    )
    constructor = source[source.index("    def __init__") : source.index("    @ti.kernel")]
    # The original method object first compiles against an always-present
    # declaration. A later optional receiver must not reuse that presence fact.
    always = source.replace("class Model:", "class Always:").replace("        self.extra = self.other = None\n", "")
    for flag, field in (("first", "extra"), ("second", "other")):
        always = always.replace(f"        if {flag}:\n            scope.{field}", f"        scope.{field}")
        always = always.replace(f"        else:\n            scope.record('{field}', None, optional=True)\n", "")
    return always + "\nclass Model(Always):\n" + constructor


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_optional_fields_infer_presence_without_new_declarations(tmp_path, monkeypatch):
    import itertools

    package = _source_package(tmp_path, monkeypatch, _optional_layout_source())
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 13
    assert sys.modules["definition_fixture.model"].constructors == 0
    families = manifest["entry_families"]
    assert sorted(len(family["variants"]) for family in families.values()) == [2, 2]
    assert all(len(family["domain"]) == 1 for family in families.values())
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        assert len([item for item in archive.infolist() if item.filename.endswith((".ll", ".bc", ".spv"))]) == 6
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as loaded:
        for first, second in itertools.product((False, True), repeat=2):
            original, current = package.Model(7, 5, first, second), loaded.host.Model(7, 5, first, second)
            for owner in (original, current):
                owner.first_step()
                owner.second_step()
                owner.common(4.0)
            for field in ("values", "extra", "other"):
                expected, actual = getattr(original, field), getattr(current, field)
                if expected is None:
                    assert actual is None
                else:
                    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
        resources = set(runtime._resources)
        with pytest.raises(ValueError, match="requires a boolean"):
            loaded.host.Model(7, 5, 1, False)
        assert resources == set(runtime._resources)
    probe = tmp_path / "consume_optional.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc, sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'definition_fixture','taichi','taichi_forge'}:
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a,b=code.host.Model(3,-1,False,False),code.host.Model(19,11,True,True)
            a.first_step()
            b.first_step()
            a.second_step()
            b.second_step()
            np.testing.assert_array_equal(a.values.to_numpy(),20)
            np.testing.assert_array_equal(b.other.to_numpy(),3*np.arange(11))
            assert a.extra is None and a.other is None
            a.tree.destroy()
            b.common(4.0)
            np.testing.assert_array_equal(b.values.to_numpy(),4*(np.arange(19)+1))
        print('OPTIONAL_LAYOUT_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OPTIONAL_LAYOUT_OK" in result.stdout


@pytest.mark.parametrize("change", ["wrong_absence", "mutated_parameter", "unknown_domain"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_optional_fields_reject_ambiguous_presence(tmp_path, monkeypatch, change):
    source = _optional_layout_source()
    if change == "wrong_absence":
        source = source.replace(
            "scope.record('extra', None, optional=True)", "scope.record('different', None, optional=True)"
        )
    elif change == "mutated_parameter":
        source = source.replace("        if first:", "        first = not first\n        if first:")
    else:
        source = source.replace("first: bool", "first")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="unconditional declared boolean"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_optional_fields_reuse_explicit_and_negated_presence(tmp_path, monkeypatch):
    source = _optional_layout_source()
    constructor = source.index("class Model(Always)")
    source = source[:constructor] + source[constructor:].replace(
        "        if first:", "        scope.static('first', choices=(False, True))\n        if not self.first:"
    ).replace("        if second:", "        scope.static('second', choices=(False, True))\n        if self.second:")
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 13
    assert not any("presence_parameter" in item for owner in manifest["owners"].values() for item in owner.values())
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as loaded:
        a, b = loaded.host.Model(3, 2, False, False), loaded.host.Model(7, 5, True, True)
        a.first_step()
        b.first_step()
        assert b.extra is None
        np.testing.assert_array_equal(b.values.to_numpy(), 10)
        np.testing.assert_array_equal(a.extra.to_numpy(), np.arange(3)[:, None] + np.arange(3)[None, :])


STATIC_DOMAIN_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, first: bool, second: bool, third: bool, unused: int):
        global constructors
        constructors += 1
        self.n = n
        self.first, self.second, self.third, self.unused = first, second, third, unused
        scope = ti.experimental.ResourceScope(self)
        scope.static('first', choices=(False, True))
        scope.static('second', choices=(False, True), kernel_only=True)
        scope.static('third', choices=(False, True), kernel_only=True)
        scope.static('unused', choices=(0, 1, 2, 3, 4, 5, 6, 7))
        self.values = ti.field(ti.i32, shape=n)

    @ti.func
    def mode(self):
        result = 0
        if ti.static(self.first):
            if ti.static(self.second):
                result = 10
            else:
                result = 20
        else:
            if ti.static(self.third):
                result = 30
            else:
                result = 40
        return result

    @ti.kernel
    def compute(self):
        for i in range(self.n):
            self.values[i] = self.mode() + i

    @ti.kernel
    def other(self):
        for i in range(self.n):
            if ti.static(self.second):
                self.values[i] = 5
            else:
                self.values[i] = -5

    @ti.kernel
    def plain(self):
        for i in range(self.n):
            self.values[i] = i

class Equivalent(Model):
    pass
"""


@pytest.mark.parametrize("helper_kind", ["func", "real_func"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_specializes_only_observed_static_dependencies(tmp_path, monkeypatch, helper_kind):
    import itertools

    if helper_kind == "real_func" and ti.cfg.arch == ti.vulkan:
        pytest.skip("Real functions require LLVM; the separate rejection test covers Vulkan")
    source = STATIC_DOMAIN_SOURCE
    if helper_kind == "real_func":
        source = source.replace("@ti.func\n    def mode(self):", "@ti.real_func\n    def mode(self) -> ti.i32:")
    package = _source_package(tmp_path, monkeypatch, source, initializer="from .model import Model, Equivalent\n")
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert manifest["version"] == 11
    families = manifest["entry_families"]
    assert sorted(len(f["variants"]) for f in families.values()) == [2, 2, 4, 4]
    assert len(manifest["entries"]) == 8  # 4 compute + 2 other + plain + unused public function
    assert all("unused" not in f["domain"] for f in families.values())
    branching = next(f for f in families.values() if len(f["variants"]) == 4)
    assert {tuple(sorted(v["selection"])) for v in branching["variants"]} == {("first", "second"), ("first", "third")}
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        code = [item for item in archive.infolist() if item.filename.endswith((".ll", ".bc", ".spv"))]
        assert len(code) == 8
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as loaded:
        a, b = loaded.host.Model(7, False, False, False, 1), loaded.host.Equivalent(19, True, True, True, 7)
        bound = a.compute
        for first, second, third in itertools.product((False, True), repeat=3):
            original = package.Model(7, first, second, third, 0)
            a.first, a.second, a.third = first, second, third
            original.compute()
            bound()
            np.testing.assert_array_equal(a.values.to_numpy(), original.values.to_numpy())
        b.compute()
        np.testing.assert_array_equal(b.values.to_numpy(), np.arange(19) + 10)
        a.first, a.second = True, False
        builder = loaded.module("definition_fixture.model").ti.graph.GraphBuilder()
        builder.dispatch(a.compute)
        graph = builder.compile()
        with graph.bind({}) as prepared:
            a.first = False
            prepared.run()
            np.testing.assert_array_equal(a.values.to_numpy(), np.arange(7) + 20)
        a.first = 1  # bool and int are distinct static identities
        with pytest.raises(ValueError, match="outside.*domain"):
            bound()
        resources = set(runtime._resources)
        with pytest.raises(ValueError, match="outside.*domain"):
            loaded.host.Model(3, False, False, False, 99)
        assert set(runtime._resources) == resources
    probe = tmp_path / "consume_static.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'definition_fixture','taichi','taichi_forge'}:
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a,b=code.host.Model(3,False,True,False,0),code.host.Model(19,True,True,False,6)
            a.compute()
            b.compute()
            np.testing.assert_array_equal(a.values.to_numpy(),np.arange(3)+40)
            np.testing.assert_array_equal(b.values.to_numpy(),np.arange(19)+10)
            a.second=False
            a.other()
            np.testing.assert_array_equal(a.values.to_numpy(),-5)
        print('STATIC_DOMAINS_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STATIC_DOMAINS_OK" in result.stdout


@pytest.mark.parametrize("choices", ["()", "(False, False)", "(self.first,)", "(object(),)"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproven_static_domains(tmp_path, monkeypatch, choices):
    source = STATIC_DOMAIN_SOURCE.replace("'first', choices=(False, True)", f"'first', choices={choices}")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="Static domain"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_an_uncompiled_declared_static_branch(tmp_path, monkeypatch):
    source = STATIC_DOMAIN_SOURCE.replace("result = 10", "result = self.unqualified_parameter")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="undeclared capture self.unqualified_parameter"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


TARGET_FIELDS_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, redirect=False):
        global constructors
        constructors += 1
        self.n = n
        scope = ti.experimental.ResourceScope(self)
        self.arch = ti.lang.impl.current_cfg().arch
        if redirect:
            self.redirect()
        if self.arch == ti.cuda:
            extent = scope.dim('n') * 2
            scope.a = scope.field(ti.f32, shape=extent)
        else:
            extent = tuple(scope.dim(name) + 1 for name in ('n',))
            scope.a = scope.field(ti.f32, shape=extent)

    def redirect(self):
        self.arch = ti.cpu if self.arch == ti.cuda else ti.cuda

    def backend_name(self):
        return self.arch.name

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.a:
            self.a[i] = gain * (i + self.n)
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_selects_backend_resource_branches_with_runtime_guards(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, TARGET_FIELDS_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch)
    assert manifest["version"] == 6
    assert sys.modules["definition_fixture.model"].constructors == 0
    selections = manifest["target_selections"]["definition_fixture.model"]
    assert len(selections) == 1 and selections[0]["taken"] == (ti.cfg.arch == ti.cuda)
    expected = package.Model(7)
    expected.step(2.0)
    probe = tmp_path / "consume_target.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(7), code.host.Model(19)
            for model in (a, b):
                model.step(2.0)
                count = model.n * 2 if runtime.arch == 'cuda' else model.n + 1
                assert model.a.shape == (count,)
                assert model.backend_name() == runtime.arch
                np.testing.assert_array_equal(np.frombuffer(model.a.read(), np.float32), 2 * (np.arange(count) + model.n))
            registered = set(runtime._resources)
            try:
                code.host.Model(3, redirect=True)
            except ValueError as error:
                assert 'Host backend condition changed after export' in str(error)
            else:
                raise AssertionError('Changed backend branch silently selected another layout')
            assert runtime._resources == registered
        print('TARGET_FIELDS_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TARGET_FIELDS_SOURCE_FREE_OK" in result.stdout
    count = 14 if ti.cfg.arch == ti.cuda else 8
    np.testing.assert_array_equal(expected.a.to_numpy(), 2 * (np.arange(count) + 7))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_does_not_compile_other_backend_resources_or_select_model_branches(tmp_path, monkeypatch):
    source = TARGET_FIELDS_SOURCE.replace(
        "scope.a = scope.field(ti.f32, shape=extent)",
        "scope.a = scope.field(ti.f32, shape=extent, placement=None)",
        1,
    )
    package = _source_package(tmp_path, monkeypatch, source)
    ti.aot.export_package(package, tmp_path / "sdk")  # CUDA-only unsupported placement is absent.
    module = tmp_path / "definition_fixture/model.py"
    module.write_text(source.replace("if self.arch == ti.cuda:", "if n > 4:"), encoding="utf8")
    monkeypatch.delitem(sys.modules, "definition_fixture.model")
    importlib.import_module("definition_fixture.model")
    with pytest.raises(ti.aot.PackageExportError, match="unconditional supported resource declaration"):
        ti.aot.export_package(package, tmp_path / "model_branch")
    assert not (tmp_path / "model_branch").exists()


@pytest.mark.parametrize("shadow", ["frontend", "tuple_argument", "tuple_global"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_shadowed_backend_and_dimension_names(tmp_path, monkeypatch, shadow):
    source = TARGET_FIELDS_SOURCE
    if shadow == "frontend":
        source = source.replace("redirect=False):", "redirect=False, ti=None):")
        message = "shadowed frontend name"
    else:
        source = (
            source.replace("redirect=False):", "redirect=False, tuple=None):")
            if shadow == "tuple_argument"
            else "tuple = list\n" + source
        )
        message = "Field shape requires declared dimensions"
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match=message):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


FREE_RECEIVER_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.kernel
def publish(gain: ti.f32, solver: ti.template()):
    for i in solver.values:
        solver.values[i] = gain * (i + solver.n)

def forward(receiver, gain):
    publish(gain, solver=receiver)

@ti.data_oriented
class Model:
    def __init__(self, n: int):
        global constructors
        constructors += 1
        self.n = n
        scope = ti.experimental.ResourceScope(self)
        scope.values = scope.field(ti.f32, shape=scope.dim('n'))
    def update(self, gain):
        alias = self
        forward(alias, gain)
    def graph(self):
        builder = ti.graph.GraphBuilder()
        self.record(builder)
        return builder.compile()
    def record(self, builder):
        gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'gain', ti.f32)
        builder.dispatch(publish, gain, template_args={'solver': self})
    def writes(self):
        return ti.aot.written_resources(self, self.record)

class Derived(Model):
    pass
"""


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_free_kernel_receiver_preserves_shared_code_and_instance_binding(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, FREE_RECEIVER_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 50
    assert sys.modules["definition_fixture.model"].constructors == 0
    function = manifest["function_entries"]["definition_fixture.model.publish"]
    assert function["receiver"] == "solver" and function["parameters"] == ["gain", "solver"]
    assert len(function["owners"]) == 2
    assert len({case["symbol"] for cases in function["owners"].values() for case in cases}) == 1
    assert len([item for item in manifest["entries"].values() if item["definition"].endswith(".publish")]) == 1
    host = (tmp_path / "sdk/host/model.py").read_text()
    assert "def forward" in host and "def update" in host
    assert "solver.values[i]" not in host and "@ti.kernel" not in host
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(5), code.host.model.Derived(11)
        assert a.writes() == (a.values,)
        a.update(2)
        code.host.model.publish(gain=3, solver=b)
        prepared = a.graph().bind(dict(gain=4))
        b.update(7)
        prepared.run()
        prepared.close()
        np.testing.assert_array_equal(a.values.to_numpy(), 4 * (np.arange(5) + 5))
        np.testing.assert_array_equal(b.values.to_numpy(), 7 * (np.arange(11) + 11))
        with pytest.raises(ValueError, match="declared receiver"):
            code.host.model.publish(1, object())
        with load_package(runtime, tmp_path / "sdk") as other:
            with pytest.raises(ValueError, match="same package"):
                code.host.model.publish(1, other.host.Model(3))
    probe = tmp_path / "free_receiver_consume.py"
    probe.write_text(
        """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class RejectSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError(fullname)
sys.meta_path.insert(0, RejectSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    a, b = code.host.Model(17), code.host.model.Derived(65)
    assert a.writes() == (a.values,)
    prepared = a.graph().bind(dict(gain=3))
    b.update(5)
    prepared.run()
    prepared.close()
    np.testing.assert_array_equal(a.values.to_numpy(), 3 * (np.arange(17) + 17))
    np.testing.assert_array_equal(b.values.to_numpy(), 5 * (np.arange(65) + 65))
print('FREE_RECEIVER_SOURCE_FREE_OK')
""",
        encoding="utf8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FREE_RECEIVER_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("change", ["unknown", "rebound", "uncovered", "starred", "recursive_context"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_free_receiver_rejects_unproven_call_domains(tmp_path, monkeypatch, change):
    source = FREE_RECEIVER_SOURCE
    if change == "unknown":
        source = source.replace("forward(alias, gain)", "forward(object(), gain)")
    elif change == "rebound":
        source = source.replace("forward(alias, gain)", "alias = object()\n        forward(alias, gain)")
    elif change == "uncovered":
        source += "\ndef arbitrary(receiver):\n    publish(1.0, receiver)\n"
    elif change == "recursive_context":
        source = source.replace(
            "def forward(receiver, gain):", "def forward(receiver, gain):\n    forward((receiver,), gain)"
        )
    else:
        source = source.replace("publish(gain, solver=receiver)", "publish(*unknown_arguments())")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(
        ti.aot.PackageExportError, match="template input|receiver|argument sources|recursive Kernel input"
    ):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("change", ["version", "receiver", "owner", "abi", "missing_code"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_free_receiver_rejects_malformed_contract_before_loading(tmp_path, monkeypatch, change):
    package = _source_package(tmp_path, monkeypatch, FREE_RECEIVER_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    function = manifest["function_entries"]["definition_fixture.model.publish"]
    if change == "version":
        manifest["version"] = 48
    elif change == "receiver":
        function["receiver"] = "missing"
    elif change == "owner":
        function["owners"]["unknown.Owner"] = next(iter(function["owners"].values()))
    elif change == "abi":
        function["parameters"][0] = "other"
    else:
        function["owners"][next(iter(function["owners"]))][0]["symbol"] = "missing"
    (tmp_path / "sdk/package.json").write_text(json.dumps(manifest), encoding="utf8")
    with Runtime(manifest["target"]) as runtime:

        def must_not_load(*args, **kwargs):
            raise AssertionError("Invalid binding reached native code loading")

        monkeypatch.setattr(runtime, "load_module", must_not_load)
        with pytest.raises(ValueError, match="receiver|numerical|format|ABI"):
            load_package(runtime, tmp_path / "sdk")


FREE_GRAPH_RECEIVER_SOURCE = (
    FREE_RECEIVER_SOURCE.replace(
        "def forward(receiver, gain):\n    publish(gain, solver=receiver)",
        "def forward(builder, kernel, inputs, *arguments):\n    builder.dispatch(kernel, *arguments, template_args=inputs)",
    )
    .replace(
        "alias = self\n        forward(alias, gain)",
        "graph = self.graph()\n        graph.run({'gain': gain})",
    )
    .replace(
        "builder.dispatch(publish, gain, template_args={'solver': self})",
        "region = builder.create_sequential()\n        inputs = {'solver': self}\n"
        "        forward(region, publish, inputs, gain)\n        builder.append(region)",
    )
)


@pytest.mark.parametrize("mapping", ["direct", "filtered"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_graph_only_free_receiver_is_discovered_without_recording(tmp_path, monkeypatch, mapping):
    source = FREE_GRAPH_RECEIVER_SOURCE
    if mapping == "filtered":
        source = source.replace(
            "def forward(builder, kernel, inputs, *arguments):\n    builder.dispatch(kernel, *arguments, template_args=inputs)",
            "class Adapter:\n    pass\n\ndef forward(builder, kernel, inputs, *arguments):\n"
            "    compile_args = {name: item.source if isinstance(item, Adapter) else item for name, item in inputs.items()}\n"
            "    named = {name: item.source for name, item in inputs.items() if isinstance(item, Adapter)}\n"
            "    scratch = []\n    scratch.append(named)\n"
            "    builder.dispatch(kernel, *arguments, template_args=compile_args)",
        )
    package = _source_package(tmp_path, monkeypatch, source)

    def no_recording(*args, **kwargs):
        raise AssertionError("Input discovery executed a Graph")

    with monkeypatch.context() as guard:
        guard.setattr(ti.graph.GraphBuilder, "__init__", no_recording)
        manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    owners = manifest["function_entries"]["definition_fixture.model.publish"]["owners"]
    assert len({case["symbol"] for cases in owners.values() for case in cases}) == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7), code.host.model.Derived(13)
        a.update(3)
        prepared = a.graph().bind(dict(gain=5))
        b.update(11)
        prepared.run()
        prepared.close()
        np.testing.assert_array_equal(a.values.to_numpy(), 5 * (np.arange(7) + 7))
        np.testing.assert_array_equal(b.values.to_numpy(), 11 * (np.arange(13) + 13))
    probe = tmp_path / "graph_only_consume.py"
    probe.write_text(
        """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class NoSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError(fullname)
sys.meta_path.insert(0, NoSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    a, b = code.host.Model(17), code.host.model.Derived(129)
    prepared = a.graph().bind(dict(gain=2))
    b.update(3)
    prepared.run()
    prepared.close()
    np.testing.assert_array_equal(a.values.to_numpy(), 2 * (np.arange(17) + 17))
    np.testing.assert_array_equal(b.values.to_numpy(), 3 * (np.arange(129) + 129))
print('GRAPH_FREE_RECEIVER_OK')
""",
        encoding="utf8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GRAPH_FREE_RECEIVER_OK" in result.stdout


@pytest.mark.parametrize(
    "change", ["mutation", "escape", "fake_builder", "alias_mutation", "fake_dispatch", "known_mutator"]
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_graph_free_receiver_rejects_unproven_forwarding(tmp_path, monkeypatch, change):
    source = FREE_GRAPH_RECEIVER_SOURCE
    if change == "mutation":
        source = source.replace(
            "inputs = {'solver': self}", "inputs = {'solver': self}\n        inputs['solver'] = object()"
        )
    elif change == "escape":
        source = source.replace("inputs = {'solver': self}", "inputs = {'solver': self}\n        mutate(inputs)")
    elif change == "alias_mutation":
        source = source.replace(
            "inputs = {'solver': self}", "inputs = {'solver': self}\n        other = inputs\n        other.clear()"
        )
    elif change == "fake_dispatch":
        source = source.replace(
            "inputs = {'solver': self}", "inputs = {'solver': self}\n        object().dispatch(inputs)"
        )
    elif change == "known_mutator":
        source += "\ndef mutate(inputs):\n    inputs.clear()\n"
        source = source.replace("inputs = {'solver': self}", "inputs = {'solver': self}\n        mutate(inputs)")
    else:
        source = source.replace("builder = ti.graph.GraphBuilder()", "builder = fake_builder()")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="receiver|template input"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


FREE_STATIC_INPUT_SOURCE = (
    FREE_RECEIVER_SOURCE.replace(
        "def publish(gain: ti.f32, solver: ti.template()):",
        "def publish(gain: ti.f32, solver: ti.template(), enabled: ti.template(), increment: ti.template(), note: ti.template()):",
    )
    .replace(
        "solver.values[i] = gain * (i + solver.n)",
        "factor = gain\n        if ti.static(enabled):\n            factor += increment\n        solver.values[i] = factor * (i + solver.n)",
    )
    .replace(
        "publish(gain, solver=receiver)",
        "publish(gain, receiver, False, 2, 'first')\n    publish(gain, receiver, False, 17, 'second')",
    )
    .replace(
        "template_args={'solver': self}",
        "template_args={'solver': self, 'enabled': True, 'increment': 3, 'note': 'graph'}",
    )
    .replace(
        "    def writes(self):",
        "    def offset(self, gain):\n        publish(gain, self, True, 3, 'offset')\n"
        "    def other_offset(self, gain):\n        publish(gain, self, True, 7, 'offset')\n"
        "    def writes(self):",
    )
)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_free_static_input_tuples_share_only_actual_numerical_variants(tmp_path, monkeypatch):
    from zipfile import ZipFile

    package = _source_package(tmp_path, monkeypatch, FREE_STATIC_INPUT_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 50
    assert sys.modules["definition_fixture.model"].constructors == 0
    function = manifest["function_entries"]["definition_fixture.model.publish"]
    assert function["templates"] == ["solver", "enabled", "increment", "note"]
    cases = function["owners"]["definition_fixture.model.Model"]
    # Five actual call tuples, rather than enabled x increment x note products.
    assert len(cases) == 5
    assert len({case["symbol"] for case in cases}) == 3
    assert len([item for item in manifest["entries"].values() if item["definition"].endswith(".publish")]) == 3
    with ZipFile(tmp_path / "sdk/code.tcm") as archive:
        # The automatic Field accessor is a distinct required numerical entry.
        assert len([name for name in archive.namelist() if name.endswith(".bc")]) == len(manifest["entries"]) == 4
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(11), code.host.model.Derived(29)
        a.update(2)
        np.testing.assert_array_equal(a.values.to_numpy(), 2 * (np.arange(11) + 11))
        prepared = a.graph().bind(dict(gain=5))
        b.other_offset(2)
        prepared.run()
        prepared.close()
        np.testing.assert_array_equal(a.values.to_numpy(), 8 * (np.arange(11) + 11))
        np.testing.assert_array_equal(b.values.to_numpy(), 9 * (np.arange(29) + 29))
        for enabled, increment, note in ((True, 9, "graph"), (1, 3, "graph"), (True, 3, object())):
            with pytest.raises(ValueError, match="definition-proven domain|immutable"):
                code.host.model.publish(1, a, enabled, increment, note)
    probe = tmp_path / "static_input_consume.py"
    probe.write_text(
        """
import importlib.abc, sys
import numpy as np
from taichi_forge_runtime.aot import Runtime, load_package
class NoSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
            raise ImportError(fullname)
sys.meta_path.insert(0, NoSource())
with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
    a, b = code.host.Model(17), code.host.model.Derived(65)
    prepared = a.graph().bind(dict(gain=5))
    b.other_offset(4)
    prepared.run()
    prepared.close()
    np.testing.assert_array_equal(a.values.to_numpy(), 8 * (np.arange(17) + 17))
    np.testing.assert_array_equal(b.values.to_numpy(), 11 * (np.arange(65) + 65))
print('STATIC_INPUT_SOURCE_FREE_OK')
""",
        encoding="utf8",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STATIC_INPUT_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("change", ["dynamic", "duplicate_case", "template_signature", "invalid_value"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_static_input_domains_reject_unknown_and_malformed_sources(tmp_path, monkeypatch, change):
    source = FREE_STATIC_INPUT_SOURCE
    if change == "dynamic":
        source = source.replace(
            "publish(gain, receiver, False, 2, 'first')", "publish(gain, receiver, False, gain, 'first')"
        )
    package = _source_package(tmp_path, monkeypatch, source)
    if change == "dynamic":
        with pytest.raises(ti.aot.PackageExportError, match="unproven template input"):
            ti.aot.export_package(package, tmp_path / "sdk")
        assert not (tmp_path / "sdk").exists()
        return
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    function = manifest["function_entries"]["definition_fixture.model.publish"]
    cases = function["owners"]["definition_fixture.model.Model"]
    if change == "duplicate_case":
        cases.append(cases[0])
    elif change == "template_signature":
        function["templates"].remove("enabled")
    else:
        cases[0]["values"]["enabled"] = ["bool", 1]
    (tmp_path / "sdk/package.json").write_text(json.dumps(manifest), encoding="utf8")
    with Runtime(manifest["target"]) as runtime:

        def must_not_load(*args, **kwargs):
            raise AssertionError("Malformed static cases reached native loading")

        monkeypatch.setattr(runtime, "load_module", must_not_load)
        with pytest.raises(ValueError, match="template input|static value"):
            load_package(runtime, tmp_path / "sdk")
