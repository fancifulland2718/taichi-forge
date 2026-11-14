"""Work-count and binding contracts for package-local Dense slot reuse."""

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils
from tests.python.test_aot_package import _source_package, TYPED_FIELD_INPUT_SOURCE
from taichi_forge.aot._component import _Capture
from taichi_forge_runtime.aot import Runtime, load_package


def _copy_source(first="scope.field(ti.f32", second="scope.field(ti.f32"):
    return TYPED_FIELD_INPUT_SOURCE.replace(
        "scope.scalar = scope.field(ti.f32", "scope.scalar = " + first
    ).replace(
        "scope.record = scope.struct_field({'x': ti.f32, 'v': ti.types.vector(self.width, ti.f32), 'id': ti.i32}, shape=scope.dim('n'), initialization='zero')",
        "scope.record = " + second + ", shape=scope.dim('width'), initialization='zero')",
    )


def _capture_counts(monkeypatch):
    counts = {}
    original = _Capture.__init__

    def observe(self, component, owner, kernel, *args, **kwargs):
        name = kernel.func.__qualname__
        counts[name] = counts.get(name, 0) + 1
        original(self, component, owner, kernel, *args, **kwargs)

    monkeypatch.setattr(_Capture, "__init__", observe)
    return counts


@pytest.mark.parametrize("field", ["scope.field(ti.f32", "scope.vector_field(2, ti.f32"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_dense_slots_reuse_capture_and_retain_source_admission(tmp_path, monkeypatch, field):
    counts = _capture_counts(monkeypatch)
    package = _source_package(tmp_path, monkeypatch, _copy_source(field, field))
    original = package.Model(3, 7)
    original.first.save()
    kernel = original.first.save._primal
    jit_compiled = dict(kernel.compiled_kernels)
    jit_candidates = tuple(kernel._component_candidates)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert counts["Checkpoint.save"] == counts["Checkpoint.restore"] == 1
    assert len(manifest["input_families"]["definition_fixture.model.Checkpoint"]) == 2
    assert dict(kernel.compiled_kernels) == jit_compiled
    assert tuple(kernel._component_candidates) == jit_candidates
    bindings = [binding for binding in manifest["entry_bindings"].values() if "captures" in binding]
    assert bindings
    assert any(any(item.get("layout_member") == "record" for item in binding["captures"])
               for binding in bindings)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        for n, width in ((3, 7), (11, 2)):
            model = code.host.Model(n, width)
            for source, backup in ((model.storage.scalar, model.first), (model.storage.record, model.second)):
                values = np.arange(np.prod(source.to_numpy().shape), dtype=np.float32).reshape(source.to_numpy().shape)
                source.from_numpy(values)
                backup.save()
                source.from_numpy(np.zeros_like(values))
                backup.restore()
                np.testing.assert_array_equal(source.to_numpy(), values)
            source = model.second.source
            model.second.source = model.first.source
            with pytest.raises(ValueError, match="Borrowed Field"):
                model.second.save()
            model.second.source = source


@pytest.mark.parametrize("second", ["scope.field(ti.i32", "scope.vector_field(2, ti.f32"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_dense_slots_separate_element_abi(tmp_path, monkeypatch, second):
    counts = _capture_counts(monkeypatch)
    package = _source_package(tmp_path, monkeypatch, _copy_source(second=second))
    ti.aot.export_package(package, tmp_path / "sdk")
    assert counts["Checkpoint.save"] == counts["Checkpoint.restore"] == 2


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_dense_slots_separate_compact_and_affine_abi(tmp_path, monkeypatch):
    counts = _capture_counts(monkeypatch)
    field = "scope.vector_field(2, ti.f32"
    source = _copy_source(field, "scope.struct_field({'v': ti.types.vector(2, ti.f32), 'id': ti.i32}").replace(
        "Checkpoint(self.storage.record)", "Checkpoint(self.storage.record.v)"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    ti.aot.export_package(package, tmp_path / "sdk")
    assert counts["Checkpoint.save"] == counts["Checkpoint.restore"] == 2


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_dense_slots_retain_runtime_source_aliases(tmp_path, monkeypatch):
    counts = _capture_counts(monkeypatch)
    source = _copy_source().replace(
        "def __init__(self, values):", "def __init__(self, values, peer):"
    ).replace(
        "scope.borrow('source', values, source_scope=True)",
        "scope.borrow('source', values, source_scope=True)\n        scope.borrow('peer', peer, source_scope=True)",
    ).replace(
        "self.backup[i] = self.source[i]", "self.backup[i] = self.source[i] + self.peer[0]"
    ).replace(
        "Checkpoint(self.storage.scalar)", "Checkpoint(self.storage.scalar, self.storage.scalar)"
    ).replace(
        "Checkpoint(self.storage.record)", "Checkpoint(self.storage.record, self.storage.scalar)"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    # Typed source scopes supply independent descriptors even when a consumer
    # passes the same Field twice. Keep those slots and both source admissions.
    assert counts["Checkpoint.save"] == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(3, 7)
        model.storage.scalar.from_numpy(np.arange(3, dtype=np.float32) + 4)
        model.storage.record.from_numpy(np.arange(7, dtype=np.float32) + 9)
        model.first.save()
        model.second.save()
        np.testing.assert_array_equal(model.first.backup.to_numpy(), np.arange(3) + 8)
        np.testing.assert_array_equal(model.second.backup.to_numpy(), np.arange(7) + 13)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_dense_slots_reject_changed_native_alias_topology(monkeypatch):
    from taichi_forge.aot._component import Component, DenseField, Dimension
    from taichi_forge.aot._package_compilation import PackageCompilation

    @ti.data_oriented
    class Owner:
        def __init__(self, alias):
            self.n = 3
            self.x, self.y, self.out = (ti.field(ti.f32) for _ in range(3))
            builder = ti.FieldsBuilder()
            builder.dense(ti.i, self.n).place(self.x, self.y, self.out)
            self.tree = builder.finalize()
            self.source, self.peer, self.backup = self.x, self.x if alias else self.y, self.out

        @ti.kernel
        def save(self):
            for i in self.source:
                self.backup[i] = self.source[i] + self.peer[i]

    component = Component(n=Dimension("extent"), **{
        name: DenseField(ti.f32, ("n",), "state") for name in ("x", "y", "out", "source", "peer", "backup")
    })
    first, second = Owner(True), Owner(False)
    try:
        with PackageCompilation() as compilation:
            aliased = component.compile(compilation.entry(first.save))
            separate = component.compile(compilation.entry(second.save))
            assert aliased.key != separate.key
            assert len(aliased.dynamic) == 2
            assert len(separate.dynamic) == 3
    finally:
        first.tree.destroy()
        second.tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_dense_slots_preserve_static_algorithm_choices(tmp_path, monkeypatch):
    counts = _capture_counts(monkeypatch)
    source = _copy_source().replace(
        "def __init__(self, values):", "def __init__(self, values, mode: bool):"
    ).replace(
        "scope.borrow('source', values, source_scope=True)",
        "self.mode = mode\n        scope.static('mode', choices=(False, True))\n        scope.borrow('source', values, source_scope=True)",
    ).replace(
        "self.backup[i] = self.source[i]",
        "if ti.static(self.mode):\n                self.backup[i] = self.source[i] + 1\n            else:\n                self.backup[i] = self.source[i]",
    ).replace(
        "Checkpoint(self.storage.scalar)", "Checkpoint(self.storage.scalar, False)"
    ).replace(
        "Checkpoint(self.storage.record)", "Checkpoint(self.storage.record, True)"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert counts["Checkpoint.save"] == 2
    assert counts["Checkpoint.restore"] == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(3, 7)
        model.first.save()
        model.second.save()
        np.testing.assert_array_equal(model.first.backup.to_numpy(), 0)
        np.testing.assert_array_equal(model.second.backup.to_numpy(), 1)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_dense_capture_binding_rejects_abi_changes_and_old_version(tmp_path, monkeypatch):
    import copy
    import json

    package = _source_package(tmp_path, monkeypatch, _copy_source())
    output = tmp_path / "sdk"
    manifest = ti.aot.export_package(package, output)
    symbol = next(name for name, binding in manifest["entry_bindings"].items() if "captures" in binding)
    assert manifest["version"] >= 84
    with Runtime(manifest["target"]) as runtime:
        for mutation in ("version", "name", "kind", "slots", "path"):
            invalid = copy.deepcopy(manifest)
            captures = invalid["entry_bindings"][symbol]["captures"]
            if mutation == "version":
                invalid["version"] = 83
            elif mutation == "slots":
                captures.pop()
            elif mutation == "path":
                captures[0]["path"] = ["backup"]
            else:
                captures[0][mutation] = "invalid"
            (output / "package.json").write_text(json.dumps(invalid), encoding="utf8")
            with pytest.raises(ValueError, match="Logical capture binding"):
                load_package(runtime, output)


@pytest.mark.parametrize("explicit_dimension", [False, True])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_dense_equal_layouts_still_project_distinct_source_members(tmp_path, monkeypatch, explicit_dimension):
    counts = _capture_counts(monkeypatch)
    source = _copy_source().replace(
        "        scope.record = scope.field(ti.f32, shape=scope.dim('width'), initialization='zero')\n", ""
    )
    storage = source.split("@ti.data_oriented\nclass Storage:", 1)[1].split("@ti.data_oriented\nclass Checkpoint:", 1)[0]
    other = "@ti.data_oriented\nclass Other:" + storage.replace("scope.scalar", "scope.record")
    source = source.replace("@ti.data_oriented\nclass Checkpoint:", other + "@ti.data_oriented\nclass Checkpoint:")
    source = source.replace("scope.storage = Storage(n, width)", "scope.storage = Storage(n, width)\n        scope.other = Other(n, width)")
    source = source.replace("Checkpoint(self.storage.record)", "Checkpoint(self.other.record)")
    if explicit_dimension:
        source = source.replace("def __init__(self, values):", "def __init__(self, values, count):")
        source = source.replace("scope.borrow('source', values, source_scope=True)",
                                "self.count = count\n        scope.dim('count')\n        scope.borrow('source', values, source_scope=True)")
        source = source.replace("self.backup[i] = self.source[i]", "self.backup[i] = self.source[i] + self.count")
        source = source.replace("Checkpoint(self.storage.scalar)", "Checkpoint(self.storage.scalar, n)")
        source = source.replace("Checkpoint(self.other.record)", "Checkpoint(self.other.record, n)")
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert counts["Checkpoint.save"] == 1
    assert counts["Checkpoint.restore"] == 1
    bindings = [binding for binding in manifest["entry_bindings"].values() if "captures" in binding]
    assert any(any(item.get("layout_member") == "record" for item in binding["captures"]) for binding in bindings)
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        model = code.host.Model(7, 11)
        model.other.record.from_numpy(np.arange(7, dtype=np.float32))
        model.second.save()
        np.testing.assert_array_equal(model.second.backup.to_numpy(), np.arange(7) + (7 if explicit_dimension else 0))
