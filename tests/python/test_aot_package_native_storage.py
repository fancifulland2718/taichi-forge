"""Compile symbolic sparse placement and exercise the original public Field semantics."""

from contextlib import closing
import importlib
import json
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime, load_package

from tests import test_utils
from tests.python.test_aot_package import _source_package


_GRID = """
import taichi_forge as ti

@ti.data_oriented
class Model:
    def __init__(self, rows, cols, width, diagnostics):
        self.rows, self.cols, self.width = rows, cols, width
        self.diagnostics = diagnostics
        scope = ti.experimental.ResourceScope(self)
        scope.dim("rows")
        scope.dim("cols")
        scope.dim("width")
        scope.static("diagnostics", choices=(False, True))
        self.pointer = ti.root.pointer(ti.ij, (self.rows, self.cols))
        self.cells = self.pointer.bitmasked(ti.ij, self.width)
        shape = (self.rows * self.width, self.cols * self.width)
        scope.grid = scope.struct_field(dict(mass=ti.f32, vel=ti.types.vector(2, ti.f32)),
                                        shape=shape, placement=self.cells)
        if self.diagnostics:
            scope.extra = scope.field(ti.f32, shape=shape, placement=self.cells)
        scope.last_value = scope.field(ti.f32, shape=(), initialization="zero")

    @ti.kernel
    def write(self, i: ti.i32, j: ti.i32, value: ti.f32):
        self.grid[i, j].mass += value
        self.grid[i, j].vel = ti.Vector([value, -value])
        self.last_value[None] = value

    @ti.kernel
    def report(self, i: ti.i32, j: ti.i32, value: ti.f32):
        if ti.static(self.diagnostics):
            self.extra[i, j] = value

    @ti.kernel
    def read_extra(self, i: ti.i32, j: ti.i32) -> ti.f32:
        return self.extra[i, j]

    @ti.kernel
    def sum_active(self) -> ti.f32:
        total = 0.0
        for i, j in self.grid.mass:
            total += self.grid[i, j].mass
            if ti.static(self.diagnostics):
                total += self.extra[i, j]
        return total

    def clear(self):
        self.pointer.deactivate_all()

    def append_clear(self, builder):
        self.pointer.deactivate_all(builder=builder)
"""


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_pointer_bitmasked_groups_preserve_optional_placement_and_field_io(tmp_path, monkeypatch):
    definitions = _source_package(tmp_path, monkeypatch, _GRID)
    sdk = tmp_path / "native"
    manifest = ti.aot.export_package(definitions, sdk, arch=ti.cpu, artifact_format="unified-bc-library")
    model = "definition_fixture.model.Model"
    group = manifest["base"]["native_storage_groups"][model + ":pointer"]
    assert [node["op"] for node in group["nodes"]] == ["pointer", "bitmasked"]
    assert group["dimensions"]["n0_axis0"]["expression"] == ["dim", "rows"]
    assert group["dimensions"]["n0_axis1"]["expression"] == ["dim", "cols"]
    assert len(group["slots"]) == 2
    for slot in group["slots"]:
        assert ("extra" in slot["members"]) is slot["conditions"]["diagnostics"]
        assert [len(slot["members"][name]["places"]) for name in ("grid.mass", "grid.vel")] == [1, 2]
    reference = []
    shapes = ((3, 5, 2, False), (5, 7, 4, True))
    for rows, cols, width, enabled in shapes:
        instance = definitions.Model(rows, cols, width, enabled)
        i, j = rows * width - 1, cols * width - 1
        instance.write(i=i, j=j, value=8.)
        instance.report(i, j, 2.)
        reference.append((instance.grid.mass.to_numpy(), instance.grid.vel.to_numpy(), instance.sum_active()))
    ti.reset()

    with Runtime("x64") as runtime, load_package(runtime, sdk) as code:
        models = [code.host.Model(*shape) for shape in shapes]
        for obj, shape, expected in zip(models, shapes, reference):
            rows, cols, width, enabled = shape
            i, j = rows * width - 1, cols * width - 1
            obj.write(i=i, j=j, value=8.)
            obj.report(i, j, 2.)
            assert obj.last_value[None] == 8.
            if enabled:
                assert obj.read_extra(i, j) == 2.
            else:
                with pytest.raises(ValueError):
                    obj.read_extra(i, j)
            np.testing.assert_array_equal(obj.grid.mass.to_numpy(), expected[0])
            np.testing.assert_array_equal(obj.grid.vel.to_numpy(), expected[1])
            assert obj.sum_active() == expected[2] == (10 if enabled else 8)
            source = np.full((rows * width, cols * width), 27., np.float32)
            obj.grid.mass.from_numpy(source)
            expected_mass = np.zeros_like(source)
            expected_mass[i, j] = 27.
            np.testing.assert_array_equal(obj.grid.mass.to_numpy(), expected_mass)
        builder = code.graph_builder()
        models[0].append_clear(builder)
        with closing(builder.compile()) as graph:
            graph.run({})
        assert models[0].sum_active() == 0.
        assert models[1].sum_active() == 29.
        models[1].clear()
        assert models[1].sum_active() == 0.


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_incremental_sparse_storage_reuses_only_matching_layouts(tmp_path, monkeypatch):
    path = tmp_path / "sparse_edits.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    changed = _GRID.replace("scope.extra = scope.field(ti.f32", "scope.extra = scope.field(ti.f64")
    # Reload the updated module even on filesystems with coarse timestamps.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    for label, source in (("seed", _GRID), ("repeat", _GRID), ("edited", changed), ("clean", changed)):
        path.write_text(source, encoding="utf8")
        definitions = importlib.import_module(path.stem)
        cache = tmp_path / ("fresh" if label == "clean" else "cache")
        try:
            ti.aot.export_package(definitions, tmp_path / label, arch=ti.cpu,
                                  artifact_format="unified-bc-library", build_cache=cache)
        finally:
            sys.modules.pop(path.stem)
        counts = json.loads((cache / "unified/latest.json").read_bytes())["produce"]
        if label == "repeat":
            assert counts["unified_root_emissions"] == 0
            assert counts["unified_capture_restores"] > 0
        elif label == "edited":
            assert counts["unified_root_emissions"] > 0
            assert counts["unified_capture_restores"] > 0
    assert (tmp_path / "seed/code.tcm").read_bytes() != (tmp_path / "edited/code.tcm").read_bytes()
    for left, right in (("seed", "repeat"), ("edited", "clean")):
        for name in ("code.tcm", "package.json"):
            assert (tmp_path / left / name).read_bytes() == (tmp_path / right / name).read_bytes()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_native_storage_requires_proven_field_extents(tmp_path, monkeypatch):
    mismatched = _GRID.replace("shape = (self.rows * self.width, self.cols * self.width)",
                               "shape = (self.rows * self.width, self.cols * self.width + 1)")
    definitions = _source_package(tmp_path, monkeypatch, mismatched)
    with pytest.raises(ti.aot.PackageExportError, match="Sparse Field shape must retain"):
        ti.aot.export_package(definitions, tmp_path / "invalid", arch=ti.cpu, artifact_format="unified-bc-library")
    assert not (tmp_path / "invalid").exists()
