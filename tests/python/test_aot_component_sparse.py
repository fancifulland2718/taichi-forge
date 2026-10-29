"""Normal fixed-SNode declarations retain original numerical Field lowering."""
import json
import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import ExecutionPlan
from tests import test_utils
from tests.python.test_aot_snode_instances import _consume_isolated


@ti.data_oriented
class _SparseState:
    def __init__(self, capacity=16, active=5, offset=10, bump=2):
        self.active, self.offset, self.bump = active, offset, bump
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        leaf = builder.pointer(ti.i, capacity // 4).dense(ti.i, 4)
        scope.grid = scope.field(ti.i32, shape=capacity, placement=leaf)
        scope.alias = self.grid
        # An unaccessed sibling keeps its native placement/allocation semantics.
        scope.sibling = scope.field(ti.f64, shape=capacity, placement=leaf)
        for name in ("active", "offset", "bump"):
            scope.value(name, ti.i32)
        self.tree = builder.finalize()

    @ti.func
    def sample(self, i):
        return self.offset + i

    @ti.kernel
    def evaluate(self, out: ti.types.ndarray(ti.i32, 1), drop: ti.i32):
        if drop:
            ti.deactivate(self.grid.parent().parent(), 0)
        for i in range(self.active):
            self.grid[i] = self.sample(i)
        for i in self.grid:
            self.grid[i] += self.bump
        for i in range(self.grid.shape[0]):
            out[i] = self.alias[i]


@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.sparse, offline_cache=False)
def test_fixed_snode_capture_preserves_source_binding_and_deploys_empty_instances(tmp_path):
    first = _SparseState()
    environment = Component.from_owner(first)
    out = ti.ndarray(ti.i32, 16)
    first.evaluate(out, 0)
    reference = out.to_numpy()
    first.grid.parent().parent().deactivate_all()
    code = environment.compile(first.evaluate, out, 0)
    code.run(environment, first, out, 0)
    np.testing.assert_array_equal(out.to_numpy(), reference)
    assert environment.compile(first.evaluate, out, 1) is code
    assert len(code.contract["fixed_groups"]) == 1
    assert set(code.contract["fixed_members"]) == {"grid", "alias"}
    assert code.contract["fixed_members"]["grid"] == code.contract["fixed_members"]["alias"]
    assert {name for name, _ in code.contract["captures"]} == {"active", "offset", "bump"}
    schema = next(iter(code.contract["fixed_groups"].values()))
    assert schema["allocation"]["root_size"] > 0
    assert schema["allocation"]["tree_id"] == schema["allocation"]["tree_generation"] == 0

    second = _SparseState(active=9, offset=100, bump=3)
    other_environment = Component.from_owner(second)
    other = other_environment.compile(second.evaluate, out, 0)
    assert other is not code  # Source bindings differ, deployable code identity does not.
    assert other.semantic_key == code.semantic_key
    with pytest.raises(ValueError, match="bindings"):
        code.run(other_environment, second, out, 0)
    other.run(other_environment, second, out, 0)
    np.testing.assert_array_equal(out.to_numpy()[:9], np.arange(9) + 103)
    np.testing.assert_array_equal(first.grid.to_numpy()[:5], np.arange(5) + 12)
    larger = _SparseState(32)
    larger_code = Component.from_owner(larger).compile(larger.evaluate, ti.ndarray(ti.i32, 32), 0)
    assert larger_code.semantic_key != code.semantic_key  # D3 remains explicitly fixed.

    artifact = ti.aot.Module()
    code.export(artifact, "evaluate")
    arguments = ["out", "drop"] + ["__capture_" + name for name, _ in code.contract["captures"]]
    artifact.add_execution_plan("evaluate", ExecutionPlan([
        dict(kind="kernel", entry="evaluate", arguments=arguments)]))
    folder = tmp_path / "artifact"
    artifact.save(folder)
    assert not (folder / "data").exists()
    projected = json.loads((folder / "component_entries/evaluate.json").read_text())
    assert projected["initialization"]["fixed_snode"] == "empty_activation_and_zero_values"
    _consume_isolated(folder, tmp_path, "capture")


@test_utils.test(arch=ti.cpu, require=ti.extension.sparse, offline_cache=False)
def test_fixed_snode_declaration_rejects_dynamic_shape_and_foreign_placement():
    class Owner:
        count = 16

    owner = Owner()
    builder, foreign = ti.FieldsBuilder(), ti.FieldsBuilder()
    scope = ti.experimental.ResourceScope(owner, builder=builder)
    leaf = builder.pointer(ti.i, 4).dense(ti.i, 4)
    other = foreign.pointer(ti.i, 4).dense(ti.i, 4)
    with pytest.raises(ValueError, match="dynamic extents"):
        scope.field(ti.i32, shape=scope.dim("count"), placement=leaf)
    with pytest.raises(ValueError, match="another FieldsBuilder"):
        scope.field(ti.i32, shape=16, placement=other)
    with pytest.raises(ValueError, match="shape differs"):
        scope.field(ti.i32, shape=8, placement=leaf)
    scope.grid = scope.field(ti.i32, shape=16, placement=leaf)
    other.place(ti.field(ti.i32))
    builder.finalize()
    foreign.finalize()
    assert Component.from_owner(owner).declarations["grid"].schema[0] == "fixed_snode_field"
