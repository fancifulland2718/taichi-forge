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


@ti.data_oriented
class _SparseGraphState(_SparseState):
    def __init__(self):
        super().__init__()
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.total = scope.field(ti.i32, shape=())
        self.total_tree = builder.finalize()

    @ti.kernel
    def summarize(self, out: ti.types.ndarray(ti.i32, 1)):
        self.total[None] = 0
        for i in range(out.shape[0]):
            self.total[None] += out[i]

    def record(self, builder):
        out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
        drop = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "drop", ti.i32)
        builder.dispatch(self.evaluate, out, drop, template_args={"self": self})
        builder.dispatch(self.summarize, out, template_args={"self": self})


@ti.data_oriented
class _SparseStructState:
    def __init__(self, implicit_root=False, bitmasked=True):
        builder = None if implicit_root else ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        pointer = (ti.root if implicit_root else builder).pointer(ti.ijk, 2)
        leaf = pointer.bitmasked(ti.ijk, 4) if bitmasked else pointer.dense(ti.ijk, 4)
        scope.grid = scope.struct_field(
            dict(mass=ti.f32, momentum=ti.types.vector(3, ti.f32), velocity=ti.types.vector(3, ti.f32)),
            shape=(8, 8, 8), placement=leaf)
        self.tree = None if implicit_root else builder.finalize()
        self.gravity = ti.Vector([1.0, -2.0, 3.0])
        scope.value("gravity", ti.types.vector(3, ti.f32))

    @ti.kernel
    def seed(self):
        for i in range(6):
            self.grid[i, 2, 3].mass = i + 1
            self.grid[i, 2, 3].momentum = ti.Vector([i, 2 * i, 3 * i])

    @ti.func
    def update_node(self, I):
        if self.grid[I].mass > 0:
            self.grid[I].velocity = self.grid[I].momentum / self.grid[I].mass + self.gravity

    @ti.kernel
    def update(self):
        for I in ti.grouped(self.grid):
            self.update_node(I)

    @ti.kernel
    def read(self, out: ti.types.ndarray(ti.f32, 2)):
        for i in range(8):
            for j in ti.static(range(3)):
                out[i, j] = self.grid[i, 2, 3].velocity[j]

    def record(self, builder):
        out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=2)
        builder.dispatch(self.seed, template_args={"self": self})
        builder.dispatch(self.update, template_args={"self": self})
        builder.dispatch(self.read, out, template_args={"self": self})

    def record_update(self, builder, *, enabled=True):
        if enabled:
            builder.dispatch(self.update, template_args={"self": self})

    def record_with_clear(self, builder):
        self.record(builder)
        clear = builder.create_sequential()
        self.grid.parent(2).deactivate_all(builder=clear)
        builder.append(clear)
        empty = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "empty", ti.f32, ndim=2)
        builder.dispatch(self.read, empty, template_args={"self": self})


@pytest.mark.parametrize("bitmasked", [True, False])
@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.sparse, offline_cache=False)
def test_normal_snode_deactivation_records_shared_kernels_and_instance_bindings(tmp_path, bitmasked):
    from taichi_forge._kernels import snode_deactivate

    state = _SparseStructState(bitmasked=bitmasked)
    state.seed()
    state.grid.parent(2).deactivate_all()
    assert state.grid.mass[2, 2, 3] == 0
    state.seed()
    builder = ti.graph.GraphBuilder()
    state.record_with_clear(builder)
    assert state.grid.mass[2, 2, 3] == 3  # Recording does not execute deactivation.
    ordinary = builder.compile()
    out, empty = ti.ndarray(ti.f32, (8, 3)), ti.ndarray(ti.f32, (8, 3))
    ordinary.run(dict(out=out, empty=empty))
    assert np.count_nonzero(out.to_numpy()) > 0
    np.testing.assert_array_equal(empty.to_numpy(), 0)
    environment = Component.from_owner(state)
    root = environment.compile_graph(state.record_with_clear)
    # Foreign SNodes cannot be treated as arbitrary static template objects.
    foreign = _SparseStructState()
    with pytest.raises(ti.TaichiCompilationError, match="SNode template.*declared fixed"):
        environment.compile(snode_deactivate, owner=state, template_args=dict(b=foreign.grid.parent(2)))
    module = ti.aot.Module()
    root.export(module, "step")
    folder = tmp_path / "artifact"
    module.save(folder)
    _consume_isolated(folder, tmp_path, "clear")


@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.sparse, offline_cache=False)
def test_fixed_component_coordinate_io_consumes_new_state_without_source(tmp_path):
    from taichi_forge.aot._component import RootOption
    from taichi_forge.aot._component_variants import Variant, build_variants

    state = _SparseStructState()
    state.seed()
    component = Component.from_owner(state, enabled=RootOption(choices=(True, False)))
    built = build_variants([Variant(component, state.record_update, enabled=True)])
    root = built.select(dict(enabled=True))
    report = built.report
    assert len(report["kernel_variants"]) == 7  # Update plus two transfers per actual member.
    assert all(value is None for value in report["budget"].values())
    module = ti.aot.Module()
    root.export(module, "step")
    family = built.export(module, "updates")
    assert len(family["exported_kernel_entries"]) == 7
    folder = tmp_path / "artifact"
    module.save(folder)
    assert not (folder / "data").exists()
    _consume_isolated(folder, tmp_path, "coordinates")


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_field_coordinate_transfers_preserve_offsets_and_matrix_elements():
    shifted = ti.field(ti.i32, shape=4, offset=-2)
    matrix = ti.Matrix.field(2, 3, ti.f32, shape=())
    shifted.write_at([[-2], [1]], [5, 9])
    np.testing.assert_array_equal(shifted.read_at([[-2], [-1], [1]]), [5, 0, 9])
    expected = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    matrix.write_at([[]], expected)
    np.testing.assert_array_equal(matrix.read_at([[]]), expected)
    assert matrix.read_at([]).shape == (0, 2, 3)
    with pytest.raises(ValueError, match="outside"):
        shifted.write_at([[2]], [3])


@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.sparse, offline_cache=False)
def test_fixed_snode_struct_elements_keep_normal_field_lowering(tmp_path):
    state = _SparseStructState(implicit_root=ti.cfg.arch == ti.cpu)
    out = ti.ndarray(ti.f32, (8, 3))
    state.seed()
    state.update()
    state.read(out)
    reference = out.to_numpy()
    expected = np.zeros((8, 3), dtype=np.float32)
    expected[:6] = np.arange(6)[:, None] / np.arange(1, 7)[:, None] * [1, 2, 3] + [1, -2, 3]
    np.testing.assert_allclose(reference, expected, rtol=1e-6, atol=1e-6)
    component = Component.from_owner(state)
    root = component.compile_graph(state.record)
    assert len(root.contract["fixed_groups"]) == 1
    assert len(root.contract["resources"]) == 3
    # Reuse the original struct indexing, traversal and helper with captured
    # runtime gravity; do not rewrite the numerical expression as member arrays.
    code = component.compile(state.update)
    state.gravity = ti.Vector([-3.0, 2.0, -1.0])
    code.run(component, state)
    state.read(out)
    expected[:6] += [-4, 4, -4]
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6, atol=1e-6)
    module = ti.aot.Module()
    root.export(module, "step")
    folder = tmp_path / "artifact"
    module.save(folder)
    _consume_isolated(folder, tmp_path, "struct")


@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.sparse, offline_cache=False)
def test_coordinate_field_transfers_share_fixed_component_templates():
    from taichi_forge._kernels import field_write_coordinates, field_read_coordinates

    state = _SparseStructState()
    coordinates = np.array([[1, 2, 3], [6, 2, 3]], dtype=np.int32)
    state.grid.mass.write_at(coordinates, [2, 4])
    state.grid.momentum.write_at(coordinates, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(state.grid.mass.read_at([[1, 2, 3], [7, 2, 3]]), [2, 0])
    np.testing.assert_array_equal(state.grid.momentum.read_at(coordinates), [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="distinct"):
        state.grid.mass.write_at([[1, 2, 3], [1, 2, 3]], [1, 2])
    with pytest.raises(ValueError, match="outside"):
        state.grid.mass.write_at([[8, 2, 3]], [1])
    environment = Component.from_owner(state)
    indices = ti.ndarray(ti.i32, (2, 3))
    values = ti.ndarray(ti.types.vector(3, ti.f32), 2)
    indices.from_numpy(coordinates)
    values.from_numpy(np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32))
    writer = environment.compile(field_write_coordinates, indices, values, owner=state,
                                 template_args={"field": state.grid.momentum})
    writer.run(environment, state, indices, values)
    reader = environment.compile(field_read_coordinates, indices, values, owner=state,
                                 template_args={"field": state.grid.momentum})
    values.fill(0)
    reader.run(environment, state, indices, values)
    np.testing.assert_array_equal(values.to_numpy(), [[7, 8, 9], [10, 11, 12]])


@test_utils.test(arch=[ti.cpu, ti.cuda], require=ti.extension.sparse, offline_cache=False)
def test_fixed_snode_normal_component_graph_exports_and_instantiates(tmp_path):
    spare = _SparseState(32)
    state = _SparseGraphState()
    builder = ti.graph.GraphBuilder()
    state.record(builder)
    ordinary = builder.compile()
    out = ti.ndarray(ti.i32, 16)
    ordinary.run(dict(out=out, drop=0))
    reference = out.to_numpy()
    assert state.total[None] == int(reference.sum())
    undeclared = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="snapshot_state=True"):
        undeclared.add_execution_plan("missing_declaration", ordinary)
    assert not undeclared._data_payloads and not undeclared._exported_kernels
    environment = Component.from_owner(state)
    root = environment.compile_graph(state.record)
    assert root.contract["version"] == 7
    assert len(root.contract["fixed_groups"]) == len(root.contract["groups"]) == 1
    assert len(root.contract["entries"]) == 2
    assert any("group" in effect for entry in root.contract["compilation"]["entries"] for effect in entry["effects"]), root.contract["compilation"]
    module = ti.aot.Module()
    plan = root.export(module, "step")
    assert len(plan.kernel_entries) == 2 and not plan.data_entries
    # A later export changes the final local tree numbering. Bindings must
    # follow native dump relocation, not be guessed during root recording.
    Component.from_owner(spare).compile(spare.evaluate, ti.ndarray(ti.i32, 32), 0).export(module, "spare")
    folder = tmp_path / "artifact"
    module.save(folder)
    metadata = json.loads((folder / "components/step.json").read_text())
    assert list(metadata["fixed_bindings"].values()) == [1]
    assert not (folder / "data").exists()
    _consume_isolated(folder, tmp_path, "component")


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
