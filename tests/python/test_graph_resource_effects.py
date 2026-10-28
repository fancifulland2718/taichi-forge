"""Resource closure survives accesses that cannot be pointwise-fused."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_ir_metadata_shares_readonly_facts_and_preserves_snapshot_isolation():
    from collections.abc import Mapping
    from types import SimpleNamespace
    from taichi_forge.graph._graph import _compiled_dispatch_ir_nodes

    total = ti.field(ti.i32, shape=())

    @ti.kernel
    def update(x: ti.types.ndarray(ti.i32, ndim=1), delta: ti.i32):
        for i in x:
            x[i] += delta
            ti.atomic_add(total[None], delta)

    a = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "a", ti.i32, ndim=1)
    b = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "b", ti.i32, ndim=1)
    delta = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "delta", ti.i32)
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    for _ in range(8):
        builder.dispatch(update, a, delta)
        builder.dispatch(update, b, delta)
    graph = builder.compile()
    compiled = graph._spec.nodes[0].compiled_graph
    views = compiled._dispatch_ir_metadata
    snapshots = compiled._dispatch_metadata
    assert len(views) == len(snapshots) == 16
    assert all(not row["opaque"] and row["synchronization"] for row in views)
    assert all("atomic" in row["side_effects"] and not row["elementwise"] for row in views)
    assert any(effect["access"] == "atomic" for effect in views[0]["effects"])
    assert len({id(row["effects"]) for row in views}) == 1
    assert len({id(row["effects"]) for row in snapshots}) == 16

    def normalize(value):
        if isinstance(value, Mapping):
            return {key: normalize(item) for key, item in value.items() if key != "footprint"}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        return value

    assert normalize(views) == normalize(snapshots)
    fallback = graph._spec.nodes[0].ir_node.children
    assert len(fallback) == len(views)
    assert fallback[0].effects is fallback[2].effects
    assert fallback[0].bindings is fallback[2].bindings
    assert fallback[0].effects != fallback[1].effects
    assert fallback[0].bindings != fallback[1].bindings
    plan = graph._spec._aot_graph_builder
    assert plan._items[0][-1] is plan._items[2][-1]
    assert plan._items[0][-1] != plan._items[1][-1]
    assert not plan._runtime_arg_name_sets, "Frozen plans must not retain the authoring interner"
    assert _compiled_dispatch_ir_nodes(compiled, fallback) == _compiled_dispatch_ir_nodes(
        SimpleNamespace(_dispatch_metadata=snapshots), fallback
    )
    with pytest.raises(TypeError):
        views[0]["opaque"] = True
    with pytest.raises(TypeError):
        views[0]["effects"][0]["access"] = "read"
    with pytest.raises(TypeError):
        views[0]["symbolic_args"][0]["name"] = "changed"
    snapshots[0]["effects"][0]["access"] = "changed"
    snapshots[0]["effects"][0]["footprint"]["pattern"] = "changed"
    assert snapshots[1]["effects"][0]["access"] != "changed"
    assert compiled._dispatch_metadata[0]["effects"][0]["footprint"]["pattern"] != "changed"

    x, y = ti.ndarray(ti.i32, 4), ti.ndarray(ti.i32, 4)
    x.fill(0)
    y.fill(10)
    graph.run({"a": x, "b": y, "delta": 2})
    graph.run({"a": y, "b": x, "delta": 3})
    assert x.to_numpy().tolist() == [40] * 4
    assert y.to_numpy().tolist() == [50] * 4
    assert total[None] == 320
    graph.close()


def test_opaque_ir_metadata_does_not_convert_discarded_effects(monkeypatch):
    from types import SimpleNamespace
    from taichi_forge.graph import _graph
    from taichi_forge.graph._ir import DispatchNode, GraphAccess, ResourceEffect

    fallback = DispatchNode(name="opaque", effects=(ResourceEffect("x", GraphAccess.READ_WRITE),))

    def forbidden(*_args):
        raise AssertionError("opaque metadata must retain conservative effects directly")

    monkeypatch.setattr(_graph, "_metadata_resource_effect", forbidden)
    compiled = SimpleNamespace(_dispatch_ir_metadata=[{
        "available": True, "opaque": True, "effects": [{}],
    }])
    result, = _graph._compiled_dispatch_ir_nodes(compiled, [fallback])
    assert result.opaque and result.effects == fallback.effects


def test_ir_descriptor_sharing_preserves_custom_subresource_identity():
    from types import SimpleNamespace
    from taichi_forge.graph._graph import _compiled_dispatch_ir_nodes
    from taichi_forge.graph._ir import DispatchNode, GraphAccess, ResourceEffect

    class MutableRegion:
        def __init__(self):
            self.offset = 0

        def __hash__(self):
            return 1

        def __eq__(self, other):
            return isinstance(other, MutableRegion) and self.offset == other.offset

    first, second = MutableRegion(), MutableRegion()
    nodes = tuple(DispatchNode(name="extension", effects=(
        ResourceEffect("x", GraphAccess.READ, subresource=region),
    )) for region in (first, second))
    compiled = SimpleNamespace(_dispatch_ir_metadata=[{"opaque": True}] * 2)
    left, right = _compiled_dispatch_ir_nodes(compiled, nodes)
    assert left.effects is nodes[0].effects and right.effects is nodes[1].effects
    first.offset = 4
    assert left.effects[0].subresource.offset == 4
    assert right.effects[0].subresource.offset == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_snapshot_field_rows_and_scalar_publication_keep_effects():
    source = [ti.Vector.field(3, ti.f32, shape=17) for _ in range(2)]
    target = [ti.Vector.field(3, ti.f32, shape=(3, 17)) for _ in range(2)]
    epochs = ti.field(ti.i64, shape=3)

    def copier(src, dst):
        @ti.kernel
        def copy(slot: ti.i32, count: ti.i32):
            for i in range(count):
                dst[slot, i] = src[i]
        return copy

    copies = [copier(src, dst) for src, dst in zip(source, target)]

    @ti.kernel
    def publish(slot: ti.i32, epoch: ti.i64):
        epochs[slot] = epoch

    slot = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "slot", ti.i32)
    count = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
    epoch = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "epoch", ti.i64)
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    for copy in copies:
        builder.dispatch(copy, slot, count)
    builder.dispatch(publish, slot, epoch)
    graph = builder.compile()
    records = graph._spec.nodes[0].compiled_graph._dispatch_metadata
    assert len(records) == 3
    for record in records:
        assert record["version"] == 5 and not record["opaque"]
        assert not record["elementwise"] and not record["synchronization"]
        assert not record["side_effects"]
    for record in records[:2]:
        effects = record["effects"]
        assert len(effects) == 6  # all three source and destination components
        assert sorted(e["access"] for e in effects) == ["read"] * 3 + ["write"] * 3
        assert all(e["resource_kind"] == "snode" for e in effects)
        assert all(e["footprint"]["pattern"] == "opaque" for e in effects if e["access"] == "write")
    assert len(records[2]["effects"]) == 1
    assert records[2]["effects"][0]["access"] == "write"
    report = graph._parallel_candidate_report(((0,), (1,)))
    assert report.safe and not report.conflicts and not report.blockers, report
    assert graph._ir_debug_info["fusion_plan"]["eligible_dispatches"] == 0
    for turn in range(2):
        for lane, src in enumerate(source):
            src.from_numpy(np.arange(51, dtype=np.float32).reshape(17, 3) + lane + turn)
        graph.run(dict(slot=turn, count=17, epoch=(1 << 40) + turn))
        for src, dst in zip(source, target):
            np.testing.assert_array_equal(dst.to_numpy()[turn], src.to_numpy())
        assert epochs[turn] == (1 << 40) + turn

    conflict = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    conflict.dispatch(copies[0], slot, count)
    conflict.dispatch(copies[0], slot, count)
    other = conflict.compile()
    report = other._parallel_candidate_report(((0,), (1,)))
    assert not report.safe and report.conflicts, report
    other.close()
    component = target[0].get_scalar_field(2)

    @ti.kernel
    def overwrite_component(slot: ti.i32, count: ti.i32):
        for i in range(count):
            component[slot, i] = -1

    # A whole-vector store must overlap a direct non-first scalar component.
    component_builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    component_builder.dispatch(copies[0], slot, count)
    component_builder.dispatch(overwrite_component, slot, count)
    component_graph = component_builder.compile()
    report = component_graph._parallel_candidate_report(((0,), (1,)))
    assert not report.safe and report.conflicts, report
    component_graph.close()
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scalar_runtime_effects_still_block_parallel_analysis():
    output = ti.field(ti.f32, shape=())

    @ti.kernel
    def random_value():
        output[None] = ti.random(ti.f32)

    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    builder.dispatch(random_value)
    builder.dispatch(random_value)
    graph = builder.compile()
    records = graph._spec.nodes[0].compiled_graph._dispatch_metadata
    assert all(r["opaque"] and r["side_effects"] for r in records)
    assert not graph._parallel_candidate_report(((0,), (1,))).safe
    graph.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_subword_field_rows_need_physical_storage_proof():
    value = ti.field(ti.i16, shape=(2, 3))

    @ti.kernel
    def write(slot: ti.i32):
        for i in range(3):
            value[slot, i] = i

    slot = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "slot", ti.i32)
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    builder.dispatch(write, slot)
    graph = builder.compile()
    record = graph._spec.nodes[0].compiled_graph._dispatch_metadata[0]
    assert record["opaque"] and record["blocker"] == "unsupported_field_storage"


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_multiple_and_nested_loops_keep_resource_effects_without_fusion():
    @ti.kernel
    def update(
        source: ti.types.ndarray(ti.i32, ndim=1),
        output: ti.types.ndarray(ti.i32, ndim=1),
        n: ti.i32,
        increment: ti.types.vector(2, ti.i32),
    ):
        for i in range(n):
            output[i] = 0
        for i in range(n):
            j = 0
            total = 0
            while j < 3:
                j += 1
                if j == 2:
                    continue
                total += source[i] + j + increment[0]
            output[i] = total

    source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    count = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
    increment = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "increment", ti.types.vector(2, ti.i32))
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    builder.dispatch(update, source, output, count, increment)
    graph = builder.compile()
    metadata = graph._spec.nodes[0].compiled_graph._dispatch_metadata[0]
    assert not metadata["opaque"] and not metadata["elementwise"]
    assert metadata["blocker"] == "multiple_iteration_domains"
    assert {(tuple(effect["arg_id"]), effect["access"]) for effect in metadata["effects"]} == {
        ((0,), "read"), ((1,), "write")}
    assert all(effect["footprint"]["pattern"] == "opaque" for effect in metadata["effects"])
    assert graph._ir_debug_info["fusion_plan"]["eligible_dispatches"] == 0
    values, result = ti.ndarray(ti.i32, 17), ti.ndarray(ti.i32, 17)
    values.from_numpy(np.arange(17, dtype=np.int32))
    graph.run(dict(source=values, output=result, count=17, increment=ti.Vector([3, 5])))
    np.testing.assert_array_equal(result.to_numpy(), 2 * np.arange(17) + 10)
    graph.close()
