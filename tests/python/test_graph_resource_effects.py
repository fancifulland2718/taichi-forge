"""Resource closure survives accesses that cannot be pointwise-fused."""

import numpy as np
import taichi_forge as ti
from tests import test_utils


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
    graph.close()
