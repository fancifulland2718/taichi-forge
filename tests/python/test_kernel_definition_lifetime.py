"""Frontend reclamation follows real consumers, independently of code caching."""

import gc
import weakref

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._graph import _CompiledCGraphNode, gen_cpp_kernel
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_sealed_aot_export_releases_source_and_reserves_name(tmp_path):
    runtime = ti.lang.impl.get_runtime()
    baseline = runtime.prog._debug_kernel_definition_count()
    aot = runtime.prog.make_aot_module_builder(ti.lang.impl.current_cfg().arch, [])
    arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)

    def export():
        @ti.kernel
        def fill(out: ti.types.ndarray(ti.i32, ndim=1)):
            for i in out:
                out[i] = i + 9

        native = gen_cpp_kernel(fill, [arg])
        aot.add('fill', native)
        aot.add('alias', native)
        aot._seal_kernel_export('fill')
        aot._seal_kernel_export('fill')
        aot._seal_kernel_export('alias')
        return weakref.ref(fill._primal), weakref.ref(native)

    source, native = export()
    gc.collect()
    assert source() is None
    assert native() is None
    runtime.collect_unused_kernel_definitions(force=True)
    assert runtime.prog._debug_kernel_definition_count() == baseline
    effects = aot._kernel_resource_effects('fill')
    assert effects['available'] and not effects['opaque']
    assert effects == aot._kernel_resource_effects('alias')
    # The native builder must still own enough code/metadata to write a module.
    aot.dump(str(tmp_path), '')

    # Check rejection separately: exception tracebacks may retain call frames,
    # which are legitimate source owners until the exception is released.
    @ti.kernel
    def other(out: ti.types.ndarray(ti.i32, ndim=1)):
        for i in out:
            out[i] = 2 * i

    replacement = gen_cpp_kernel(other, [arg])
    with pytest.raises(RuntimeError, match='sealed'):
        aot.add('fill', replacement)
    aot.add('other', replacement)
    aot._seal_kernel_export('other')
    with pytest.raises(RuntimeError, match='sealed'):
        aot.add('other', replacement)
    with pytest.raises(RuntimeError, match='unknown'):
        aot._seal_kernel_export('missing')
    with pytest.raises(RuntimeError, match='No compiled resource effects'):
        aot._kernel_resource_effects('missing')


@test_utils.test(arch=[ti.cuda], offline_cache=False)
def test_rejected_aot_export_does_not_publish_stale_effects():
    runtime = ti.lang.impl.get_runtime()
    aot = runtime.prog.make_aot_module_builder(ti.cuda, [])
    arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "data", ti.i32, ndim=1)

    @ti.kernel
    def read(data: ti.types.ndarray(ti.i32, ndim=1)) -> ti.i32:
        return data[0]

    @ti.kernel
    def write(data: ti.types.ndarray(ti.i32, ndim=1)):
        for i in data:
            data[i] = 9

    with pytest.raises(RuntimeError, match="requires void"):
        aot._add_with_launch_abi("retry", gen_cpp_kernel(read, [arg]), "cuda_graph_gate_v1")
    with pytest.raises(RuntimeError, match="No compiled resource effects"):
        aot._kernel_resource_effects("retry")
    aot.add("retry", gen_cpp_kernel(write, [arg]))
    effects = aot._kernel_resource_effects("retry")
    assert effects["available"] and not effects["opaque"]
    assert {(tuple(effect["arg_id"]), effect["access"]) for effect in effects["effects"]} == {((0,), "write")}


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_runtime_reset_releases_last_frontend_handle():
    arch = ti.lang.impl.current_cfg().arch

    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, ndim=1)):
        for i in out:
            out[i] = i + 4

    out = ti.ndarray(ti.i32, 3)
    fill(out)
    source = weakref.ref(fill._primal.kernel_cpp)
    assert source() is not None
    ti.reset()
    assert fill._primal.kernel_cpp is None
    gc.collect()
    assert source() is None
    # The decorated callable survives reset; its previous Program handle must
    # not survive or alias a new definition if native addresses are reused.
    ti.init(arch=arch, enable_fallback=False, offline_cache=False)
    fresh = ti.ndarray(ti.i32, 5)
    fill(fresh)
    np.testing.assert_array_equal(fresh.to_numpy(), np.arange(5) + 4)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_temporary_definitions_are_reclaimed_on_materialization():
    runtime = ti.lang.impl.get_runtime()
    baseline = runtime.prog._debug_kernel_definition_count()
    output = ti.ndarray(ti.i32, 4)

    def invoke():
        @ti.kernel
        def fill(dst: ti.types.ndarray(ti.i32, ndim=1)):
            for i in dst:
                dst[i] = i + 3

        fill(output)
        return weakref.ref(fill._primal)

    for _ in range(16):
        source = invoke()
        gc.collect()
        assert source() is None
        # Counting must not itself trigger a sweep. Each next materialization
        # reclaims old definitions while preserving the executable cache.
        assert runtime.prog._debug_kernel_definition_count() <= baseline + 3
    np.testing.assert_array_equal(output.to_numpy(), [3, 4, 5, 6])
    runtime.collect_unused_kernel_definitions(force=True)
    assert runtime.prog._debug_kernel_definition_count() == baseline
    assert runtime.debug_kernel_executable_lifecycle_stats()["resident_specializations"] == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_native_graph_and_aot_builder_keep_source_definition():
    runtime = ti.lang.impl.get_runtime()
    baseline = runtime.prog._debug_kernel_definition_count()
    arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)

    def make_builder():
        @ti.kernel
        def fill(out: ti.types.ndarray(ti.i32, ndim=1)):
            for i in out:
                out[i] = i + 9

        native = gen_cpp_kernel(fill, [arg])
        builder = core.GraphBuilder()
        builder.dispatch(native, [arg])
        sequence = builder.create_sequential()
        sequence.dispatch(native, [arg])
        builder.seq().append(sequence)
        builder.dispatch(native, [arg])
        return builder, weakref.ref(fill._primal)

    builder, source = make_builder()
    gc.collect()
    assert source() is None
    runtime.collect_unused_kernel_definitions(force=True)
    assert runtime.prog._debug_kernel_definition_count() == baseline + 1
    compiled = builder.compile()
    assert compiled._composer_stats["physical_dispatches"] == 3
    assert compiled._composer_stats["unique_jit_invocations"] == 1
    # The sharing table belongs to a compilation, not the builder or Program.
    again = builder.compile()
    assert again._composer_stats["unique_jit_invocations"] == 1
    del again
    del builder
    gc.collect()
    runtime.collect_unused_kernel_definitions(force=True)
    graph = ti.graph.Graph(_CompiledCGraphNode(compiled, 3, ("out",)))
    output = ti.ndarray(ti.i32, 4)
    graph.run({"out": output})
    np.testing.assert_array_equal(output.to_numpy(), [9, 10, 11, 12])

    # Export and re-export after the Graph source wrappers have disappeared.
    aot = runtime.prog.make_aot_module_builder(ti.lang.impl.current_cfg().arch, [])
    entries = aot.add_graph("fill", compiled)
    native = next(iter(entries.values()))
    name = next(iter(entries))
    del graph, compiled, entries
    gc.collect()
    runtime.collect_unused_kernel_definitions(force=True)
    aot.add(name, native)
    del aot
    gc.collect()
    runtime.collect_unused_kernel_definitions(force=True)
    # A recreated source wrapper must keep its export owner alive as well.
    assert not native.definition_retired()
    del native
    gc.collect()
    runtime.collect_unused_kernel_definitions(force=True)
    assert runtime.prog._debug_kernel_definition_count() == baseline


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_invocation_sharing_keeps_distinct_launch_policies():
    from taichi_forge.lang._offload_execution_plan import (
        _OffloadExecutionPlan,
        _bind_offload_execution_plan,
    )

    values = ti.ndarray(ti.i32, 64)

    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(64):
            out[i] = i * 5 + 11

    baseline = _OffloadExecutionPlan.from_task_manifests(fill.task_manifest(values))
    task, = (task.task_index for task in baseline.tasks if task.task_kind == "range_for")
    variants = [
        _bind_offload_execution_plan(fill, baseline.replace_task(task, range_work_per_thread_target=target))
        for target in (2, 4)
    ]
    symbol = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    for variant in variants * 2:
        builder.dispatch(variant, symbol)
    graph = builder.compile()
    native = graph._spec.nodes[0].compiled_graph
    assert native._composer_stats["physical_dispatches"] == 4
    assert native._composer_stats["unique_jit_invocations"] == 2
    assert len({row["kernel_name"] for row in native._dispatch_ir_metadata}) == 2
    graph.run({"out": values})
    np.testing.assert_array_equal(values.to_numpy(), np.arange(64) * 5 + 11)
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_retired_field_definitions_do_not_accumulate():
    runtime = ti.lang.impl.get_runtime()
    baseline = runtime.prog._debug_kernel_definition_count()

    def cycle():
        field = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, 4).place(field)
        tree = builder.finalize()

        @ti.kernel
        def fill():
            for i in field:
                field[i] = i

        fill()
        tree.destroy()

    for _ in range(12):
        cycle()
        gc.collect()
        assert runtime.prog._debug_kernel_definition_count() <= baseline + 3
    runtime.collect_unused_kernel_definitions(force=True)
    assert runtime.prog._debug_kernel_definition_count() == baseline
    assert runtime.debug_kernel_executable_lifecycle_stats()["resident_specializations"] == 0
