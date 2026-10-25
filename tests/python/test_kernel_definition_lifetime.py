"""Frontend reclamation follows real consumers, independently of code caching."""

import gc
import weakref

import numpy as np
import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._graph import _CompiledCGraphNode, gen_cpp_kernel
from tests import test_utils


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
        return builder, weakref.ref(fill._primal)

    builder, source = make_builder()
    gc.collect()
    assert source() is None
    runtime.collect_unused_kernel_definitions(force=True)
    assert runtime.prog._debug_kernel_definition_count() == baseline + 1
    compiled = builder.compile()
    del builder
    gc.collect()
    runtime.collect_unused_kernel_definitions(force=True)
    graph = ti.graph.Graph(_CompiledCGraphNode(compiled, 1, ("out",)))
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
