import gc
import threading
import time
import weakref
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiCompilationError, TaichiRuntimeError
from tests import test_utils


_DENSE_GRAPH_ARCHS = [ti.cpu, ti.cuda, ti.vulkan]
@test_utils.test(arch=_DENSE_GRAPH_ARCHS)
def test_dense_field_graph_global_zero_args_matches_direct():
    counter = ti.field(dtype=ti.i32, shape=())
    line = ti.field(dtype=ti.i32, shape=8)
    plane = ti.field(dtype=ti.f32, shape=(4, 3))
    vectors = ti.Vector.field(3, dtype=ti.f32, shape=8)
    matrices = ti.Matrix.field(2, 2, dtype=ti.f32, shape=4)

    @ti.kernel
    def clear():
        counter[None] = 0
        for i in line:
            line[i] = 0
            vectors[i] = ti.Vector.zero(ti.f32, 3)
        for i, j in plane:
            plane[i, j] = 0.0
        for i in matrices:
            matrices[i] = ti.Matrix.zero(ti.f32, 2, 2)

    @ti.kernel
    def advance():
        counter[None] += 3
        for i in line:
            line[i] += i + 1
            vectors[i] += ti.Vector(
                [ti.cast(i, ti.f32), 2.0, -1.0]
            )
        for i, j in plane:
            plane[i, j] += ti.cast(i * 3 + j, ti.f32) * 0.25
        for i in matrices:
            value = ti.cast(i, ti.f32)
            matrices[i] += ti.Matrix(
                [[value, 1.0], [-2.0, value + 0.5]]
            )

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance)
    graph = builder.compile()

    clear()
    advance()
    ti.sync()
    direct = {
        "counter": counter.to_numpy(),
        "line": line.to_numpy(),
        "plane": plane.to_numpy(),
        "vectors": vectors.to_numpy(),
        "matrices": matrices.to_numpy(),
    }

    clear()
    graph.run({})
    ti.sync()
    np.testing.assert_array_equal(counter.to_numpy(), direct["counter"])
    np.testing.assert_array_equal(line.to_numpy(), direct["line"])
    np.testing.assert_allclose(plane.to_numpy(), direct["plane"], rtol=0.0)
    np.testing.assert_allclose(
        vectors.to_numpy(), direct["vectors"], rtol=0.0
    )
    np.testing.assert_allclose(
        matrices.to_numpy(), direct["matrices"], rtol=0.0
    )

    graph.run({})
    ti.sync()
    np.testing.assert_array_equal(line.to_numpy(), direct["line"] * 2)
    np.testing.assert_allclose(
        vectors.to_numpy(), direct["vectors"] * 2, rtol=0.0
    )


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_dense_field_single_dispatch_reports_insufficient_dispatches():
    values = ti.field(dtype=ti.i32, shape=32)

    @ti.kernel
    def advance():
        for i in values:
            values[i] += i + 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance)
    graph = builder.compile()
    initial = graph.execution_stats()
    assert initial.execution_path == "not_run"
    assert initial.counters_complete
    graph.run({})
    graph.run({})
    ti.sync()

    stats = graph._graph_stats[0]
    assert stats["backend"] == "vulkan"
    assert stats["attempts"] == 2
    assert stats["ordinary_fallbacks"] == 2
    assert stats["records"] == 0
    assert stats["replays"] == 0
    assert stats["structural_fallbacks"] == 0
    assert stats["last_path"] == "ordinary_fallback"
    assert stats["last_fallback_reason"] == "insufficient_dispatches"
    assert stats["known_persistent_argument_bytes"] == 0
    report = graph.execution_stats()
    assert report.execution_path == "ordinary_fallback"
    assert report.fallback_reason == "insufficient_dispatches"
    assert report.ordinary_fallback_segments == 1
    assert report.backend_graph_segments == 0
    np.testing.assert_array_equal(
        values.to_numpy(), (np.arange(32, dtype=np.int32) + 1) * 2
    )


@test_utils.test(
    arch=[ti.cuda, ti.vulkan], debug=True, offline_cache=False
)
def test_dense_field_graph_report_classifies_debug_mode_fallback():
    values = ti.field(dtype=ti.i32, shape=32)

    @ti.kernel
    def advance():
        for i in values:
            values[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance)
    builder.dispatch(advance)
    graph = builder.compile()
    assert graph.execution_stats().execution_path == "not_run"
    graph.run({})
    graph.run({})
    ti.sync()

    report = graph.execution_stats()
    segment = report.segments[0]
    assert report.execution_path == "ordinary_fallback"
    assert report.fallback_reason == "debug_mode"
    assert report.backend_graph_segments == 0
    assert report.backend_replay_segments == 0
    assert report.ordinary_fallback_segments == 1
    assert segment.counters.attempts == 2
    assert segment.counters.ordinary_fallbacks == 2
    np.testing.assert_array_equal(
        values.to_numpy(), np.full(32, 4, dtype=np.int32)
    )


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_dense_field_multi_tree_graph_records_and_replays():
    values = ti.field(dtype=ti.i32)
    vectors = ti.Vector.field(3, dtype=ti.f32)
    matrices = ti.Matrix.field(2, 2, dtype=ti.f32)
    first_builder = ti.FieldsBuilder()
    first_builder.dense(ti.i, 64).place(values, vectors)
    first_tree = first_builder.finalize()
    second_builder = ti.FieldsBuilder()
    second_builder.dense(ti.i, 64).place(matrices)
    second_tree = second_builder.finalize()

    @ti.kernel
    def advance_values():
        for i in values:
            values[i] += i + 1
            vectors[i] += ti.Vector([1.0, 2.0, 3.0])

    @ti.kernel
    def advance_matrices():
        for i in matrices:
            matrices[i] += ti.Matrix([[1.0, 2.0], [3.0, 4.0]])

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance_values)
    builder.dispatch(advance_matrices)
    graph = builder.compile()
    assert graph._spec.snode_tree_dependencies == {
        (first_tree.id, first_tree.generation),
        (second_tree.id, second_tree.generation),
    }
    assert graph.execution_stats().execution_path == "not_run"

    # Populate all eight fixed replay slots, synchronize so slot zero is
    # reusable, then prove the ninth launch replays its recorded command list.
    for _ in range(8):
        graph.run({})
    ti.sync()
    graph.run({})
    ti.sync()

    stats = graph._graph_stats[0]
    assert stats["backend"] == "vulkan"
    assert stats["attempts"] == 9
    assert stats["records"] == 8
    assert stats["replays"] == 1
    assert stats["ordinary_fallbacks"] == 0
    assert stats["last_path"] == "vulkan_replay"
    assert stats["last_fallback_reason"] == "none"
    assert stats["known_persistent_argument_bytes"] == 0
    np.testing.assert_array_equal(
        values.to_numpy(), (np.arange(64, dtype=np.int32) + 1) * 9
    )
    np.testing.assert_allclose(
        vectors.to_numpy(), np.tile([9.0, 18.0, 27.0], (64, 1)),
        rtol=0.0,
    )
    np.testing.assert_allclose(
        matrices.to_numpy(),
        np.tile([[9.0, 18.0], [27.0, 36.0]], (64, 1, 1)),
        rtol=0.0,
    )
    first_tree.destroy()
    second_tree.destroy()


@test_utils.test(arch=_DENSE_GRAPH_ARCHS, offline_cache=False)
def test_dense_field_graph_execution_report_explains_backend_path():
    values = ti.field(dtype=ti.i32, shape=64)

    @ti.kernel
    def advance():
        for i in values:
            values[i] += i + 1

    @ti.kernel
    def scale():
        for i in values:
            values[i] *= 2

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance)
    builder.dispatch(scale)
    graph = builder.compile()

    initial = graph.execution_stats()
    assert isinstance(initial, ti.graph.GraphExecutionReport)
    assert initial.schema_version == 1
    assert initial.lifecycle_state == "ready"
    assert initial.node_count == 1
    assert initial.cgraph_segment_count == 1
    assert initial.native_node_count == 0
    assert initial.dispatch_count == 2
    assert initial.compiled_task_count is None
    assert initial.runtime_arg_count == 0
    assert initial.static_dependency_count == 1
    assert len(initial.static_layout_fingerprint) == 16
    assert initial.execution_path == "not_run"
    assert initial.fallback_reason == "none"
    assert initial.counters_complete
    assert initial.segments[0].runtime_arg_count == 0
    with pytest.raises(FrozenInstanceError):
        initial.arch = "mutated"

    arch = ti.lang.impl.current_cfg().arch
    run_count = 9 if arch == ti.vulkan else 2
    for run_index in range(run_count):
        if arch == ti.vulkan and run_index == 8:
            ti.sync()
        graph.run({})
    ti.sync()

    report = graph.execution_stats()
    segment = report.segments[0]
    assert report.compiled_task_count is not None
    assert report.compiled_task_count >= 2
    assert segment.compiled_task_count == report.compiled_task_count
    assert segment.persistent_argument_bytes == 0
    assert segment.counters_complete
    if arch == ti.cuda:
        assert report.execution_path == "cuda_exact_replay"
        assert report.backend_graph_segments == 1
        assert report.backend_replay_segments == 1
        assert report.ordinary_fallback_segments == 0
        assert segment.zero_arg_eligible
        assert segment.counters.captures == 1
        assert segment.counters.exact_replays == 1
    elif arch == ti.vulkan:
        assert report.execution_path == "vulkan_replay"
        assert report.backend_graph_segments == 1
        assert report.backend_replay_segments == 1
        assert report.ordinary_fallback_segments == 0
        assert segment.counters.records == 8
        assert segment.counters.replays == 1
    else:
        assert report.execution_path == "ordinary"
        assert report.backend_graph_segments == 0
        assert report.backend_replay_segments == 0
        assert report.ordinary_fallback_segments == 1


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_dense_field_graph_report_layout_fingerprint_is_structural():
    @ti.kernel
    def touch(bound: ti.template()):
        for i in bound:
            bound[i] += 1

    trees = []
    graphs = []
    fingerprints = []
    for shape in (32, 32, 48):
        field = ti.field(dtype=ti.f32)
        fields_builder = ti.FieldsBuilder()
        fields_builder.dense(ti.i, shape).place(field)
        tree = fields_builder.finalize()
        graph_builder = ti.graph.GraphBuilder()
        graph_builder.dispatch(touch, template_args={"bound": field})
        graph = graph_builder.compile()
        report = graph.execution_stats()
        assert report.static_dependency_count == 1
        assert len(report.static_layout_fingerprint) == 16
        int(report.static_layout_fingerprint, 16)
        trees.append(tree)
        graphs.append(graph)
        fingerprints.append(report.static_layout_fingerprint)

    assert fingerprints[0] == fingerprints[1]
    assert fingerprints[0] != fingerprints[2]
    trees[0].destroy()

    stale = graphs[0].execution_stats()
    assert stale.lifecycle_state == "stale_field_dependency"
    assert stale.execution_path == "stale_field_dependency"
    assert stale.fallback_reason == "stale_field_dependency"
    assert stale.segments[0].last_path == "unavailable"
    trees[1].destroy()
    trees[2].destroy()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_dense_field_graph_report_marks_pre_opt_in_gpu_counters_incomplete():
    values = ti.field(dtype=ti.i32, shape=32)

    @ti.kernel
    def advance():
        for i in values:
            values[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance)
    builder.dispatch(advance)
    graph = builder.compile()
    graph.run({})
    ti.sync()

    first = graph.execution_stats()
    assert first.execution_path in ("cuda_capture", "vulkan_record")
    assert not first.counters_complete
    assert first.segments[0].counters.attempts == 0

    graph.run({})
    ti.sync()
    second = graph.execution_stats()
    # Lifetime totals remain explicitly incomplete because the first launch
    # happened before opt-in; later interval deltas are still valid.
    assert not second.counters_complete
    assert second.segments[0].counters.attempts == 1


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=4, offline_cache=False)
def test_dense_field_graph_execution_report_multithread_reads_are_immutable():
    values = ti.field(dtype=ti.i32, shape=64)

    @ti.kernel
    def advance():
        for i in values:
            values[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance)
    graph = builder.compile()
    graph.run({})

    start = threading.Barrier(4)
    reports = []
    errors = []

    def read_report():
        try:
            start.wait(timeout=10.0)
            for _ in range(64):
                reports.append(graph.execution_stats())
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=read_report) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20.0)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert len(reports) == 256
    assert all(report.execution_path == "ordinary" for report in reports)
    assert all(report.compiled_task_count is not None for report in reports)


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=4, offline_cache=False)
def test_cpu_dense_field_same_graph_two_host_callers_are_serialized():
    values = ti.field(dtype=ti.i32, shape=1 << 14)

    @ti.kernel
    def advance():
        for i in values:
            values[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance)
    graph = builder.compile()
    graph.run({})
    values.fill(0)
    start = threading.Barrier(2)
    errors = []

    def submit():
        try:
            start.wait(timeout=10.0)
            for _ in range(32):
                graph.run({})
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    np.testing.assert_array_equal(
        values.to_numpy(), np.full(1 << 14, 64, dtype=np.int32)
    )


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=4, offline_cache=False)
def test_cpu_dense_field_independent_graph_callers_share_bounded_pool():
    fields = [ti.field(dtype=ti.i32, shape=1 << 14) for _ in range(4)]

    @ti.kernel
    def advance(field: ti.template()):
        for i in field:
            field[i] += 1

    graphs = []
    for field in fields:
        builder = ti.graph.GraphBuilder()
        builder.dispatch(advance, template_args={"field": field})
        graphs.append(builder.compile())
    for graph in graphs:
        graph.run({})
    for field in fields:
        field.fill(0)

    start = threading.Barrier(len(graphs))
    errors = []

    def submit(graph):
        try:
            start.wait(timeout=10.0)
            for _ in range(16):
                graph.run({})
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=submit, args=(graph,)) for graph in graphs
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    for field in fields:
        np.testing.assert_array_equal(
            field.to_numpy(), np.full(1 << 14, 16, dtype=np.int32)
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_dense_field_zero_arg_graph_captures_without_arg_storage():
    scalar = ti.field(dtype=ti.i32, shape=())
    values = ti.field(dtype=ti.i32, shape=64)
    vectors = ti.Vector.field(3, dtype=ti.f32, shape=64)
    matrices = ti.Matrix.field(2, 2, dtype=ti.f32, shape=64)

    @ti.kernel
    def advance():
        scalar[None] += 1
        for i in values:
            values[i] += i + 1
            vectors[i] += ti.Vector([1.0, 2.0, 3.0])
            matrices[i] += ti.Matrix([[1.0, 2.0], [3.0, 4.0]])

    builder = ti.graph.GraphBuilder()
    for _ in range(4):
        builder.dispatch(advance)
    graph = builder.compile()

    # The first explicit snapshot enables lazy counters before capture. It
    # must not allocate a runtime argument buffer for this static-Field graph.
    assert graph._graph_stats[0]["zero_arg_captures"] == 0
    graph.run({})
    ti.sync()
    first = graph._graph_stats[0]
    assert first["last_path"] == "cuda_capture"
    assert first["zero_arg_eligible"]
    assert first["zero_arg_captures"] == 1
    assert first["known_persistent_argument_bytes"] == 0
    assert scalar[None] == 4

    graph.run({})
    ti.sync()
    second = graph._graph_stats[0]
    assert second["last_path"] == "cuda_exact_replay"
    assert second["exact_replays"] >= 1
    assert second["known_persistent_argument_bytes"] == 0
    assert scalar[None] == 8
    np.testing.assert_array_equal(
        values.to_numpy(), (np.arange(64, dtype=np.int32) + 1) * 8
    )
    np.testing.assert_allclose(
        vectors.to_numpy(), np.tile([8.0, 16.0, 24.0], (64, 1)),
        rtol=0.0,
    )
    np.testing.assert_allclose(
        matrices.to_numpy(),
        np.tile([[8.0, 16.0], [24.0, 32.0]], (64, 1, 1)),
        rtol=0.0,
    )

    cache = graph._instance._backend_executable._jit_cache
    cache.clear_runtime_state()
    cache.clear_runtime_state()
    graph.run({})
    ti.sync()
    rebuilt = graph._graph_stats[0]
    assert rebuilt["last_path"] == "cuda_capture"
    assert rebuilt["zero_arg_eligible"]
    assert rebuilt["known_persistent_argument_bytes"] == 0
    assert scalar[None] == 12


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_dense_field_graph_concurrent_simulation_and_display_submission():
    simulation = ti.field(dtype=ti.i32, shape=1 << 14)
    image = ti.Vector.field(4, dtype=ti.u8, shape=(96, 64))

    @ti.kernel
    def simulation_step():
        for i in simulation:
            simulation[i] += 1

    @ti.kernel
    def render_frame():
        for i, j in image:
            image[i, j] = ti.cast(
                ti.Vector([i % 251, j % 251, (i * 3 + j * 5) % 251, 255]),
                ti.u8,
            )

    simulation_builder = ti.graph.GraphBuilder()
    for _ in range(4):
        simulation_builder.dispatch(simulation_step)
    simulation_graph = simulation_builder.compile()
    display_builder = ti.graph.GraphBuilder()
    for _ in range(2):
        display_builder.dispatch(render_frame)
    display_graph = display_builder.compile()

    # Enable lazy diagnostics before the first concurrent capture. Separate
    # Fields avoid introducing an application-level data race into this host
    # submission safety test.
    assert simulation_graph._graph_stats[0]["attempts"] == 0
    assert display_graph._graph_stats[0]["attempts"] == 0
    start = threading.Barrier(2)
    errors = []

    def submit(graph):
        try:
            start.wait(timeout=10.0)
            for _ in range(128):
                graph.run({})
        except BaseException as exc:
            errors.append(exc)

    simulation_thread = threading.Thread(target=submit, args=(simulation_graph,))
    display_thread = threading.Thread(target=submit, args=(display_graph,))
    simulation_thread.start()
    display_thread.start()
    simulation_thread.join(timeout=30.0)
    display_thread.join(timeout=30.0)
    assert not simulation_thread.is_alive()
    assert not display_thread.is_alive()
    assert not errors
    ti.sync()

    np.testing.assert_array_equal(
        simulation.to_numpy(),
        np.full(1 << 14, 128 * 4, dtype=np.int32),
    )
    expected = np.empty((96, 64, 4), dtype=np.uint8)
    for i in range(96):
        for j in range(64):
            expected[i, j] = [i % 251, j % 251, (i * 3 + j * 5) % 251, 255]
    np.testing.assert_array_equal(image.to_numpy(), expected)
    for graph in (simulation_graph, display_graph):
        stats = graph._graph_stats[0]
        assert stats["zero_arg_eligible"]
        assert stats["captures"] == 1
        assert stats["exact_replays"] == 127
        assert stats["known_persistent_argument_bytes"] == 0


@test_utils.test(arch=_DENSE_GRAPH_ARCHS)
def test_dense_field_graph_updates_scalar_runtime_argument():
    values = ti.field(dtype=ti.i32, shape=16)

    @ti.kernel
    def transform(scale: ti.i32, bias: ti.i32):
        for i in values:
            values[i] = values[i] * scale + bias + i

    sym_scale = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "scale", ti.i32
    )
    sym_bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.i32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(transform, sym_scale, sym_bias)
    graph = builder.compile()

    values.fill(2)
    graph.run({"scale": 3, "bias": 5})
    np.testing.assert_array_equal(
        values.to_numpy(), np.arange(16, dtype=np.int32) + 11
    )

    graph.run({"scale": -2, "bias": 7})
    expected = -(np.arange(16, dtype=np.int32) + 11) * 2
    expected += np.arange(16, dtype=np.int32) + 7
    np.testing.assert_array_equal(values.to_numpy(), expected)


@test_utils.test(arch=_DENSE_GRAPH_ARCHS)
def test_dense_field_graph_accepts_ndarray_runtime_input():
    output = ti.field(dtype=ti.f32, shape=32)

    @ti.kernel
    def copy_and_bias(
        source: ti.types.ndarray(dtype=ti.f32, ndim=1), bias: ti.f32
    ):
        for i in output:
            output[i] = source[i] * 1.5 + bias

    sym_source = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1
    )
    sym_bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.f32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(copy_and_bias, sym_source, sym_bias)
    graph = builder.compile()

    source_np = np.linspace(-2.0, 3.0, 32, dtype=np.float32)
    source = ti.ndarray(dtype=ti.f32, shape=32)
    source.from_numpy(source_np)
    graph.run({"source": source, "bias": 0.25})
    np.testing.assert_allclose(
        output.to_numpy(), source_np * 1.5 + 0.25, rtol=1e-6
    )


@pytest.mark.parametrize("container", ["builder", "sequential"])
@test_utils.test(arch=_DENSE_GRAPH_ARCHS)
def test_dense_field_graph_template_owner_uses_two_dense_trees(container):
    source = ti.field(dtype=ti.i32)
    target = ti.field(dtype=ti.i32)
    source_builder = ti.FieldsBuilder()
    source_builder.dense(ti.i, 24).place(source)
    source_tree = source_builder.finalize()
    target_builder = ti.FieldsBuilder()
    target_builder.dense(ti.i, 24).place(target)
    target_tree = target_builder.finalize()

    @ti.data_oriented
    class DenseOwner:
        def __init__(self, target_field):
            self.target = target_field

        @ti.kernel
        def accumulate(
            self, source_field: ti.template(), bias: ti.i32
        ):
            for i in self.target:
                self.target[i] += source_field[i] + bias

    @ti.kernel
    def initialize():
        for i in source:
            source[i] = i * 2
            target[i] = 1

    initialize()
    owner = DenseOwner(target)
    sym_bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.i32)
    builder = ti.graph.GraphBuilder()
    dispatch_target = builder
    if container == "sequential":
        dispatch_target = builder.create_sequential()
    dispatch_target.dispatch(
        owner.accumulate,
        sym_bias,
        template_args={"self": owner, "source_field": source},
    )
    if container == "sequential":
        builder.append(dispatch_target)
    graph = builder.compile()
    assert graph._spec.snode_tree_dependencies == {
        (source_tree.id, source_tree.generation),
        (target_tree.id, target_tree.generation),
    }

    graph.run({"bias": 3})
    graph.run({"bias": -2})
    expected = 1 + np.arange(24, dtype=np.int32) * 4 + 1
    np.testing.assert_array_equal(target.to_numpy(), expected)

    graph._compiled_graph.jit_run(
        ti.lang.impl.current_cfg(), {"bias": 4}
    )
    ti.sync()
    expected += np.arange(24, dtype=np.int32) * 2 + 4
    np.testing.assert_array_equal(target.to_numpy(), expected)

    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert owner_ref() is None
    graph.run({"bias": 0})

    assert source_tree is not target_tree


@test_utils.test(arch=_DENSE_GRAPH_ARCHS)
def test_dense_field_graph_destroyed_tree_is_rejected():
    values = ti.field(dtype=ti.i32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, 32).place(values)
    tree = builder.finalize()

    @ti.kernel
    def advance():
        for i in values:
            values[i] += i + 1

    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(advance)
    graph = graph_builder.compile()
    dependency = (tree.id, tree.generation)
    assert graph._spec.snode_tree_dependencies == {dependency}

    graph.run({})
    tree.destroy()
    with pytest.raises(
        TaichiRuntimeError, match="destroyed SNodeTree.*rebuild the Graph"
    ):
        graph.run({})


@test_utils.test(arch=_DENSE_GRAPH_ARCHS)
def test_destroying_unrelated_tree_keeps_dense_field_graph_valid():
    values = ti.field(dtype=ti.i32)
    unrelated = ti.field(dtype=ti.i32)
    values_builder = ti.FieldsBuilder()
    values_builder.dense(ti.i, 16).place(values)
    values_tree = values_builder.finalize()
    unrelated_builder = ti.FieldsBuilder()
    unrelated_builder.dense(ti.i, 8).place(unrelated)
    unrelated_tree = unrelated_builder.finalize()

    @ti.kernel
    def advance():
        for i in values:
            values[i] += 2

    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(advance)
    graph = graph_builder.compile()
    assert graph._spec.snode_tree_dependencies == {
        (values_tree.id, values_tree.generation)
    }

    graph.run({})
    unrelated_tree.destroy()
    graph.run({})
    ti.sync()
    np.testing.assert_array_equal(
        values.to_numpy(), np.full(16, 4, dtype=np.int32)
    )
    values_tree.destroy()


@test_utils.test(arch=_DENSE_GRAPH_ARCHS)
def test_reused_tree_id_does_not_revive_stale_dense_field_graph():
    old_values = ti.field(dtype=ti.i32)
    old_builder = ti.FieldsBuilder()
    old_builder.dense(ti.i, 8).place(old_values)
    old_tree = old_builder.finalize()

    @ti.kernel
    def old_advance():
        for i in old_values:
            old_values[i] += 1

    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(old_advance)
    old_graph = graph_builder.compile()
    old_identity = (old_tree.id, old_tree.generation)
    old_tree.destroy()

    new_values = ti.field(dtype=ti.i32)
    new_builder = ti.FieldsBuilder()
    new_builder.dense(ti.i, 8).place(new_values)
    new_tree = new_builder.finalize()
    assert new_tree.id == old_identity[0]
    assert new_tree.generation > old_identity[1]

    with pytest.raises(TaichiRuntimeError, match="destroyed SNodeTree"):
        old_graph.run({})

    @ti.kernel
    def new_advance():
        for i in new_values:
            new_values[i] += 3

    new_graph_builder = ti.graph.GraphBuilder()
    new_graph_builder.dispatch(new_advance)
    new_graph = new_graph_builder.compile()
    assert new_graph._spec.snode_tree_dependencies == {
        (new_tree.id, new_tree.generation)
    }
    new_graph.run({})
    ti.sync()
    np.testing.assert_array_equal(
        new_values.to_numpy(), np.full(8, 3, dtype=np.int32)
    )
    new_tree.destroy()


@test_utils.test(arch=_DENSE_GRAPH_ARCHS)
def test_tree_destroy_waits_for_active_graph_host_transaction():
    values = ti.field(dtype=ti.i32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, 64).place(values)
    tree = builder.finalize()

    @ti.kernel
    def advance():
        for i in values:
            values[i] += i

    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(advance)
    graph = graph_builder.compile()

    entered = threading.Event()
    release = threading.Event()
    run_errors = []
    destroy_errors = []
    destroy_done = threading.Event()
    original_run = graph._run_impl

    def gated_run(args):
        entered.set()
        assert release.wait(timeout=10.0)
        original_run(args)

    def invoke():
        try:
            graph.run({})
        except BaseException as exc:
            run_errors.append(exc)

    def destroy():
        try:
            tree.destroy()
        except BaseException as exc:
            destroy_errors.append(exc)
        finally:
            destroy_done.set()

    graph._run_impl = gated_run
    run_thread = threading.Thread(target=invoke)
    run_thread.start()
    assert entered.wait(timeout=10.0)
    destroy_thread = threading.Thread(target=destroy)
    destroy_thread.start()
    time.sleep(0.05)
    assert not destroy_done.is_set()
    release.set()
    run_thread.join(timeout=20.0)
    destroy_thread.join(timeout=20.0)
    assert not run_thread.is_alive()
    assert not destroy_thread.is_alive()
    assert not run_errors
    assert not destroy_errors
    with pytest.raises(TaichiRuntimeError, match="destroyed SNodeTree"):
        graph.run({})


@test_utils.test(arch=_DENSE_GRAPH_ARCHS, offline_cache=False)
def test_dense_field_graph_tree_churn_reuses_generation_safely():
    @ti.kernel
    def add_value(field: ti.template(), value: ti.i32):
        for i in field:
            field[i] += value + i

    sym_value = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "value", ti.i32
    )
    identities = []
    for iteration in range(12):
        values = ti.field(dtype=ti.i32)
        fields_builder = ti.FieldsBuilder()
        fields_builder.dense(ti.i, 16).place(values)
        tree = fields_builder.finalize()
        identity = (tree.id, tree.generation)
        identities.append(identity)

        graph_builder = ti.graph.GraphBuilder()
        graph_builder.dispatch(
            add_value,
            sym_value,
            template_args={"field": values},
        )
        graph = graph_builder.compile()
        assert graph._spec.snode_tree_dependencies == {identity}
        graph.run({"value": iteration})
        ti.sync()
        np.testing.assert_array_equal(
            values.to_numpy(),
            np.arange(16, dtype=np.int32) + iteration,
        )

        tree.destroy()
        with pytest.raises(TaichiRuntimeError, match="destroyed SNodeTree"):
            graph.run({"value": iteration})

    assert {tree_id for tree_id, _ in identities} == {identities[0][0]}
    generations = [generation for _, generation in identities]
    assert generations == sorted(set(generations))


@test_utils.test(arch=_DENSE_GRAPH_ARCHS, offline_cache=False)
def test_dense_field_graph_reset_keeps_whole_runtime_invalidation():
    values = ti.field(dtype=ti.i32, shape=8)

    @ti.kernel
    def advance():
        for i in values:
            values[i] += i + 1

    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(advance)
    graph = graph_builder.compile()
    graph.run({})
    ti.sync()

    arch = ti.lang.impl.current_cfg().arch
    ti.reset()
    assert graph._spec is None
    assert graph._instance is None
    assert graph._instances == {}
    ti.init(arch=arch, enable_fallback=False, offline_cache=False)
    with pytest.raises(TaichiRuntimeError, match="compiled before ti.reset"):
        graph.run({})


@test_utils.test(arch=_DENSE_GRAPH_ARCHS)
def test_dense_field_graph_ndarray_compile_exemplar_stays_runtime_bound():
    output = ti.field(dtype=ti.f32, shape=32)

    @ti.kernel
    def transform(
        source: ti.types.ndarray(ndim=1), scale: ti.f32
    ):
        for i in output:
            output[i] = source[i] * scale

    sym_source = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1
    )
    sym_scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.f32)
    exemplar = ti.ndarray(dtype=ti.f32, shape=4)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        transform,
        sym_source,
        sym_scale,
        template_args={"source": exemplar},
    )
    graph = builder.compile()

    source_np = np.arange(32, dtype=np.float32)
    source = ti.ndarray(dtype=ti.f32, shape=32)
    source.from_numpy(source_np)
    graph.run({"source": source, "scale": 1.25})
    np.testing.assert_allclose(
        output.to_numpy(), source_np * 1.25, rtol=1e-6
    )


@test_utils.test(arch=ti.cpu)
def test_graph_template_args_reject_invalid_bindings_at_build_time():
    values = ti.field(dtype=ti.i32, shape=8)

    @ti.data_oriented
    class Owner:
        @ti.kernel
        def update(self, value: ti.i32):
            for i in values:
                values[i] = value

    owner = Owner()
    sym_value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)

    with pytest.raises(
        TaichiCompilationError,
        match="Missing required Graph template arguments: self",
    ):
        ti.graph.GraphBuilder().dispatch(owner.update, sym_value)

    with pytest.raises(
        TaichiCompilationError, match="Unknown Graph template arguments: typo"
    ):
        ti.graph.GraphBuilder().dispatch(
            owner.update,
            sym_value,
            template_args={"self": owner, "typo": values},
        )

    with pytest.raises(
        TaichiCompilationError, match="invalid: value"
    ):
        ti.graph.GraphBuilder().dispatch(
            owner.update,
            sym_value,
            template_args={"self": owner, "value": 3},
        )

    with pytest.raises(
        TaichiCompilationError, match="received 2 symbolic arguments"
    ):
        ti.graph.GraphBuilder().dispatch(
            owner.update,
            sym_value,
            sym_value,
            template_args={"self": owner},
        )

    with pytest.raises(
        TaichiCompilationError, match="template_args must be a dict"
    ):
        ti.graph.GraphBuilder().dispatch(
            owner.update, sym_value, template_args=[owner]
        )


@test_utils.test(arch=ti.cpu)
def test_graph_ndarray_template_exemplar_must_match_symbolic_arg():
    @ti.kernel
    def copy(source: ti.types.ndarray(ndim=1)):
        for i in source:
            source[i] = source[i]

    sym_source = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1
    )
    wrong_dtype = ti.ndarray(dtype=ti.i32, shape=8)
    with pytest.raises(
        TaichiCompilationError, match="does not match its symbolic ndarray"
    ):
        ti.graph.GraphBuilder().dispatch(
            copy,
            sym_source,
            template_args={"source": wrong_dtype},
        )


@test_utils.test(arch=ti.cpu)
def test_graph_template_injection_cache_is_weak_and_identity_safe():
    values = ti.field(dtype=ti.i32, shape=8)

    @ti.data_oriented
    class Owner:
        @ti.kernel
        def update(self, value: ti.i32):
            for i in values:
                values[i] = value

    owner = Owner()
    sym_value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    sym_value_ref = weakref.ref(sym_value)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        owner.update,
        sym_value,
        template_args={"self": owner},
    )
    graph = builder.compile()

    graph.run({"value": 7})
    np.testing.assert_array_equal(
        values.to_numpy(), np.full(8, 7, dtype=np.int32)
    )

    del builder
    del graph
    del sym_value
    gc.collect()
    assert sym_value_ref() is None

    replacement = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "value", ti.f32
    )
    with pytest.raises(TaichiCompilationError, match="doesn't match"):
        ti.graph.GraphBuilder().dispatch(
            owner.update,
            replacement,
            template_args={"self": owner},
        )


@test_utils.test(arch=_DENSE_GRAPH_ARCHS, offline_cache=False)
def test_dense_field_graph_heterogeneous_blocks_run_on_independent_threads():
    @ti.data_oriented
    class DenseBlock:
        def __init__(self, envs, items, scale):
            self.envs = envs
            self.items = items
            self.scale = scale
            self.state = ti.field(ti.i32)
            self.snapshot = ti.field(ti.i32)
            self.counter = ti.field(ti.i32)
            self.domain = ti.field(ti.i32)
            builder = ti.FieldsBuilder()
            builder.dense(ti.ij, (envs, items)).place(
                self.state,
                self.snapshot,
            )
            builder.dense(ti.i, envs).place(
                self.counter,
                self.domain,
            )
            self.tree = builder.finalize()

        @ti.kernel
        def initialize(self):
            for env in self.counter:
                self.counter[env] = 0
                self.domain[env] = env + 1
            for env, item in self.state:
                self.state[env, item] = item
                self.snapshot[env, item] = 0

        @ti.kernel
        def advance(self):
            for env in self.counter:
                self.counter[env] += 1
            for env, item in self.state:
                self.state[env, item] += (
                    self.domain[env] * ti.static(self.scale)
                )

        @ti.kernel
        def publish(self):
            for env, item in self.snapshot:
                self.snapshot[env, item] = self.state[env, item]

    blocks = [
        DenseBlock(2, 8, 1),
        DenseBlock(3, 12, 2),
        DenseBlock(4, 16, 3),
    ]
    graphs = []
    for block in blocks:
        block.initialize()
        builder = ti.graph.GraphBuilder()
        for kernel in (
            block.advance,
            block.publish,
            block.advance,
            block.publish,
        ):
            builder.dispatch(
                kernel,
                template_args={"self": block},
            )
        graph = builder.compile()
        graph.execution_stats()
        graph.run({})
        graphs.append(graph)
    ti.sync()

    barrier = threading.Barrier(len(graphs))
    errors = []
    error_lock = threading.Lock()

    def producer(graph):
        try:
            barrier.wait(timeout=10.0)
            # Vulkan deliberately fills all eight bounded replay slots before
            # rotating back to an already-recorded slot.
            for _ in range(10):
                graph.run({})
                ti.sync()
        except BaseException as exc:
            with error_lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=producer, args=(graph,))
        for graph in graphs
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    ti.sync()

    fingerprints = set()
    for block, graph in zip(blocks, graphs):
        np.testing.assert_array_equal(
            block.counter.to_numpy(),
            np.full(block.envs, 22, dtype=np.int32),
        )
        report = graph.execution_stats()
        fingerprints.add(report.static_layout_fingerprint)
        assert report.static_dependency_count == 1
        assert report.dispatch_count == 4
        assert report.runtime_arg_count == 0
        assert report.counters_complete
        assert all(
            segment.persistent_argument_bytes == 0
            for segment in report.segments
        )
        if ti.lang.impl.current_cfg().arch == ti.cpu:
            assert report.execution_path == "ordinary"
        elif ti.lang.impl.current_cfg().arch == ti.cuda:
            assert report.execution_path == "cuda_exact_replay"
            assert report.backend_replay_segments == 1
        else:
            assert report.execution_path == "vulkan_replay"
            assert report.backend_replay_segments == 1
    assert len(fingerprints) == len(blocks)
