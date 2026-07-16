import threading

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


def _run_concurrently(workers):
    """Start all workers together and re-raise failures on the test thread."""

    start = threading.Barrier(len(workers))
    failures = []
    failure_lock = threading.Lock()

    def run(worker):
        try:
            start.wait(timeout=10)
            worker()
        except BaseException as exc:  # propagate thread failures to pytest
            with failure_lock:
                failures.append(exc)

    threads = [threading.Thread(target=run, args=(worker,)) for worker in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert all(not thread.is_alive() for thread in threads), "worker deadlocked"
    if failures:
        raise failures[0]


@test_utils.test(arch=ti.cpu)
def test_graph_run_serializes_the_complete_dispatch_loop():
    """Two callers must not interleave nodes inside one Graph.run()."""

    from taichi_forge.graph._graph import Graph, _GraphSpec

    trace = []
    trace_lock = threading.Lock()
    second_caller_entered_first_node = threading.Event()
    first_node_calls = 0

    class RecordingNode:
        needs_runtime_args = True
        runtime_arg_names = frozenset()
        snode_tree_dependencies = frozenset()
        snode_tree_dependency_info = frozenset()

        def __init__(self, index):
            self.index = index

        def run(self, context):
            nonlocal first_node_calls
            thread_id = threading.get_ident()
            with trace_lock:
                trace.append((self.index, thread_id))
                if self.index == 0:
                    first_node_calls += 1
                    call_index = first_node_calls
                else:
                    call_index = 0
            if self.index != 0:
                return
            if call_index == 1:
                # Without a whole-Graph transaction, the second caller enters
                # node 0 and releases this wait, producing 0,0,1,1.
                second_caller_entered_first_node.wait(timeout=0.1)
            else:
                second_caller_entered_first_node.set()

        @property
        def debug_info(self):
            return {"kind": "recording", "index": self.index}

    graph = Graph(_GraphSpec([RecordingNode(0), RecordingNode(1)]))
    _run_concurrently([lambda: graph.run({}), lambda: graph.run({})])

    assert [index for index, _ in trace] == [0, 1, 0, 1]
    assert trace[0][1] == trace[1][1]
    assert trace[2][1] == trace[3][1]
    assert trace[0][1] != trace[2][1]


@test_utils.test(arch=ti.cpu)
def test_graph_invalidation_waits_for_an_active_invocation():
    """Invalidation must not clear an instance while its run is active."""

    from taichi_forge.graph._graph import Graph, _GraphSpec
    from taichi_forge.lang.exception import TaichiRuntimeError

    entered = threading.Event()
    release = threading.Event()
    invalidated = threading.Event()
    failures = []

    class BlockingNode:
        needs_runtime_args = True
        runtime_arg_names = frozenset()
        snode_tree_dependencies = frozenset()
        snode_tree_dependency_info = frozenset()

        def run(self, context):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release the active graph run")

        @property
        def debug_info(self):
            return {"kind": "blocking"}

    graph = Graph(_GraphSpec([BlockingNode()]))

    def run_graph():
        try:
            graph.run({})
        except BaseException as exc:
            failures.append(exc)

    def invalidate_graph():
        graph._invalidate_runtime()
        invalidated.set()

    run_thread = threading.Thread(target=run_graph)
    run_thread.start()
    assert entered.wait(timeout=5)
    invalidate_thread = threading.Thread(target=invalidate_graph)
    invalidate_thread.start()
    assert not invalidated.wait(timeout=0.05)
    release.set()
    run_thread.join(timeout=5)
    invalidate_thread.join(timeout=5)

    assert not failures
    assert not run_thread.is_alive()
    assert not invalidate_thread.is_alive()
    assert invalidated.is_set()
    with pytest.raises(TaichiRuntimeError, match="compiled before ti.reset"):
        graph.run({})


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    cpu_max_num_threads=4,
)
def test_async_graph_and_cold_kernel_registration_are_thread_safe():
    """Keep a producer graph running while a distinct kernel registers.

    Graph execution releases the GIL, matching an async simulation producer.
    The second kernel has a scalar argument and two ndarray bindings, matching
    the GGUI staging-kernel descriptor shape that exposed the Vulkan handle
    mix-up. The buffers are deliberately disjoint so failures belong to the
    backend runtime rather than application data ownership.
    """

    n = 1 << 15
    # Keep enough overlap to catch short capture/replay versus ordinary-launch
    # races without turning the cross-backend CI test into a benchmark.
    iterations = 128
    simulation = ti.ndarray(ti.i32, shape=n)
    render_source = ti.ndarray(ti.i32, shape=n)
    render_output = ti.ndarray(ti.i32, shape=n)
    simulation.fill(0)
    render_source.from_numpy(np.arange(n, dtype=np.int32))
    render_output.fill(0)

    @ti.kernel
    def simulate(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    @ti.kernel
    def stage_for_display(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        bias: ti.i32,
    ):
        for i in output:
            output[i] = source[i] + bias

    symbolic_values = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(simulate, symbolic_values)
    producer_graph = builder.compile()

    # Finish Python/IR compilation on the owner thread, but deliberately do
    # not launch/register this kernel. This isolates the native registration
    # race from Python compiler callbacks that are not part of the worker-safe
    # submission contract.
    primal = stage_for_display._primal
    key = primal.ensure_compiled(render_source, render_output, 7)
    kernel_cpp = primal.compiled_kernels[key]
    prog = impl.get_runtime().prog
    prog.compile_kernel(prog.config(), prog.get_device_caps(), kernel_cpp)

    def producer():
        for _ in range(iterations):
            producer_graph.run({"values": simulation})

    def renderer():
        # Intentionally cold: registration races the already-running producer
        # on runtimes that do not protect their kernel/context tables.
        for _ in range(iterations):
            stage_for_display(render_source, render_output, 7)

    _run_concurrently([producer, renderer])
    ti.sync()
    np.testing.assert_array_equal(
        simulation.to_numpy(), np.full(n, iterations, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        render_output.to_numpy(), np.arange(n, dtype=np.int32) + 7
    )


@test_utils.test(arch=ti.cuda)
def test_cuda_graph_dynamic_patch_is_safe_for_two_host_callers():
    n = 1 << 14
    iterations = 128

    @ti.kernel
    def transform(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        bias: ti.i32,
    ):
        for i in output:
            output[i] = source[i] + bias

    sym_source = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1
    )
    sym_output = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    sym_bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.i32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(transform, sym_source, sym_output, sym_bias)
    graph = builder.compile()

    source_a_np = np.arange(n, dtype=np.int32)
    source_b_np = np.arange(n, dtype=np.int32) * -2
    source_a = ti.ndarray(ti.i32, shape=n)
    source_b = ti.ndarray(ti.i32, shape=n)
    output_a = ti.ndarray(ti.i32, shape=n)
    output_b = ti.ndarray(ti.i32, shape=n)
    source_a.from_numpy(source_a_np)
    source_b.from_numpy(source_b_np)

    def run_a():
        for _ in range(iterations):
            graph.run({"source": source_a, "output": output_a, "bias": 11})

    def run_b():
        for _ in range(iterations):
            graph.run({"source": source_b, "output": output_b, "bias": -7})

    _run_concurrently([run_a, run_b])
    ti.sync()
    np.testing.assert_array_equal(output_a.to_numpy(), source_a_np + 11)
    np.testing.assert_array_equal(output_b.to_numpy(), source_b_np - 7)


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=4)
def test_cpu_native_primitives_are_safe_for_two_gil_released_callers():
    """Exercise the real native bindings on two disjoint buffers concurrently."""

    prog = impl.get_runtime().prog
    required = (
        "cpu_transform_available",
        "cpu_reduce_available",
        "cpu_indexed_copy_available",
        "cpu_scatter_add_available",
    )
    if not all(hasattr(prog, name) and getattr(prog, name)() for name in required):
        import pytest

        pytest.skip("CPU native primitive support is unavailable in this runtime.")

    # Native primitive bindings release the GIL. Materialize on the owner
    # thread first: LLVM module construction deliberately remains a
    # main-thread-only runtime transition and is not part of the worker-safe
    # submission contract exercised below.
    impl.get_runtime().materialize()

    n = 1 << 18
    iterations = 6
    source_data = np.ones(n, dtype=np.int32)
    reverse_indices = np.arange(n - 1, -1, -1, dtype=np.int32)
    results = []

    def make_worker():
        src = ti.ndarray(ti.i32, shape=n)
        transformed = ti.ndarray(ti.i32, shape=n)
        copied = ti.ndarray(ti.i32, shape=n)
        scatter_sum = ti.ndarray(ti.i32, shape=n)
        indices = ti.ndarray(ti.i32, shape=n)
        reduced = ti.ndarray(ti.i32, shape=1)
        src.from_numpy(source_data)
        indices.from_numpy(reverse_indices)
        copied.fill(0)
        scatter_sum.fill(0)
        transform_workspace = ti.algorithms.TransformWorkspace(max_items=n)
        reduce_workspace = ti.algorithms.ReduceWorkspace(max_items=n)
        copy_workspace = ti.algorithms.IndexedCopyWorkspace(max_items=n)
        scatter_workspace = ti.algorithms.ScatterAddWorkspace(max_items=n)

        def worker():
            for _ in range(iterations):
                ti.algorithms.experimental_transform(
                    src,
                    transformed,
                    scale=3,
                    bias=2,
                    method="cpu_native",
                    workspace=transform_workspace,
                )
                reduced.fill(0)
                ti.algorithms.experimental_reduce(
                    transformed,
                    reduced,
                    op="sum",
                    method="cpu_native",
                    workspace=reduce_workspace,
                )
                ti.algorithms.experimental_scatter(
                    transformed,
                    indices,
                    copied,
                    method="cpu_native",
                    workspace=copy_workspace,
                )
                ti.algorithms.experimental_scatter_add(
                    transformed,
                    indices,
                    scatter_sum,
                    method="cpu_native",
                    workspace=scatter_workspace,
                )
            results.append(
                (
                    transformed.to_numpy(),
                    copied.to_numpy(),
                    scatter_sum.to_numpy(),
                    reduced.to_numpy(),
                )
            )

        return worker

    _run_concurrently([make_worker(), make_worker()])
    assert len(results) == 2
    for transformed, copied, scatter_sum, reduced in results:
        np.testing.assert_array_equal(transformed, np.full(n, 5, dtype=np.int32))
        np.testing.assert_array_equal(copied, np.full(n, 5, dtype=np.int32))
        np.testing.assert_array_equal(
            scatter_sum, np.full(n, 5 * iterations, dtype=np.int32)
        )
        assert reduced[0] == n * 5


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=4)
def test_cpu_native_workspace_does_not_grow_with_historical_threads():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "cpu_scatter_add_available")
        and prog.cpu_scatter_add_available()
    ):
        pytest.skip("CPU native scatter-add support is unavailable.")

    impl.get_runtime().materialize()
    n = 1 << 18
    groups = 64
    worker_count = 32
    indices_np = np.arange(n, dtype=np.int32) % groups
    workspaces = []
    outputs = []
    workers = []

    for _ in range(worker_count):
        values = ti.ndarray(ti.i32, shape=n)
        indices = ti.ndarray(ti.i32, shape=n)
        output = ti.ndarray(ti.i32, shape=groups)
        values.fill(1)
        indices.from_numpy(indices_np)
        output.fill(0)
        workspace = ti.algorithms.ScatterAddWorkspace(
            max_items=n, max_groups=groups
        )
        workspaces.append(workspace)
        outputs.append(output)

        def worker(
            values=values,
            indices=indices,
            output=output,
            workspace=workspace,
        ):
            ti.algorithms.experimental_scatter_add(
                values,
                indices,
                output,
                method="cpu_native",
                workspace=workspace,
            )

        workers.append(worker)

    _run_concurrently(workers)
    snapshot = prog._primitive_workspace_stats()
    assert 0 < snapshot["entries"] <= 16
    for output in outputs:
        np.testing.assert_array_equal(
            output.to_numpy(), np.full(groups, n // groups, dtype=np.int32)
        )

    workspaces[0].clear()
    assert prog._primitive_workspace_stats()["entries"] == 0


@test_utils.test(arch=ti.vulkan, exclude=[(ti.vulkan, "Darwin")])
def test_vulkan_native_workspace_arena_is_safe_for_two_callers():
    """One Program cache may be shared; per-call bindings must not leak."""

    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_transform_available")
        and prog.vulkan_transform_available()
    ):
        pytest.skip("Vulkan native transform is unavailable in this runtime.")

    impl.get_runtime().materialize()
    n = 1 << 16
    iterations = 8
    source_data = np.arange(n, dtype=np.int32)
    results = []
    stats_before = prog._primitive_workspace_stats()

    def make_worker(scale, bias):
        source = ti.ndarray(ti.i32, shape=n)
        output = ti.ndarray(ti.i32, shape=n)
        source.from_numpy(source_data)
        workspace = ti.algorithms.TransformWorkspace(max_items=n)

        def worker():
            for _ in range(iterations):
                ti.algorithms.experimental_transform(
                    source,
                    output,
                    scale=scale,
                    bias=bias,
                    method="vulkan_native",
                    workspace=workspace,
                )
            results.append((scale, bias, output.to_numpy()))

        return worker

    _run_concurrently([make_worker(3, 11), make_worker(-2, 7)])
    assert len(results) == 2
    for scale, bias, output in results:
        np.testing.assert_array_equal(output, source_data * scale + bias)

    stats_after = prog._primitive_workspace_stats()
    assert stats_after["entries"] >= stats_before["entries"] + 1
    assert (
        stats_after["acquisitions"]
        >= stats_before["acquisitions"] + iterations * 2
    )
    assert stats_after["active_leases"] == 0


@test_utils.test(arch=ti.vulkan, exclude=[(ti.vulkan, "Darwin")])
def test_vulkan_workspace_clear_cannot_cross_native_enqueue():
    prog = impl.get_runtime().prog
    if not (
        hasattr(prog, "vulkan_transform_available")
        and prog.vulkan_transform_available()
    ):
        pytest.skip("Vulkan native transform is unavailable in this runtime.")

    impl.get_runtime().materialize()
    n = 1 << 15
    source_data = np.arange(n, dtype=np.int32)
    source = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    source.from_numpy(source_data)
    workspace = ti.algorithms.TransformWorkspace(max_items=n)
    first_submission = threading.Event()

    def submitter():
        for iteration in range(16):
            ti.algorithms.experimental_transform(
                source,
                output,
                scale=5,
                bias=-3,
                method="vulkan_native",
                workspace=workspace,
            )
            if iteration == 0:
                first_submission.set()

    def clearer():
        assert first_submission.wait(timeout=10)
        for _ in range(8):
            prog.vulkan_transform_clear_workspace()

    _run_concurrently([submitter, clearer])
    ti.sync()
    np.testing.assert_array_equal(output.to_numpy(), source_data * 5 - 3)
    prog.vulkan_transform_clear_workspace()
    assert prog.vulkan_transform_workspace_bytes() == 0
    assert prog._primitive_workspace_stats()["active_leases"] == 0


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=4)
def test_compiled_graph_cache_serializes_two_gil_released_callers():
    """One executable/cache may be shared; run-local arguments must not leak."""

    @ti.kernel
    def add_one(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    n = 1 << 16
    iterations = 12
    symbolic_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(add_one, symbolic_values)
    graph = builder.compile()
    first = ti.ndarray(ti.i32, shape=n)
    second = ti.ndarray(ti.i32, shape=n)
    first.fill(0)
    second.fill(100)

    def make_worker(values):
        def worker():
            for _ in range(iterations):
                graph.run({"values": values})

        return worker

    _run_concurrently([make_worker(first), make_worker(second)])
    np.testing.assert_array_equal(first.to_numpy(), np.full(n, iterations, dtype=np.int32))
    np.testing.assert_array_equal(
        second.to_numpy(), np.full(n, 100 + iterations, dtype=np.int32)
    )
