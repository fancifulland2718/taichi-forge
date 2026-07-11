import threading

import numpy as np

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
    iterations = 32
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
