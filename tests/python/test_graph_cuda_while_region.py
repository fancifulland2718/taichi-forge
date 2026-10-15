import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from tests import test_utils


def _build_step(budget, *, embedded=False, lowering_mode="native_required"):
    @ti.kernel
    def initialize(state: ti.types.ndarray(ti.i32, 1), counter: ti.types.ndarray(ti.i32, 0)):
        counter[None] = 0
        for i in state:
            state[i] = 0

    @ti.kernel
    def condition(
        state: ti.types.ndarray(ti.i32, 1), targets: ti.types.ndarray(ti.i32, 1), predicate: ti.types.ndarray(ti.i32, 0)
    ):
        predicate[None] = 0
        for i in state:
            if state[i] < targets[i]:
                ti.atomic_max(predicate[None], 1)

    @ti.kernel
    def solve_group(
        state: ti.types.ndarray(ti.i32, 1), targets: ti.types.ndarray(ti.i32, 1), counter: ti.types.ndarray(ti.i32, 0)
    ):
        for i in state:
            for _ in ti.static(range(4)):
                if state[i] < targets[i]:
                    state[i] += 1
        counter[None] += 1

    @ti.kernel
    def finalize(
        state: ti.types.ndarray(ti.i32, 1),
        counter: ti.types.ndarray(ti.i32, 0),
        result: ti.types.ndarray(ti.i32, 1),
        scale: ti.i32,
    ):
        for i in state:
            result[i] = scale * state[i] + counter[None]

    @ti.kernel
    def publish(result: ti.types.ndarray(ti.i32, 1)):
        for i in result:
            result[i] += 1000

    def array(name, ndim=1):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=ndim)

    state, targets, result = (array(name) for name in ("state", "targets", "result"))
    counter, predicate = (array(name, 0) for name in ("counter", "predicate"))
    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.i32)
    builder = ti.graph.GraphBuilder()
    step = builder.create_sequential() if embedded else builder
    step.dispatch(initialize, state, counter)
    cond = builder.create_sequential()
    cond.dispatch(condition, state, targets, predicate)
    body = builder.create_sequential()
    body.dispatch(solve_group, state, targets, counter)
    step.while_loop(
        cond,
        body,
        predicate=predicate,
        counter=counter,
        control_inputs=(state, targets),
        carried_state=(state,),
        max_iterations=budget,
        lowering_mode=lowering_mode,
        name="solver",
    )
    step.dispatch(finalize, state, counter, result, scale)
    if embedded:
        builder.append(step)
        builder.dispatch(publish, result)
    return builder.compile()


def _arguments(targets):
    arguments = {name: ti.ndarray(ti.i32, shape=4) for name in ("state", "targets", "result")}
    arguments.update({name: ti.ndarray(ti.i32, shape=()) for name in ("counter", "predicate")})
    arguments["targets"].from_numpy(np.array(targets, dtype=np.int32))
    arguments["scale"] = 3
    return arguments


def _require_conditional():
    capabilities = dict(ti_core.cuda_conditional_graph_capabilities())
    if not (
        capabilities["driver_version_eligible"]
        and capabilities["conditional_graph_symbols_loaded"]
        and capabilities["general_device_setter_lowering_compiled"]
    ):
        pytest.skip("CUDA conditional graphs are unavailable")


@pytest.mark.parametrize("budget,embedded", [(0, False), (4, False), (4, True)])
@test_utils.test(arch=ti.cuda)
def test_cuda_while_region_single_submission(budget, embedded):
    _require_conditional()
    graph = _build_step(budget, embedded=embedded)
    arguments = _arguments([0, 2, 99, -1])
    program = impl.get_runtime().prog
    # Changed values, scalar bindings and allocations must all preserve the
    # prefix/condition/body/suffix order. The last two invocations become miss
    # (zero-iteration) replays after a previous active invocation.
    for index, targets in enumerate(([0, 2, 99, -1], [1, 5, 9, 0], [0, 0, 0, -1], [0, 0, 0, 0])):
        if index == 2:
            arguments = _arguments(targets)
        else:
            arguments["targets"].from_numpy(np.array(targets, dtype=np.int32))
        arguments["scale"] = 3 + index
        before = program._runtime_statistics_snapshot()["submission"]
        graph.submit(arguments).wait()
        after = program._runtime_statistics_snapshot()["submission"]
        assert after["graph_backend_submissions"] - before["graph_backend_submissions"] == 1
        expected = np.minimum(np.maximum(targets, 0), budget * 4)
        iterations = min(budget, (max(max(targets), 0) + 3) // 4)
        np.testing.assert_array_equal(arguments["state"].to_numpy(), expected)
        np.testing.assert_array_equal(
            arguments["result"].to_numpy(), expected * arguments["scale"] + iterations + (1000 if embedded else 0)
        )
        assert arguments["counter"].to_numpy()[()] == iterations


@test_utils.test(arch=ti.cuda)
def test_cuda_while_region_keeps_standalone_diagnostic_positions():
    _require_conditional()
    graph = _build_step(4)
    arguments = _arguments([0, 2, 5, -1])
    control = graph._instance._executable._cuda_submission_regions[0].control
    graph._graph_stats  # Enable counters before either cache is used.
    graph.run(arguments)
    assert control.debug_graph_stats[0]["captures"] == 1
    assert control.debug_graph_stats[-1]["captures"] == 0
    graph.submit(arguments).wait()
    graph.submit(arguments).wait()
    for stats in (control.debug_graph_stats, control.snapshot_graph_stats):
        assert stats[0]["captures"] == 1
        assert stats[-1]["captures"] == 1
        assert stats[-1]["exact_replays"] >= 1
        assert stats[-1]["last_path"] == "cuda_exact_replay"
    np.testing.assert_array_equal(arguments["result"].to_numpy(), [2, 8, 17, 2])


@test_utils.test(arch=ti.cuda)
def test_cuda_while_region_telemetry_and_invalidation():
    _require_conditional()
    graph = _build_step(4, embedded=True)
    arguments = _arguments([0, 2, 5, -1])
    graph.submit(arguments).wait()
    # Telemetry retains before/after snapshots at the original boundaries.
    report = graph.submit(arguments, telemetry="summary").telemetry()
    assert len(report.regions) == 1
    region = report.regions[0]
    assert region.logical_iterations == 2
    assert region.initial_counter == 0
    assert region.final_counter == 2
    graph._spec.invalidate_runtime()
    before = impl.get_runtime().prog._runtime_statistics_snapshot()["submission"]
    graph.submit(arguments).wait()
    after = impl.get_runtime().prog._runtime_statistics_snapshot()["submission"]
    assert after["graph_backend_submissions"] - before["graph_backend_submissions"] == 1
    np.testing.assert_array_equal(arguments["result"].to_numpy(), [1002, 1008, 1017, 1002])
    # Reset retires the fused native cache before the shared device.
    graph.submit(arguments)
    ti.reset()
    assert graph._spec is None


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_cuda_while_region_preserves_other_backends():
    graph = _build_step(4, lowering_mode="portable")
    arguments = _arguments([0, 2, 5, -1])
    graph.run(arguments)
    assert not graph._instance._executable._cuda_submission_regions
    np.testing.assert_array_equal(arguments["result"].to_numpy(), [2, 8, 17, 2])


@test_utils.test(arch=ti.cuda)
def test_cuda_while_region_failed_preparation_preserves_fallback():
    _require_conditional()
    graph = _build_step(4, embedded=True)
    arguments = _arguments([0, 2, 5, -1])
    region = graph._instance._executable._cuda_submission_regions[0]
    compiled = region.compiled_graph
    invalid_prefix = region.prefix_count + region.suffix_count + region.control._chunks[1].dispatch_count + 1

    class UnavailableRegion:
        def jit_submit_bounded_cuda_region_cached(self, config, args, cache, predicate, budget, prefix, suffix):
            # Native structural failure also exercises its retry state.
            return compiled.jit_submit_bounded_cuda_region_cached(
                config, args, cache, predicate, budget, invalid_prefix, suffix
            )

    region.compiled_graph = UnavailableRegion()
    for _ in range(2):
        graph.submit(arguments).wait()
        np.testing.assert_array_equal(arguments["result"].to_numpy(), [1002, 1008, 1017, 1002])


@test_utils.test(arch=ti.cuda)
def test_cuda_while_region_field_dependencies_and_vector_control():
    _require_conditional()
    state = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 4).place(state)
    tree = fields.finalize()

    @ti.kernel
    def initialize(counter: ti.types.ndarray(ti.i32, 1)):
        counter[0] = 0
        for i in state:
            state[i] = 0

    @ti.kernel
    def condition(predicate: ti.types.ndarray(ti.i32, 1)):
        predicate[0] = 0
        for i in state:
            if state[i] < i + 1:
                ti.atomic_or(predicate[0], 1)

    @ti.kernel
    def step(counter: ti.types.ndarray(ti.i32, 1)):
        for i in state:
            if state[i] < i + 1:
                state[i] += 1
        counter[0] += 1

    @ti.kernel
    def finalize():
        for i in state:
            state[i] *= 10

    predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=1)
    counter = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, counter)
    cond, body = builder.create_sequential(), builder.create_sequential()
    cond.dispatch(condition, predicate)
    body.dispatch(step, counter)
    builder.while_loop(
        cond, body, predicate=predicate, counter=counter, max_iterations=8, lowering_mode="native_required"
    )
    builder.dispatch(finalize)
    graph = builder.compile()
    arguments = {name: ti.ndarray(ti.i32, shape=1) for name in ("predicate", "counter")}
    before = impl.get_runtime().prog._runtime_statistics_snapshot()["submission"]
    graph.submit(arguments).wait()
    after = impl.get_runtime().prog._runtime_statistics_snapshot()["submission"]
    assert after["graph_backend_submissions"] - before["graph_backend_submissions"] == 1
    np.testing.assert_array_equal(state.to_numpy(), [10, 20, 30, 40])
    assert arguments["counter"].to_numpy()[0] == 4
    tree.destroy()
    with pytest.raises(RuntimeError, match="destroy|invalid"):
        graph.submit(arguments)
