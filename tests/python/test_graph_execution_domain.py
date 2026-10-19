"""Whole selected JIT recipes use live state and independent fixed instances."""

import gc
import weakref

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_execution_domain_live_fields_and_frozen_bindings():
    field = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 8).place(field)
    tree = fields.finalize()
    field.fill(10)

    @ti.kernel
    def advance(out: ti.types.ndarray(ti.i64, 1), delta: ti.i64):
        for i in field:
            field[i] += 1
            out[i] = ti.cast(field[i], ti.i64) + delta

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        advance,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i64, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "delta", ti.i64),
    )
    graph = builder.compile()
    out = ti.ndarray(ti.i64, 8)
    first_domain = ti.graph.execution_domain(instance_capacity=4)
    second_domain = ti.graph.execution_domain(instance_capacity=4)
    assert first_domain._runtime is second_domain._runtime
    bindings = graph.bind(dict(out=out, delta=1 << 40))
    first = graph.prepare(bindings, execution_domain=first_domain)
    bindings.update(dict(out=out, delta=37))
    second = graph.prepare(bindings, execution_domain=second_domain)
    first.initialize()
    second.initialize()
    ready = first.submit()
    ready.wait()
    np.testing.assert_array_equal(out.to_numpy(), np.full(8, (1 << 40) + 11))
    second.submit(wait_for=(ready,)).wait()
    np.testing.assert_array_equal(out.to_numpy(), np.full(8, 49))
    np.testing.assert_array_equal(field.to_numpy(), np.full(8, 12))
    # Collection of source/call wrappers cannot close the live code loan.
    source_ref = weakref.ref(graph)
    del bindings, graph
    first.close()
    del first
    gc.collect()
    assert source_ref() is not None
    second.submit().wait()
    np.testing.assert_array_equal(out.to_numpy(), np.full(8, 50))
    tree.destroy()
    assert first_domain.info["closed"] and second_domain.info["closed"]
    with pytest.raises(RuntimeError, match="closed|retired|live"):
        second.submit()
    second.close()
    first_domain.close()
    second_domain.close()


def _loop_graph():
    @ti.kernel
    def prefix(
        work: ti.types.ndarray(ti.i32, 1),
        counter: ti.types.ndarray(ti.i32, 1),
        seed: ti.i32,
    ):
        work[0] = seed
        counter[0] = 0

    @ti.kernel
    def condition(
        counter: ti.types.ndarray(ti.i32, 1),
        flag: ti.types.ndarray(ti.i32, 1),
        limit: ti.i32,
    ):
        flag[0] = counter[0] < limit

    @ti.kernel
    def advance(
        work: ti.types.ndarray(ti.i32, 1), counter: ti.types.ndarray(ti.i32, 1)
    ):
        work[0] *= 2
        counter[0] += 1

    @ti.kernel
    def suffix(
        work: ti.types.ndarray(ti.i32, 1),
        counter: ti.types.ndarray(ti.i32, 1),
        out: ti.types.ndarray(ti.i32, 1),
    ):
        out[0] = work[0] + 7
        out[1] = counter[0]

    builder = ti.graph.GraphBuilder()
    work = builder.private_ndarray("work", ti.i32, (1,))
    counter = builder.private_ndarray("counter", ti.i32, (1,))
    flag = builder.private_ndarray("flag", ti.i32, (1,))
    seed = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "seed", ti.i32)
    limit = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "limit", ti.i32)
    out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    whole = builder.create_sequential()
    whole.dispatch(prefix, work, counter, seed)
    cond = builder.create_sequential()
    cond.dispatch(condition, counter, flag, limit)
    body = builder.create_sequential()
    body.dispatch(advance, work, counter)
    whole.while_loop(
        cond,
        body,
        predicate=flag,
        counter=counter,
        control_inputs=(counter,),
        carried_state=(work,),
        max_iterations=4,
        lowering_mode="native_required",
    )
    whole.dispatch(suffix, work, counter, out)
    builder.append(whole)
    return builder.compile()


@pytest.mark.parametrize("masked", (False, True))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_execution_domain_whole_loop_and_private_instances(masked, monkeypatch):
    monkeypatch.setenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "1" if masked else "0")
    graph = _loop_graph()
    arrays = [ti.ndarray(ti.i32, 2) for _ in range(3)]
    domains = [ti.graph.execution_domain() for _ in range(2)]
    calls = [
        graph.prepare(
            dict(out=arrays[0], seed=3, limit=2), execution_domain=domains[0]
        ),
        graph.prepare(
            dict(out=arrays[1], seed=5, limit=4), execution_domain=domains[1]
        ),
    ]
    for call in calls:
        call.initialize()
        assert call.info["dispatch_count"] > 0
        assert call.info["execution_kind"] == "cuda_graph"
    # Source Graph's own workspace and prepared instances stay independent.
    graph.run(dict(out=arrays[2], seed=11, limit=3))
    ti.sync()
    for _ in range(3):
        tickets = [call.submit() for call in calls]
        for ticket in tickets:
            ticket.wait()
    np.testing.assert_array_equal(arrays[0].to_numpy(), [19, 2])
    np.testing.assert_array_equal(arrays[1].to_numpy(), [87, 4])
    np.testing.assert_array_equal(arrays[2].to_numpy(), [95, 3])
    for call in calls:
        call.close()
    for domain in domains:
        domain.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_execution_domain_reset_and_cold_rejection():
    @ti.kernel
    def random_fill(out: ti.types.ndarray(ti.i32, 1)):
        out[0] = ti.random(ti.i32)

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        random_fill, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    )
    graph = builder.compile()
    domain = ti.graph.execution_domain()
    with pytest.raises((ValueError, RuntimeError), match="random"):
        graph.prepare(dict(out=ti.ndarray(ti.i32, 1)), execution_domain=domain)
    assert domain.latest_completion().info["sequence"] == 0
    assert domain.info["accepting"]
    owner = domain._runtime
    domain.release()
    del domain
    gc.collect()
    assert not owner._borrower.closed
    ti.reset()
    assert owner._borrower.closed
    ti.init(arch=ti.cuda, offline_cache=False)
    current = ti.graph.execution_domain()
    assert current._runtime is not owner
    with pytest.raises(RuntimeError, match="reset|retired|invalid|closed"):
        graph.prepare(dict(out=ti.ndarray(ti.i32, 1)), execution_domain=current)
    current.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_execution_domain_selected_parallel():
    from tests.python.test_cuda_parallel_capture import parallel_graph

    graph, names = parallel_graph(3, 257)
    arrays = {name: ti.ndarray(ti.f32, 257) for name in names}
    source = np.arange(257, dtype=np.float32) / 10
    arrays["source"].from_numpy(source)
    domain = ti.graph.execution_domain()
    call = graph.prepare(dict(arrays, scale=1.0005), execution_domain=domain)
    call.initialize()
    assert call.info["execution_kind"] == "cuda_graph"
    # Prepared physical code remains independent of the source wrapper state.
    graph.close()
    for _ in range(3):
        call.submit().wait()
    expected = source + 1
    for _ in range(32):
        expected = expected * 1.0005 + 0.001
    np.testing.assert_allclose(
        arrays["result"].to_numpy(), expected * 3, rtol=3e-5, atol=3e-5
    )
    call.close()
    domain.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_execution_domain_manual_ad_and_automatic_ad_exclusion():
    @ti.kernel
    def square(
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in x:
            y[i] = x[i] * x[i]

    builder = ti.graph.GraphBuilder()
    x_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)
    y_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.f32, ndim=1)
    builder.dispatch(square, x_arg, y_arg)
    builder.dispatch(square.grad, x_arg, y_arg)
    graph = builder.compile()
    x = ti.ndarray(ti.f32, 17, needs_grad=True)
    y = ti.ndarray(ti.f32, 17, needs_grad=True)
    x.fill(3)
    x.grad.fill(0)
    y.grad.fill(2)
    domain = ti.graph.execution_domain()
    call = graph.prepare(dict(x=x, y=y), execution_domain=domain)
    call.initialize()
    call.submit().wait()
    np.testing.assert_allclose(y.to_numpy(), np.full(17, 9))
    np.testing.assert_allclose(x.grad.to_numpy(), np.full(17, 12))
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    with ti.ad.Tape(loss):
        with pytest.raises(RuntimeError, match="automatic AD"):
            call.submit()
    owner = ti.lang.impl.get_runtime()
    assert owner._active_graph_submissions == 0
    call.close()
    domain.close()


@pytest.mark.parametrize("masked", (False, True))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_execution_domain_branch_latches_selected_body(masked, monkeypatch):
    monkeypatch.setenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "1" if masked else "0")

    @ti.kernel
    def select(flag: ti.types.ndarray(ti.i32, 1), choice: ti.i32):
        flag[0] = choice

    @ti.kernel
    def body(
        flag: ti.types.ndarray(ti.i32, 1),
        out: ti.types.ndarray(ti.i32, 1),
        value: ti.i32,
    ):
        out[0] = value
        flag[0] = 0

    @ti.kernel
    def tail(out: ti.types.ndarray(ti.i32, 1)):
        out[0] += 9

    builder = ti.graph.GraphBuilder()
    flag = builder.private_ndarray("flag", ti.i32, (1,))
    choice = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "choice", ti.i32)
    out_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    cond = builder.create_sequential()
    cond.dispatch(select, flag, choice)
    branches = []
    for value in (30, 70):
        branch = builder.create_sequential()
        constant = branch._bind_internal_scalar(f"value{value}", ti.i32, value)
        branch.dispatch(body, flag, out_arg, constant)
        branch.dispatch(tail, out_arg)
        branches.append(branch)
    builder.if_then_else(
        cond,
        branches[0],
        predicate=flag,
        else_region=branches[1],
        lowering_mode="native_required",
    )
    graph = builder.compile()
    domain = ti.graph.execution_domain()
    outputs = [ti.ndarray(ti.i32, 1) for _ in range(2)]
    calls = [
        graph.prepare(dict(choice=choice, out=out), execution_domain=domain)
        for choice, out in enumerate(outputs)
    ]
    for call in calls:
        call.initialize()
        call.submit().wait()
    assert outputs[0].to_numpy()[0] == 79
    assert outputs[1].to_numpy()[0] == 39
    for call in calls:
        call.close()
    domain.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_execution_domain_keeps_each_dispatch_launch_policy():
    from taichi_forge._lib import core
    from taichi_forge.lang._offload_execution_plan import (
        _OffloadExecutionPlan,
        _bind_offload_execution_plan,
    )

    count = 1 << 16
    arrays = [ti.ndarray(ti.i32, count) for _ in range(2)]

    @ti.kernel
    def fill(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(count):
            out[i] = i * 3 + 5

    baseline = _OffloadExecutionPlan.from_task_manifests(fill.task_manifest(arrays[0]))
    (index,) = (
        task.task_index for task in baseline.tasks if task.task_kind == "range_for"
    )
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    for i, target in enumerate((2, 4)):
        selected = baseline.replace_task(index, range_work_per_thread_target=target)
        builder.dispatch(
            _bind_offload_execution_plan(fill, selected),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, f"out{i}", ti.i32, ndim=1),
        )
    graph = builder.compile()
    assert graph._spec.nodes[0].physical_dispatch_count == 2
    domain = ti.graph.execution_domain()
    call = graph.prepare(dict(out0=arrays[0], out1=arrays[1]), execution_domain=domain)
    call.initialize()
    assert core.query_int64("cuda_range_coarsening_last_work_per_thread_target") == 4
    resolved = core.query_int64("cuda_range_coarsening_resolution_calls")
    call.submit().wait()
    assert core.query_int64("cuda_range_coarsening_resolution_calls") == resolved
    for array in arrays:
        np.testing.assert_array_equal(
            array.to_numpy(), np.arange(count, dtype=np.int32) * 3 + 5
        )
    call.close()
    domain.close()


@pytest.mark.parametrize("matrix", (False, True))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_graph_execution_domain_tensor_arrays_and_real_func(matrix):
    element = ti.types.matrix(2, 2, ti.f32) if matrix else ti.types.vector(2, ti.f32)

    @ti.func
    def tensor_sum(value):
        return value.sum()

    @ti.real_func
    def evaluate(value: element) -> ti.f32:
        return tensor_sum(value) * 3

    @ti.kernel
    def stage(values: ti.types.ndarray(element, 1), out: ti.types.ndarray(ti.f32, 1)):
        for i in values:
            out[i] = evaluate(values[i])

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        stage,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", element, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1),
    )
    graph = builder.compile()
    values = ti.ndarray(element, 7)
    out = ti.ndarray(ti.f32, 7)
    host = np.arange(28 if matrix else 14, dtype=np.float32).reshape(
        (7, 2, 2) if matrix else (7, 2)
    )
    values.from_numpy(host)
    domain = ti.graph.execution_domain()
    call = graph.prepare(dict(values=values, out=out), execution_domain=domain)
    call.initialize()
    call.submit().wait()
    np.testing.assert_array_equal(out.to_numpy(), host.reshape(7, -1).sum(axis=1) * 3)
    call.close()
    domain.close()
