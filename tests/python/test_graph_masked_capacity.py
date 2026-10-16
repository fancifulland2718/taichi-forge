"""A selected flat masked route is limited by representation and resources."""

import json
import time

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_flat_masked_expanded_dispatch_capacity(tmp_path, monkeypatch):
    monkeypatch.setenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "1")

    @ti.kernel
    def reset(state: ti.types.ndarray(ti.i32, 1), checks: ti.types.ndarray(ti.i32, 1)):
        state[0], checks[0] = 0, 0

    @ti.kernel
    def condition(
        state: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
        predicate: ti.types.ndarray(ti.i32, 1),
        limit: ti.i32,
    ):
        predicate[0] = state[0] < limit
        checks[0] += 1

    @ti.kernel
    def advance(state: ti.types.ndarray(ti.i32, 1)):
        state[0] += 1

    begin = time.perf_counter()
    builder = ti.graph.GraphBuilder()
    state = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.i32, ndim=1)
    checks = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "checks", ti.i32, ndim=1)
    predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=1)
    limit = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "limit", ti.i32)
    builder.dispatch(reset, state, checks)
    cond, body = builder.create_sequential(), builder.create_sequential()
    cond.dispatch(condition, state, checks, predicate, limit)
    body.dispatch(advance, state)
    budget = 2049  # Two dispatches per iteration: 4098, just past the old cap.
    builder.while_loop(
        cond, body, predicate=predicate, counter=state, max_iterations=budget, lowering_mode="native_required"
    )
    graph = builder.compile()
    costs = dict(build_ms=(time.perf_counter() - begin) * 1000, complete_ms=[])
    state_array, predicate_array, checks_array = (ti.ndarray(ti.i32, 1) for _ in range(3))
    arrays = dict(state=state_array, predicate=predicate_array, checks=checks_array)
    for value in (0, 2, budget + 1, 0):
        begin = time.perf_counter()
        graph.submit(dict(arrays, limit=value)).wait()
        costs["complete_ms"].append((time.perf_counter() - begin) * 1000)
        steps = min(value, budget)
        assert state_array.to_numpy().tolist() == [steps]
        assert checks_array.to_numpy().tolist() == [steps + 1]
    assert graph._spec.structured_control_nodes[0]._cuda_control_lowering == "cuda_masked_bounded_graph"
    assert ti.graph.structured_control_capabilities()["device_control"]["max_encoded_dispatches"] is None
    costs["host_us"], costs["warm_complete_us"] = [], []
    for _ in range(2):
        begin = time.perf_counter()
        ticket = graph.submit(dict(arrays, limit=2))
        submitted = time.perf_counter()
        ticket.wait()
        costs["host_us"].append((submitted - begin) * 1e6)
        costs["warm_complete_us"].append((time.perf_counter() - begin) * 1e6)
    assert state_array.to_numpy().tolist() == [2]
    assert checks_array.to_numpy().tolist() == [3]
    graph.close()
    (tmp_path / "costs.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")
