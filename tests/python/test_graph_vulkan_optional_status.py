import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("outer_status,inner_status", [(False, True), (True, False)])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_nested_optional_status_bindings_and_replay(outer_status, inner_status):
    if not ti.graph.structured_control_capabilities()["nested_native_lowering"]:
        pytest.skip("Native nested Vulkan control is unavailable")

    @ti.kernel
    def evaluate(
        count: ti.types.ndarray(ti.i32, 0),
        predicate: ti.types.ndarray(ti.i32, 0),
        status: ti.types.ndarray(ti.i32, 0),
        limit: ti.i32,
    ):
        predicate[None] = int(count[None] < limit)
        status[None] = 10 + count[None]

    @ti.kernel
    def step(count: ti.types.ndarray(ti.i32, 0), predicate: ti.types.ndarray(ti.i32, 0)):
        if predicate[None] != 0:
            count[None] += 1

    @ti.kernel
    def reset(count: ti.types.ndarray(ti.i32, 0)):
        count[None] = 0

    names = ("outer", "op", "os", "inner", "ip", "is")
    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0) for name in names}
    target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
    inner_target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "inner_target", ti.i32)
    builder = ti.graph.GraphBuilder()
    oc, ob, ic, ib = (builder.create_sequential() for _ in range(4))
    oc.dispatch(evaluate, symbols["outer"], symbols["op"], symbols["os"], target)
    ic.dispatch(evaluate, symbols["inner"], symbols["ip"], symbols["is"], inner_target)
    ib.dispatch(step, symbols["inner"], symbols["ip"])
    ob.dispatch(reset, symbols["inner"])
    ob.while_loop(
        ic,
        ib,
        predicate=symbols["ip"],
        counter=symbols["inner"],
        status=symbols["is"] if inner_status else None,
        max_iterations=4,
        chunk_size=2,
        masked_execution=True,
        name="inner",
    )
    ob.dispatch(step, symbols["outer"], symbols["op"])
    builder.while_loop(
        oc,
        ob,
        predicate=symbols["op"],
        counter=symbols["outer"],
        status=symbols["os"] if outer_status else None,
        max_iterations=4,
        masked_execution=True,
        name="outer",
    )
    graph = builder.compile()
    arrays = None
    for index, limit in enumerate((2, 1, 0)):
        if index < 2:
            arrays = {name: ti.ndarray(ti.i32, shape=()) for name in names}
        for array in arrays.values():
            array.fill(0)
        report = graph.submit({**arrays, "target": limit, "inner_target": 3}, telemetry="summary").telemetry()
        assert int(arrays["outer"].to_numpy()) == limit
        assert int(arrays["inner"].to_numpy()) == (3 if limit else 0)
        outer = report.regions[0]
        assert outer.logical_iterations == limit
        # Submission summary samples entry resources before condition kernels.
        assert outer.initial_status == (0 if outer_status else None)
        assert outer.final_status == (10 + limit if outer_status else None)
        if limit:
            inner = report.regions[1]
            assert inner.logical_iterations == 3
            assert inner.initial_status == (0 if inner_status else None)
            assert inner.final_status == (13 if inner_status else None)
    graph.close()
