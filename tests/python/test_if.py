import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("compile_tier", ["fast", "balanced"])
@pytest.mark.parametrize("swap_branches", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_nested_if_preserves_extra_else(compile_tier, swap_branches):
    ti.cfg.compile_tier = compile_tier
    ti.cfg.advanced_optimization = compile_tier != "fast"
    result = ti.field(ti.i32, shape=())

    @ti.kernel
    def run(a: ti.i32, b: ti.i32):
        result[None] = 0
        if a != ti.static(int(swap_branches)):
            if b:
                result[None] = 1
        else:
            if b:
                result[None] = 1
            else:
                result[None] = 2

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        run,
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "a", ti.i32),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "b", ti.i32),
    )
    graph = builder.compile()
    try:
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1), (0, 0)):
            expected = 1 if b else (2 if a == int(swap_branches) else 0)
            run(a, b)
            assert result[None] == expected
            result[None] = -1
            graph.run({"a": a, "b": b})
            assert result[None] == expected
    finally:
        graph.close()


@test_utils.test()
def test_ifexpr_vector():
    n_grids = 10

    g_v = ti.Vector.field(3, float, (n_grids, n_grids, n_grids))
    g_m = ti.field(float, (n_grids, n_grids, n_grids))

    @ti.kernel
    def func():
        for I in ti.grouped(g_m):
            cond = (I < 3) & (g_v[I] < 0) | (I > n_grids - 3) & (g_v[I] > 0)
            g_v[I] = 0 if cond else g_v[I]

    with pytest.raises(ti.TaichiSyntaxError, match='Please use "ti.select" instead.'):
        func()


@test_utils.test()
def test_ifexpr_scalar():
    n_grids = 10

    g_v = ti.Vector.field(3, float, (n_grids, n_grids, n_grids))
    g_m = ti.field(float, (n_grids, n_grids, n_grids))

    @ti.kernel
    def func():
        for I in ti.grouped(g_m):
            cond = (I[0] < 3) and (g_v[I][0] < 0) or (I[0] > n_grids - 3) and (g_v[I][0] > 0)
            g_v[I] = 0 if cond else g_v[I]

    func()


@pytest.mark.parametrize("move_true_branch", [True, False])
@test_utils.test(arch=[ti.cuda, ti.vulkan], compile_tier="balanced",
                 advanced_optimization=True, offline_cache=False)
def test_adjacent_if_merge_transfers_branch_owner(move_true_branch):
    result = ti.field(ti.i32, shape=4)

    @ti.kernel
    def run():
        for i in result:
            condition = i < 2
            if ti.static(move_true_branch):
                if condition:
                    result[i] = 3
                if condition:
                    pass
                else:
                    result[i] = 5
            else:
                if condition:
                    pass
                else:
                    result[i] = 5
                if condition:
                    result[i] = 3

    run()
    assert result.to_numpy().tolist() == [3, 3, 5, 5]
    builder = ti.graph.GraphBuilder()
    builder.dispatch(run)
    graph = builder.compile()
    result.fill(-1)
    graph.run({})
    assert result.to_numpy().tolist() == [3, 3, 5, 5]
