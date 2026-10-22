"""Graph logical pack members retain the compiled physical argument mapping."""

import gc

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


def test_graph_argpack_symbol_distinguishes_resource_contracts():
    from taichi_forge.types._argument_descriptor import describe_symbolic_arg

    types = [
        ti.types.argpack(values=ti.types.ndarray(dtype, ndim, needs_grad=grad))
        for dtype, ndim, grad in ((ti.f32, 1, False), (ti.i32, 1, False), (ti.f32, 2, False), (ti.f32, 1, True))
    ]
    assert all(dtype.dtype == types[0].dtype for dtype in types)
    symbols = [ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", dtype) for dtype in types]
    assert len({describe_symbolic_arg(symbol).structural_key() for symbol in symbols}) == 4


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_argpack_nested_resources_positions_and_rebinding():
    leaf = ti.types.argpack(offset=ti.f32, values=ti.types.ndarray(ti.f32, 1))
    record = ti.types.struct(weights=ti.types.vector(2, ti.f32))
    pack_type = ti.types.argpack(left=leaf, right=leaf, record=record)
    x, y, out = (ti.ndarray(ti.f32, 2) for _ in range(3))
    x.from_numpy(np.array([3, 4], np.float32))
    y.from_numpy(np.array([7, 9], np.float32))
    pack = pack_type(left=leaf(offset=1, values=x), right=leaf(offset=2, values=y), record=record(weights=[10, 1]))

    @ti.kernel
    def first(p: pack_type, result: ti.types.ndarray(ti.f32, 1)):
        for i in range(2):
            result[i] = p.record.weights[0] * p.left.values[i] + p.right.values[i] + p.left.offset

    @ti.kernel
    def second(result: ti.types.ndarray(ti.f32, 1), p: pack_type):
        for i in range(2):
            result[i] += p.record.weights[1] * p.right.offset

    symbol = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", pack_type)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(first, symbol, output)
    builder.dispatch(second, output, symbol)
    graph = builder.compile()
    graph.run(dict(pack=pack, out=out))
    np.testing.assert_array_equal(out.to_numpy(), [40, 52])
    bindings = graph.bind(dict(pack=pack, out=out))
    pack.left.values = y
    pack.right = leaf(offset=4, values=x)
    pack.record = record(weights=[2, 3])
    gc.collect()
    graph.run(bindings)
    np.testing.assert_array_equal(out.to_numpy(), [30, 35])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_argpack_explicit_adjoint():
    member = ti.types.argpack(values=ti.types.ndarray(ti.f32, 1, needs_grad=True))
    pack_type = ti.types.argpack(left=member, right=member)
    x, y, loss = (ti.ndarray(ti.f32, 1, needs_grad=True) for _ in range(3))
    x.fill(3)
    y.fill(7)
    loss.fill(0)
    x.grad.fill(0)
    y.grad.fill(0)
    loss.grad.fill(1)
    pack = pack_type(left=member(values=x), right=member(values=y))

    @ti.kernel
    def energy(p: pack_type, total: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.left.values:
            total[0] += p.left.values[i] ** 2 + 3 * p.right.values[i]

    symbol = ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", pack_type)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(energy, symbol, output)
    builder.dispatch(energy.grad, symbol, output)
    graph = builder.compile()
    graph.run(dict(pack=pack, loss=loss))
    np.testing.assert_allclose(loss.to_numpy(), [30])
    np.testing.assert_allclose(x.grad.to_numpy(), [6])
    np.testing.assert_allclose(y.grad.to_numpy(), [3])
    pack.left.values = ti.ndarray(ti.f32, 1)
    with pytest.raises(RuntimeError, match="matching gradient owner"):
        graph.run(dict(pack=pack, loss=loss))
    np.testing.assert_allclose(loss.to_numpy(), [30])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_argpack_rejects_member_type_changes():
    pack_type = ti.types.argpack(gain=ti.f32, values=ti.types.ndarray(ti.f32, 1))
    wrong_type = ti.types.argpack(gain=ti.i32, values=ti.types.ndarray(ti.f32, 1))
    x, out = ti.ndarray(ti.f32, 1), ti.ndarray(ti.f32, 1)

    @ti.kernel
    def evaluate(p: pack_type, result: ti.types.ndarray(ti.f32, 1)):
        result[0] = p.gain * p.values[0]

    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    with pytest.raises(Exception, match="dtype.*doesn't match"):
        builder.dispatch(evaluate, ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "p", wrong_type), output)
    builder.dispatch(evaluate, ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "p", pack_type), output)
    graph = builder.compile()
    with pytest.raises(Exception, match="ArgPack.*value type does not match"):
        graph.run(dict(p=wrong_type(gain=2, values=x), out=out))
    pack = pack_type(gain=2, values=x)
    pack.values = ti.ndarray(ti.i32, 1)
    with pytest.raises(Exception, match="dtype|type"):
        graph.run(dict(p=pack, out=out))
    pack.values = x
    x.fill(5)
    graph.run(dict(p=pack, out=out))
    np.testing.assert_array_equal(out.to_numpy(), [10])
    ti.reset()
    with pytest.raises(RuntimeError, match="reset|runtime|retired"):
        graph.run(dict(p=pack, out=out))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_argpack_affine_member_and_submission_retirement():
    pack_type = ti.types.argpack(gain=ti.f32, values=ti.types.ndarray(ti.f32, 1))
    values, output = ti.ndarray(ti.f32, 4), ti.ndarray(ti.f32, 2)
    values.from_numpy(np.array([1, 4, 9, 16], np.float32))
    pack = pack_type(gain=2, values=values)

    @ti.kernel
    def evaluate(p: pack_type, result: ti.types.ndarray(ti.f32, 1)):
        for i in result:
            result[i] = p.gain * p.values[i]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        evaluate,
        ti.graph.Arg(ti.graph.ArgKind.ARGPACK, "pack", pack_type),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1),
    )
    graph = builder.compile()
    pack.values = ti.experimental.ndarray_view(values, slices=slice(1, None, 2))
    graph.run(dict(pack=pack, out=output))
    np.testing.assert_array_equal(output.to_numpy(), [8, 32])
    pack.values = ti.experimental.ndarray_view(values, slices=slice(0, None, 2))
    graph.run(dict(pack=pack, out=output))
    graph.close()
    del graph, pack, values
    gc.collect()
    np.testing.assert_array_equal(output.to_numpy(), [2, 18])
