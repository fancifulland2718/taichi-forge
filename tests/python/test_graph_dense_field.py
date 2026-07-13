import numpy as np

import taichi_forge as ti
from taichi_forge.aot.utils import produce_injected_args_from_template
from taichi_forge.graph._graph import flatten_args
from tests import test_utils


_DENSE_GRAPH_ARCHS = [ti.cpu, ti.cuda, ti.vulkan]


def _dispatch_template_baseline(
    builder, kernel_fn, symbolic_args, template_args
):
    """Compile the current private template path used by GeoPhys.

    DF0 records existing dense Field behavior without changing the public API.
    DF1 must replace this helper with GraphBuilder.dispatch(template_args=...).
    """

    kernel = kernel_fn._primal
    injected_args = produce_injected_args_from_template(kernel, template_args)
    key = kernel.ensure_compiled(*injected_args)
    kernel_cpp = kernel.compiled_kernels[key]
    unzipped_args = flatten_args(symbolic_args)
    builder._aot_graph_plan.dispatch(kernel_cpp, unzipped_args)
    builder._ensure_runtime_graph_builder().dispatch(
        kernel_cpp, unzipped_args
    )
    builder._runtime_graph_arg_names.update(
        arg.name for arg in unzipped_args
    )
    builder._dispatch_count += 1


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


@test_utils.test(arch=_DENSE_GRAPH_ARCHS)
def test_dense_field_graph_template_owner_uses_two_dense_trees():
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
    _dispatch_template_baseline(
        builder,
        owner.accumulate,
        (sym_bias,),
        {"self": owner, "source_field": source},
    )
    graph = builder.compile()

    graph.run({"bias": 3})
    graph.run({"bias": -2})
    expected = 1 + np.arange(24, dtype=np.int32) * 4 + 1
    np.testing.assert_array_equal(target.to_numpy(), expected)

    # Keep both trees alive through the final graph use. Explicit destruction
    # while a graph is alive belongs to DF3's generation/lifetime contract.
    assert source_tree is not target_tree
