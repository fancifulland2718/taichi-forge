import gc
import weakref

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiCompilationError
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

    # Keep both trees alive through the final graph use. Explicit destruction
    # while a graph is alive belongs to DF3's generation/lifetime contract.
    assert source_tree is not target_tree


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
