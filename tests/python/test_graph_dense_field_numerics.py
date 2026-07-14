import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


_DENSE_GRAPH_ARCHS = [ti.cpu, ti.cuda, ti.vulkan]
_RUNS = 10
_SIZE = 64


def _assert_replay_path(report):
    arch = ti.lang.impl.current_cfg().arch
    if arch == ti.cpu:
        assert report.execution_path == "ordinary"
    elif arch == ti.cuda:
        assert report.execution_path == "cuda_exact_replay"
    else:
        assert report.execution_path == "vulkan_replay"


@test_utils.test(arch=_DENSE_GRAPH_ARCHS, offline_cache=False)
def test_dense_field_graph_integer_aos_soa_multiple_trees_is_exact():
    @ti.data_oriented
    class IntegerBlock:
        def __init__(self, layout):
            self.scalar = ti.field(ti.i32)
            self.vector = ti.Vector.field(3, ti.i32)
            self.matrix = ti.Matrix.field(2, 2, ti.i32)
            self.copy = ti.field(ti.i32)
            self.total = ti.field(ti.i32)
            builder = ti.FieldsBuilder()
            if layout == "aos":
                builder.dense(ti.i, _SIZE).place(
                    self.scalar,
                    self.vector,
                    self.matrix,
                    self.copy,
                )
            else:
                builder.dense(ti.i, _SIZE).place(self.scalar)
                builder.dense(ti.i, _SIZE).place(self.vector)
                builder.dense(ti.i, _SIZE).place(self.matrix)
                builder.dense(ti.i, _SIZE).place(self.copy)
            builder.place(self.total)
            self.tree = builder.finalize()

        @ti.kernel
        def initialize(self):
            self.total[None] = 0
            for i in self.scalar:
                self.scalar[i] = i * 3 - 17
                self.vector[i] = ti.Vector([i, -2 * i, 5 - i])
                self.matrix[i] = ti.Matrix(
                    [[i + 1, 2 - i], [3 * i, 7]]
                )
                self.copy[i] = 0

        @ti.kernel
        def advance(self):
            for i in self.scalar:
                self.scalar[i] = self.scalar[i] * 3 + i - 11
                self.vector[i] += ti.Vector([i + 1, 2 * i, -i])
                self.matrix[i] += ti.Matrix(
                    [[i, 1], [-2, i + 3]]
                )

        @ti.kernel
        def gather(self):
            for i in self.copy:
                self.copy[i] = (
                    self.scalar[i]
                    + self.vector[i][1]
                    + self.matrix[i][0, 1]
                )

        @ti.kernel
        def clear_total(self):
            self.total[None] = 0

        @ti.kernel
        def reduce(self):
            for i in self.copy:
                ti.atomic_add(self.total[None], self.copy[i])

    direct_blocks = [IntegerBlock("aos"), IntegerBlock("soa")]
    graph_blocks = [IntegerBlock("aos"), IntegerBlock("soa")]
    for block in direct_blocks + graph_blocks:
        block.initialize()

    builder = ti.graph.GraphBuilder()
    for block in graph_blocks:
        for kernel in (
            block.advance,
            block.gather,
            block.clear_total,
            block.reduce,
        ):
            builder.dispatch(kernel, template_args={"self": block})
    graph = builder.compile()
    graph.execution_stats()

    for _ in range(_RUNS):
        for block in direct_blocks:
            block.advance()
            block.gather()
            block.clear_total()
            block.reduce()
        graph.run({})
    ti.sync()

    for direct, captured in zip(direct_blocks, graph_blocks):
        np.testing.assert_array_equal(
            captured.scalar.to_numpy(), direct.scalar.to_numpy()
        )
        np.testing.assert_array_equal(
            captured.vector.to_numpy(), direct.vector.to_numpy()
        )
        np.testing.assert_array_equal(
            captured.matrix.to_numpy(), direct.matrix.to_numpy()
        )
        np.testing.assert_array_equal(
            captured.copy.to_numpy(), direct.copy.to_numpy()
        )
        assert captured.total[None] == direct.total[None]

    report = graph.execution_stats()
    assert report.static_dependency_count == 2
    assert report.runtime_arg_count == 0
    _assert_replay_path(report)


def _run_real_layout_matrix(dtype, *, rtol, atol):
    @ti.data_oriented
    class RealBlock:
        def __init__(self, layout):
            self.scalar = ti.field(dtype)
            self.vector = ti.Vector.field(2, dtype)
            self.matrix = ti.Matrix.field(2, 2, dtype)
            self.total = ti.field(dtype)
            builder = ti.FieldsBuilder()
            if layout == "aos":
                builder.dense(ti.i, _SIZE).place(
                    self.scalar,
                    self.vector,
                    self.matrix,
                )
            else:
                builder.dense(ti.i, _SIZE).place(self.scalar)
                builder.dense(ti.i, _SIZE).place(self.vector)
                builder.dense(ti.i, _SIZE).place(self.matrix)
            builder.place(self.total)
            self.tree = builder.finalize()

        @ti.kernel
        def initialize(self):
            self.total[None] = 0
            for i in self.scalar:
                value = ti.cast(i + 1, dtype) * ti.cast(0.003, dtype)
                self.scalar[i] = value
                self.vector[i] = ti.Vector([value, value * 0.5])
                self.matrix[i] = ti.Matrix(
                    [[value + 1.0, value], [-value, value + 0.5]]
                )

        @ti.kernel
        def advance(self):
            for i in self.scalar:
                bias = ti.cast(i % 7 - 3, dtype) * ti.cast(1e-4, dtype)
                self.scalar[i] = (
                    self.scalar[i] * ti.cast(0.9993, dtype) + bias
                )
                self.vector[i] = (
                    self.matrix[i] @ self.vector[i]
                    * ti.cast(0.5001, dtype)
                )
                self.matrix[i] += ti.Matrix(
                    [
                        [bias, ti.cast(2e-5, dtype)],
                        [ti.cast(-3e-5, dtype), -bias],
                    ]
                )

        @ti.kernel
        def clear_total(self):
            self.total[None] = 0

        @ti.kernel
        def reduce(self):
            for i in self.scalar:
                value = (
                    self.scalar[i]
                    + self.vector[i].dot(self.vector[i])
                    + self.matrix[i].trace()
                )
                ti.atomic_add(self.total[None], value)

    direct_blocks = [RealBlock("aos"), RealBlock("soa")]
    graph_blocks = [RealBlock("aos"), RealBlock("soa")]
    for block in direct_blocks + graph_blocks:
        block.initialize()

    builder = ti.graph.GraphBuilder()
    for block in graph_blocks:
        for kernel in (
            block.advance,
            block.clear_total,
            block.reduce,
        ):
            builder.dispatch(kernel, template_args={"self": block})
    graph = builder.compile()
    graph.execution_stats()

    for _ in range(_RUNS):
        for block in direct_blocks:
            block.advance()
            block.clear_total()
            block.reduce()
        graph.run({})
    ti.sync()

    for direct, captured in zip(direct_blocks, graph_blocks):
        np.testing.assert_allclose(
            captured.scalar.to_numpy(),
            direct.scalar.to_numpy(),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            captured.vector.to_numpy(),
            direct.vector.to_numpy(),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            captured.matrix.to_numpy(),
            direct.matrix.to_numpy(),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            captured.total[None],
            direct.total[None],
            rtol=rtol,
            atol=atol,
        )

    report = graph.execution_stats()
    assert report.static_dependency_count == 2
    _assert_replay_path(report)


@test_utils.test(arch=_DENSE_GRAPH_ARCHS, offline_cache=False)
def test_dense_field_graph_f32_layout_matrix_matches_direct():
    _run_real_layout_matrix(ti.f32, rtol=1e-5, atol=1e-6)


@test_utils.test(
    arch=_DENSE_GRAPH_ARCHS,
    require=ti.extension.data64,
    offline_cache=False,
)
def test_dense_field_graph_f64_layout_matrix_matches_direct():
    _run_real_layout_matrix(ti.f64, rtol=1e-12, atol=1e-13)


@test_utils.test(arch=_DENSE_GRAPH_ARCHS, offline_cache=False)
def test_graph_rejects_active_reverse_and_forward_ad():
    x = ti.field(
        ti.f32,
        shape=(),
        needs_grad=True,
        needs_dual=True,
    )
    loss = ti.field(
        ti.f32,
        shape=(),
        needs_grad=True,
        needs_dual=True,
    )

    @ti.kernel
    def square():
        loss[None] = x[None] * x[None]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(square)
    graph = builder.compile()
    x[None] = 3.0
    graph.run({})
    ti.sync()
    assert loss[None] == pytest.approx(9.0)

    with pytest.raises(
        TaichiRuntimeError,
        match=r"primal-only.*ti\.ad\.Tape",
    ):
        with ti.ad.Tape(loss):
            graph.run({})
    assert ti.lang.impl.get_runtime().target_tape is None

    with pytest.raises(
        TaichiRuntimeError,
        match=r"primal-only.*ti\.ad\.FwdMode",
    ):
        with ti.ad.FwdMode(loss=loss, param=x):
            graph.run({})
    assert ti.lang.impl.get_runtime().fwd_mode_manager is None


@test_utils.test(arch=_DENSE_GRAPH_ARCHS, offline_cache=False)
def test_explicit_grad_kernel_graph_runs_outside_tape():
    x = ti.field(ti.f32, shape=(), needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def square():
        loss[None] = x[None] * x[None]

    x[None] = 3.0
    square()
    ti.sync()
    x.grad[None] = 0.0
    loss.grad[None] = 1.0

    builder = ti.graph.GraphBuilder()
    builder.dispatch(square.grad)
    grad_graph = builder.compile()
    grad_graph.run({})
    ti.sync()

    assert x.grad[None] == pytest.approx(6.0)
    assert ti.lang.impl.get_runtime().target_tape is None
