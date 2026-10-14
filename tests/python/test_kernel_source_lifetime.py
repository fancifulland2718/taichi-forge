"""Source definitions must retire even when codegen removes their Field use."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("compiled", [False, True])
@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cfg_optimization=True, advanced_optimization=True
)
def test_retire_source_without_changing_field_free_execution(compiled):
    source = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    fields.place(source)
    tree = fields.finalize()

    @ti.kernel
    def calculate(x: ti.types.ndarray(ti.i32, 1), y: ti.types.ndarray(ti.i32, 1)):
        value = source[None]
        value = 7
        y[0] = x[0] + value

    x = ti.ndarray(ti.i32, shape=1)
    y = ti.ndarray(ti.i32, shape=1)
    x.fill(11)
    key = calculate._primal.ensure_compiled(x, y)
    kernel = calculate._primal.compiled_kernels[key]
    if compiled:
        calculate(x, y)
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            calculate,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.i32, ndim=1),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.i32, ndim=1),
        )
        graph = builder.compile()
        assert not graph._spec.snode_tree_dependencies
        np.testing.assert_array_equal(y.to_numpy(), [18])
    tree_id, tree_generation = tree.id, tree.generation
    tree.destroy()
    assert kernel.definition_retired()
    replacement = ti.field(ti.i64)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 8).place(replacement)
    replacement_tree = fields.finalize()
    assert replacement_tree.id == tree_id
    assert replacement_tree.generation != tree_generation
    if compiled:
        x.fill(19)
        graph.run({"x": x, "y": y})
        np.testing.assert_array_equal(y.to_numpy(), [26])
        calculate(x, y)
        np.testing.assert_array_equal(y.to_numpy(), [26])
        assert calculate._primal.compiled_kernels[key] is kernel
    config = ti.lang.impl.current_cfg()
    config.advanced_optimization = not config.advanced_optimization
    program = ti.lang.impl.get_runtime().prog
    with pytest.raises(RuntimeError, match="retired kernel definition"):
        program.compile_kernel(config, program.get_device_caps(), kernel)


@test_utils.test(arch=ti.cpu, offline_cache=False, advanced_optimization=True)
def test_aot_recompile_rejects_source_eliminated_only_by_jit_optimization():
    source = ti.field(ti.f32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 4).place(source)
    tree = fields.finalize()

    @ti.kernel
    def calculate(x: ti.types.ndarray(ti.f32, 1), y: ti.types.ndarray(ti.f32, 1)):
        for i in range(4):
            a = source[i]
            y[i] = x[i]
            b = source[i]
            y[i] = a - b + x[i]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        calculate,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.f32, ndim=1),
    )
    graph = builder.compile()
    assert not graph._compiled_graph._snode_tree_dependency_info
    tree.destroy()
    replacement = ti.field(ti.f64)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 8).place(replacement)
    fields.finalize()
    module = ti.aot.Module(compile_options={"advanced_optimization": False, "llvm_opt_level": 0})
    with pytest.raises((RuntimeError, ti.TaichiCompilationError), match="retired kernel definition|Field.*destroyed"):
        module.add_kernel(calculate)
    graph_module = ti.aot.Module(compile_options={"advanced_optimization": False, "llvm_opt_level": 0})
    with pytest.raises(RuntimeError, match="retired kernel definition"):
        graph_module.add_execution_plan("retired", graph)


@pytest.mark.parametrize("vector", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_new_specialization_rejects_destroyed_field_without_blocking_unrelated_fields(vector):
    source = ti.Vector.field(2, ti.f32, needs_grad=True) if vector else ti.field(ti.f32, needs_grad=True)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 4).place(source, source.grad)
    fields.finalize().destroy()
    replacement = ti.field(ti.f32, shape=4)
    replacement.fill(13)
    np.testing.assert_array_equal(replacement.to_numpy(), [13] * 4)

    @ti.kernel
    def read_retired() -> ti.f32:
        value = 0.0
        if ti.static(vector):
            value = source[0][0]
        else:
            value = source[0]
        return value

    with pytest.raises((RuntimeError, ti.TaichiCompilationError), match="Field.*destroyed"):
        read_retired()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_retiring_separate_gradient_root_preserves_primal_compilation():
    source = ti.field(ti.f32)
    primal = ti.FieldsBuilder()
    primal.place(source)
    primal.finalize()
    adjoint = ti.FieldsBuilder()
    adjoint.place(source.grad)
    gradient_tree = adjoint.finalize()
    source[None] = 13
    gradient_tree.destroy()

    @ti.kernel
    def read_primal() -> ti.f32:
        return source[None]

    assert read_primal() == 13
    with pytest.raises(RuntimeError, match="Field.*destroyed"):
        source.grad.to_numpy()


@pytest.mark.parametrize("cached", [False, True])
@test_utils.test(arch=ti.cpu, offline_cache=False, cfg_optimization=True, advanced_optimization=True)
def test_retired_real_function_cannot_be_reused_by_a_new_kernel(cached, tmp_path):
    config = ti.lang.impl.current_cfg()
    config.offline_cache = cached
    config.offline_cache_file_path = str(tmp_path / "source-cache")
    source = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    fields.place(source)
    tree = fields.finalize()

    @ti.real_func
    def read_source() -> ti.i32:
        value = source[None]
        value = 7
        return value

    @ti.kernel
    def first() -> ti.i32:
        return read_source()

    @ti.kernel
    def second() -> ti.i32:
        return read_source() + 1

    assert first() == 7
    tree.destroy()
    assert first() == 7
    with pytest.raises(RuntimeError, match="retired function definition"):
        second()
