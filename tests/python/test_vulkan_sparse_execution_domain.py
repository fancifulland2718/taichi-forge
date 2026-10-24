"""Public sparse preparation uses one loan for all phases and selected code."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_sparse_control import export_control
from tests.python.test_aot_sparse_retirement import _sparse_module


def export_domain_cases(module):
    export_control(module)
    initial = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    fields.pointer(ti.i, 4).place(initial)
    fields.finalize()

    @ti.kernel
    def initialize_only():
        initial[1] = 71

    @ti.kernel
    def read_initial(out: ti.types.ndarray(ti.i32, 1)):
        out[0] = initial[1]

    for kernel in (initialize_only, read_initial):
        module.add_kernel(kernel)
    module.add_execution_plan(
        "initial_only",
        ti.aot.ExecutionPlan(
            [dict(kind="fill32", destination="state", value=0)],
            initialization=[dict(kind="kernel", entry="initialize_only", arguments=[])],
        ),
    )
    # Host branch execution is deliberately outside a native domain. Its
    # rejection happens after the complete sparse loan was reserved; rollback
    # must return even trees which only occur in a branch body.
    module.add_execution_plan(
        "unsupported_branch",
        ti.aot.ExecutionPlan([
            dict(kind="branch", selector="state", selection="nonzero", has_default=False,
                 lowering=dict(kind="vulkan_host", version=1), condition=[],
                 bodies=[[dict(kind="kernel", entry="initialize_only", arguments=[])]])
        ]),
    )


@pytest.mark.parametrize("pool", ["independent", "chunked"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_vulkan_sparse_domain_control_and_retirement(tmp_path, pool):
    from taichi_forge_runtime import runtime_contract_manifest
    from taichi_forge._lib import core

    feature = "vulkan_execution_domain_sparse_fields"
    assert runtime_contract_manifest()["features"][feature]
    assert core.get_runtime_contract_manifest()["features"][feature]
    ti.cfg.vulkan_pointer_independent_pool = True
    ti.cfg.vulkan_pointer_allocator_kind = "chunked" if pool == "chunked" else "bump"
    ti.cfg.vulkan_pointer_max_chunks = 2 if pool == "chunked" else 1
    _sparse_module(tmp_path, None, export_extra=export_domain_cases,
                   consumer="aot_sparse_domain_probe.py")


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_jit_sparse_domain_selected_code_gradient_and_tree_retirement():
    values = ti.field(ti.f32, needs_grad=True)
    fields = ti.FieldsBuilder()
    blocks = fields.pointer(ti.i, 4)
    blocks.bitmasked(ti.i, 2).place(values, values.grad)
    tree = fields.finalize()

    @ti.kernel
    def seed(scale: ti.f32, loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        values[2] = scale
        values[5] = 2 * scale
        values.grad[2] = 0
        values.grad[5] = 0
        loss[0] = 0
        loss.grad[0] = 1

    @ti.kernel
    def energy(loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in values:
            loss[0] += 0.5 * values[i] * values[i]

    @ti.kernel
    def gather(gradient: ti.types.ndarray(ti.f32, 1)):
        gradient[0] = values.grad[2] + values.grad[5]

    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.f32)
    loss_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    gradient_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "gradient", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(seed, scale, loss_arg)
    builder.dispatch(energy, loss_arg)
    builder.dispatch(energy.grad, loss_arg)
    builder.dispatch(gather, gradient_arg)
    graph = builder.compile()
    loss = ti.ndarray(ti.f32, shape=1, needs_grad=True)
    gradient = ti.ndarray(ti.f32, shape=1)
    arguments = dict(scale=3, loss=loss, gradient=gradient)
    # Ordinary Graph.run does not bind ndarray adjoints; direct JIT kernels
    # provide the existing reference while Graph.prepare imports those binds.
    seed(3, loss)
    energy(loss)
    energy.grad(loss)
    gather(gradient)
    ti.sync()
    program = ti.lang.impl.get_runtime().prog
    registered = program._debug_kernel_lifecycle_stats()["registered_executables"]
    with ti.aot.borrow_runtime() as runtime:
        domain = runtime.execution_domain(capacity=2, instance_capacity=1)
        call = graph.prepare(arguments, execution_domain=domain)
        assert program._debug_kernel_lifecycle_stats()["registered_executables"] == registered
        with pytest.raises((ValueError, RuntimeError), match="loan|conflict"):
            seed(3, loss)
        call.initialize()
        for _ in range(2):
            call.submit().wait()
            np.testing.assert_array_equal(loss.to_numpy(), [22.5])
            np.testing.assert_array_equal(gradient.to_numpy(), [9])
        call.close()
        seed(3, loss)
        energy(loss)
        energy.grad(loss)
        gather(gradient)
        ti.sync()
        np.testing.assert_array_equal(gradient.to_numpy(), [9])
        call = graph.prepare(dict(arguments, scale=-3), execution_domain=domain)
        call.initialize()
        pending = call.submit()
        tree.destroy()
        assert domain.info["closed"]
        pending.wait()
        with pytest.raises(RuntimeError, match="retired|closed|live"):
            call.submit()
        call.close()
        domain.close()
