import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("prepare", [ti.linalg.prepare_matrix_free_cg, ti.linalg.prepare_matrix_free_bicgstab])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_prepared_field_solve_reuses_kernels_with_live_inputs(prepare):
    x = ti.field(ti.f32, shape=(4, 8))
    b = ti.field(ti.f32, shape=x.shape)

    @ti.kernel
    def apply(src: ti.template(), dst: ti.template()):
        for i, j in src:
            dst[i, j] = (1 + j % 2) * src[i, j]

    compiled = ti.lang.impl.get_runtime().get_num_compiled_functions
    with prepare(ti.linalg.FieldLinearOperator(apply), b, x, tol=1e-4) as plan:
        for k in (1, 2, 3):
            b.fill(k)
            x.fill(-k)
            assert plan.solve()
            np.testing.assert_allclose(x.to_numpy(), np.tile(k / (1 + np.arange(8) % 2), (4, 1)), atol=1e-5)
            if k == 1:
                first = compiled()
            else:
                assert compiled() == first
    plan.close()
    with pytest.raises(RuntimeError, match="retired"):
        plan.solve()


@test_utils.test(arch=ti.cpu)
def test_prepared_field_solve_rejection_and_exception_recovery():
    x = ti.field(ti.f32, 8)
    builder = ti.FieldsBuilder()
    b = ti.field(ti.f32)
    builder.dense(ti.i, 8).place(b)
    tree = builder.finalize()
    b.fill(3)
    reenter = True

    @ti.kernel
    def identity(src: ti.template(), dst: ti.template()):
        for i in src:
            dst[i] = src[i]

    def apply(src, dst):
        if reenter:
            plan.solve()
        identity(src, dst)

    with ti.linalg.prepare_matrix_free_cg(ti.linalg.FieldLinearOperator(apply), b, x) as plan:
        with pytest.raises(RuntimeError, match="already in use"):
            plan.solve()
        reenter = False
        assert plan.solve()
        tree.destroy()
        with pytest.raises(RuntimeError, match="retired"):
            plan.solve()


@pytest.mark.parametrize("prepare", [ti.linalg.prepare_matrix_free_cg, ti.linalg.prepare_matrix_free_bicgstab])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_sparse_initial_guess_can_drop_active_cells_between_solves(prepare):
    b = ti.field(ti.f32, 8)
    x = ti.field(ti.f32)
    blocks = ti.root.pointer(ti.i, 2)
    blocks.dense(ti.i, 4).place(x)

    @ti.kernel
    def identity(src: ti.template(), dst: ti.template()):
        for i in dst:
            dst[i] = src[i]

    x[0] = 1
    with prepare(ti.linalg.FieldLinearOperator(identity), b, x, maxiter=0) as plan:
        assert not plan.solve()
        blocks.deactivate_all()
        assert plan.solve()


@test_utils.test(arch=ti.cpu)
def test_retiring_one_workspace_keeps_other_bindings_live_until_reset():
    b = ti.field(ti.f32, 8)
    x = ti.field(ti.f32, 8)
    y = ti.field(ti.f32, 8)

    @ti.kernel
    def identity(src: ti.template(), dst: ti.template()):
        for i in src:
            dst[i] = src[i]

    operator = ti.linalg.FieldLinearOperator(identity)
    b.fill(2)
    with ti.linalg.prepare_matrix_free_cg(operator, b, y) as second:
        with ti.linalg.prepare_matrix_free_cg(operator, b, x) as first:
            assert first.solve() and second.solve()
        b.fill(3)
        assert second.solve()
        np.testing.assert_array_equal(y.to_numpy(), np.full(8, 3))
        ti.reset()
        with pytest.raises(RuntimeError, match="retired"):
            second.solve()
