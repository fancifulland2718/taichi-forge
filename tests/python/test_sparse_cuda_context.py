"""Fixed sparse resources must own their CUDA context at public boundaries."""

import gc
import weakref

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from tests.python.test_external_cuda_context import _ambient_context


def _array(values, dtype):
    result = ti.ndarray(dtype, shape=np.asarray(values).size)
    result.from_numpy(np.asarray(values, dtype=np.int32 if dtype == ti.i32 else np.float32).reshape(-1))
    return result


@pytest.mark.parametrize("mode", ["empty", "foreign"])
@pytest.mark.parametrize("layout", ["csr", "bsr"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fixed_sparse_context_construction_replay_refresh_and_retirement(mode, layout):
    dense = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    exact = np.asarray([1.25, -0.75], dtype=np.float32)
    with _ambient_context(mode) as (current, ambient, _):

        def checked(call, *args, **kwargs):
            result = call(*args, **kwargs)
            assert current() == ambient
            return result

        if layout == "csr":
            offsets = _array([0, 2, 4], ti.i32)
            columns = _array([0, 1, 0, 1], ti.i32)
            bad_columns = _array([0, 2, 0, 1], ti.i32)
            factory, shape = ti.linalg.SparsePattern.csr, (2, 2)
        else:
            offsets = _array([0, 1], ti.i32)
            columns = _array([0], ti.i32)
            bad_columns = _array([1], ti.i32)
            factory, shape = ti.linalg.SparsePattern.bsr, (1, 1, 2)
        assert current() == ambient

        # This failure reads actual device indices, before allocating the
        # retained pattern. A failed construction must restore the caller.
        with pytest.raises(RuntimeError, match="column"):
            factory(*shape, offsets, bad_columns)
        assert current() == ambient
        pattern = checked(factory, *shape, offsets, columns)
        values = _array(dense, ti.f32)
        wrong_values = _array([1.0], ti.f32)
        with pytest.raises(RuntimeError, match="entries"):
            pattern.matrix(wrong_values)
        assert current() == ambient
        matrix = checked(pattern.matrix, values)
        vector = _array(exact, ti.f32)
        product = checked(matrix.spmv, vector, method="provider")
        np.testing.assert_allclose(checked(product.to_numpy), dense @ exact, rtol=1e-6)
        operator = checked(
            ti.linalg.LinearOperator.from_sparse_matrix,
            matrix,
            traits=ti.linalg.OperatorTraits(self_adjoint=True, positive_definite=True, singular=False),
        )
        plan = checked(
            ti.linalg.experimental.SolvePlan,
            operator,
            method="minres",
            preconditioner="jacobi" if layout == "csr" else "block_jacobi",
            max_iterations=8,
            atol=1e-6,
            rtol=1e-6,
            execution_policy="host_check_every_k",
            check_interval=4,
        )
        rhs, output = _array(dense @ exact, ti.f32), _array([0.0, 0.0], ti.f32)
        for _ in range(2):
            checked(output.fill, 0)
            result = checked(plan.solve, rhs, out=output)
            assert result.converged
            np.testing.assert_allclose(checked(output.to_numpy), exact, rtol=4e-5, atol=4e-5)
        statistics = plan.statistics()["operations"]
        assert statistics["solver_chunk_builds"] > 0
        assert statistics["solver_chunk_replays"] > 0

        # Refresh the same captured solver, then reject an invalid diagonal
        # and recover using the same owner and caller context.
        checked(values.from_numpy, (dense * 2).reshape(-1))
        checked(matrix.update_values, values)
        checked(output.fill, 0)
        assert checked(plan.solve, rhs, out=output).converged
        np.testing.assert_allclose(checked(output.to_numpy), exact / 2, rtol=4e-5, atol=4e-5)
        checked(values.fill, 0)
        checked(matrix.update_values, values)
        with pytest.raises(RuntimeError, match="zero|positive definite"):
            plan.solve(rhs, out=output)
        assert current() == ambient
        checked(values.from_numpy, dense.reshape(-1))
        checked(matrix.update_values, values)
        checked(output.fill, 0)
        assert checked(plan.solve, rhs, out=output).converged
        np.testing.assert_allclose(checked(output.to_numpy), exact, rtol=4e-5, atol=4e-5)

        checked(ti.sync)
        retired = [weakref.ref(owner) for owner in (plan, operator, matrix, pattern)]
        del plan, operator, matrix, pattern
        gc.collect()
        assert all(owner() is None for owner in retired)
        assert current() == ambient
