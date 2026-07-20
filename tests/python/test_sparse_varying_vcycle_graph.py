import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.impl import get_runtime
from taichi_forge.linalg._sparse_varying_vcycle_graph import (
    _SparseRecursiveVaryingBlockVcycleGraphPlan,
)
from tests import test_utils
from tests.sparse_runtime_stats import (
    assert_sparse_graph_cache_attribution,
)
from tests.python.test_sparse_varying_bsr_hierarchy import (
    _dense_first_transfer,
    _dense_second_transfer,
    _f32_array,
    _snapshot_dense,
)
from tests.python.test_sparse_varying_vcycle_inputs import (
    _hierarchy,
    _inputs,
)


def _block_inverse_apply(inverse, vector, block_rows, block_size):
    blocks = inverse.reshape(block_rows, block_size, block_size).astype(
        np.float64
    )
    vector_blocks = np.asarray(vector, dtype=np.float64).reshape(
        block_rows, block_size
    )
    return np.einsum("bij,bj->bi", blocks, vector_blocks).reshape(-1)


def _varying_vcycle_oracle(hierarchy, numeric_sources, rhs):
    inverse_numpy = numeric_sources[3]
    damping_numpy = numeric_sources[4]
    bottom_inverse_numpy = numeric_sources[5]
    matrices = [_snapshot_dense(level) for level in hierarchy._levels]
    transfers = (
        _dense_first_transfer().astype(np.float64),
        _dense_second_transfer().astype(np.float64),
    )

    def apply(level_index, level_rhs):
        if level_index + 1 == hierarchy.level_count:
            bottom_size = hierarchy._levels[-1].rows
            bottom_inverse = bottom_inverse_numpy.reshape(
                bottom_size, bottom_size
            ).astype(np.float64)
            return bottom_inverse @ level_rhs
        level = hierarchy._levels[level_index]
        damping = float(damping_numpy[level_index])
        pre = damping * _block_inverse_apply(
            inverse_numpy[level_index],
            level_rhs,
            level.block_rows,
            level.block_size,
        )
        residual = level_rhs - matrices[level_index] @ pre
        coarse_rhs = transfers[level_index].T @ residual
        coarse_solution = apply(level_index + 1, coarse_rhs)
        corrected = pre + transfers[level_index] @ coarse_solution
        post_residual = level_rhs - matrices[level_index] @ corrected
        return corrected + damping * _block_inverse_apply(
            inverse_numpy[level_index],
            post_residual,
            level.block_rows,
            level.block_size,
        )

    return apply(0, np.asarray(rhs, dtype=np.float64))


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_recursive_varying_block_vcycle_matches_host_oracle():
    hierarchy = _hierarchy()
    inputs, numeric_sources = _inputs(hierarchy)
    with pytest.raises(RuntimeError, match="capacity overflow"):
        _SparseRecursiveVaryingBlockVcycleGraphPlan(
            inputs, explicit_array_capacity_bytes=7283
        )
    plan = _SparseRecursiveVaryingBlockVcycleGraphPlan(
        inputs, explicit_array_capacity_bytes=7284
    )

    rhs_host = np.linspace(-0.65, 0.95, 15, dtype=np.float32)
    expected = _varying_vcycle_oracle(
        hierarchy, numeric_sources, rhs_host
    )
    rhs = _f32_array(rhs_host)
    output = ti.ndarray(ti.f32, shape=15)
    output.fill(0.0)
    plan.apply(rhs, output)
    ti.sync()
    np.testing.assert_allclose(
        output.to_numpy(), expected, rtol=5e-5, atol=2e-3
    )
    with pytest.raises(RuntimeError, match="alias"):
        plan.apply(rhs, rhs)

    native = plan.create_native_operator()
    with pytest.raises(RuntimeError, match="at most one"):
        plan.create_native_operator()
    native_output = ti.ndarray(ti.f32, shape=15)
    native_output.fill(0.0)
    native.spmv(get_runtime().prog, rhs.arr, native_output.arr)
    ti.sync()
    np.testing.assert_allclose(
        native_output.to_numpy(), expected, rtol=5e-5, atol=2e-3
    )
    native_stats = native._debug_runtime_stats()
    assert native_stats["resources"]["pattern_reserved_bytes"] == 316
    assert native_stats["resources"]["values_reserved_bytes"] == 2888
    assert native_stats["resources"][
        "spmv_workspace_reserved_bytes"
    ] == 360
    assert native_stats["resources"][
        "operator_owned_reserved_bytes"
    ] == 3564

    stats = plan.debug_runtime_stats()
    assert stats["identity"] == {
        "backend_family": hierarchy._backend,
        "method": "recursive_symmetric_varying_block_vcycle",
        "size": 15,
        "level_count": 3,
        "level_block_rows": (5, 2, 1),
        "level_block_sizes": (3, 6, 6),
        "level_scalar_rows": (15, 12, 6),
        "topology_version": 43,
        "numeric_version": 47,
    }
    assert stats["operations"] == {
        "apply_calls": 1,
        "rejected_apply_calls": 1,
        "graph_node_count": 1,
        "graph_dispatch_count": 13,
        "kernel_dispatches_per_apply": 13,
        "pre_restrict_dispatches_per_level": 3,
        "post_dispatches_per_level": 3,
        "host_graph_submissions_per_apply": 1,
        "explicit_apply_host_synchronizations": 0,
        "native_operator_publishes": 1,
    }
    assert assert_sparse_graph_cache_attribution(
        stats, expected_cache_object_count=1
    ) == {
        "borrowed_execution_inputs_reserved_bytes": 3360,
        "topology_argument_reserved_bytes": 316,
        "numeric_argument_reserved_bytes": 2888,
        "plan_workspace_reserved_bytes": 360,
        "native_operator_reserved_bytes": 3564,
        "build_peak_explicit_array_bytes": 7284,
        "explicit_array_capacity_bytes": 7284,
    }
    assert stats["transfers"] == {
        "device_to_host_bytes": 0,
        "device_to_device_bytes": 3564,
        "device_kernel_workspace_initialization_bytes": 360,
        "device_payload_readback_bytes": 0,
    }
    assert stats["contract"][
        "deterministic_transpose_schedules_bound_directly"
    ]
    assert stats["contract"][
        "fine_applied_workspace_reused_for_post_spmv"
    ]
    assert stats["contract"]["all_dispatches_share_one_compiled_graph"]
    assert not stats["contract"]["compiled_transfer_subgraphs_nested"]
    assert not stats["contract"]["floating_atomic_restriction_required"]
    assert not stats["contract"]["pcg_constructed"]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_recursive_varying_block_vcycle_rejects_after_reset():
    with pytest.raises(RuntimeError, match="requires varying block"):
        _SparseRecursiveVaryingBlockVcycleGraphPlan(
            object(), explicit_array_capacity_bytes=7284
        )
    hierarchy = _hierarchy()
    inputs, _ = _inputs(hierarchy)
    plan = _SparseRecursiveVaryingBlockVcycleGraphPlan(
        inputs, explicit_array_capacity_bytes=7284
    )
    source = ti.ndarray(ti.f32, shape=15)
    destination = ti.ndarray(ti.f32, shape=15)
    ti.reset()
    with pytest.raises(RuntimeError, match="runtime has been reset"):
        plan.debug_runtime_stats()
    with pytest.raises(RuntimeError, match="runtime has been reset"):
        plan.apply(source, destination)
