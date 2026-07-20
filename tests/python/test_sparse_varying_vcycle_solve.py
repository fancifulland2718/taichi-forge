import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.impl import get_runtime
from taichi_forge.linalg._sparse_solve_publication import (
    _SparseSolvePublicationRegistry,
)
from taichi_forge.linalg._sparse_varying_vcycle_solve import (
    _SparseVaryingBlockVcycleSolvePublicationBuilder,
)
from tests import test_utils
from tests.python.test_sparse_varying_bsr_hierarchy import (
    _f32_array,
    _snapshot_dense,
)
from tests.python.test_sparse_varying_vcycle_inputs import (
    _hierarchy,
    _inputs,
)


def _solve(publication, hierarchy, exact_host):
    fine_dense = _snapshot_dense(hierarchy._levels[0])
    rhs = _f32_array((fine_dense @ exact_host).astype(np.float32))
    solution = ti.ndarray(ti.f32, shape=len(exact_host))
    solution.fill(0.0)
    publication.solve(solution, rhs)
    ti.sync()
    return solution.to_numpy(), publication.solver.is_success()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_varying_block_vcycle_pcg_publishes_immutable_generation():
    hierarchy = _hierarchy()
    inputs, _ = _inputs(hierarchy)
    backend = ti.lang.impl.current_cfg().arch
    expected_solver_bytes = 280 if backend == ti.vulkan else 240
    expected_steady_bytes = 4868 if backend == ti.vulkan else 4828
    expected_build_peak = 8588 if backend == ti.vulkan else 8548
    with pytest.raises(RuntimeError, match="capacity overflow"):
        _SparseVaryingBlockVcycleSolvePublicationBuilder(
            inputs,
            max_iterations=64,
            absolute_tolerance=1e-5,
            explicit_array_capacity_bytes=expected_build_peak - 1,
        )
    builder = _SparseVaryingBlockVcycleSolvePublicationBuilder(
        inputs,
        max_iterations=64,
        absolute_tolerance=1e-5,
        explicit_array_capacity_bytes=expected_build_peak,
    )
    assert builder.estimated_steady_device_bytes == expected_steady_bytes
    assert builder.estimated_build_peak_device_bytes == expected_build_peak
    publication = builder.build()

    exact = np.linspace(-0.75, 1.05, 15, dtype=np.float32)
    solution, success = _solve(publication, hierarchy, exact)
    assert success
    np.testing.assert_allclose(solution, exact, rtol=0.0, atol=6e-4)
    solver_stats = publication.solver._debug_runtime_stats()
    expected_method = (
        "pcg_compiled_graph_bounded_masked_probe"
        if backend == ti.vulkan
        else "pcg_compiled_graph"
    )
    assert solver_stats["identity"]["method"] == expected_method
    assert solver_stats["identity"]["preconditioner_method"] == (
        "compiled_graph_inverse_apply"
    )
    target_stats = publication.target._debug_runtime_stats()
    inverse_stats = publication.inverse._debug_runtime_stats()
    assert target_stats["identity"]["pattern_version"] == 7
    assert target_stats["identity"]["numeric_version"] == 11
    assert target_stats["resources"]["operator_owned_reserved_bytes"] == 1024
    assert inverse_stats["identity"]["pattern_version"] == 43
    assert inverse_stats["identity"]["numeric_version"] == 47
    assert inverse_stats["resources"][
        "operator_owned_reserved_bytes"
    ] == 3564
    graph_cache = publication.graph_cache_stats()
    assert graph_cache["operator_cache_count"] == 2
    if backend == ti.cpu:
        assert graph_cache["known_persistent_device_argument_bytes"] == 0
        assert graph_cache["opaque_cache_count"] == 0
        assert graph_cache["total_owned_device_bytes_reported"]
    else:
        assert graph_cache["known_persistent_device_argument_bytes"] > 0
        expected_opaque_caches = 2 if backend == ti.cuda else 1
        assert graph_cache["opaque_cache_count"] == expected_opaque_caches
        assert not graph_cache["total_owned_device_bytes_reported"]

    stats = builder.debug_runtime_stats()
    assert stats["identity"] == {
        "backend_family": hierarchy._backend,
        "method": "immutable_varying_block_vcycle_graph_pcg",
        "block_rows": 5,
        "block_size": 3,
        "size": 15,
        "level_block_sizes": (3, 6, 6),
        "topology_version": 43,
        "numeric_version": 47,
        "target_component_topology_version": 7,
        "target_component_numeric_version": 11,
        "max_iterations": 64,
        "absolute_tolerance": 1e-5,
    }
    assert stats["operations"] == {
        "build_attempts": 1,
        "successful_builds": 1,
        "failed_builds": 0,
    }
    resources = stats["resources"]
    assert resources["borrowed_execution_inputs_reserved_bytes"] == 3360
    assert resources["target_operator_reservation_bytes"] == 1024
    assert resources["inverse_operator_reservation_bytes"] == 3564
    assert resources["inverse_workspace_reservation_bytes"] == 360
    assert resources["solver_workspace_reservation_bytes"] == (
        expected_solver_bytes
    )
    assert resources["estimated_steady_device_bytes"] == expected_steady_bytes
    assert resources["estimated_build_peak_device_bytes"] == expected_build_peak
    expected_materialized = 0 if backend == ti.cuda else expected_solver_bytes
    assert resources["last_report"][
        "solver_workspace_materialized_bytes"
    ] == expected_materialized
    assert resources["live_publication_count"] == 1
    assert resources["live_publication_steady_device_bytes"] == (
        expected_steady_bytes
    )
    assert stats["contract"][
        "publication_identity_uses_hierarchy_generation"
    ]
    assert stats["contract"][
        "component_operator_lineage_versions_preserved"
    ]
    assert stats["contract"]["compiled_preconditioner_and_pcg_reused"]
    assert not stats["contract"]["host_inversion_performed"]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_varying_block_vcycle_registry_keeps_old_generation():
    first_hierarchy = _hierarchy(numeric_version=47)
    first_inputs, _ = _inputs(first_hierarchy)
    first_builder = _SparseVaryingBlockVcycleSolvePublicationBuilder(
        first_inputs,
        max_iterations=64,
        absolute_tolerance=1e-5,
        explicit_array_capacity_bytes=8548,
    )
    registry = _SparseSolvePublicationRegistry(
        get_runtime().prog, capacity_bytes=13376
    )
    first_result = registry.publish(
        expected_generation=0,
        topology_version=43,
        numeric_version=47,
        estimated_steady_device_bytes=(
            first_builder.estimated_steady_device_bytes
        ),
        estimated_build_peak_device_bytes=(
            first_builder.estimated_build_peak_device_bytes
        ),
        builder=first_builder.build,
    )
    assert first_result["published"]
    old_lease = registry.acquire()

    second_hierarchy = _hierarchy(numeric_version=48)
    second_inputs, _ = _inputs(second_hierarchy)
    second_builder = _SparseVaryingBlockVcycleSolvePublicationBuilder(
        second_inputs,
        max_iterations=64,
        absolute_tolerance=1e-5,
        explicit_array_capacity_bytes=8548,
    )
    second_result = registry.publish(
        expected_generation=1,
        topology_version=43,
        numeric_version=48,
        estimated_steady_device_bytes=(
            second_builder.estimated_steady_device_bytes
        ),
        estimated_build_peak_device_bytes=(
            second_builder.estimated_build_peak_device_bytes
        ),
        builder=second_builder.build,
    )
    assert second_result == {
        "published": True,
        "status": "published",
        "builder_invoked": True,
        "generation": 2,
        "topology_version": 43,
        "numeric_version": 48,
        "steady_device_bytes": 4828,
        "build_peak_device_bytes": 8548,
        "old_plus_new_steady_device_bytes": 9656,
    }
    new_lease = registry.acquire()
    exact = np.linspace(-0.55, 0.85, 15, dtype=np.float32)
    old_solution = ti.ndarray(ti.f32, shape=15)
    new_solution = ti.ndarray(ti.f32, shape=15)
    rhs = _f32_array(
        (_snapshot_dense(first_hierarchy._levels[0]) @ exact).astype(
            np.float32
        )
    )
    old_solution.fill(0.0)
    new_solution.fill(0.0)
    old_lease.solve(old_solution, rhs)
    new_lease.solve(new_solution, rhs)
    ti.sync()
    np.testing.assert_allclose(
        old_solution.to_numpy(), exact, rtol=0.0, atol=6e-4
    )
    np.testing.assert_allclose(
        new_solution.to_numpy(), exact, rtol=0.0, atol=6e-4
    )
    overlap_stats = registry.debug_runtime_stats()
    assert overlap_stats["resources"][
        "publish_overlap_peak_device_bytes"
    ] == 13376
    assert overlap_stats["resources"][
        "live_generation_steady_device_bytes"
    ] == 9656
    workspace = overlap_stats["resources"]
    assert workspace["program_primitive_workspace_reserved_bytes"] >= 0
    assert workspace["program_primitive_workspace_peak_reserved_bytes"] >= (
        workspace["program_primitive_workspace_reserved_bytes"]
    )
    assert workspace[
        "program_sparse_relevant_workspace_reserved_bytes"
    ] <= workspace["program_primitive_workspace_reserved_bytes"]
    assert isinstance(workspace["program_primitive_workspace_groups"], list)
    assert overlap_stats["contract"][
        "program_primitive_workspace_bytes_reported"
    ]
    assert overlap_stats["contract"][
        "program_primitive_workspace_current_groups_exact"
    ]
    assert not overlap_stats["contract"][
        "program_primitive_workspace_group_peak_reported"
    ]
    assert not overlap_stats["contract"][
        "program_primitive_workspace_in_explicit_capacity"
    ]
    old_lease.release()

    def fail_build():
        raise RuntimeError("injected varying generation failure")

    failed = registry.publish(
        expected_generation=2,
        topology_version=43,
        numeric_version=49,
        estimated_steady_device_bytes=4828,
        estimated_build_peak_device_bytes=8548,
        builder=fail_build,
    )
    assert not failed["published"]
    assert failed["status"] == "build_failed"
    current_after_failure = registry.acquire()
    assert current_after_failure.numeric_version == 48
    current_after_failure.release()

    tight_registry = _SparseSolvePublicationRegistry(
        get_runtime().prog, capacity_bytes=13375
    )
    tight_first = tight_registry.publish(
        expected_generation=0,
        topology_version=43,
        numeric_version=47,
        estimated_steady_device_bytes=4828,
        estimated_build_peak_device_bytes=8548,
        builder=first_builder.build,
    )
    assert tight_first["published"]
    tight_registry.acquire()
    tight_second = tight_registry.publish(
        expected_generation=1,
        topology_version=43,
        numeric_version=48,
        estimated_steady_device_bytes=4828,
        estimated_build_peak_device_bytes=8548,
        builder=second_builder.build,
    )
    assert tight_second == {
        "published": False,
        "status": "capacity_overflow",
        "builder_invoked": False,
    }
    new_lease.release()
