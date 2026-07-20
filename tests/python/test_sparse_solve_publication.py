import threading

import pytest

from taichi_forge.linalg._sparse_runtime_memory import (
    _graph_cache_memory_attribution,
)
from taichi_forge.linalg._sparse_solve_publication import (
    _SparseSolvePublication,
    _SparseSolvePublicationRegistry,
)


class _Array:
    def __init__(self, value=None):
        self.arr = [] if value is None else value


class _ImmediateSolver:
    def solve(self, program, solution, rhs):
        solution.append((program, rhs))


class _BlockingSolver:
    def __init__(self, started, finish):
        self._started = started
        self._finish = finish

    def solve(self, program, solution, rhs):
        self._started.set()
        assert self._finish.wait(timeout=2.0)
        solution.append((program, rhs))


class _GraphOperator:
    def __init__(self, stats):
        self._stats = dict(stats)

    def _debug_graph_cache_stats(self):
        return dict(self._stats)


class _CompiledGraph:
    def __init__(self, *snapshots):
        self._snapshots = tuple(dict(snapshot) for snapshot in snapshots)

    @property
    def _graph_stats(self):
        return [dict(snapshot) for snapshot in self._snapshots]


class _ProgramWorkspace:
    def _primitive_workspace_detailed_stats(self):
        total = {
            "reserved_bytes": 128,
            "peak_reserved_bytes": 256,
            "in_use_bytes": 0,
            "persistent_bytes": 16,
            "reclaimable_bytes": 112,
            "entries": 4,
        }
        return {
            "total": total,
            "groups": [
                {"backend": "cuda", "family": "ordering", "reserved_bytes": 32},
                {"backend": "cuda", "family": "scan", "reserved_bytes": 48},
                {"backend": "cuda", "family": "transform", "reserved_bytes": 48},
            ],
            "current_group_totals_match": True,
            "historical_peak_group_breakdown_available": False,
        }


def _publication(
    program,
    topology_version,
    solver=None,
    numeric_version=1,
    target=None,
    inverse=None,
):
    return _SparseSolvePublication(
        program=program,
        topology_version=topology_version,
        numeric_version=numeric_version,
        size=4,
        target=object() if target is None else target,
        inverse=object() if inverse is None else inverse,
        preconditioner=object(),
        solver=_ImmediateSolver() if solver is None else solver,
        numeric_publisher=None,
        target_operator_bytes=10,
        inverse_operator_bytes=20,
        solver_workspace_bytes=30,
        solver_workspace_materialized_bytes=0,
        build_peak_device_bytes=80,
    )


def _graph_stats(device_bytes, host_bytes, *, opaque, retained_leases):
    return {
        "known_persistent_device_argument_bytes": device_bytes,
        "known_deferred_host_argument_bytes": host_bytes,
        "retained_allocation_lease_count": retained_leases,
        "opaque_driver_runtime_state_present": opaque,
        "total_owned_device_bytes_reported": not opaque,
    }


def test_sparse_graph_cache_memory_attribution_deduplicates_graphs():
    exact = _CompiledGraph(
        _graph_stats(7, 2, opaque=False, retained_leases=1)
    )
    opaque = _CompiledGraph(
        _graph_stats(11, 3, opaque=True, retained_leases=2)
    )
    attribution = _graph_cache_memory_attribution(exact, exact, opaque)
    assert attribution["resources"] == {
        "graph_runtime_cache_known_device_argument_bytes": 18,
        "graph_runtime_cache_known_deferred_host_argument_bytes": 5,
        "graph_runtime_cache_retained_allocation_lease_count": 3,
        "graph_runtime_cache_object_count": 2,
        "graph_runtime_cache_opaque_cache_count": 1,
        "graph_runtime_cache_owned_device_bytes": None,
    }
    assert attribution["contract"] == {
        "graph_runtime_cache_bytes_reported": False,
        "graph_runtime_cache_in_explicit_capacity": False,
        "graph_runtime_cache_retained_allocation_leases_are_borrowed": True,
        "graph_runtime_cache_query_materializes_device_resources": False,
    }
    exact_only = _graph_cache_memory_attribution(exact)
    assert exact_only["resources"][
        "graph_runtime_cache_owned_device_bytes"
    ] == 7
    assert exact_only["contract"]["graph_runtime_cache_bytes_reported"]


def test_sparse_publication_attributes_target_inverse_and_program_caches_once():
    program = _ProgramWorkspace()
    target = _GraphOperator(
        _graph_stats(11, 3, opaque=True, retained_leases=2)
    )
    inverse = _GraphOperator(
        _graph_stats(17, 5, opaque=False, retained_leases=1)
    )
    publication = _publication(
        program,
        topology_version=1,
        target=target,
        inverse=inverse,
    )
    direct = publication.graph_cache_stats()
    assert direct == {
        "operator_cache_count": 2,
        "known_persistent_device_argument_bytes": 28,
        "known_deferred_host_argument_bytes": 8,
        "retained_allocation_lease_count": 3,
        "opaque_cache_count": 1,
        "opaque_driver_runtime_state_present": True,
        "total_owned_device_bytes_reported": False,
    }

    shared = _publication(
        program,
        topology_version=2,
        target=target,
        inverse=target,
    )
    assert shared.graph_cache_stats()["operator_cache_count"] == 1
    assert shared.graph_cache_stats()[
        "known_persistent_device_argument_bytes"
    ] == 11

    registry = _SparseSolvePublicationRegistry(program, capacity_bytes=256)
    assert _publish(registry, publication, 0)["published"]
    resources = registry.debug_runtime_stats()["resources"]
    assert resources["graph_runtime_cache_object_count"] == 2
    assert resources["graph_runtime_cache_known_device_argument_bytes"] == 28
    assert resources[
        "graph_runtime_cache_known_deferred_host_argument_bytes"
    ] == 8
    assert resources[
        "graph_runtime_cache_retained_allocation_lease_count"
    ] == 3
    assert resources["graph_runtime_cache_opaque_cache_count"] == 1
    assert resources["graph_runtime_cache_opaque_generation_count"] == 1
    assert resources["graph_runtime_cache_device_bytes"] is None
    assert resources["program_primitive_workspace_reserved_bytes"] == 128
    assert resources[
        "program_primitive_workspace_peak_reserved_bytes"
    ] == 256
    assert resources["program_primitive_workspace_in_use_bytes"] == 0
    assert resources["program_primitive_workspace_persistent_bytes"] == 16
    assert resources["program_primitive_workspace_reclaimable_bytes"] == 112
    assert resources["program_primitive_workspace_entries"] == 4
    assert resources[
        "program_sparse_relevant_workspace_reserved_bytes"
    ] == 80
    contract = registry.debug_runtime_stats()["contract"]
    assert contract["program_primitive_workspace_bytes_reported"]
    assert contract["program_primitive_workspace_current_groups_exact"]
    assert not contract["program_primitive_workspace_group_peak_reported"]
    assert not contract["program_primitive_workspace_in_explicit_capacity"]
    assert contract[
        "graph_runtime_cache_retained_allocation_leases_are_borrowed"
    ]
    assert not contract[
        "graph_runtime_cache_query_materializes_device_resources"
    ]
    assert contract[
        "graph_runtime_cache_opaque_generation_count_is_legacy_alias"
    ]

    old_lease = registry.acquire()
    replacement = _publication(
        program,
        topology_version=2,
        target=target,
        inverse=inverse,
    )
    assert _publish(registry, replacement, 1)["published"]
    overlap = registry.debug_runtime_stats()["resources"]
    assert overlap["graph_runtime_cache_object_count"] == 2
    assert overlap["graph_runtime_cache_known_device_argument_bytes"] == 28
    assert overlap["graph_runtime_cache_opaque_cache_count"] == 1
    old_lease.release()


def _publish(registry, publication, expected_generation):
    return registry.publish(
        expected_generation=expected_generation,
        topology_version=publication.topology_version,
        estimated_steady_device_bytes=publication.steady_device_bytes,
        estimated_build_peak_device_bytes=publication.build_peak_device_bytes,
        builder=lambda: publication,
    )


def _publish_numeric(registry, publication, expected_generation, builder=None):
    return registry.publish(
        expected_generation=expected_generation,
        topology_version=publication.topology_version,
        numeric_version=publication.numeric_version,
        estimated_steady_device_bytes=publication.steady_device_bytes,
        estimated_build_peak_device_bytes=publication.build_peak_device_bytes,
        builder=(lambda: publication) if builder is None else builder,
    )


def test_sparse_publication_swaps_same_topology_numeric_generations_atomically(
):
    program = object()
    registry = _SparseSolvePublicationRegistry(program, capacity_bytes=199)
    initial = _publication(program, topology_version=3, numeric_version=7)
    initial_result = _publish_numeric(registry, initial, 0)
    assert initial_result["published"]
    assert initial_result["numeric_version"] == 7
    old_lease = registry.acquire()
    assert old_lease.topology_version == 3
    assert old_lease.numeric_version == 7

    builder_calls = 0

    def must_not_build():
        nonlocal builder_calls
        builder_calls += 1
        raise AssertionError("version rejection must precede builder")

    same_version = registry.publish(
        expected_generation=1,
        topology_version=3,
        numeric_version=7,
        estimated_steady_device_bytes=initial.steady_device_bytes,
        estimated_build_peak_device_bytes=initial.build_peak_device_bytes,
        builder=must_not_build,
    )
    assert same_version["status"] == "numeric_version_not_monotonic"
    assert not same_version["builder_invoked"]
    assert builder_calls == 0

    numeric_eight = _publication(
        program, topology_version=3, numeric_version=8
    )

    def fail_build():
        raise RuntimeError("injected numeric generation failure")

    failed = _publish_numeric(registry, numeric_eight, 1, fail_build)
    assert failed["status"] == "build_failed"
    assert failed["builder_invoked"]
    assert "injected numeric generation failure" in failed["error"]
    assert registry.acquire().numeric_version == 7

    numeric_eight_result = _publish_numeric(registry, numeric_eight, 1)
    assert numeric_eight_result["published"]
    assert numeric_eight_result["generation"] == 2
    assert numeric_eight_result["topology_version"] == 3
    assert numeric_eight_result["numeric_version"] == 8
    current_lease = registry.acquire()
    assert current_lease.generation == 2
    assert current_lease.numeric_version == 8
    assert old_lease.generation == 1
    assert old_lease.numeric_version == 7

    old_solution = _Array()
    new_solution = _Array()
    rhs = _Array(value="rhs")
    old_lease.solve(old_solution, rhs)
    current_lease.solve(new_solution, rhs)
    assert old_solution.arr == [(program, "rhs")]
    assert new_solution.arr == [(program, "rhs")]

    numeric_nine = _publication(
        program, topology_version=3, numeric_version=9
    )
    retained_capacity = _publish_numeric(registry, numeric_nine, 2)
    assert retained_capacity["status"] == "capacity_overflow"
    assert not retained_capacity["builder_invoked"]
    assert registry.acquire().numeric_version == 8

    assert old_lease.release()
    numeric_nine_result = _publish_numeric(registry, numeric_nine, 2)
    assert numeric_nine_result["published"]
    assert numeric_nine_result["generation"] == 3
    newest_lease = registry.acquire()
    assert newest_lease.topology_version == 3
    assert newest_lease.numeric_version == 9

    stats = registry.debug_runtime_stats()
    assert stats["identity"] == {
        "generation": 3,
        "topology_version": 3,
        "numeric_version": 9,
    }
    assert stats["operations"]["publish_attempts"] == 6
    assert stats["operations"]["successful_publishes"] == 3
    assert stats["operations"]["rejected_publishes"] == 3
    assert stats["operations"]["build_failures"] == 1
    assert stats["contract"][
        "same_topology_numeric_generations_supported"
    ]
    current_lease.release()
    newest_lease.release()


def test_sparse_publication_release_waits_for_inflight_solve():
    program = object()
    started = threading.Event()
    finish = threading.Event()
    publication = _publication(
        program,
        topology_version=1,
        solver=_BlockingSolver(started, finish),
    )
    registry = _SparseSolvePublicationRegistry(program, capacity_bytes=256)
    assert _publish(registry, publication, expected_generation=0)["published"]
    lease = registry.acquire()
    solution = _Array()
    rhs = _Array(value="rhs")

    solve_thread = threading.Thread(target=lease.solve, args=(solution, rhs))
    solve_thread.start()
    assert started.wait(timeout=2.0)

    candidate = _publication(program, topology_version=2)
    builder_started = threading.Event()
    finish_builder = threading.Event()
    publish_result = []

    def builder():
        builder_started.set()
        assert finish_builder.wait(timeout=2.0)
        return candidate

    def publish():
        publish_result.append(
            registry.publish(
                expected_generation=1,
                topology_version=2,
                estimated_steady_device_bytes=candidate.steady_device_bytes,
                estimated_build_peak_device_bytes=(
                    candidate.build_peak_device_bytes
                ),
                builder=builder,
            )
        )

    publish_thread = threading.Thread(target=publish)
    publish_thread.start()
    assert builder_started.wait(timeout=2.0)
    finish_builder.set()
    publish_thread.join(timeout=2.0)
    assert not publish_thread.is_alive()
    assert publish_result[0]["published"]
    assert publish_result[0]["generation"] == 2

    release_result = []
    release_started = threading.Event()
    release_done = threading.Event()

    def release():
        release_started.set()
        release_result.append(lease.release())
        release_done.set()

    release_thread = threading.Thread(target=release)
    release_thread.start()
    assert release_started.wait(timeout=2.0)
    assert not release_done.wait(timeout=0.05)
    finish.set()
    solve_thread.join(timeout=2.0)
    release_thread.join(timeout=2.0)
    assert not solve_thread.is_alive()
    assert not release_thread.is_alive()
    assert release_result == [True]
    assert solution.arr == [(program, "rhs")]

    sentinel = _Array(value=["sentinel"])
    with pytest.raises(RuntimeError, match="lease was released"):
        lease.solve(sentinel, rhs)
    assert sentinel.arr == ["sentinel"]


def test_sparse_publication_acquire_continues_while_builder_is_blocked():
    program = object()
    registry = _SparseSolvePublicationRegistry(program, capacity_bytes=256)
    initial = _publication(program, topology_version=1)
    assert _publish(registry, initial, expected_generation=0)["published"]
    old_lease = registry.acquire()

    builder_started = threading.Event()
    finish_builder = threading.Event()
    candidate = _publication(program, topology_version=2)
    publish_result = []

    def builder():
        builder_started.set()
        assert finish_builder.wait(timeout=2.0)
        return candidate

    def publish():
        publish_result.append(
            registry.publish(
                expected_generation=1,
                topology_version=2,
                estimated_steady_device_bytes=candidate.steady_device_bytes,
                estimated_build_peak_device_bytes=(
                    candidate.build_peak_device_bytes
                ),
                builder=builder,
            )
        )

    publish_thread = threading.Thread(target=publish)
    publish_thread.start()
    assert builder_started.wait(timeout=2.0)
    during_build_lease = registry.acquire()
    assert during_build_lease.generation == 1
    assert during_build_lease.topology_version == 1

    finish_builder.set()
    publish_thread.join(timeout=2.0)
    assert not publish_thread.is_alive()
    assert publish_result[0]["published"]
    assert publish_result[0]["generation"] == 2
    new_lease = registry.acquire()
    assert new_lease.generation == 2
    assert new_lease.topology_version == 2
    assert old_lease.generation == 1

    stats = registry.debug_runtime_stats()
    assert stats["operations"] == {
        "publish_attempts": 2,
        "successful_publishes": 2,
        "rejected_publishes": 0,
        "build_failures": 0,
        "active_leases": 3,
    }
    assert stats["contract"]["publish_builders_serialized"]
    assert stats["contract"]["acquire_allowed_during_build"]
    assert stats["contract"]["lease_release_waits_for_solve"]

    old_lease.release()
    during_build_lease.release()
    new_lease.release()
