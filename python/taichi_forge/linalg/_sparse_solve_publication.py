"""Private transactional publication for prebuilt sparse solve plans.

This module owns no hierarchy builder and selects no solver algorithm.  It
publishes already constructed target/inverse/solver generations with explicit
resource reservations, so fixed CSR/BSR, matrix-free and multilevel providers
can share the same failure and retirement contract without entering the public
``SparseCG`` surface prematurely.
"""

import threading
import weakref

from ._sparse_runtime_memory import (
    _aggregate_graph_cache_snapshots,
    _program_workspace_attribution,
)


def _aggregate_graph_cache_stats(*operators):
    snapshots = []
    seen = set()
    for operator in operators:
        if id(operator) in seen:
            continue
        seen.add(id(operator))
        debug = getattr(operator, "_debug_graph_cache_stats", None)
        if debug is not None:
            snapshots.append(debug())
    aggregate = _aggregate_graph_cache_snapshots(snapshots)
    return {
        "operator_cache_count": aggregate["cache_object_count"],
        "known_persistent_device_argument_bytes": aggregate[
            "known_persistent_device_argument_bytes"
        ],
        "known_deferred_host_argument_bytes": aggregate[
            "known_deferred_host_argument_bytes"
        ],
        "retained_allocation_lease_count": aggregate[
            "retained_allocation_lease_count"
        ],
        "opaque_cache_count": aggregate["opaque_cache_count"],
        "opaque_driver_runtime_state_present": (
            aggregate["opaque_cache_count"] != 0
        ),
        "total_owned_device_bytes_reported": aggregate[
            "total_owned_device_bytes_reported"
        ],
    }


class _SparseSolvePublication:
    """Self-contained target/inverse/solver generation."""

    def __init__(
        self,
        *,
        program,
        topology_version,
        numeric_version,
        size,
        target,
        inverse,
        preconditioner,
        solver,
        numeric_publisher,
        target_operator_bytes,
        inverse_operator_bytes,
        solver_workspace_bytes,
        solver_workspace_materialized_bytes,
        build_peak_device_bytes,
    ):
        if topology_version <= 0 or numeric_version <= 0 or size <= 0:
            raise ValueError("sparse publication versions and size must be positive")
        self.program = program
        self.topology_version = topology_version
        self.numeric_version = numeric_version
        self.size = size
        self.target = target
        self.inverse = inverse
        self.preconditioner = preconditioner
        self.solver = solver
        self.numeric_publisher = numeric_publisher
        self.target_operator_bytes = int(target_operator_bytes)
        self.inverse_operator_bytes = int(inverse_operator_bytes)
        self.solver_workspace_bytes = int(solver_workspace_bytes)
        self.solver_workspace_materialized_bytes = int(
            solver_workspace_materialized_bytes
        )
        if min(
            self.target_operator_bytes,
            self.inverse_operator_bytes,
            self.solver_workspace_bytes,
            self.solver_workspace_materialized_bytes,
        ) < 0:
            raise ValueError("sparse publication byte counts cannot be negative")
        if self.solver_workspace_materialized_bytes > self.solver_workspace_bytes:
            raise ValueError(
                "materialized solver workspace cannot exceed its reservation"
            )
        publisher_stats = (
            {
                "host_topology_metadata_bytes": 0,
                "device_reserved_bytes": 0,
            }
            if numeric_publisher is None
            else numeric_publisher.debug_runtime_stats()
        )
        self.publisher_host_metadata_bytes = int(
            publisher_stats["host_topology_metadata_bytes"]
        )
        self.publisher_device_reserved_bytes = int(
            publisher_stats["device_reserved_bytes"]
        )
        if min(
            self.publisher_host_metadata_bytes,
            self.publisher_device_reserved_bytes,
        ) < 0:
            raise ValueError("sparse publisher byte counts cannot be negative")
        self.steady_device_bytes = (
            self.target_operator_bytes
            + self.inverse_operator_bytes
            + self.solver_workspace_bytes
            + self.publisher_device_reserved_bytes
        )
        self.build_peak_device_bytes = int(build_peak_device_bytes)
        if self.build_peak_device_bytes < self.steady_device_bytes:
            raise ValueError("sparse build peak cannot be smaller than steady bytes")

    def solve(self, solution, rhs):
        solution_core = getattr(solution, "arr", solution)
        rhs_core = getattr(rhs, "arr", rhs)
        self.solver.solve(self.program, solution_core, rhs_core)

    def graph_cache_stats(self):
        return _aggregate_graph_cache_stats(self.target, self.inverse)


class _SparseSolvePublicationLease:
    """One generation lease; release waits for an in-progress solve."""

    def __init__(self, publication, generation):
        self._publication = publication
        self._lock = threading.Lock()
        self.generation = generation
        self.topology_version = publication.topology_version
        self.numeric_version = publication.numeric_version

    @property
    def released(self):
        with self._lock:
            return self._publication is None

    def _snapshot_publication(self):
        # A GIL-protected pointer read is enough for registry accounting.
        # Racing with release may conservatively retain/count one old snapshot;
        # it cannot omit a publication while solve holds the lease lock.
        return self._publication

    def solve(self, solution, rhs):
        with self._lock:
            if self._publication is None:
                raise RuntimeError(
                    "sparse publication lease was released; output was not mutated"
                )
            self._publication.solve(solution, rhs)

    def release(self):
        with self._lock:
            publication_was_live = self._publication is not None
            self._publication = None
            return publication_was_live

    def __enter__(self):
        if self.released:
            raise RuntimeError("cannot enter a released sparse publication lease")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False


class _SparseSolvePublicationRegistry:
    """Thread-safe generation publication with serialized off-lock builders."""

    def __init__(self, program, capacity_bytes):
        if capacity_bytes <= 0:
            raise ValueError("sparse publication capacity must be positive")
        self._program = program
        self._capacity_bytes = int(capacity_bytes)
        self._generation = 0
        self._current = None
        self._leases = weakref.WeakSet()
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._publish_attempts = 0
        self._successful_publishes = 0
        self._rejected_publishes = 0
        self._build_failures = 0
        self._publish_overlap_peak_device_bytes = 0
        self._publish_overlap_peak_publisher_host_bytes = 0

    def _live_publications_locked(self):
        publications = {}
        if self._current is not None:
            publications[id(self._current)] = self._current
        for lease in self._leases:
            publication = lease._snapshot_publication()
            if publication is not None:
                publications[id(publication)] = publication
        return list(publications.values())

    def _reject_locked(self, status, *, builder_invoked, error=None):
        self._rejected_publishes += 1
        result = {
            "published": False,
            "status": status,
            "builder_invoked": builder_invoked,
        }
        if error is not None:
            result["error"] = error
        return result

    def _preflight_locked(
        self,
        *,
        expected_generation,
        topology_version,
        estimated_steady_device_bytes,
        estimated_build_peak_device_bytes,
        numeric_version=None,
    ):
        if expected_generation != self._generation:
            return self._reject_locked(
                "generation_mismatch", builder_invoked=False
            )
        estimated_steady_device_bytes = int(estimated_steady_device_bytes)
        estimated_build_peak_device_bytes = int(
            estimated_build_peak_device_bytes
        )
        if (
            estimated_steady_device_bytes <= 0
            or estimated_build_peak_device_bytes < estimated_steady_device_bytes
        ):
            return self._reject_locked(
                "invalid_resource_estimate", builder_invoked=False
            )
        current_topology_version = (
            0 if self._current is None else self._current.topology_version
        )
        current_numeric_version = (
            0 if self._current is None else self._current.numeric_version
        )
        if numeric_version is None:
            if topology_version <= current_topology_version:
                return self._reject_locked(
                    "topology_version_not_monotonic",
                    builder_invoked=False,
                )
        else:
            numeric_version = int(numeric_version)
            if topology_version <= 0 or numeric_version <= 0:
                return self._reject_locked(
                    "invalid_publication_version", builder_invoked=False
                )
            if topology_version < current_topology_version:
                return self._reject_locked(
                    "topology_version_not_monotonic",
                    builder_invoked=False,
                )
            if (
                topology_version == current_topology_version
                and numeric_version <= current_numeric_version
            ):
                return self._reject_locked(
                    "numeric_version_not_monotonic",
                    builder_invoked=False,
                )
        live_publications = self._live_publications_locked()
        live_steady_bytes = sum(
            publication.steady_device_bytes
            for publication in live_publications
        )
        if (
            estimated_steady_device_bytes > self._capacity_bytes
            or live_steady_bytes + estimated_build_peak_device_bytes
            > self._capacity_bytes
        ):
            return self._reject_locked(
                "capacity_overflow", builder_invoked=False
            )
        return {
            "live_steady_bytes": live_steady_bytes,
            "estimated_steady_device_bytes": estimated_steady_device_bytes,
            "estimated_build_peak_device_bytes": (
                estimated_build_peak_device_bytes
            ),
        }

    def publish(
        self,
        *,
        expected_generation,
        topology_version,
        estimated_steady_device_bytes,
        estimated_build_peak_device_bytes,
        builder,
        numeric_version=None,
    ):
        # One builder per registry protects Program/Graph construction without
        # blocking acquire() or solves on the current self-contained snapshot.
        with self._build_lock:
            with self._lock:
                self._publish_attempts += 1
                preflight = self._preflight_locked(
                    expected_generation=expected_generation,
                    topology_version=topology_version,
                    estimated_steady_device_bytes=(
                        estimated_steady_device_bytes
                    ),
                    estimated_build_peak_device_bytes=(
                        estimated_build_peak_device_bytes
                    ),
                    numeric_version=numeric_version,
                )
                if "published" in preflight:
                    return preflight
            try:
                candidate = builder()
            except Exception as exc:
                with self._lock:
                    self._build_failures += 1
                    return self._reject_locked(
                        "build_failed",
                        builder_invoked=True,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            with self._lock:
                if expected_generation != self._generation:
                    return self._reject_locked(
                        "generation_changed_during_build",
                        builder_invoked=True,
                    )
                live_publications = self._live_publications_locked()
                live_steady_bytes = sum(
                    publication.steady_device_bytes
                    for publication in live_publications
                )
                live_publisher_host_bytes = sum(
                    publication.publisher_host_metadata_bytes
                    for publication in live_publications
                )
                if (
                    not isinstance(candidate, _SparseSolvePublication)
                    or candidate.program is not self._program
                    or candidate.topology_version != topology_version
                    or (
                        numeric_version is not None
                        and candidate.numeric_version != numeric_version
                    )
                    or candidate.steady_device_bytes > self._capacity_bytes
                    or candidate.steady_device_bytes
                    > preflight["estimated_steady_device_bytes"]
                    or candidate.build_peak_device_bytes
                    > preflight["estimated_build_peak_device_bytes"]
                    or live_steady_bytes + candidate.build_peak_device_bytes
                    > self._capacity_bytes
                ):
                    return self._reject_locked(
                        "candidate_contract_rejected", builder_invoked=True
                    )
                self._publish_overlap_peak_device_bytes = max(
                    self._publish_overlap_peak_device_bytes,
                    live_steady_bytes + candidate.build_peak_device_bytes,
                )
                self._publish_overlap_peak_publisher_host_bytes = max(
                    self._publish_overlap_peak_publisher_host_bytes,
                    live_publisher_host_bytes
                    + candidate.publisher_host_metadata_bytes,
                )
                self._current = candidate
                self._generation += 1
                self._successful_publishes += 1
                return {
                    "published": True,
                    "status": "published",
                    "builder_invoked": True,
                    "generation": self._generation,
                    "topology_version": topology_version,
                    "numeric_version": candidate.numeric_version,
                    "steady_device_bytes": candidate.steady_device_bytes,
                    "build_peak_device_bytes": (
                        candidate.build_peak_device_bytes
                    ),
                    "old_plus_new_steady_device_bytes": (
                        live_steady_bytes + candidate.steady_device_bytes
                    ),
                }

    def acquire(self):
        with self._lock:
            if self._current is None:
                raise RuntimeError("sparse publication registry has no current plan")
            lease = _SparseSolvePublicationLease(
                self._current, self._generation
            )
            self._leases.add(lease)
            return lease

    def debug_runtime_stats(self):
        with self._lock:
            active_leases = [
                lease
                for lease in self._leases
                if lease._snapshot_publication() is not None
            ]
            live_publications = self._live_publications_locked()
            current = self._current
            identity = {
                "generation": self._generation,
                "topology_version": (
                    0 if current is None else current.topology_version
                ),
                "numeric_version": (
                    0 if current is None else current.numeric_version
                ),
            }
            operations = {
                "publish_attempts": self._publish_attempts,
                "successful_publishes": self._successful_publishes,
                "rejected_publishes": self._rejected_publishes,
                "build_failures": self._build_failures,
                "active_leases": len(active_leases),
            }
            peak_device_bytes = self._publish_overlap_peak_device_bytes
            peak_publisher_host_bytes = (
                self._publish_overlap_peak_publisher_host_bytes
            )
        retired_publications = [
            publication
            for publication in live_publications
            if publication is not current
        ]
        graph_cache_stats = _aggregate_graph_cache_stats(
            *(
                operator
                for publication in live_publications
                for operator in (publication.target, publication.inverse)
            )
        )
        graph_cache_device_bytes_complete = graph_cache_stats[
            "total_owned_device_bytes_reported"
        ]
        graph_cache_known_device_bytes = graph_cache_stats[
            "known_persistent_device_argument_bytes"
        ]
        graph_cache_opaque_count = graph_cache_stats[
            "opaque_cache_count"
        ]
        program_workspace = _program_workspace_attribution(self._program)
        return {
            "identity": identity,
            "operations": operations,
            "resources": {
                "explicit_device_capacity_bytes": self._capacity_bytes,
                "current_steady_device_bytes": (
                    0 if current is None else current.steady_device_bytes
                ),
                "current_target_operator_bytes": (
                    0 if current is None else current.target_operator_bytes
                ),
                "current_inverse_operator_bytes": (
                    0 if current is None else current.inverse_operator_bytes
                ),
                "current_solver_workspace_bytes": (
                    0 if current is None else current.solver_workspace_bytes
                ),
                "current_solver_workspace_materialized_bytes": (
                    0
                    if current is None
                    else current.solver_workspace_materialized_bytes
                ),
                "current_publisher_device_reserved_bytes": (
                    0
                    if current is None
                    else current.publisher_device_reserved_bytes
                ),
                "live_generation_steady_device_bytes": sum(
                    publication.steady_device_bytes
                    for publication in live_publications
                ),
                "retired_lease_steady_device_bytes": sum(
                    publication.steady_device_bytes
                    for publication in retired_publications
                ),
                "publish_overlap_peak_device_bytes": peak_device_bytes,
                "current_publisher_host_metadata_bytes": (
                    0
                    if current is None
                    else current.publisher_host_metadata_bytes
                ),
                "live_publisher_host_metadata_bytes": sum(
                    publication.publisher_host_metadata_bytes
                    for publication in live_publications
                ),
                "retired_lease_publisher_host_metadata_bytes": sum(
                    publication.publisher_host_metadata_bytes
                    for publication in retired_publications
                ),
                "publish_overlap_peak_publisher_host_metadata_bytes": (
                    peak_publisher_host_bytes
                ),
                "graph_runtime_cache_known_device_argument_bytes": (
                    graph_cache_known_device_bytes
                ),
                "graph_runtime_cache_known_deferred_host_argument_bytes": (
                    graph_cache_stats["known_deferred_host_argument_bytes"]
                ),
                "graph_runtime_cache_retained_allocation_lease_count": (
                    graph_cache_stats["retained_allocation_lease_count"]
                ),
                "graph_runtime_cache_object_count": (
                    graph_cache_stats["operator_cache_count"]
                ),
                "graph_runtime_cache_opaque_cache_count": (
                    graph_cache_opaque_count
                ),
                "graph_runtime_cache_opaque_generation_count": (
                    graph_cache_opaque_count
                ),
                "graph_runtime_cache_device_bytes": (
                    graph_cache_known_device_bytes
                    if graph_cache_device_bytes_complete
                    else None
                ),
                "program_primitive_workspace_reserved_bytes": (
                    program_workspace["reserved_bytes"]
                ),
                "program_primitive_workspace_peak_reserved_bytes": (
                    program_workspace["peak_reserved_bytes"]
                ),
                "program_primitive_workspace_in_use_bytes": (
                    program_workspace["in_use_bytes"]
                ),
                "program_primitive_workspace_persistent_bytes": (
                    program_workspace["persistent_bytes"]
                ),
                "program_primitive_workspace_reclaimable_bytes": (
                    program_workspace["reclaimable_bytes"]
                ),
                "program_primitive_workspace_entries": (
                    program_workspace["entries"]
                ),
                "program_sparse_relevant_workspace_reserved_bytes": (
                    program_workspace["sparse_relevant_reserved_bytes"]
                ),
                "program_primitive_workspace_groups": (
                    program_workspace["groups"]
                ),
                "registry_device_reserved_bytes": 0,
            },
            "contract": {
                "old_leases_are_self_contained": True,
                "same_topology_numeric_generations_supported": True,
                "new_acquire_returns_current_generation_only": True,
                "capacity_checked_before_builder": True,
                "capacity_includes_all_live_publication_arrays": True,
                "graph_runtime_cache_bytes_reported": (
                    graph_cache_device_bytes_complete
                ),
                "graph_runtime_cache_in_explicit_capacity": False,
                "graph_runtime_cache_retained_allocation_leases_are_borrowed": True,
                "graph_runtime_cache_query_materializes_device_resources": False,
                "graph_runtime_cache_opaque_generation_count_is_legacy_alias": True,
                "program_primitive_workspace_bytes_reported": (
                    program_workspace["available"]
                ),
                "program_primitive_workspace_current_groups_exact": (
                    program_workspace["current_group_totals_match"]
                ),
                "program_primitive_workspace_group_peak_reported": (
                    program_workspace[
                        "historical_peak_group_breakdown_available"
                    ]
                ),
                "program_primitive_workspace_in_explicit_capacity": False,
                "publish_builders_serialized": True,
                "acquire_allowed_during_build": True,
                "lease_release_waits_for_solve": True,
                "no_host_or_dense_fallback": True,
            },
        }
