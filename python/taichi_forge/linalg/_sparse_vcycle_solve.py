"""Private immutable solve publications for recursive sparse V-cycles."""

import math
import threading
import weakref

import taichi_forge as ti
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang.exception import TaichiRuntimeError

from ._sparse_csr_graph_operator import _SparseCsrGraphOperatorPlan
from ._sparse_hierarchy_assembly import (
    _ensure_current_program,
    _positive_int,
)
from ._sparse_solve_publication import _SparseSolvePublication
from ._sparse_vcycle_graph import (
    _SparseRecursiveVcycleGraphPlan,
    _ensure_hierarchy_numeric_pair,
)


class _SparseVcycleSolvePublicationBuilder:
    """Build one self-contained target/inverse/binding/solver generation."""

    def __init__(
        self,
        hierarchy,
        numeric,
        *,
        max_iterations,
        absolute_tolerance,
        explicit_array_capacity_bytes,
    ):
        _ensure_hierarchy_numeric_pair(
            hierarchy, numeric, "V-cycle solve publication builder"
        )
        self._program = hierarchy._program
        self._backend = hierarchy._backend
        self._hierarchy = hierarchy
        self._numeric = numeric
        self._max_iterations = _positive_int(
            max_iterations, "V-cycle solve max_iterations"
        )
        self._absolute_tolerance = float(absolute_tolerance)
        if (
            not math.isfinite(self._absolute_tolerance)
            or self._absolute_tolerance <= 0.0
        ):
            raise TaichiRuntimeError(
                "V-cycle solve absolute_tolerance must be finite and positive"
            )
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "V-cycle solve explicit_array_capacity_bytes",
        )
        self._lock = threading.Lock()
        self._publications = weakref.WeakSet()
        self._build_attempts = 0
        self._successful_builds = 0
        self._failed_builds = 0
        self._last_report = None

        self._target_operator_bytes = hierarchy._levels[0].total_reserved_bytes
        topology_bytes = 0
        hierarchy_numeric_bytes = 0
        for level_index in range(hierarchy.level_count - 1):
            level = hierarchy._levels[level_index]
            topology_bytes += (
                level.pattern_reserved_bytes
                + 4 * hierarchy.level_sizes[level_index]
                + hierarchy._restriction_schedules[
                    level_index
                ].total_reserved_bytes
            )
            hierarchy_numeric_bytes += level.value_reserved_bytes
        self._inverse_workspace_bytes = 4 * (
            sum(hierarchy.level_sizes[:-1])
            + 2 * sum(hierarchy.level_sizes[1:])
        )
        self._inverse_operator_bytes = (
            topology_bytes
            + hierarchy_numeric_bytes
            + numeric.total_reserved_bytes
            + self._inverse_workspace_bytes
        )
        self._solver_vector_reservation_bytes = (
            4 * hierarchy.level_sizes[0] * 4
        )
        self._solver_scalar_reservation_bytes = (
            40 if self._backend == "vulkan" else 0
        )
        self._solver_workspace_reservation_bytes = (
            self._solver_vector_reservation_bytes
            + self._solver_scalar_reservation_bytes
        )
        self._estimated_steady_device_bytes = (
            self._target_operator_bytes
            + self._inverse_operator_bytes
            + self._solver_workspace_reservation_bytes
        )
        self._estimated_build_peak_device_bytes = (
            hierarchy.steady_reserved_bytes
            + numeric.total_reserved_bytes
            + self._inverse_workspace_bytes
            + self._target_operator_bytes
            + self._inverse_operator_bytes
            + self._solver_workspace_reservation_bytes
        )
        if self._estimated_build_peak_device_bytes > self._capacity_bytes:
            raise TaichiRuntimeError(
                "V-cycle solve explicit-array capacity overflow before build"
            )

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "V-cycle solve publication builder"
        )

    @property
    def estimated_steady_device_bytes(self):
        return self._estimated_steady_device_bytes

    @property
    def estimated_build_peak_device_bytes(self):
        return self._estimated_build_peak_device_bytes

    def _make_solver(self, target, preconditioner):
        if self._backend == "cpu":
            return _ti_core._make_cpu_compiled_kernel_pcg_solver(
                self._program,
                target,
                preconditioner,
                self._max_iterations,
                self._absolute_tolerance,
            )
        if self._backend == "cuda":
            return _ti_core._make_cuda_compiled_kernel_pcg_solver(
                self._program,
                target,
                preconditioner,
                self._max_iterations,
                self._absolute_tolerance,
                False,
            )
        if self._backend == "vulkan":
            return _ti_core._make_vulkan_compiled_kernel_pcg_convergence_plan(
                self._program,
                target,
                preconditioner,
                self._max_iterations,
                self._absolute_tolerance,
            )
        raise TaichiRuntimeError(
            "V-cycle solve builder supports CPU, CUDA, and Vulkan only"
        )

    def build(self):
        with self._lock:
            self._ensure_current()
            self._build_attempts += 1
            try:
                target_plan = _SparseCsrGraphOperatorPlan(
                    self._hierarchy._levels[0],
                    explicit_array_capacity_bytes=self._capacity_bytes,
                )
                inverse_plan = _SparseRecursiveVcycleGraphPlan(
                    self._hierarchy,
                    self._numeric,
                    explicit_array_capacity_bytes=self._capacity_bytes,
                )
                target = target_plan.create_native_operator()
                inverse = inverse_plan.create_native_operator()
                preconditioner = (
                    _ti_core._make_compiled_kernel_preconditioner_plan(
                        self._program,
                        target,
                        inverse,
                        True,
                    )
                )
                solver = self._make_solver(target, preconditioner)
                target_stats = target._debug_runtime_stats()
                inverse_stats = inverse._debug_runtime_stats()
                solver_stats = solver._debug_runtime_stats()
                target_bytes = int(
                    target_stats["resources"]["operator_owned_reserved_bytes"]
                )
                inverse_bytes = int(
                    inverse_stats["resources"]["operator_owned_reserved_bytes"]
                )
                materialized_solver_bytes = int(
                    solver_stats["resources"][
                        "persistent_vector_reserved_bytes"
                    ]
                ) + int(
                    solver_stats["resources"].get(
                        "persistent_scalar_reserved_bytes", 0
                    )
                )
                if (
                    target_bytes != self._target_operator_bytes
                    or inverse_bytes != self._inverse_operator_bytes
                    or materialized_solver_bytes
                    > self._solver_workspace_reservation_bytes
                ):
                    raise TaichiRuntimeError(
                        "V-cycle solve provider resources exceeded preflight"
                    )
                publication = _SparseSolvePublication(
                    program=self._program,
                    topology_version=self._hierarchy.topology_version,
                    numeric_version=self._hierarchy.numeric_version,
                    size=self._hierarchy.level_sizes[0],
                    target=target,
                    inverse=inverse,
                    preconditioner=preconditioner,
                    solver=solver,
                    numeric_publisher=None,
                    target_operator_bytes=target_bytes,
                    inverse_operator_bytes=inverse_bytes,
                    solver_workspace_bytes=(
                        self._solver_workspace_reservation_bytes
                    ),
                    solver_workspace_materialized_bytes=(
                        materialized_solver_bytes
                    ),
                    build_peak_device_bytes=(
                        self._estimated_build_peak_device_bytes
                    ),
                )
                self._publications.add(publication)
                self._successful_builds += 1
                self._last_report = {
                    "topology_version": publication.topology_version,
                    "numeric_version": publication.numeric_version,
                    "target_operator_bytes": target_bytes,
                    "inverse_operator_bytes": inverse_bytes,
                    "solver_workspace_reservation_bytes": (
                        self._solver_workspace_reservation_bytes
                    ),
                    "solver_workspace_materialized_bytes": (
                        materialized_solver_bytes
                    ),
                    "steady_device_bytes": publication.steady_device_bytes,
                    "build_peak_device_bytes": (
                        publication.build_peak_device_bytes
                    ),
                    "graph_runtime_cache_in_explicit_capacity": False,
                }
                return publication
            except Exception:
                self._failed_builds += 1
                raise

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            live_publications = list(self._publications)
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "topology_version": self._hierarchy.topology_version,
                    "numeric_version": self._hierarchy.numeric_version,
                    "size": self._hierarchy.level_sizes[0],
                    "max_iterations": self._max_iterations,
                    "absolute_tolerance": self._absolute_tolerance,
                },
                "operations": {
                    "build_attempts": self._build_attempts,
                    "successful_builds": self._successful_builds,
                    "failed_builds": self._failed_builds,
                },
                "resources": {
                    "target_operator_reservation_bytes": (
                        self._target_operator_bytes
                    ),
                    "inverse_operator_reservation_bytes": (
                        self._inverse_operator_bytes
                    ),
                    "solver_vector_reservation_bytes": (
                        self._solver_vector_reservation_bytes
                    ),
                    "solver_scalar_reservation_bytes": (
                        self._solver_scalar_reservation_bytes
                    ),
                    "solver_workspace_reservation_bytes": (
                        self._solver_workspace_reservation_bytes
                    ),
                    "estimated_steady_device_bytes": (
                        self._estimated_steady_device_bytes
                    ),
                    "estimated_build_peak_device_bytes": (
                        self._estimated_build_peak_device_bytes
                    ),
                    "explicit_array_capacity_bytes": self._capacity_bytes,
                    "last_report": self._last_report,
                    "live_publication_count": len(live_publications),
                    "live_publication_steady_device_bytes": sum(
                        value.steady_device_bytes
                        for value in live_publications
                    ),
                },
                "contract": {
                    "immutable_target_inverse_solver_generation": True,
                    "same_topology_numeric_refresh_rebuilds_generation": True,
                    "compiled_preconditioner_and_pcg_reused": True,
                    "solver_parameters_selected_by_caller": True,
                    "cuda_lazy_workspace_uses_reservation": True,
                    "vulkan_scalar_state_in_reservation": True,
                    "graph_runtime_cache_in_explicit_capacity": False,
                    "public_api": False,
                },
            }
