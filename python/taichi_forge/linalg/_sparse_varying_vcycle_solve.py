"""Private immutable solve publications for varying-block V-cycles."""

import math
import threading
import weakref

from taichi_forge.lang.exception import TaichiRuntimeError

from ._sparse_bsr_graph_operator import _SparseBsrGraphOperatorPlan
from ._sparse_compiled_graph_pcg import (
    _compiled_graph_pcg_materialized_workspace_bytes,
    _compiled_graph_pcg_workspace_reservation,
    _make_compiled_graph_pcg_solver,
)
from ._sparse_hierarchy_assembly import (
    _ensure_current_program,
    _positive_int,
)
from ._sparse_solve_publication import _SparseSolvePublication
from ._sparse_varying_vcycle_graph import (
    _SparseRecursiveVaryingBlockVcycleGraphPlan,
    _ensure_varying_vcycle_inputs,
    _varying_vcycle_graph_resource_requirements,
)


class _SparseVaryingBlockVcycleSolvePublicationBuilder:
    """Build one immutable varying-hierarchy Graph/PCG generation."""

    def __init__(
        self,
        inputs,
        *,
        max_iterations,
        absolute_tolerance,
        explicit_array_capacity_bytes,
    ):
        _ensure_varying_vcycle_inputs(
            inputs, "varying block V-cycle solve publication builder"
        )
        self._program = inputs._program
        self._backend = inputs._backend
        self._inputs = inputs
        self._hierarchy = inputs._hierarchy
        self._max_iterations = _positive_int(
            max_iterations, "varying block V-cycle solve max_iterations"
        )
        self._absolute_tolerance = float(absolute_tolerance)
        if (
            not math.isfinite(self._absolute_tolerance)
            or self._absolute_tolerance <= 0.0
        ):
            raise TaichiRuntimeError(
                "varying block V-cycle solve absolute_tolerance must be "
                "finite and positive"
            )
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "varying block V-cycle solve explicit_array_capacity_bytes",
        )
        self._lock = threading.Lock()
        self._publications = weakref.WeakSet()
        self._build_attempts = 0
        self._successful_builds = 0
        self._failed_builds = 0
        self._last_report = None

        fine = self._hierarchy._levels[0]
        inverse_requirements = _varying_vcycle_graph_resource_requirements(
            inputs
        )
        solver_reservation = _compiled_graph_pcg_workspace_reservation(
            self._backend, fine.rows
        )
        self._target_operator_bytes = fine.total_reserved_bytes
        self._inverse_workspace_bytes = inverse_requirements[
            "workspace_bytes"
        ]
        self._inverse_operator_bytes = inverse_requirements[
            "native_operator_bytes"
        ]
        self._solver_vector_reservation_bytes = solver_reservation[
            "vector_bytes"
        ]
        self._solver_scalar_reservation_bytes = solver_reservation[
            "scalar_bytes"
        ]
        self._solver_workspace_reservation_bytes = solver_reservation[
            "total_bytes"
        ]
        self._estimated_steady_device_bytes = (
            self._target_operator_bytes
            + self._inverse_operator_bytes
            + self._solver_workspace_reservation_bytes
        )
        self._estimated_build_peak_device_bytes = (
            inputs.steady_reserved_bytes
            + self._inverse_workspace_bytes
            + self._target_operator_bytes
            + self._inverse_operator_bytes
            + self._solver_workspace_reservation_bytes
        )
        if self._estimated_build_peak_device_bytes > self._capacity_bytes:
            raise TaichiRuntimeError(
                "varying block V-cycle solve explicit-array capacity "
                "overflow before build"
            )

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "varying block V-cycle solve publication builder"
        )

    @property
    def estimated_steady_device_bytes(self):
        return self._estimated_steady_device_bytes

    @property
    def estimated_build_peak_device_bytes(self):
        return self._estimated_build_peak_device_bytes

    def build(self):
        with self._lock:
            self._ensure_current()
            self._build_attempts += 1
            try:
                fine = self._hierarchy._levels[0]
                target_plan = _SparseBsrGraphOperatorPlan(
                    fine,
                    explicit_array_capacity_bytes=self._capacity_bytes,
                )
                inverse_plan = _SparseRecursiveVaryingBlockVcycleGraphPlan(
                    self._inputs,
                    explicit_array_capacity_bytes=self._capacity_bytes,
                )
                target = target_plan.create_native_operator()
                inverse = inverse_plan.create_native_operator()
                preconditioner, solver = _make_compiled_graph_pcg_solver(
                    program=self._program,
                    backend=self._backend,
                    target=target,
                    inverse=inverse,
                    max_iterations=self._max_iterations,
                    absolute_tolerance=self._absolute_tolerance,
                )
                target_stats = target._debug_runtime_stats()
                inverse_stats = inverse._debug_runtime_stats()
                target_bytes = int(
                    target_stats["resources"][
                        "operator_owned_reserved_bytes"
                    ]
                )
                inverse_bytes = int(
                    inverse_stats["resources"][
                        "operator_owned_reserved_bytes"
                    ]
                )
                materialized_solver_bytes = (
                    _compiled_graph_pcg_materialized_workspace_bytes(solver)
                )
                if (
                    target_bytes != self._target_operator_bytes
                    or inverse_bytes != self._inverse_operator_bytes
                    or materialized_solver_bytes
                    > self._solver_workspace_reservation_bytes
                ):
                    raise TaichiRuntimeError(
                        "varying block V-cycle solve resources exceeded "
                        "preflight"
                    )
                publication = _SparseSolvePublication(
                    program=self._program,
                    topology_version=self._inputs.topology_version,
                    numeric_version=self._inputs.numeric_version,
                    size=fine.rows,
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
                    "target_component_topology_version": (
                        fine.topology_version
                    ),
                    "target_component_numeric_version": fine.numeric_version,
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
            publications = list(self._publications)
            fine = self._hierarchy._levels[0]
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": "immutable_varying_block_vcycle_graph_pcg",
                    "block_rows": fine.block_rows,
                    "block_size": fine.block_size,
                    "size": fine.rows,
                    "level_block_sizes": tuple(
                        level.block_size
                        for level in self._hierarchy._levels
                    ),
                    "topology_version": self._inputs.topology_version,
                    "numeric_version": self._inputs.numeric_version,
                    "target_component_topology_version": (
                        fine.topology_version
                    ),
                    "target_component_numeric_version": fine.numeric_version,
                    "max_iterations": self._max_iterations,
                    "absolute_tolerance": self._absolute_tolerance,
                },
                "operations": {
                    "build_attempts": self._build_attempts,
                    "successful_builds": self._successful_builds,
                    "failed_builds": self._failed_builds,
                },
                "resources": {
                    "borrowed_execution_inputs_reserved_bytes": (
                        self._inputs.steady_reserved_bytes
                    ),
                    "target_operator_reservation_bytes": (
                        self._target_operator_bytes
                    ),
                    "inverse_operator_reservation_bytes": (
                        self._inverse_operator_bytes
                    ),
                    "inverse_workspace_reservation_bytes": (
                        self._inverse_workspace_bytes
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
                    "live_publication_count": len(publications),
                    "live_publication_steady_device_bytes": sum(
                        value.steady_device_bytes for value in publications
                    ),
                },
                "contract": {
                    "immutable_target_inverse_solver_generation": True,
                    "publication_identity_uses_hierarchy_generation": True,
                    "component_operator_lineage_versions_preserved": True,
                    "same_topology_numeric_refresh_rebuilds_generation": True,
                    "caller_provides_spd_vcycle_inverse": True,
                    "host_inversion_performed": False,
                    "compiled_graph_target_and_inverse": True,
                    "compiled_preconditioner_and_pcg_reused": True,
                    "solver_parameters_selected_by_caller": True,
                    "cuda_lazy_workspace_uses_reservation": True,
                    "vulkan_scalar_state_in_reservation": True,
                    "graph_runtime_cache_in_explicit_capacity": False,
                    "public_api": False,
                },
            }
