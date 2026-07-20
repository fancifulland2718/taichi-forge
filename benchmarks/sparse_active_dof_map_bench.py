"""Active-block to compact-DOF bridge correctness diagnostic.

The profile starts from a bounded device-produced batch of possibly duplicated
block keys. It uses an explicit backend-native sort and stable consecutive
unique, assigns DOFs by sorted block key and row-major brick offset, and builds
a compact neighbor table once per topology version. Matrix-free operator
application then touches only ndarrays; an SNode stencil is retained solely as
an oracle.
"""

import argparse
import gc
import json
import sys
import weakref
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sparse_snode_lifecycle_bench import _arch_name, _runtime_snapshot  # noqa: E402
from taichi_forge.linalg._sparse_solve_publication import (  # noqa: E402
    _SparseSolvePublication as _HierarchyPublication,
    _SparseSolvePublicationRegistry as _HierarchyPublicationRegistry,
)


SCHEMA = "taichi_forge.sparse_active_dof_map.v1"
SENTINEL_KEY = np.uint32(0xFFFFFFFF)


def _ndarray_allocation_id(value):
    arr = getattr(value, "arr", None)
    if arr is None:
        return None
    return int(arr.device_allocation_ptr())


class _StructuredActiveDofOperatorPlan:
    """Profile-private lifecycle contract for a compact stencil operator.

    The plan deliberately owns no SNode and allocates nothing in ``apply``.
    Its backend-native Taichi kernel is injected so this profile can freeze
    ownership/version semantics before a C++ solver-facing provider exists.
    """

    def __init__(
        self,
        ti,
        *,
        program,
        dimensions,
        capacity,
        active_size,
        topology_version,
        numeric_version,
        current_topology_version,
        neighbors,
        apply_kernel,
    ):
        if dimensions not in (2, 3):
            raise ValueError("structured operator dimensions must be 2 or 3")
        if capacity <= 0 or active_size <= 0 or active_size > capacity:
            raise ValueError(
                "structured operator active size must be within capacity"
            )
        if neighbors.dtype != ti.i32 or tuple(neighbors.shape) != (
            capacity,
            2 * dimensions,
        ):
            raise TypeError(
                "structured operator neighbors must be a capacity x 2d i32 "
                "ndarray"
            )
        if neighbors._runtime_prog is not program:
            raise RuntimeError(
                "structured operator neighbors must belong to the plan Program; "
                "no fallback was performed"
            )
        self._ti = ti
        self._program = program
        self._dimensions = dimensions
        self._capacity = capacity
        self._active_size = active_size
        self._topology_version = topology_version
        self._numeric_version = numeric_version
        self._current_topology_version = current_topology_version
        self._neighbors = neighbors
        self._apply_kernel = apply_kernel
        self._apply_calls = 0
        self._rejected_apply_calls = 0

    def _reject(self, message):
        self._rejected_apply_calls += 1
        raise RuntimeError(message)

    def _validate_runtime(self):
        runtime_program = self._ti.lang.impl.get_runtime().prog
        if runtime_program is not self._program:
            self._reject(
                "structured operator cannot be used after its Program was "
                "reset; no fallback was performed"
            )
        if self._neighbors.arr is None:
            self._reject("structured operator neighbor storage was retired")

    def _validate_vector(self, value, role):
        if value.dtype != self._ti.f32 or tuple(value.shape) != (
            self._capacity,
        ):
            self._reject(
                f"structured operator {role} must be a capacity-sized f32 "
                "ndarray"
            )
        if value._runtime_prog is not self._program or value.arr is None:
            self._reject(
                f"structured operator {role} must belong to the plan Program; "
                "no fallback was performed"
            )

    def apply(
        self,
        input_array,
        output_array,
        *,
        expected_topology_version=None,
        expected_numeric_version=None,
    ):
        self._validate_runtime()
        current_topology_version = int(self._current_topology_version())
        if current_topology_version != self._topology_version:
            self._reject(
                "structured operator topology version is stale; output was "
                "not mutated"
            )
        if (
            expected_topology_version is not None
            and int(expected_topology_version) != self._topology_version
        ):
            self._reject(
                "structured operator expected topology version does not match; "
                "output was not mutated"
            )
        if (
            expected_numeric_version is not None
            and int(expected_numeric_version) != self._numeric_version
        ):
            self._reject(
                "structured operator expected numeric version does not match; "
                "output was not mutated"
            )
        self._validate_vector(input_array, "input")
        self._validate_vector(output_array, "output")
        if _ndarray_allocation_id(input_array) == _ndarray_allocation_id(
            output_array
        ):
            self._reject(
                "structured operator does not support input/output alias; "
                "output was not mutated"
            )
        self._apply_kernel(
            self._active_size,
            self._neighbors,
            input_array,
            output_array,
        )
        self._apply_calls += 1

    def debug_runtime_stats(self):
        return {
            "identity": {
                "dimensions": self._dimensions,
                "capacity": self._capacity,
                "active_size": self._active_size,
                "topology_version": self._topology_version,
                "numeric_version": self._numeric_version,
            },
            "operations": {
                "apply_calls": self._apply_calls,
                "rejected_apply_calls": self._rejected_apply_calls,
                "plan_owned_apply_allocations": 0,
                "explicit_apply_host_synchronizations": 0,
            },
            "resources": {
                "neighbor_reserved_bytes": (
                    self._capacity * 2 * self._dimensions * 4
                ),
                "neighbor_active_bytes": (
                    self._active_size * 2 * self._dimensions * 4
                ),
                "owns_snode_tree": False,
                "owns_input_or_output": False,
            },
            "contract": {
                "backend_native_taichi_kernel": True,
                "profile_private_not_solver_provider": True,
                "no_host_fallback": True,
                "publisher_validates_neighbor_indices": True,
                "apply_calls_are_host_submissions": True,
                "kernel_jit_and_runtime_cache_outside_plan_telemetry": True,
            },
        }


class _TwoLevelCompactPreconditionerPlan:
    """Profile-private two-level additive Galerkin preconditioner.

    The fine-to-coarse map, coarse inverse, and level workspace are owned by
    the plan. ``apply`` is deliberately a three-kernel sequence so the profile
    records the execution boundary that a future Graph-backed provider must
    preserve; no SNode or host fallback participates in the sequence.
    """

    def __init__(
        self,
        ti,
        *,
        program,
        dimensions,
        active_size,
        coarse_size,
        topology_version,
        numeric_version,
        current_topology_version,
        current_numeric_version,
        fine_to_coarse,
        coarse_inverse,
        restrict_kernel,
        coarse_solve_kernel,
        prolong_kernel,
    ):
        if dimensions not in (2, 3):
            raise ValueError("two-level preconditioner dimensions must be 2 or 3")
        if active_size <= 0 or coarse_size <= 0 or coarse_size > active_size:
            raise ValueError("two-level preconditioner level sizes are invalid")
        fine_to_coarse = np.asarray(fine_to_coarse, dtype=np.int32)
        coarse_inverse = np.asarray(coarse_inverse, dtype=np.float32)
        if fine_to_coarse.shape != (active_size,):
            raise ValueError("fine-to-coarse map must cover every active DOF")
        if coarse_inverse.shape != (coarse_size, coarse_size):
            raise ValueError("coarse inverse must be square at the coarse size")
        if np.any(fine_to_coarse < 0) or np.any(
            fine_to_coarse >= coarse_size
        ):
            raise ValueError("fine-to-coarse map contains an invalid coarse row")

        self._ti = ti
        self._program = program
        self._dimensions = dimensions
        self._active_size = active_size
        self._coarse_size = coarse_size
        self._topology_version = topology_version
        self._numeric_version = numeric_version
        self._current_topology_version = current_topology_version
        self._current_numeric_version = current_numeric_version
        self._fine_to_coarse = ti.ndarray(ti.i32, shape=active_size)
        self._coarse_inverse = ti.ndarray(
            ti.f32, shape=(coarse_size, coarse_size)
        )
        self._coarse_rhs = ti.ndarray(ti.f32, shape=coarse_size)
        self._coarse_solution = ti.ndarray(ti.f32, shape=coarse_size)
        self._fine_to_coarse.from_numpy(fine_to_coarse)
        self._coarse_inverse.from_numpy(coarse_inverse)
        self._restrict_kernel = restrict_kernel
        self._coarse_solve_kernel = coarse_solve_kernel
        self._prolong_kernel = prolong_kernel
        sym_active_size = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "active_size", ti.i32
        )
        sym_coarse_size = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, "coarse_size", ti.i32
        )
        sym_fine_to_coarse = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "fine_to_coarse", ti.i32, ndim=1
        )
        sym_coarse_inverse = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "coarse_inverse", ti.f32, ndim=2
        )
        sym_input = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
        )
        sym_output = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
        )
        sym_coarse_rhs = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "coarse_rhs", ti.f32, ndim=1
        )
        sym_coarse_solution = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "coarse_solution", ti.f32, ndim=1
        )
        graph_builder = ti.graph.GraphBuilder()
        graph_builder.dispatch(
            restrict_kernel,
            sym_active_size,
            sym_coarse_size,
            sym_fine_to_coarse,
            sym_input,
            sym_coarse_rhs,
        )
        graph_builder.dispatch(
            coarse_solve_kernel,
            sym_coarse_size,
            sym_coarse_inverse,
            sym_coarse_rhs,
            sym_coarse_solution,
        )
        graph_builder.dispatch(
            prolong_kernel,
            sym_active_size,
            sym_fine_to_coarse,
            sym_input,
            sym_coarse_solution,
            sym_output,
        )
        self._graph = graph_builder.compile()
        self._graph_args = {
            "active_size": active_size,
            "coarse_size": coarse_size,
            "fine_to_coarse": self._fine_to_coarse,
            "coarse_inverse": self._coarse_inverse,
            "input": None,
            "output": None,
            "coarse_rhs": self._coarse_rhs,
            "coarse_solution": self._coarse_solution,
        }
        self._owned_allocation_ids = tuple(
            _ndarray_allocation_id(value)
            for value in (
                self._fine_to_coarse,
                self._coarse_inverse,
                self._coarse_rhs,
                self._coarse_solution,
            )
        )
        self._apply_calls = 0
        self._rejected_apply_calls = 0

    def _reject(self, message):
        self._rejected_apply_calls += 1
        raise RuntimeError(message)

    def _validate_runtime(self):
        runtime_program = self._ti.lang.impl.get_runtime().prog
        if runtime_program is not self._program:
            self._reject(
                "two-level preconditioner cannot be used after its Program "
                "was reset; no fallback was performed"
            )
        for name, value in (
            ("fine-to-coarse map", self._fine_to_coarse),
            ("coarse inverse", self._coarse_inverse),
            ("coarse rhs", self._coarse_rhs),
            ("coarse solution", self._coarse_solution),
        ):
            if value.arr is None:
                self._reject(f"two-level preconditioner {name} was retired")

    def _validate_vector(self, value, role):
        if value.dtype != self._ti.f32 or tuple(value.shape) != (
            self._active_size,
        ):
            self._reject(
                f"two-level preconditioner {role} must be an active-sized "
                "f32 ndarray"
            )
        if value._runtime_prog is not self._program or value.arr is None:
            self._reject(
                f"two-level preconditioner {role} must belong to the plan "
                "Program; no fallback was performed"
            )

    def apply(
        self,
        input_array,
        output_array,
        *,
        expected_topology_version=None,
        expected_numeric_version=None,
    ):
        self._validate_runtime()
        current_topology_version = int(self._current_topology_version())
        if current_topology_version != self._topology_version:
            self._reject(
                "two-level preconditioner topology version is stale; output "
                "was not mutated"
            )
        current_numeric_version = int(self._current_numeric_version())
        if current_numeric_version != self._numeric_version:
            self._reject(
                "two-level preconditioner numeric version is stale; output "
                "was not mutated"
            )
        if (
            expected_topology_version is not None
            and int(expected_topology_version) != self._topology_version
        ):
            self._reject(
                "two-level preconditioner expected topology version does not "
                "match; output was not mutated"
            )
        if (
            expected_numeric_version is not None
            and int(expected_numeric_version) != self._numeric_version
        ):
            self._reject(
                "two-level preconditioner expected numeric version does not "
                "match; output was not mutated"
            )
        self._validate_vector(input_array, "input")
        self._validate_vector(output_array, "output")
        if _ndarray_allocation_id(input_array) == _ndarray_allocation_id(
            output_array
        ):
            self._reject(
                "two-level preconditioner does not support input/output alias; "
                "output was not mutated"
            )

        self._graph_args["input"] = input_array
        self._graph_args["output"] = output_array
        try:
            self._graph.run(self._graph_args)
        finally:
            self._graph_args["input"] = None
            self._graph_args["output"] = None
        self._apply_calls += 1

    def create_native_operator(self):
        return self._program._create_compiled_graph_linear_operator(
            self._graph._compiled_graph,
            self._active_size,
            self._topology_version,
            self._numeric_version,
            {
                "active_size": self._active_size,
                "coarse_size": self._coarse_size,
            },
            {"fine_to_coarse": self._fine_to_coarse.arr},
            {"coarse_inverse": self._coarse_inverse.arr},
            {
                "coarse_rhs": self._coarse_rhs.arr,
                "coarse_solution": self._coarse_solution.arr,
            },
        )

    def debug_runtime_stats(self):
        scalar_bytes = 4
        topology_bytes = self._active_size * scalar_bytes
        numeric_bytes = self._coarse_size**2 * scalar_bytes
        workspace_bytes = 2 * self._coarse_size * scalar_bytes
        graph_execution = self._graph.execution_stats()
        graph_persistent_argument_bytes = sum(
            segment.persistent_argument_bytes
            for segment in graph_execution.segments
        )
        return {
            "identity": {
                "method": "two_level_additive_galerkin",
                "dimensions": self._dimensions,
                "fine_size": self._active_size,
                "coarse_size": self._coarse_size,
                "fine_topology_version": self._topology_version,
                "coarse_topology_version": self._topology_version,
                "numeric_version": self._numeric_version,
            },
            "operations": {
                "apply_calls": self._apply_calls,
                "rejected_apply_calls": self._rejected_apply_calls,
                "kernel_launches_per_apply": 3,
                "kernel_launches": 3 * self._apply_calls,
                "host_graph_submissions_per_apply": 1,
                "host_graph_submissions": self._apply_calls,
                "graph_execution_path": graph_execution.execution_path,
                "graph_backend_segments": (
                    graph_execution.backend_graph_segments
                ),
                "graph_replay_segments": (
                    graph_execution.backend_replay_segments
                ),
                "plan_owned_apply_allocations": 0,
                "explicit_apply_host_synchronizations": 0,
            },
            "resources": {
                "topology_reserved_bytes": topology_bytes,
                "numeric_reserved_bytes": numeric_bytes,
                "workspace_reserved_bytes": workspace_bytes,
                "plan_owned_reserved_bytes": (
                    topology_bytes + numeric_bytes + workspace_bytes
                ),
                "owns_level_topology": True,
                "owns_level_numeric_data": True,
                "owns_level_workspace": True,
                "owns_snode_tree": False,
                "owns_input_or_output": False,
                "last_bound_caller_vector_bytes_not_plan_owned": (
                    2 * self._active_size * scalar_bytes
                ),
                "graph_persistent_argument_bytes_outside_plan_payload": (
                    graph_persistent_argument_bytes
                ),
                "graph_fast_arg_cache_retains_last_native_binding": True,
                "owned_allocation_identity_stable": (
                    self._owned_allocation_ids
                    == tuple(
                        _ndarray_allocation_id(value)
                        for value in (
                            self._fine_to_coarse,
                            self._coarse_inverse,
                            self._coarse_rhs,
                            self._coarse_solution,
                        )
                    )
                ),
                "plan_argument_dict_retains_last_input_or_output": (
                    self._graph_args["input"] is not None
                    or self._graph_args["output"] is not None
                ),
            },
            "contract": {
                "aggregation": "one_sorted_active_brick_per_coarse_dof",
                "restriction": "piecewise_constant_transpose_sum",
                "coarse_operator": "host_built_galerkin_snapshot",
                "coarse_solve": "exact_dense_inverse_apply",
                "fine_term": "diagonal_jacobi",
                "additive_spd_contract": True,
                "backend_native_taichi_kernels": True,
                "profile_private_not_solver_provider": True,
                "no_host_fallback": True,
                "snode_accesses_per_apply": 0,
                "compiled_single_kernel_eligible": False,
                "graph_sequence_candidate": True,
                "cached_graph_execution_integrated": True,
                "backend_replay_required_for_correctness": False,
                "one_host_graph_run_per_apply": True,
                "graph_runtime_resource_lease_managed_by_program": True,
                "graph_node_count": graph_execution.node_count,
                "graph_dispatch_count": graph_execution.dispatch_count,
                "graph_runtime_arg_count": graph_execution.runtime_arg_count,
                "solver_integrated": False,
            },
        }


def _csr_from_rows_reference(rows):
    row_offsets = [0]
    column_indices = []
    values = []
    for entries in rows:
        for column, value in sorted(entries.items()):
            if value != 0.0:
                column_indices.append(int(column))
                values.append(float(value))
        row_offsets.append(len(column_indices))
    return (
        np.asarray(row_offsets, dtype=np.int32),
        np.asarray(column_indices, dtype=np.int32),
        np.asarray(values, dtype=np.float64),
    )


def _csr_to_dense_reference(row_offsets, column_indices, values):
    size = len(row_offsets) - 1
    matrix = np.zeros((size, size), dtype=np.float64)
    for row in range(size):
        for offset in range(row_offsets[row], row_offsets[row + 1]):
            matrix[row, int(column_indices[offset])] += float(values[offset])
    return matrix


def _galerkin_csr_reference(
    row_offsets, column_indices, values, fine_to_coarse, coarse_size
):
    """Project arbitrary CSR with piecewise-constant aggregates."""
    fine_size = len(row_offsets) - 1
    fine_to_coarse = np.asarray(fine_to_coarse, dtype=np.int32)
    if fine_to_coarse.shape != (fine_size,):
        raise ValueError("aggregate map must cover every fine row")
    if coarse_size <= 0 or np.any(fine_to_coarse < 0) or np.any(
        fine_to_coarse >= coarse_size
    ):
        raise ValueError("aggregate map contains an invalid coarse row")
    if len(np.unique(fine_to_coarse)) != coarse_size:
        raise ValueError("every coarse aggregate must be non-empty")
    coarse_rows = [dict() for _ in range(coarse_size)]
    for fine_row in range(fine_size):
        coarse_row = int(fine_to_coarse[fine_row])
        for offset in range(row_offsets[fine_row], row_offsets[fine_row + 1]):
            coarse_column = int(fine_to_coarse[int(column_indices[offset])])
            coarse_rows[coarse_row][coarse_column] = (
                coarse_rows[coarse_row].get(coarse_column, 0.0)
                + float(values[offset])
            )
    coarse_csr = _csr_from_rows_reference(coarse_rows)
    return {
        "row_offsets": coarse_csr[0],
        "column_indices": coarse_csr[1],
        "values": coarse_csr[2],
        "reconstructed": _csr_to_dense_reference(*coarse_csr),
    }


def _csr_component_reference(row_offsets, column_indices, values):
    size = len(row_offsets) - 1
    labels = np.full(size, -1, dtype=np.int32)
    component_sizes = []
    for seed in range(size):
        if labels[seed] >= 0:
            continue
        label = len(component_sizes)
        labels[seed] = label
        pending = [seed]
        component_size = 0
        while pending:
            row = pending.pop()
            component_size += 1
            for offset in range(row_offsets[row], row_offsets[row + 1]):
                column = int(column_indices[offset])
                if (
                    column != row
                    and values[offset] != 0.0
                    and labels[column] < 0
                ):
                    labels[column] = label
                    pending.append(column)
        component_sizes.append(component_size)
    return labels, component_sizes


def _coordinate_poisson_csr_reference(coordinates, dimensions):
    coordinates = np.asarray(coordinates, dtype=np.int32)
    if coordinates.ndim != 2 or coordinates.shape[1] != dimensions:
        raise ValueError("coordinate rows do not match hierarchy dimensions")
    coordinate_to_row = {
        tuple(int(value) for value in coordinate): row
        for row, coordinate in enumerate(coordinates)
    }
    if len(coordinate_to_row) != len(coordinates):
        raise ValueError("coordinate rows must be unique")
    rows = []
    for row, coordinate in enumerate(coordinates):
        entries = {row: float(2 * dimensions)}
        for axis in range(dimensions):
            for offset in (-1, 1):
                neighbor = coordinate.copy()
                neighbor[axis] += offset
                column = coordinate_to_row.get(tuple(int(x) for x in neighbor))
                if column is not None:
                    entries[column] = -1.0
        rows.append(entries)
    return _csr_from_rows_reference(rows)


def _directional_csr_eligible(
    row_offsets, column_indices, values, coordinates
):
    for row in range(len(row_offsets) - 1):
        for offset in range(row_offsets[row], row_offsets[row + 1]):
            column = int(column_indices[offset])
            if column == row or values[offset] == 0.0:
                continue
            difference = np.abs(coordinates[column] - coordinates[row])
            if np.count_nonzero(difference) != 1 or int(difference.sum()) != 1:
                return False
    return True


def _build_recursive_csr_hierarchy_reference(
    *,
    dimensions,
    coordinates,
    row_offsets,
    column_indices,
    values,
    bottom_component_cap,
):
    """Build a deterministic component-aware geometric Galerkin hierarchy."""
    if dimensions not in (2, 3):
        raise ValueError("hierarchy dimensions must be 2 or 3")
    if bottom_component_cap <= 0:
        raise ValueError("bottom component cap must be positive")
    current_coordinates = np.asarray(coordinates, dtype=np.int32)
    current_csr = (
        np.asarray(row_offsets, dtype=np.int32),
        np.asarray(column_indices, dtype=np.int32),
        np.asarray(values, dtype=np.float64),
    )
    levels = []
    reference_levels = []
    while True:
        dense = _csr_to_dense_reference(*current_csr)
        size = len(current_csr[0]) - 1
        labels, component_sizes = _csr_component_reference(*current_csr)
        row_nnz = np.diff(current_csr[0])
        level = {
            "size": size,
            "actual_nnz": int(len(current_csr[1])),
            "max_row_nnz": int(row_nnz.max(initial=0)),
            "component_sizes": sorted(component_sizes, reverse=True),
            "symmetric": bool(
                np.allclose(dense, dense.T, rtol=0.0, atol=0.0)
            ),
            "min_eigenvalue": float(np.linalg.eigvalsh(dense)[0]),
            "directional_stencil_eligible": _directional_csr_eligible(
                *current_csr, current_coordinates
            ),
            "operator_pattern_bytes": (
                size + 1 + len(current_csr[1])
            )
            * 4,
            "operator_value_bytes": len(current_csr[2]) * 4,
            "map_to_coarse_bytes": 0,
            "workspace_upper_bytes": 3 * size * 4,
            "galerkin_matches_dense_oracle": True,
        }
        reference_level = {
            "row_offsets": current_csr[0].copy(),
            "column_indices": current_csr[1].copy(),
            "values": current_csr[2].copy(),
            "fine_to_coarse": None,
        }
        if max(component_sizes) <= bottom_component_cap:
            levels.append(level)
            reference_levels.append(reference_level)
            break

        origins = []
        for component in range(len(component_sizes)):
            origins.append(current_coordinates[labels == component].min(axis=0))
        parent_keys = []
        for row, coordinate in enumerate(current_coordinates):
            component = int(labels[row])
            parent = (coordinate - origins[component]) // 2
            parent_keys.append(
                (component,) + tuple(int(value) for value in parent)
            )
        ordered_parent_keys = sorted(set(parent_keys))
        key_to_coarse = {
            key: coarse for coarse, key in enumerate(ordered_parent_keys)
        }
        fine_to_coarse = np.asarray(
            [key_to_coarse[key] for key in parent_keys], dtype=np.int32
        )
        coarse_size = len(ordered_parent_keys)
        if coarse_size >= size:
            raise RuntimeError(
                "deterministic parent aggregation did not reduce a non-bottom "
                "component"
            )
        projected = _galerkin_csr_reference(
            *current_csr, fine_to_coarse, coarse_size
        )
        prolongation = np.zeros((size, coarse_size), dtype=np.float64)
        prolongation[np.arange(size), fine_to_coarse] = 1.0
        dense_oracle = prolongation.T @ dense @ prolongation
        level["map_to_coarse_bytes"] = size * 4
        level["galerkin_matches_dense_oracle"] = bool(
            np.allclose(
                projected["reconstructed"],
                dense_oracle,
                rtol=0.0,
                atol=0.0,
            )
        )
        level["coarse_size"] = coarse_size
        reference_level["fine_to_coarse"] = fine_to_coarse
        levels.append(level)
        reference_levels.append(reference_level)
        current_coordinates = np.asarray(
            [key[1:] for key in ordered_parent_keys], dtype=np.int32
        )
        current_csr = (
            projected["row_offsets"],
            projected["column_indices"],
            projected["values"],
        )
        if len(levels) >= 32:
            raise RuntimeError("hierarchy exceeded the correctness probe limit")

    pattern_bytes = sum(level["operator_pattern_bytes"] for level in levels)
    value_bytes = sum(level["operator_value_bytes"] for level in levels)
    map_bytes = sum(level["map_to_coarse_bytes"] for level in levels)
    workspace_bytes = sum(level["workspace_upper_bytes"] for level in levels)
    bottom_inverse_bytes = (
        sum(size**2 for size in levels[-1]["component_sizes"]) * 4
    )
    return {
        "levels": levels,
        "_reference_levels": reference_levels,
        "level_count": len(levels),
        "nonbottom_level_count": len(levels) - 1,
        "sum_level_dofs": sum(level["size"] for level in levels),
        "bottom_component_cap": bottom_component_cap,
        "bottom_component_inverse_bytes": bottom_inverse_bytes,
        "operator_pattern_bytes": pattern_bytes,
        "operator_value_bytes": value_bytes,
        "aggregate_map_bytes": map_bytes,
        "workspace_upper_bytes": workspace_bytes,
        "steady_reserved_bytes_upper": (
            pattern_bytes
            + value_bytes
            + map_bytes
            + workspace_bytes
            + bottom_inverse_bytes
        ),
        "logical_dispatch_upper_bound": (
            "1 + nonbottom_levels * "
            "(2 * smoother_steps_per_side + 3)"
        ),
        "smoother_steps_selected": False,
        "aggregation_policy": (
            "component_aware_axis_aligned_parent_cells_width_2"
        ),
        "operator_projection": "generic_csr_galerkin",
        "bottom_dense_scope": "independent_connected_components",
    }


def _safe_weighted_jacobi_reference(row_offsets, column_indices, values):
    dense = _csr_to_dense_reference(row_offsets, column_indices, values)
    diagonal = np.diag(dense).copy()
    if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
        raise ValueError("SPD smoother requires a finite positive diagonal")
    inverse_sqrt_diagonal = 1.0 / np.sqrt(diagonal)
    normalized = (
        inverse_sqrt_diagonal[:, None]
        * dense
        * inverse_sqrt_diagonal[None, :]
    )
    absolute_row_sum_bound = float(np.max(np.sum(np.abs(normalized), axis=1)))
    if not np.isfinite(absolute_row_sum_bound) or absolute_row_sum_bound <= 0.0:
        raise ValueError("SPD smoother spectral bound must be finite and positive")
    maximum_eigenvalue = float(np.linalg.eigvalsh(normalized)[-1])
    damping = 1.0 / absolute_row_sum_bound
    return {
        "diagonal": diagonal,
        "damping": damping,
        "normalized_absolute_row_sum_bound": absolute_row_sum_bound,
        "normalized_maximum_eigenvalue": maximum_eigenvalue,
        "strict_spd_smoother_bound": bool(
            0.0 < damping < 2.0 / maximum_eigenvalue
        ),
    }


def _apply_symmetric_vcycle_reference(hierarchy, rhs, level_index=0):
    references = hierarchy["_reference_levels"]
    reference = references[level_index]
    row_offsets = reference["row_offsets"]
    column_indices = reference["column_indices"]
    values = reference["values"]
    dense = _csr_to_dense_reference(row_offsets, column_indices, values)
    rhs = np.asarray(rhs, dtype=np.float64)
    if rhs.shape != (dense.shape[0],):
        raise ValueError("V-cycle rhs does not match level size")
    if level_index == len(references) - 1:
        labels, component_sizes = _csr_component_reference(
            row_offsets, column_indices, values
        )
        solution = np.zeros_like(rhs)
        for component in range(len(component_sizes)):
            rows = np.flatnonzero(labels == component)
            solution[rows] = np.linalg.solve(
                dense[np.ix_(rows, rows)], rhs[rows]
            )
        return solution

    smoother = _safe_weighted_jacobi_reference(
        row_offsets, column_indices, values
    )
    diagonal = smoother["diagonal"]
    damping = smoother["damping"]
    solution = damping * rhs / diagonal
    residual = rhs - dense @ solution
    fine_to_coarse = reference["fine_to_coarse"]
    coarse_rhs = np.zeros(
        hierarchy["levels"][level_index + 1]["size"], dtype=np.float64
    )
    np.add.at(coarse_rhs, fine_to_coarse, residual)
    coarse_correction = _apply_symmetric_vcycle_reference(
        hierarchy, coarse_rhs, level_index + 1
    )
    solution += coarse_correction[fine_to_coarse]
    residual = rhs - dense @ solution
    solution += damping * residual / diagonal
    return solution


def _assemble_symmetric_vcycle_reference(hierarchy):
    size = hierarchy["levels"][0]["size"]
    inverse_operator = np.column_stack(
        [
            _apply_symmetric_vcycle_reference(
                hierarchy, np.eye(size, dtype=np.float64)[:, column]
            )
            for column in range(size)
        ]
    )
    damping = [
        _safe_weighted_jacobi_reference(
            level["row_offsets"], level["column_indices"], level["values"]
        )
        for level in hierarchy["_reference_levels"][:-1]
    ]
    left = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    right = np.cos(np.arange(size, dtype=np.float64))
    combined = _apply_symmetric_vcycle_reference(
        hierarchy, 1.25 * left - 0.75 * right
    )
    separate = 1.25 * _apply_symmetric_vcycle_reference(
        hierarchy, left
    ) - 0.75 * _apply_symmetric_vcycle_reference(hierarchy, right)
    symmetry_error = np.abs(inverse_operator - inverse_operator.T)
    return {
        "inverse_operator": inverse_operator,
        "damping": damping,
        "smoother_numeric_bytes": len(damping) * 4,
        "steady_reserved_bytes_upper": (
            hierarchy["steady_reserved_bytes_upper"] + len(damping) * 4
        ),
        "linearity_difference_linf": float(
            np.abs(combined - separate).max(initial=0.0)
        ),
        "symmetry_difference_linf": float(symmetry_error.max(initial=0.0)),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(inverse_operator)[0]),
        "pre_smoother_steps": 1,
        "post_smoother_steps": 1,
        "logical_dispatch_upper_bound": (
            1 + hierarchy["nonbottom_level_count"] * 5
        ),
        "host_algebra_correctness_only": True,
        "fixed_linear_operator": True,
        "symmetric_pre_post_composition": True,
        "bottom_inverse_spd": True,
    }


def _recursive_vcycle_topology_matches(reference, replacement):
    reference_levels = reference["_reference_levels"]
    replacement_levels = replacement["_reference_levels"]
    if len(reference_levels) != len(replacement_levels):
        return False
    for reference_level, replacement_level in zip(
        reference_levels, replacement_levels
    ):
        for name in ("row_offsets", "column_indices"):
            if not np.array_equal(
                reference_level[name], replacement_level[name]
            ):
                return False
        reference_map = reference_level["fine_to_coarse"]
        replacement_map = replacement_level["fine_to_coarse"]
        if (reference_map is None) != (replacement_map is None):
            return False
        if reference_map is not None and not np.array_equal(
            reference_map, replacement_map
        ):
            return False
    return True


def _recursive_vcycle_numeric_payload_reference(hierarchy):
    payload = {}
    references = hierarchy["_reference_levels"]
    for level_index, reference in enumerate(references[:-1]):
        prefix = f"l{level_index}"
        smoother = _safe_weighted_jacobi_reference(
            reference["row_offsets"],
            reference["column_indices"],
            reference["values"],
        )
        payload[f"{prefix}_values"] = reference["values"].astype(np.float32)
        payload[f"{prefix}_diagonal"] = smoother["diagonal"].astype(np.float32)
        payload[f"{prefix}_damping"] = np.asarray(
            [smoother["damping"]], dtype=np.float32
        )
    bottom = references[-1]
    labels, component_sizes = _csr_component_reference(
        bottom["row_offsets"], bottom["column_indices"], bottom["values"]
    )
    bottom_dense = _csr_to_dense_reference(
        bottom["row_offsets"], bottom["column_indices"], bottom["values"]
    )
    inverse_values = []
    for component in range(len(component_sizes)):
        rows = np.flatnonzero(labels == component)
        inverse = np.linalg.inv(bottom_dense[np.ix_(rows, rows)])
        inverse_values.extend(float(value) for value in inverse.flat)
    payload["bottom_inverse_values"] = np.asarray(
        inverse_values, dtype=np.float32
    )
    return payload


class _RecursiveVcycleNumericPublisher:
    """Host topology guard that creates numeric-only device publish buffers."""

    def __init__(self, ti, program, hierarchy):
        self._ti = ti
        self._program = program
        self._topology = []
        for level in hierarchy["_reference_levels"]:
            self._topology.append(
                {
                    "row_offsets": level["row_offsets"].copy(),
                    "column_indices": level["column_indices"].copy(),
                    "fine_to_coarse": (
                        None
                        if level["fine_to_coarse"] is None
                        else level["fine_to_coarse"].copy()
                    ),
                }
            )
        payload = _recursive_vcycle_numeric_payload_reference(hierarchy)
        self._role_shapes = {
            name: tuple(value.shape) for name, value in payload.items()
        }

    def _topology_matches(self, replacement):
        replacement_levels = replacement["_reference_levels"]
        if len(self._topology) != len(replacement_levels):
            return False
        for reference, candidate in zip(self._topology, replacement_levels):
            for name in ("row_offsets", "column_indices"):
                if not np.array_equal(reference[name], candidate[name]):
                    return False
            reference_map = reference["fine_to_coarse"]
            candidate_map = candidate["fine_to_coarse"]
            if (reference_map is None) != (candidate_map is None):
                return False
            if reference_map is not None and not np.array_equal(
                reference_map, candidate_map
            ):
                return False
        return True

    def create_sources(self, replacement_hierarchy):
        if self._ti.lang.impl.get_runtime().prog is not self._program:
            raise RuntimeError(
                "recursive V-cycle numeric publisher cannot cross Program reset"
            )
        if not self._topology_matches(replacement_hierarchy):
            raise ValueError(
                "recursive V-cycle numeric update requires identical level "
                "CSR patterns and aggregate maps"
            )
        payload = _recursive_vcycle_numeric_payload_reference(
            replacement_hierarchy
        )
        if set(payload) != set(self._role_shapes):
            raise ValueError(
                "recursive V-cycle numeric update role set does not match"
            )
        sources = {}
        for name, source in payload.items():
            if tuple(source.shape) != self._role_shapes[name]:
                raise ValueError(
                    f"recursive V-cycle numeric role {name!r} changed shape"
                )
            value = self._ti.ndarray(self._ti.f32, shape=source.shape)
            value.from_numpy(source)
            sources[name] = value
        return sources

    def debug_runtime_stats(self):
        return {
            "host_topology_metadata_bytes": sum(
                value.nbytes
                for level in self._topology
                for value in level.values()
                if value is not None
            ),
            "device_reserved_bytes": 0,
            "numeric_role_count": len(self._role_shapes),
            "numeric_payload_bytes": sum(
                int(np.prod(shape)) * 4 for shape in self._role_shapes.values()
            ),
        }


class _RecursiveVcycleGraphPlan:
    """Profile-private fixed symmetric V-cycle compiled as one CGraph."""

    def __init__(
        self,
        ti,
        *,
        program,
        hierarchy,
        topology_version,
        numeric_version,
        pre_kernel,
        restrict_kernel,
        bottom_kernel,
        post_kernel,
    ):
        self._ti = ti
        self._program = program
        self._hierarchy = hierarchy
        self._topology_version = topology_version
        self._numeric_version = numeric_version
        self._size = hierarchy["levels"][0]["size"]
        self._topology = {}
        self._numeric = {}
        self._workspace = {}
        self._resource_types = {}
        self._scalars = {}
        self._apply_calls = 0
        self._rejected_apply_calls = 0

        def add_resource(role, name, dtype, source):
            source = np.asarray(source)
            value = ti.ndarray(dtype, shape=source.shape)
            value.from_numpy(source)
            getattr(self, f"_{role}")[name] = value
            self._resource_types[name] = (dtype, source.ndim)

        references = hierarchy["_reference_levels"]
        numeric_payload = _recursive_vcycle_numeric_payload_reference(hierarchy)
        for level_index, reference in enumerate(references):
            prefix = f"l{level_index}"
            self._scalars[f"{prefix}_size"] = hierarchy["levels"][
                level_index
            ]["size"]
            if level_index < len(references) - 1:
                add_resource(
                    "topology",
                    f"{prefix}_row_offsets",
                    ti.i32,
                    reference["row_offsets"].astype(np.int32),
                )
                add_resource(
                    "topology",
                    f"{prefix}_columns",
                    ti.i32,
                    reference["column_indices"].astype(np.int32),
                )
                add_resource(
                    "numeric",
                    f"{prefix}_values",
                    ti.f32,
                    numeric_payload[f"{prefix}_values"],
                )
                add_resource(
                    "topology",
                    f"{prefix}_fine_to_coarse",
                    ti.i32,
                    reference["fine_to_coarse"].astype(np.int32),
                )
                add_resource(
                    "numeric",
                    f"{prefix}_diagonal",
                    ti.f32,
                    numeric_payload[f"{prefix}_diagonal"],
                )
                add_resource(
                    "numeric",
                    f"{prefix}_damping",
                    ti.f32,
                    numeric_payload[f"{prefix}_damping"],
                )
                add_resource(
                    "workspace",
                    f"{prefix}_pre_solution",
                    ti.f32,
                    np.zeros(hierarchy["levels"][level_index]["size"], np.float32),
                )
                self._scalars[f"{prefix}_coarse_size"] = hierarchy["levels"][
                    level_index + 1
                ]["size"]
            if level_index > 0:
                level_size = hierarchy["levels"][level_index]["size"]
                add_resource(
                    "workspace",
                    f"{prefix}_rhs",
                    ti.f32,
                    np.zeros(level_size, np.float32),
                )
                add_resource(
                    "workspace",
                    f"{prefix}_solution",
                    ti.f32,
                    np.zeros(level_size, np.float32),
                )

        bottom = references[-1]
        labels, component_sizes = _csr_component_reference(
            bottom["row_offsets"], bottom["column_indices"], bottom["values"]
        )
        component_offsets = [0]
        component_rows = []
        inverse_offsets = [0]
        row_component = np.empty(len(labels), dtype=np.int32)
        row_local = np.empty(len(labels), dtype=np.int32)
        for component in range(len(component_sizes)):
            rows = np.flatnonzero(labels == component).astype(np.int32)
            component_rows.extend(int(row) for row in rows)
            component_offsets.append(len(component_rows))
            inverse_offsets.append(inverse_offsets[-1] + len(rows) ** 2)
            for local, row in enumerate(rows):
                row_component[row] = component
                row_local[row] = local
        for name, source in (
            ("bottom_component_offsets", component_offsets),
            ("bottom_component_rows", component_rows),
            ("bottom_inverse_offsets", inverse_offsets),
            ("bottom_row_component", row_component),
            ("bottom_row_local", row_local),
        ):
            add_resource(
                "topology", name, ti.i32, np.asarray(source, dtype=np.int32)
            )
        add_resource(
            "numeric",
            "bottom_inverse_values",
            ti.f32,
            numeric_payload["bottom_inverse_values"],
        )

        symbols = {
            name: ti.graph.Arg(
                ti.graph.ArgKind.NDARRAY, name, dtype, ndim=ndim
            )
            for name, (dtype, ndim) in self._resource_types.items()
        }
        scalar_symbols = {
            name: ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, ti.i32)
            for name in self._scalars
        }
        sym_input = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
        )
        sym_output = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
        )
        graph_builder = ti.graph.GraphBuilder()
        for level_index in range(len(references) - 1):
            prefix = f"l{level_index}"
            rhs = sym_input if level_index == 0 else symbols[f"{prefix}_rhs"]
            graph_builder.dispatch(
                pre_kernel,
                scalar_symbols[f"{prefix}_size"],
                symbols[f"{prefix}_diagonal"],
                symbols[f"{prefix}_damping"],
                rhs,
                symbols[f"{prefix}_pre_solution"],
            )
            graph_builder.dispatch(
                restrict_kernel,
                scalar_symbols[f"{prefix}_size"],
                scalar_symbols[f"{prefix}_coarse_size"],
                symbols[f"{prefix}_row_offsets"],
                symbols[f"{prefix}_columns"],
                symbols[f"{prefix}_fine_to_coarse"],
                symbols[f"{prefix}_values"],
                rhs,
                symbols[f"{prefix}_pre_solution"],
                symbols[f"l{level_index + 1}_rhs"],
            )
        bottom_index = len(references) - 1
        bottom_prefix = f"l{bottom_index}"
        bottom_rhs = (
            sym_input
            if bottom_index == 0
            else symbols[f"{bottom_prefix}_rhs"]
        )
        bottom_output = (
            sym_output
            if bottom_index == 0
            else symbols[f"{bottom_prefix}_solution"]
        )
        graph_builder.dispatch(
            bottom_kernel,
            scalar_symbols[f"{bottom_prefix}_size"],
            symbols["bottom_component_offsets"],
            symbols["bottom_component_rows"],
            symbols["bottom_inverse_offsets"],
            symbols["bottom_row_component"],
            symbols["bottom_row_local"],
            symbols["bottom_inverse_values"],
            bottom_rhs,
            bottom_output,
        )
        for level_index in range(len(references) - 2, -1, -1):
            prefix = f"l{level_index}"
            rhs = sym_input if level_index == 0 else symbols[f"{prefix}_rhs"]
            output = (
                sym_output
                if level_index == 0
                else symbols[f"{prefix}_solution"]
            )
            graph_builder.dispatch(
                post_kernel,
                scalar_symbols[f"{prefix}_size"],
                symbols[f"{prefix}_row_offsets"],
                symbols[f"{prefix}_columns"],
                symbols[f"{prefix}_fine_to_coarse"],
                symbols[f"{prefix}_values"],
                symbols[f"{prefix}_diagonal"],
                symbols[f"{prefix}_damping"],
                rhs,
                symbols[f"{prefix}_pre_solution"],
                symbols[f"l{level_index + 1}_solution"],
                output,
            )
        self._graph = graph_builder.compile()
        self._graph_args = dict(self._scalars)
        self._graph_args.update(self._topology)
        self._graph_args.update(self._numeric)
        self._graph_args.update(self._workspace)
        self._graph_args.update({"input": None, "output": None})

    def _reject(self, message):
        self._rejected_apply_calls += 1
        raise RuntimeError(message)

    def _validate_vector(self, value, role):
        if value.dtype != self._ti.f32 or tuple(value.shape) != (self._size,):
            self._reject(f"recursive V-cycle {role} must be a size-matched f32 ndarray")
        if value._runtime_prog is not self._program or value.arr is None:
            self._reject(
                f"recursive V-cycle {role} must belong to the plan Program"
            )

    def apply(self, input_array, output_array):
        if self._ti.lang.impl.get_runtime().prog is not self._program:
            self._reject("recursive V-cycle cannot be used after Program reset")
        self._validate_vector(input_array, "input")
        self._validate_vector(output_array, "output")
        if _ndarray_allocation_id(input_array) == _ndarray_allocation_id(
            output_array
        ):
            self._reject("recursive V-cycle input/output alias is not supported")
        self._graph_args["input"] = input_array
        self._graph_args["output"] = output_array
        try:
            self._graph.run(self._graph_args)
        finally:
            self._graph_args["input"] = None
            self._graph_args["output"] = None
        self._apply_calls += 1

    def create_native_operator(self):
        return self._program._create_compiled_graph_linear_operator(
            self._graph._compiled_graph,
            self._size,
            self._topology_version,
            self._numeric_version,
            dict(self._scalars),
            {name: value.arr for name, value in self._topology.items()},
            {name: value.arr for name, value in self._numeric.items()},
            {name: value.arr for name, value in self._workspace.items()},
        )

    def create_numeric_publisher(self):
        return _RecursiveVcycleNumericPublisher(
            self._ti, self._program, self._hierarchy
        )

    def debug_runtime_stats(self):
        topology_bytes = sum(
            int(np.prod(value.shape)) * 4 for value in self._topology.values()
        )
        numeric_bytes = sum(
            int(np.prod(value.shape)) * 4 for value in self._numeric.values()
        )
        workspace_bytes = sum(
            int(np.prod(value.shape)) * 4 for value in self._workspace.values()
        )
        graph_execution = self._graph.execution_stats()
        return {
            "identity": {
                "method": "recursive_symmetric_vcycle",
                "size": self._size,
                "level_count": self._hierarchy["level_count"],
                "topology_version": self._topology_version,
                "numeric_version": self._numeric_version,
            },
            "operations": {
                "apply_calls": self._apply_calls,
                "rejected_apply_calls": self._rejected_apply_calls,
                "graph_node_count": graph_execution.node_count,
                "graph_dispatch_count": graph_execution.dispatch_count,
                "kernel_dispatches_per_apply": (
                    1 + 3 * self._hierarchy["nonbottom_level_count"]
                ),
                "host_graph_submissions_per_apply": 1,
                "explicit_apply_host_synchronizations": 0,
            },
            "resources": {
                "topology_reserved_bytes": topology_bytes,
                "numeric_reserved_bytes": numeric_bytes,
                "workspace_reserved_bytes": workspace_bytes,
                "plan_owned_reserved_bytes": (
                    topology_bytes + numeric_bytes + workspace_bytes
                ),
            },
            "contract": {
                "operator_projection": "generic_csr_galerkin",
                "pre_smoother_steps": 1,
                "post_smoother_steps": 1,
                "damping": "normalized_absolute_row_sum_safe_bound",
                "bottom_inverse": "exact_per_connected_component",
                "fixed_linear_spd_assumption_verified_by_host_oracle": True,
                "no_host_fallback": True,
                "profile_private_not_public_solver": True,
            },
        }


def _build_structured_coarse_reference(
    *, dimensions, cells_per_block, neighbors, fine_to_coarse, coarse_size
):
    """Assemble ``P.T @ A @ P`` from fine adjacency without dense storage."""
    active_size = len(fine_to_coarse)
    directional_neighbors = np.full(
        (coarse_size, 2 * dimensions), -1, dtype=np.int32
    )
    directional_values = np.zeros(
        (coarse_size, 2 * dimensions), dtype=np.float64
    )
    diagonal = np.full(
        coarse_size,
        cells_per_block * 2 * dimensions,
        dtype=np.float64,
    )
    for fine_row in range(active_size):
        coarse_row = int(fine_to_coarse[fine_row])
        for direction, fine_neighbor in enumerate(neighbors[fine_row]):
            if fine_neighbor < 0:
                continue
            coarse_column = int(fine_to_coarse[int(fine_neighbor)])
            if coarse_column == coarse_row:
                diagonal[coarse_row] -= 1.0
                continue
            previous = int(directional_neighbors[coarse_row, direction])
            if previous not in (-1, coarse_column):
                raise ValueError(
                    "one coarse direction reaches multiple aggregates; "
                    "the structured coarse stencil contract does not hold"
                )
            directional_neighbors[coarse_row, direction] = coarse_column
            directional_values[coarse_row, direction] -= 1.0

    reconstructed = np.zeros((coarse_size, coarse_size), dtype=np.float64)
    row_offsets = [0]
    column_indices = []
    values = []
    max_row_nnz = 0
    for row in range(coarse_size):
        entries = {row: float(diagonal[row])}
        for direction in range(2 * dimensions):
            column = int(directional_neighbors[row, direction])
            if column >= 0:
                entries[column] = entries.get(column, 0.0) + float(
                    directional_values[row, direction]
                )
        sorted_entries = sorted(entries.items())
        max_row_nnz = max(max_row_nnz, len(sorted_entries))
        for column, value in sorted_entries:
            reconstructed[row, column] = value
            column_indices.append(column)
            values.append(value)
        row_offsets.append(len(column_indices))

    row_offsets = np.asarray(row_offsets, dtype=np.int32)
    column_indices = np.asarray(column_indices, dtype=np.int32)
    values = np.asarray(values, dtype=np.float32)
    scalar_bytes = 4
    directional_topology_bytes = (
        coarse_size * 2 * dimensions * scalar_bytes
    )
    directional_numeric_bytes = (
        coarse_size * (2 * dimensions + 1) * scalar_bytes
    )
    csr_pattern_bytes = (
        coarse_size + 1 + len(column_indices)
    ) * scalar_bytes
    csr_value_bytes = len(values) * scalar_bytes
    offdiagonal = reconstructed.copy()
    np.fill_diagonal(offdiagonal, 0.0)
    reciprocal = all(
        reconstructed[row, column] == reconstructed[column, row]
        for row in range(coarse_size)
        for column in column_indices[
            row_offsets[row] : row_offsets[row + 1]
        ]
    )
    component_sizes = []
    unvisited = set(range(coarse_size))
    while unvisited:
        pending = [unvisited.pop()]
        component_size = 0
        while pending:
            row = pending.pop()
            component_size += 1
            for column in column_indices[
                row_offsets[row] : row_offsets[row + 1]
            ]:
                column = int(column)
                if column != row and column in unvisited:
                    unvisited.remove(column)
                    pending.append(column)
        component_sizes.append(component_size)
    component_sizes.sort(reverse=True)
    return {
        "directional_neighbors": directional_neighbors,
        "directional_values": directional_values.astype(np.float32),
        "diagonal": diagonal.astype(np.float32),
        "csr_row_offsets": row_offsets,
        "csr_column_indices": column_indices,
        "csr_values": values,
        "reconstructed": reconstructed,
        "positive_diagonal": bool(np.all(diagonal > 0.0)),
        "nonpositive_offdiagonal": bool(np.all(offdiagonal <= 0.0)),
        "reciprocal_weighted_adjacency": bool(reciprocal),
        "actual_nnz": int(len(values)),
        "max_row_nnz": int(max_row_nnz),
        "row_nnz_upper_bound": 2 * dimensions + 1,
        "component_sizes": component_sizes,
        "directional_topology_bytes": directional_topology_bytes,
        "directional_numeric_bytes": directional_numeric_bytes,
        "directional_total_bytes": (
            directional_topology_bytes + directional_numeric_bytes
        ),
        "csr_pattern_bytes": csr_pattern_bytes,
        "csr_value_bytes": csr_value_bytes,
        "csr_total_bytes": csr_pattern_bytes + csr_value_bytes,
        "dense_inverse_bytes": coarse_size**2 * scalar_bytes,
        "component_dense_inverse_bytes": (
            sum(size**2 for size in component_sizes) * scalar_bytes
        ),
    }


def _build_two_level_reference(
    *, dimensions, cells_per_block, neighbors, active_size
):
    """Build deterministic profile-only Galerkin data and host references."""
    neighbors = np.asarray(neighbors, dtype=np.int32)
    if neighbors.shape != (active_size, 2 * dimensions):
        raise ValueError("neighbor snapshot does not match the active fine level")
    if active_size % cells_per_block != 0:
        raise ValueError("active fine level does not contain complete bricks")
    coarse_size = active_size // cells_per_block
    fine_to_coarse = (
        np.arange(active_size, dtype=np.int32) // cells_per_block
    )
    fine_operator = np.eye(active_size, dtype=np.float64) * (2 * dimensions)
    for row in range(active_size):
        for neighbor in neighbors[row]:
            if neighbor >= 0:
                fine_operator[row, int(neighbor)] = -1.0
    prolongation = np.zeros((active_size, coarse_size), dtype=np.float64)
    prolongation[np.arange(active_size), fine_to_coarse] = 1.0
    coarse_operator = prolongation.T @ fine_operator @ prolongation
    structured_coarse = _build_structured_coarse_reference(
        dimensions=dimensions,
        cells_per_block=cells_per_block,
        neighbors=neighbors,
        fine_to_coarse=fine_to_coarse,
        coarse_size=coarse_size,
    )
    coarse_inverse = np.linalg.inv(coarse_operator)
    preconditioner = (
        np.eye(active_size, dtype=np.float64) / (2 * dimensions)
        + prolongation @ coarse_inverse @ prolongation.T
    )
    return {
        "fine_to_coarse": fine_to_coarse,
        "coarse_inverse": coarse_inverse.astype(np.float32),
        "preconditioner": preconditioner,
        "coarse_storage": {
            "directional_reconstructs_dense_galerkin": bool(
                np.array_equal(
                    structured_coarse["reconstructed"], coarse_operator
                )
            ),
            "positive_diagonal": structured_coarse["positive_diagonal"],
            "nonpositive_offdiagonal": structured_coarse[
                "nonpositive_offdiagonal"
            ],
            "reciprocal_weighted_adjacency": structured_coarse[
                "reciprocal_weighted_adjacency"
            ],
            "actual_nnz": structured_coarse["actual_nnz"],
            "max_row_nnz": structured_coarse["max_row_nnz"],
            "row_nnz_upper_bound": structured_coarse[
                "row_nnz_upper_bound"
            ],
            "directional_topology_bytes": structured_coarse[
                "directional_topology_bytes"
            ],
            "directional_numeric_bytes": structured_coarse[
                "directional_numeric_bytes"
            ],
            "directional_total_bytes": structured_coarse[
                "directional_total_bytes"
            ],
            "csr_pattern_bytes": structured_coarse["csr_pattern_bytes"],
            "csr_value_bytes": structured_coarse["csr_value_bytes"],
            "csr_total_bytes": structured_coarse["csr_total_bytes"],
            "dense_inverse_bytes": structured_coarse["dense_inverse_bytes"],
            "component_count": len(structured_coarse["component_sizes"]),
            "component_sizes": structured_coarse["component_sizes"],
            "component_dense_inverse_bytes": structured_coarse[
                "component_dense_inverse_bytes"
            ],
            "assembly": "fine_adjacency_plus_aggregate_map",
            "directional_bound_scope": (
                "axis_aligned_complete_brick_scalar_poisson_only"
            ),
            "generic_sparse_system_storage": "csr_or_bsr",
            "directional_stencil_is_optional_specialization": True,
            "host_dense_matrix_role": "small_correctness_oracle_only",
            "solve_workspace_included": False,
        },
        "fine_symmetric": bool(
            np.allclose(fine_operator, fine_operator.T, rtol=0.0, atol=0.0)
        ),
        "coarse_symmetric": bool(
            np.allclose(coarse_operator, coarse_operator.T, rtol=0.0, atol=0.0)
        ),
        "preconditioner_symmetric": bool(
            np.allclose(preconditioner, preconditioner.T, rtol=0.0, atol=1e-14)
        ),
        "fine_min_eigenvalue": float(np.linalg.eigvalsh(fine_operator)[0]),
        "coarse_min_eigenvalue": float(
            np.linalg.eigvalsh(coarse_operator)[0]
        ),
        "preconditioner_min_eigenvalue": float(
            np.linalg.eigvalsh(preconditioner)[0]
        ),
    }


def _backend_methods(arch_name):
    methods = {
        "cpu": ("cpu_native", "cpu_native"),
        "cuda": ("cuda_device", "cuda_device"),
        "vulkan": ("vulkan_native_radix_u32", "vulkan_native"),
    }
    try:
        return methods[arch_name]
    except KeyError as exc:
        raise RuntimeError(
            f"active DOF map does not support backend {arch_name!r}; "
            "no fallback was performed"
        ) from exc


def _encode_block(coordinates, root_blocks):
    key = 0
    for coordinate in coordinates:
        if coordinate < 0 or coordinate >= root_blocks:
            raise ValueError("active block coordinate is outside the root domain")
        key = key * root_blocks + coordinate
    return key


def _producer_coordinates(blocks, root_blocks, capacity, overflow=False):
    keys = [_encode_block(block, root_blocks) for block in blocks]
    if len(set(keys)) != len(keys):
        raise ValueError("canonical active blocks must be unique")
    raw = list(reversed(blocks))
    raw.extend((blocks[0], blocks[-1]))
    if overflow:
        raw.extend((blocks[1], blocks[-2]))
    if len(raw) > capacity:
        raise ValueError("producer capacity is smaller than the fixture")
    coordinates = np.zeros((capacity, len(blocks[0])), dtype=np.int32)
    coordinates[: len(raw)] = np.asarray(raw, dtype=np.int32)
    return coordinates, len(raw), sorted(keys)


def _topologies(dimensions):
    if dimensions == 2:
        return (
            ((1, 1), (1, 2), (2, 1), (3, 3)),
            ((1, 2), (2, 2), (3, 2), (3, 3)),
        )
    if dimensions == 3:
        return (
            ((1, 1, 1), (1, 1, 2), (2, 1, 1), (3, 3, 3)),
            ((1, 1, 2), (2, 2, 2), (3, 2, 2), (3, 3, 3)),
        )
    raise ValueError("dimensions must be 2 or 3")


def run_initialized(
    ti,
    *,
    dimensions=3,
    root_blocks=5,
    block_size=2,
    candidate_capacity=8,
):
    """Build, migrate, and validate a compact active-block DOF map."""
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    for name, value in {
        "root_blocks": root_blocks,
        "block_size": block_size,
        "candidate_capacity": candidate_capacity,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    initial_blocks, migrated_blocks = _topologies(dimensions)
    expected_blocks = len(initial_blocks)
    expected_dofs = expected_blocks * block_size**dimensions
    if root_blocks <= 3:
        raise ValueError("root_blocks must be at least 4 for the canonical topologies")
    if candidate_capacity < 8:
        raise ValueError("candidate_capacity must be at least 8")

    arch_name = _arch_name(ti)
    sort_method, unique_method = _backend_methods(arch_name)
    cells_per_block = block_size**dimensions
    max_active_blocks = candidate_capacity - 1
    max_dofs = max_active_blocks * cells_per_block
    domain_size = root_blocks * block_size
    producer_capacity = candidate_capacity + 2

    producer_blocks = ti.ndarray(
        ti.i32, shape=(producer_capacity, dimensions)
    )
    candidates = ti.ndarray(ti.u32, shape=candidate_capacity)
    candidate_count = ti.ndarray(ti.i32, shape=1)
    candidate_status = ti.ndarray(ti.i32, shape=1)
    unique_keys = ti.ndarray(ti.u32, shape=candidate_capacity)
    unique_count = ti.ndarray(ti.i32, shape=1)
    dof_count = ti.ndarray(ti.i32, shape=1)
    dof_coordinates = ti.ndarray(ti.i32, shape=(max_dofs, dimensions))
    neighbors = ti.ndarray(ti.i32, shape=(max_dofs, 2 * dimensions))
    compact_x = ti.ndarray(ti.f32, shape=max_dofs)
    compact_ax = ti.ndarray(ti.f32, shape=max_dofs)
    gathered_x = ti.ndarray(ti.f32, shape=max_dofs)
    oracle_ax = ti.ndarray(ti.f32, shape=max_dofs)
    native_input = ti.ndarray(ti.f32, shape=expected_dofs)
    native_output = ti.ndarray(ti.f32, shape=expected_dofs)
    sort_workspace = ti.algorithms.SortWorkspace(max_items=candidate_capacity)
    unique_workspace = ti.algorithms.RunLengthWorkspace(
        max_items=candidate_capacity
    )

    builder = ti.FieldsBuilder()
    grid_x = ti.field(ti.f32)
    grid_ax = ti.field(ti.f32)
    dof_plus_one = ti.field(ti.i32)
    axes = ti.ij if dimensions == 2 else ti.ijk
    pointer_kwargs = (
        {"vk_max_active": max_active_blocks}
        if arch_name in ("cuda", "vulkan")
        else {}
    )
    pointer = builder.pointer(
        axes, (root_blocks,) * dimensions, **pointer_kwargs
    )
    pointer.dense(axes, (block_size,) * dimensions).place(
        grid_x, grid_ax, dof_plus_one
    )
    tree = builder.finalize()
    ti.lang.impl.get_runtime().materialize()
    ti.sync()
    tree_identity = [int(tree.id), int(tree.generation)]

    @ti.func
    def decode_block(key):
        coordinates = ti.Vector.zero(ti.i32, dimensions)
        remaining = ti.cast(key, ti.i32)
        for axis in ti.static(range(dimensions - 1, -1, -1)):
            coordinates[axis] = remaining % root_blocks
            remaining //= root_blocks
        return coordinates

    @ti.func
    def decode_local(local_linear):
        coordinates = ti.Vector.zero(ti.i32, dimensions)
        remaining = local_linear
        for axis in ti.static(range(dimensions - 1, -1, -1)):
            coordinates[axis] = remaining % block_size
            remaining //= block_size
        return coordinates

    @ti.kernel
    def reset_candidate_staging(
        candidates_arg: ti.types.ndarray(dtype=ti.u32, ndim=1),
        candidate_count_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        candidate_status_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        candidate_count_arg[0] = 0
        candidate_status_arg[0] = 0
        for index in range(candidate_capacity):
            candidates_arg[index] = ti.u32(0xFFFFFFFF)

    @ti.kernel
    def emit_candidate_keys(
        producer_blocks_arg: ti.types.ndarray(dtype=ti.i32, ndim=2),
        producer_count: ti.i32,
        candidates_arg: ti.types.ndarray(dtype=ti.u32, ndim=1),
        candidate_count_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        candidate_status_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for producer in range(producer_count):
            valid = True
            key = ti.u32(0)
            for axis in ti.static(range(dimensions)):
                coordinate = producer_blocks_arg[producer, axis]
                valid = valid and coordinate >= 0 and coordinate < root_blocks
                key = key * ti.u32(root_blocks) + ti.cast(coordinate, ti.u32)
            if valid:
                slot = ti.atomic_add(candidate_count_arg[0], 1)
                # Reserve the final slot for a sentinel so full-capacity sort
                # and consecutive unique never need a host active-prefix size.
                if slot < candidate_capacity - 1:
                    candidates_arg[slot] = key
                else:
                    ti.atomic_max(candidate_status_arg[0], 1)
            else:
                ti.atomic_max(candidate_status_arg[0], 2)

    @ti.kernel
    def build_dof_map(
        unique_keys_arg: ti.types.ndarray(dtype=ti.u32, ndim=1),
        unique_count_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dof_count_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dof_coordinates_arg: ti.types.ndarray(dtype=ti.i32, ndim=2),
        compact_x_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        dof_count_arg[0] = (unique_count_arg[0] - 1) * cells_per_block
        for block_ordinal in range(candidate_capacity):
            if block_ordinal < unique_count_arg[0]:
                key = unique_keys_arg[block_ordinal]
                if key != ti.u32(0xFFFFFFFF):
                    block = decode_block(key)
                    for local_linear in range(cells_per_block):
                        local = decode_local(local_linear)
                        coordinate = block * block_size + local
                        dof = block_ordinal * cells_per_block + local_linear
                        dof_plus_one[coordinate] = dof + 1
                        for axis in ti.static(range(dimensions)):
                            dof_coordinates_arg[dof, axis] = coordinate[axis]
                        compact_x_arg[dof] = ti.cast(
                            1 + (dof * 7 + block_ordinal * 11) % 19,
                            ti.f32,
                        )

    @ti.kernel
    def build_neighbor_table(
        dof_count_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dof_coordinates_arg: ti.types.ndarray(dtype=ti.i32, ndim=2),
        neighbors_arg: ti.types.ndarray(dtype=ti.i32, ndim=2),
    ):
        for dof in range(max_dofs):
            if dof < dof_count_arg[0]:
                coordinate = ti.Vector.zero(ti.i32, dimensions)
                for axis in ti.static(range(dimensions)):
                    coordinate[axis] = dof_coordinates_arg[dof, axis]
                for axis in ti.static(range(dimensions)):
                    offset = ti.Vector.zero(ti.i32, dimensions)
                    offset[axis] = 1
                    lower = -1
                    upper = -1
                    if coordinate[axis] > 0:
                        lower = dof_plus_one[coordinate - offset] - 1
                    if coordinate[axis] + 1 < domain_size:
                        upper = dof_plus_one[coordinate + offset] - 1
                    neighbors_arg[dof, 2 * axis] = lower
                    neighbors_arg[dof, 2 * axis + 1] = upper

    @ti.kernel
    def scatter_compact_to_grid(
        dof_count_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dof_coordinates_arg: ti.types.ndarray(dtype=ti.i32, ndim=2),
        compact_x_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for dof in range(max_dofs):
            if dof < dof_count_arg[0]:
                coordinate = ti.Vector.zero(ti.i32, dimensions)
                for axis in ti.static(range(dimensions)):
                    coordinate[axis] = dof_coordinates_arg[dof, axis]
                grid_x[coordinate] = compact_x_arg[dof]

    @ti.kernel
    def gather_grid_x(
        dof_count_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dof_coordinates_arg: ti.types.ndarray(dtype=ti.i32, ndim=2),
        gathered_x_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for dof in range(max_dofs):
            if dof < dof_count_arg[0]:
                coordinate = ti.Vector.zero(ti.i32, dimensions)
                for axis in ti.static(range(dimensions)):
                    coordinate[axis] = dof_coordinates_arg[dof, axis]
                gathered_x_arg[dof] = grid_x[coordinate]

    @ti.kernel
    def apply_compact_operator(
        active_size: ti.i32,
        neighbors_arg: ti.types.ndarray(dtype=ti.i32, ndim=2),
        compact_x_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
        compact_ax_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for dof in range(max_dofs):
            if dof < active_size:
                value = ti.cast(2 * dimensions, ti.f32) * compact_x_arg[dof]
                for slot in ti.static(range(2 * dimensions)):
                    neighbor = neighbors_arg[dof, slot]
                    if neighbor >= 0:
                        value -= compact_x_arg[neighbor]
                compact_ax_arg[dof] = value

    @ti.kernel
    def restrict_two_level(
        active_size: ti.i32,
        coarse_size: ti.i32,
        fine_to_coarse_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        input_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
        coarse_rhs_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for coarse in range(coarse_size):
            value = 0.0
            for fine in range(active_size):
                if fine_to_coarse_arg[fine] == coarse:
                    value += input_arg[fine]
            coarse_rhs_arg[coarse] = value

    @ti.kernel
    def solve_two_level_coarse(
        coarse_size: ti.i32,
        coarse_inverse_arg: ti.types.ndarray(dtype=ti.f32, ndim=2),
        coarse_rhs_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
        coarse_solution_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(coarse_size):
            value = 0.0
            for column in range(coarse_size):
                value += (
                    coarse_inverse_arg[row, column]
                    * coarse_rhs_arg[column]
                )
            coarse_solution_arg[row] = value

    @ti.kernel
    def prolong_two_level(
        active_size: ti.i32,
        fine_to_coarse_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        input_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
        coarse_solution_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for fine in range(active_size):
            coarse = fine_to_coarse_arg[fine]
            output_arg[fine] = (
                input_arg[fine] / ti.cast(2 * dimensions, ti.f32)
                + coarse_solution_arg[coarse]
            )

    @ti.kernel
    def apply_snode_oracle():
        for coordinate in ti.grouped(grid_x):
            value = ti.cast(2 * dimensions, ti.f32) * grid_x[coordinate]
            for axis in ti.static(range(dimensions)):
                offset = ti.Vector.zero(ti.i32, dimensions)
                offset[axis] = 1
                if coordinate[axis] > 0:
                    value -= grid_x[coordinate - offset]
                if coordinate[axis] + 1 < domain_size:
                    value -= grid_x[coordinate + offset]
            grid_ax[coordinate] = value

    @ti.kernel
    def gather_oracle_ax(
        dof_count_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dof_coordinates_arg: ti.types.ndarray(dtype=ti.i32, ndim=2),
        oracle_ax_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for dof in range(max_dofs):
            if dof < dof_count_arg[0]:
                coordinate = ti.Vector.zero(ti.i32, dimensions)
                for axis in ti.static(range(dimensions)):
                    coordinate[axis] = dof_coordinates_arg[dof, axis]
                oracle_ax_arg[dof] = grid_ax[coordinate]

    @ti.kernel
    def compact_difference_l1(
        dof_count_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        compact_ax_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
        oracle_ax_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ) -> ti.f32:
        difference = 0.0
        for dof in range(max_dofs):
            if dof < dof_count_arg[0]:
                difference += ti.abs(
                    compact_ax_arg[dof] - oracle_ax_arg[dof]
                )
        return difference

    @ti.kernel
    def gather_difference_l1(
        dof_count_arg: ti.types.ndarray(dtype=ti.i32, ndim=1),
        compact_x_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
        gathered_x_arg: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ) -> ti.f32:
        difference = 0.0
        for dof in range(max_dofs):
            if dof < dof_count_arg[0]:
                difference += ti.abs(
                    compact_x_arg[dof] - gathered_x_arg[dof]
                )
        return difference

    root_ranges = (root_blocks,) * dimensions

    @ti.kernel
    def count_active_blocks() -> ti.i32:
        count = 0
        for block in ti.grouped(ti.ndrange(*root_ranges)):
            if ti.is_active(pointer, block):
                count += 1
        return count

    initial_probe = tuple(value * block_size for value in initial_blocks[0])

    @ti.kernel
    def stale_initial_probe_l1() -> ti.f32:
        return ti.abs(grid_x[initial_probe])

    def build_two_level_plan():
        active_size = int(dof_count.to_numpy()[0])
        active_blocks = int(unique_count.to_numpy()[0]) - 1
        neighbor_snapshot = neighbors.to_numpy()[
            :active_size, : 2 * dimensions
        ]
        reference = _build_two_level_reference(
            dimensions=dimensions,
            cells_per_block=cells_per_block,
            neighbors=neighbor_snapshot,
            active_size=active_size,
        )
        if active_blocks != active_size // cells_per_block:
            raise RuntimeError("coarse aggregation does not match active bricks")
        ordered_keys = unique_keys.to_numpy()[:active_blocks].astype(np.uint32)
        fine_to_coarse_source = reference["fine_to_coarse"].copy()
        coarse_inverse_source = reference["coarse_inverse"].copy()
        fine_to_coarse_source_ref = weakref.ref(fine_to_coarse_source)
        coarse_inverse_source_ref = weakref.ref(coarse_inverse_source)
        plan = _TwoLevelCompactPreconditionerPlan(
            ti,
            program=ti.lang.impl.get_runtime().prog,
            dimensions=dimensions,
            active_size=active_size,
            coarse_size=active_blocks,
            topology_version=topology_version,
            numeric_version=level_numeric_version,
            current_topology_version=lambda: topology_version,
            current_numeric_version=lambda: level_numeric_version,
            fine_to_coarse=fine_to_coarse_source,
            coarse_inverse=coarse_inverse_source,
            restrict_kernel=restrict_two_level,
            coarse_solve_kernel=solve_two_level_coarse,
            prolong_kernel=prolong_two_level,
        )
        del fine_to_coarse_source
        del coarse_inverse_source
        gc.collect()
        source_snapshots_released = (
            fine_to_coarse_source_ref() is None
            and coarse_inverse_source_ref() is None
        )
        return (
            plan,
            reference,
            [int(value) for value in ordered_keys],
            source_snapshots_released,
        )

    def validate_two_level(plan, reference, ordered_keys):
        input_host = compact_x.to_numpy()[:expected_dofs].astype(
            np.float32, copy=True
        )
        native_input.from_numpy(input_host)
        native_output.fill(-137.0)
        plan.apply(
            native_input,
            native_output,
            expected_topology_version=topology_version,
            expected_numeric_version=level_numeric_version,
        )
        actual = native_output.to_numpy()
        expected = reference["preconditioner"] @ input_host.astype(np.float64)
        difference = np.abs(actual.astype(np.float64) - expected)
        fine_to_coarse = reference["fine_to_coarse"]
        return {
            "fine_size": expected_dofs,
            "coarse_size": len(ordered_keys),
            "coarse_ordered_keys": ordered_keys,
            "coarse_ordering_matches_fine_bricks": (
                ordered_keys == sorted(ordered_keys)
            ),
            "fine_to_coarse_complete": bool(
                np.array_equal(
                    fine_to_coarse,
                    np.arange(expected_dofs, dtype=np.int32)
                    // cells_per_block,
                )
            ),
            "fine_operator_symmetric": reference["fine_symmetric"],
            "fine_operator_min_eigenvalue": reference[
                "fine_min_eigenvalue"
            ],
            "coarse_operator_symmetric": reference["coarse_symmetric"],
            "coarse_operator_min_eigenvalue": reference[
                "coarse_min_eigenvalue"
            ],
            "coarse_storage": dict(reference["coarse_storage"]),
            "preconditioner_symmetric": reference[
                "preconditioner_symmetric"
            ],
            "preconditioner_min_eigenvalue": reference[
                "preconditioner_min_eigenvalue"
            ],
            "output_difference_l1": float(difference.sum()),
            "output_difference_linf": float(difference.max(initial=0.0)),
        }

    topology_version = 0
    level_numeric_version = 1
    operator_plan = None

    def rebuild(blocks, *, force_overflow=False):
        nonlocal operator_plan, topology_version
        version_before = topology_version
        producer_input, producer_count, expected = _producer_coordinates(
            blocks,
            root_blocks,
            producer_capacity,
            overflow=force_overflow,
        )
        producer_blocks.from_numpy(producer_input)
        reset_candidate_staging(
            candidates, candidate_count, candidate_status
        )
        emit_candidate_keys(
            producer_blocks,
            producer_count,
            candidates,
            candidate_count,
            candidate_status,
        )
        ti.sync()
        emitted_count = int(candidate_count.to_numpy()[0])
        status = int(candidate_status.to_numpy()[0])
        attempt = {
            "producer_count": producer_count,
            "emitted_count": emitted_count,
            "status_code": status,
            "status": {0: "success", 1: "capacity_overflow", 2: "invalid_key"}[
                status
            ],
            "version_before": version_before,
            "version_after": version_before,
            "published": False,
            "old_topology_preserved": status != 0,
        }
        if status != 0:
            return expected, attempt

        ti.algorithms.sort(
            candidates,
            method=sort_method,
            workspace=sort_workspace,
        )
        ti.algorithms.experimental_unique(
            candidates,
            unique_keys,
            unique_count,
            method=unique_method,
            workspace=unique_workspace,
        )
        ti.sync()
        staged_unique_count = int(unique_count.to_numpy()[0])
        staged_keys = unique_keys.to_numpy()
        if (
            staged_unique_count < 1
            or staged_keys[staged_unique_count - 1] != SENTINEL_KEY
        ):
            raise RuntimeError(
                "active DOF staging did not preserve the reserved sentinel"
            )
        # Capacity/provider/staging failures occur before the first SNode
        # mutation. This is a host-mediated two-phase publish boundary.
        if topology_version != 0:
            pointer.deactivate_all()
        build_dof_map(
            unique_keys,
            unique_count,
            dof_count,
            dof_coordinates,
            compact_x,
        )
        build_neighbor_table(dof_count, dof_coordinates, neighbors)
        ti.sync()
        topology_version += 1
        operator_plan = _StructuredActiveDofOperatorPlan(
            ti,
            program=ti.lang.impl.get_runtime().prog,
            dimensions=dimensions,
            capacity=max_dofs,
            active_size=(staged_unique_count - 1) * cells_per_block,
            topology_version=topology_version,
            numeric_version=1,
            current_topology_version=lambda: topology_version,
            neighbors=neighbors,
            apply_kernel=apply_compact_operator,
        )
        attempt.update(
            version_after=topology_version,
            published=True,
            old_topology_preserved=False,
            staged_active_blocks=staged_unique_count - 1,
        )
        return expected, attempt

    def validate(expected_keys, plan):
        scatter_compact_to_grid(dof_count, dof_coordinates, compact_x)
        gather_grid_x(dof_count, dof_coordinates, gathered_x)
        plan.apply(
            compact_x,
            compact_ax,
            expected_topology_version=topology_version,
            expected_numeric_version=1,
        )
        apply_snode_oracle()
        gather_oracle_ax(dof_count, dof_coordinates, oracle_ax)
        ti.sync()
        active_count = int(unique_count.to_numpy()[0]) - 1
        ordered = unique_keys.to_numpy()[:active_count].astype(np.uint32)
        return {
            "active_blocks": count_active_blocks(),
            "active_dofs": int(dof_count.to_numpy()[0]),
            "ordered_keys": [int(value) for value in ordered],
            "ordering_matches": np.array_equal(
                ordered, np.asarray(expected_keys, dtype=np.uint32)
            ),
            "gather_difference_l1": gather_difference_l1(
                dof_count, compact_x, gathered_x
            ),
            "operator_difference_l1": compact_difference_l1(
                dof_count, compact_ax, oracle_ax
            ),
        }

    initial_expected, initial_attempt = rebuild(initial_blocks)
    initial_operator_plan = operator_plan
    initial = validate(initial_expected, initial_operator_plan)
    (
        initial_two_level_plan,
        initial_two_level_reference,
        initial_coarse_keys,
        initial_two_level_sources_released,
    ) = build_two_level_plan()
    current_two_level_plan = initial_two_level_plan
    initial_two_level = validate_two_level(
        initial_two_level_plan,
        initial_two_level_reference,
        initial_coarse_keys,
    )
    overflow_expected, overflow_attempt = rebuild(
        migrated_blocks, force_overflow=True
    )
    overflow_preserved_operator_plan = operator_plan is initial_operator_plan
    overflow_preserved_two_level_plan = (
        current_two_level_plan is initial_two_level_plan
    )
    after_overflow = validate(initial_expected, operator_plan)
    migrated_expected, migrated_attempt = rebuild(migrated_blocks)
    migrated_operator_plan = operator_plan
    (
        migrated_two_level_plan,
        migrated_two_level_reference,
        migrated_coarse_keys,
        migrated_two_level_sources_released,
    ) = build_two_level_plan()
    current_two_level_plan = migrated_two_level_plan
    stale_two_level_output_before = np.full(
        expected_dofs, -211.0, dtype=np.float32
    )
    native_output.from_numpy(stale_two_level_output_before)
    stale_two_level_plan_rejected = False
    try:
        initial_two_level_plan.apply(native_input, native_output)
    except RuntimeError as exc:
        stale_two_level_plan_rejected = "topology version is stale" in str(exc)
    stale_two_level_output_preserved = np.array_equal(
        stale_two_level_output_before, native_output.to_numpy()
    )
    migrated_two_level = validate_two_level(
        migrated_two_level_plan,
        migrated_two_level_reference,
        migrated_coarse_keys,
    )
    two_level_alias_input_before = native_input.to_numpy()
    two_level_alias_rejected = False
    try:
        migrated_two_level_plan.apply(native_input, native_input)
    except RuntimeError as exc:
        two_level_alias_rejected = "does not support input/output alias" in str(
            exc
        )
    two_level_alias_input_preserved = np.array_equal(
        two_level_alias_input_before, native_input.to_numpy()
    )
    initial_two_level_plan_stats = initial_two_level_plan.debug_runtime_stats()
    initial_two_level_plan = None
    initial_two_level_reference = None
    gc.collect()

    level_numeric_version += 1
    stale_two_level_numeric_output_before = np.full(
        expected_dofs, -223.0, dtype=np.float32
    )
    native_output.from_numpy(stale_two_level_numeric_output_before)
    stale_two_level_numeric_plan_rejected = False
    try:
        migrated_two_level_plan.apply(native_input, native_output)
    except RuntimeError as exc:
        stale_two_level_numeric_plan_rejected = "numeric version is stale" in str(
            exc
        )
    stale_two_level_numeric_output_preserved = np.array_equal(
        stale_two_level_numeric_output_before, native_output.to_numpy()
    )
    migrated_two_level_plan_stats = migrated_two_level_plan.debug_runtime_stats()
    (
        refreshed_two_level_plan,
        refreshed_two_level_reference,
        refreshed_coarse_keys,
        refreshed_two_level_sources_released,
    ) = build_two_level_plan()
    current_two_level_plan = refreshed_two_level_plan
    refreshed_two_level = validate_two_level(
        refreshed_two_level_plan,
        refreshed_two_level_reference,
        refreshed_coarse_keys,
    )
    migrated_two_level_plan = None
    migrated_two_level_reference = None
    gc.collect()

    transient_level_input = ti.ndarray(ti.f32, shape=expected_dofs)
    transient_level_output = ti.ndarray(ti.f32, shape=expected_dofs)
    transient_level_input.from_numpy(
        compact_x.to_numpy()[:expected_dofs].astype(np.float32, copy=True)
    )
    refreshed_two_level_plan.apply(
        transient_level_input,
        transient_level_output,
        expected_topology_version=topology_version,
        expected_numeric_version=level_numeric_version,
    )
    transient_level_input_ref = weakref.ref(transient_level_input)
    transient_level_output_ref = weakref.ref(transient_level_output)
    transient_level_input_native_ref = weakref.ref(transient_level_input.arr)
    transient_level_output_native_ref = weakref.ref(transient_level_output.arr)
    del transient_level_input
    del transient_level_output
    gc.collect()
    transient_level_wrappers_released = (
        transient_level_input_ref() is None
        and transient_level_output_ref() is None
    )
    ti.sync()
    gc.collect()
    graph_cache_pinned_transient_native_views = (
        transient_level_input_native_ref() is not None
        and transient_level_output_native_ref() is not None
    )
    rebound_two_level = validate_two_level(
        refreshed_two_level_plan,
        refreshed_two_level_reference,
        refreshed_coarse_keys,
    )
    ti.sync()
    gc.collect()
    transient_native_views_released_after_rebind = (
        transient_level_input_native_ref() is None
        and transient_level_output_native_ref() is None
    )
    program = ti.lang.impl.get_runtime().prog
    native_graph_operator = refreshed_two_level_plan.create_native_operator()
    native_graph_source_plan_ref = weakref.ref(refreshed_two_level_plan)
    native_graph_alias_input_before = native_input.to_numpy()
    native_graph_alias_rejected = False
    try:
        native_graph_operator.spmv(program, native_input.arr, native_input.arr)
    except RuntimeError as exc:
        native_graph_alias_rejected = "input and output must not alias" in str(
            exc
        )
    native_graph_alias_input_preserved = np.array_equal(
        native_graph_alias_input_before, native_input.to_numpy()
    )
    native_output.fill(0.0)
    native_graph_operator.spmv(program, native_input.arr, native_output.arr)
    ti.sync()
    native_graph_reference = (
        refreshed_two_level_reference["preconditioner"]
        @ native_input.to_numpy().astype(np.float64)
    )
    native_graph_before_destroy_errors = np.abs(
        native_output.to_numpy().astype(np.float64) - native_graph_reference
    )
    native_graph_before_destroy_difference = float(
        native_graph_before_destroy_errors.sum()
    )
    native_graph_before_destroy_difference_linf = float(
        native_graph_before_destroy_errors.max(initial=0.0)
    )
    native_graph_before_destroy_stats = (
        native_graph_operator._debug_runtime_stats()
    )
    stale_output_before = compact_ax.to_numpy()
    stale_plan_rejected = False
    try:
        initial_operator_plan.apply(compact_x, compact_ax)
    except RuntimeError as exc:
        stale_plan_rejected = "topology version is stale" in str(exc)
    stale_output_preserved = np.array_equal(
        stale_output_before, compact_ax.to_numpy()
    )
    migrated = validate(migrated_expected, migrated_operator_plan)
    alias_input_before = compact_x.to_numpy()
    alias_rejected = False
    try:
        migrated_operator_plan.apply(compact_x, compact_x)
    except RuntimeError as exc:
        alias_rejected = "does not support input/output alias" in str(exc)
    alias_input_preserved = np.array_equal(
        alias_input_before, compact_x.to_numpy()
    )
    version_output_before = compact_ax.to_numpy()
    numeric_version_mismatch_rejected = False
    try:
        migrated_operator_plan.apply(
            compact_x,
            compact_ax,
            expected_numeric_version=2,
        )
    except RuntimeError as exc:
        numeric_version_mismatch_rejected = (
            "expected numeric version does not match" in str(exc)
        )
    version_output_preserved = np.array_equal(
        version_output_before, compact_ax.to_numpy()
    )
    native_input.from_numpy(
        compact_x.to_numpy()[:expected_dofs].astype(np.float32, copy=True)
    )
    native_output.fill(0.0)
    native_kernel = apply_compact_operator._primal
    native_kernel_key = native_kernel.ensure_compiled(
        expected_dofs,
        neighbors,
        native_input,
        native_output,
    )
    native_kernel_cpp = native_kernel.compiled_kernels[native_kernel_key]
    program = ti.lang.impl.get_runtime().prog
    native_operator = program._create_compiled_kernel_linear_operator(
        native_kernel_cpp,
        expected_dofs,
        topology_version,
        1,
        neighbors.arr,
    )
    native_alias_input_before = native_input.to_numpy()
    native_alias_rejected = False
    try:
        native_operator.spmv(program, native_input.arr, native_input.arr)
    except RuntimeError as exc:
        native_alias_rejected = "input and output must not alias" in str(exc)
    native_alias_input_preserved = np.array_equal(
        native_alias_input_before, native_input.to_numpy()
    )
    native_operator.spmv(program, native_input.arr, native_output.arr)
    ti.sync()
    native_reference = oracle_ax.to_numpy()[:expected_dofs]
    native_before_destroy_difference = float(
        np.abs(native_output.to_numpy() - native_reference).sum()
    )
    native_before_destroy_stats = native_operator._debug_runtime_stats()
    operator_data_source_ref = weakref.ref(neighbors)
    stale_initial = stale_initial_probe_l1()
    tree_stats = ti.lang.impl.get_runtime().prog._debug_sparse_snode_tree_stats(
        int(tree.id)
    )
    active_neighbors = int(
        np.count_nonzero(
            neighbors.to_numpy()[:expected_dofs, : 2 * dimensions] >= 0
        )
    )
    csr_actual_nnz = expected_dofs + active_neighbors
    scalar_bytes = 4
    candidate_staging_reserved_bytes = (
        producer_capacity * dimensions * scalar_bytes
        + candidate_capacity * scalar_bytes
        + 2 * scalar_bytes
    )
    dof_map_reserved_bytes = (
        candidate_capacity * scalar_bytes
        + scalar_bytes
        + scalar_bytes
        + max_dofs * dimensions * scalar_bytes
    )
    operator_reserved_bytes = max_dofs * 2 * dimensions * scalar_bytes
    diagnostic_vector_reserved_bytes = (
        4 * max_dofs + 2 * expected_dofs
    ) * scalar_bytes
    native_provider_operator_data_reserved_bytes = operator_reserved_bytes
    initial_two_level_reserved_bytes = initial_two_level_plan_stats["resources"][
        "plan_owned_reserved_bytes"
    ]
    migrated_two_level_reserved_bytes = migrated_two_level_plan_stats[
        "resources"
    ]["plan_owned_reserved_bytes"]
    refreshed_two_level_reserved_bytes = (
        refreshed_two_level_plan.debug_runtime_stats()["resources"][
            "plan_owned_reserved_bytes"
        ]
    )
    two_level_migration_overlap_reserved_bytes = (
        initial_two_level_reserved_bytes + migrated_two_level_reserved_bytes
    )
    two_level_numeric_refresh_overlap_reserved_bytes = (
        migrated_two_level_reserved_bytes + refreshed_two_level_reserved_bytes
    )
    two_level_publish_overlap_peak_reserved_bytes = max(
        two_level_migration_overlap_reserved_bytes,
        two_level_numeric_refresh_overlap_reserved_bytes,
    )
    native_graph_provider_reserved_bytes = native_graph_before_destroy_stats[
        "resources"
    ]["operator_owned_reserved_bytes"]
    two_level_native_bridge_overlap_reserved_bytes = (
        refreshed_two_level_reserved_bytes + native_graph_provider_reserved_bytes
    )
    two_level_residency_peak_reserved_bytes = max(
        two_level_publish_overlap_peak_reserved_bytes,
        two_level_native_bridge_overlap_reserved_bytes,
    )
    profile_owned_reserved_bytes = (
        candidate_staging_reserved_bytes
        + dof_map_reserved_bytes
        + operator_reserved_bytes
        + diagnostic_vector_reserved_bytes
        + two_level_residency_peak_reserved_bytes
    )
    csr_actual_pattern_bytes = (
        (expected_dofs + 1) * scalar_bytes
        + csr_actual_nnz * scalar_bytes
    )
    csr_actual_value_bytes = csr_actual_nnz * scalar_bytes
    csr_upper_nnz = expected_dofs * (2 * dimensions + 1)
    csr_upper_total_bytes = (
        (expected_dofs + 1) * scalar_bytes
        + 2 * csr_upper_nnz * scalar_bytes
    )
    memory_attribution = {
        "scalar_and_index_bytes": scalar_bytes,
        "candidate_staging_reserved_bytes": candidate_staging_reserved_bytes,
        "dof_map_reserved_bytes": dof_map_reserved_bytes,
        "structured_operator_neighbor_reserved_bytes": operator_reserved_bytes,
        "structured_operator_neighbor_active_bytes": (
            expected_dofs * 2 * dimensions * scalar_bytes
        ),
        "diagnostic_vector_count": 6,
        "diagnostic_vector_capacity_entries": (
            4 * max_dofs + 2 * expected_dofs
        ),
        "diagnostic_vector_reserved_bytes": diagnostic_vector_reserved_bytes,
        "native_provider_operator_data_reserved_bytes": (
            native_provider_operator_data_reserved_bytes
        ),
        "two_level_preconditioner_steady_reserved_bytes": (
            refreshed_two_level_reserved_bytes
        ),
        "two_level_preconditioner_migration_overlap_reserved_bytes": (
            two_level_migration_overlap_reserved_bytes
        ),
        "two_level_preconditioner_numeric_refresh_overlap_reserved_bytes": (
            two_level_numeric_refresh_overlap_reserved_bytes
        ),
        "two_level_preconditioner_publish_overlap_peak_reserved_bytes": (
            two_level_publish_overlap_peak_reserved_bytes
        ),
        "native_graph_provider_reserved_bytes": (
            native_graph_provider_reserved_bytes
        ),
        "two_level_native_bridge_overlap_reserved_bytes": (
            two_level_native_bridge_overlap_reserved_bytes
        ),
        "two_level_residency_peak_reserved_bytes": (
            two_level_residency_peak_reserved_bytes
        ),
        "profile_and_native_provider_peak_reserved_bytes": (
            profile_owned_reserved_bytes
            + native_provider_operator_data_reserved_bytes
        ),
        "profile_owned_reserved_bytes": profile_owned_reserved_bytes,
        "active_neighbor_entries": active_neighbors,
        "csr_reference_actual_nnz": csr_actual_nnz,
        "csr_reference_actual_pattern_bytes": csr_actual_pattern_bytes,
        "csr_reference_actual_value_bytes": csr_actual_value_bytes,
        "csr_reference_actual_total_bytes": (
            csr_actual_pattern_bytes + csr_actual_value_bytes
        ),
        "csr_reference_upper_nnz": csr_upper_nnz,
        "csr_reference_upper_total_bytes": csr_upper_total_bytes,
        "csr_actual_minus_structured_active_operator_bytes": (
            csr_actual_pattern_bytes
            + csr_actual_value_bytes
            - expected_dofs * 2 * dimensions * scalar_bytes
        ),
        "allocation_scopes": {
            "candidate_staging": "profile_reusable_build_workspace",
            "dof_map": "active_dof_map_plan",
            "structured_operator_neighbor_table": (
                "structured_linear_operator_plan"
            ),
            "native_provider_operator_data_snapshot": (
                "compiled_kernel_linear_operator_exclusive"
            ),
            "two_level_preconditioner": (
                "profile_private_level_plan_including_owned_workspace"
            ),
            "two_level_migration_overlap": (
                "old_stale_plan_plus_new_current_plan"
            ),
            "two_level_numeric_refresh_overlap": (
                "old_numeric_stale_plan_plus_refreshed_current_plan"
            ),
            "native_graph_provider": (
                "exclusive_typed_topology_numeric_workspace_snapshots"
            ),
            "native_graph_bridge_handoff": (
                "python_graph_source_plan_plus_native_provider_snapshot"
            ),
            "diagnostic_vectors": "correctness_profile_only_not_solver_plan",
            "sort_unique_workspace": "explicit_reusable_primitive_workspace",
            "coordinate_to_dof_field": "included_in_snode_tree_memory",
        },
    }
    before_destroy = _runtime_snapshot(ti)
    tree.destroy()
    compact_ax.fill(0.0)
    migrated_operator_plan.apply(
        compact_x,
        compact_ax,
        expected_topology_version=topology_version,
        expected_numeric_version=1,
    )
    ti.sync()
    post_destroy_operator_difference = compact_difference_l1(
        dof_count, compact_ax, oracle_ax
    )
    post_destroy_two_level = validate_two_level(
        refreshed_two_level_plan,
        refreshed_two_level_reference,
        refreshed_coarse_keys,
    )
    initial_operator_plan_stats = initial_operator_plan.debug_runtime_stats()
    migrated_operator_plan_stats = (
        migrated_operator_plan.debug_runtime_stats()
    )
    refreshed_two_level_plan_stats = (
        refreshed_two_level_plan.debug_runtime_stats()
    )
    operator_plan = None
    current_two_level_plan = None
    initial_operator_plan = None
    migrated_operator_plan = None
    migrated_two_level_plan = None
    refreshed_two_level_plan = None
    migrated_two_level_reference = None
    refreshed_two_level_reference = None
    neighbors = None
    gc.collect()
    native_graph_source_plan_released = native_graph_source_plan_ref() is None
    native_output.fill(0.0)
    native_graph_operator.spmv(program, native_input.arr, native_output.arr)
    ti.sync()
    native_graph_post_destroy_errors = np.abs(
        native_output.to_numpy().astype(np.float64) - native_graph_reference
    )
    native_graph_post_destroy_difference = float(
        native_graph_post_destroy_errors.sum()
    )
    native_graph_post_destroy_difference_linf = float(
        native_graph_post_destroy_errors.max(initial=0.0)
    )
    native_graph_post_destroy_stats = (
        native_graph_operator._debug_runtime_stats()
    )
    operator_data_source_released = operator_data_source_ref() is None
    native_output.fill(0.0)
    native_operator.spmv(program, native_input.arr, native_output.arr)
    ti.sync()
    native_post_destroy_difference = float(
        np.abs(native_output.to_numpy() - native_reference).sum()
    )
    native_post_destroy_stats = native_operator._debug_runtime_stats()
    after_destroy = _runtime_snapshot(ti)

    def coarse_storage_correct(check):
        storage = check["coarse_storage"]
        return all(
            (
                storage["directional_reconstructs_dense_galerkin"],
                storage["positive_diagonal"],
                storage["nonpositive_offdiagonal"],
                storage["reciprocal_weighted_adjacency"],
                storage["max_row_nnz"] <= storage["row_nnz_upper_bound"],
                sum(storage["component_sizes"]) == check["coarse_size"],
                storage["component_dense_inverse_bytes"]
                <= storage["dense_inverse_bytes"],
            )
        )

    correct = all(
        all(
            (
                phase["active_blocks"] == expected_blocks,
                phase["active_dofs"] == expected_dofs,
                phase["ordering_matches"],
                phase["gather_difference_l1"] == 0.0,
                phase["operator_difference_l1"] == 0.0,
            )
        )
        for phase in (initial, after_overflow, migrated)
    ) and all(
        (
            initial_attempt["published"],
            overflow_expected == migrated_expected,
            overflow_attempt["status"] == "capacity_overflow",
            not overflow_attempt["published"],
            overflow_attempt["old_topology_preserved"],
            overflow_attempt["version_before"] == 1,
            overflow_attempt["version_after"] == 1,
            migrated_attempt["published"],
            stale_initial == 0.0,
            overflow_preserved_operator_plan,
            stale_plan_rejected,
            stale_output_preserved,
            alias_rejected,
            alias_input_preserved,
            numeric_version_mismatch_rejected,
            version_output_preserved,
            post_destroy_operator_difference == 0.0,
            native_before_destroy_difference == 0.0,
            native_alias_rejected,
            native_alias_input_preserved,
            operator_data_source_released,
            native_post_destroy_difference == 0.0,
            native_graph_alias_rejected,
            native_graph_alias_input_preserved,
            native_graph_before_destroy_difference_linf <= 5e-5,
            native_graph_source_plan_released,
            native_graph_post_destroy_difference_linf <= 5e-5,
            overflow_preserved_two_level_plan,
            stale_two_level_plan_rejected,
            stale_two_level_output_preserved,
            stale_two_level_numeric_plan_rejected,
            stale_two_level_numeric_output_preserved,
            two_level_alias_rejected,
            two_level_alias_input_preserved,
            initial_two_level_sources_released,
            migrated_two_level_sources_released,
            refreshed_two_level_sources_released,
            transient_level_wrappers_released,
            graph_cache_pinned_transient_native_views,
            transient_native_views_released_after_rebind,
            initial_two_level["fine_to_coarse_complete"],
            initial_two_level["fine_operator_symmetric"],
            initial_two_level["fine_operator_min_eigenvalue"] > 0.0,
            initial_two_level["coarse_operator_symmetric"],
            initial_two_level["coarse_operator_min_eigenvalue"] > 0.0,
            coarse_storage_correct(initial_two_level),
            initial_two_level["preconditioner_symmetric"],
            initial_two_level["preconditioner_min_eigenvalue"] > 0.0,
            initial_two_level["output_difference_linf"] <= 5e-5,
            migrated_two_level["fine_to_coarse_complete"],
            migrated_two_level["fine_operator_symmetric"],
            migrated_two_level["fine_operator_min_eigenvalue"] > 0.0,
            migrated_two_level["coarse_operator_symmetric"],
            migrated_two_level["coarse_operator_min_eigenvalue"] > 0.0,
            coarse_storage_correct(migrated_two_level),
            migrated_two_level["preconditioner_symmetric"],
            migrated_two_level["preconditioner_min_eigenvalue"] > 0.0,
            migrated_two_level["output_difference_linf"] <= 5e-5,
            refreshed_two_level["fine_to_coarse_complete"],
            refreshed_two_level["fine_operator_symmetric"],
            refreshed_two_level["fine_operator_min_eigenvalue"] > 0.0,
            refreshed_two_level["coarse_operator_symmetric"],
            refreshed_two_level["coarse_operator_min_eigenvalue"] > 0.0,
            coarse_storage_correct(refreshed_two_level),
            refreshed_two_level["preconditioner_symmetric"],
            refreshed_two_level["preconditioner_min_eigenvalue"] > 0.0,
            refreshed_two_level["output_difference_linf"] <= 5e-5,
            rebound_two_level["output_difference_linf"] <= 5e-5,
            post_destroy_two_level["output_difference_linf"] <= 5e-5,
        )
    )

    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "arch": arch_name,
        "correct": correct,
        "config": {
            "dimensions": dimensions,
            "root_blocks_per_axis": root_blocks,
            "block_size_per_axis": block_size,
            "candidate_capacity": candidate_capacity,
            "candidate_payload_capacity": candidate_capacity - 1,
            "producer_capacity": producer_capacity,
            "cells_per_block": cells_per_block,
            "expected_active_blocks": expected_blocks,
            "expected_active_dofs": expected_dofs,
        },
        "topology": {
            "version": topology_version,
            "ordering": "sorted_linear_block_key_then_row_major_local_cell",
            "candidate_source": (
                "device_emitted_from_explicit_bounded_fixture_coordinates"
            ),
            "sentinel_key": int(SENTINEL_KEY),
            "sort_method": sort_method,
            "unique_method": unique_method,
            "host_active_prefix_read_required_to_build": False,
            "host_staging_status_read_required_to_publish": True,
            "publish_model": "host_mediated_two_phase_before_snode_mutation",
            "mutation_fault_rollback": False,
        },
        "operator_contract": {
            "storage": "compact_ndarray_neighbor_table",
            "snode_accesses_per_compact_apply": 0,
            "snode_struct_for_is_correctness_oracle_only": True,
            "inactive_neighbor": "dirichlet_zero",
            "plan_owned_apply_allocation_count": 0,
            "explicit_apply_host_synchronization_count": 0,
            "profile_private_not_solver_provider": True,
            "profile_plan_reuses_native_sparse_matrix_hook": False,
            "compiled_kernel_provider_reuses_native_sparse_matrix_hook": True,
            "compiled_kernel_provider_solver_integrated": False,
        },
        "checks": {
            "initial": initial,
            "after_overflow": after_overflow,
            "migrated": migrated,
            "stale_initial_probe_l1": stale_initial,
        },
        "attempts": {
            "initial": initial_attempt,
            "overflow": overflow_attempt,
            "migrated": migrated_attempt,
        },
        "resources": {
            "sort_workspace_peak_bytes": sort_workspace.workspace_bytes_peak,
            "unique_workspace_peak_bytes": unique_workspace.workspace_bytes_peak,
            "memory_attribution": memory_attribution,
            "tree_memory": dict(tree_stats["memory"]),
        },
        "operator_plans": {
            "initial": initial_operator_plan_stats,
            "migrated": migrated_operator_plan_stats,
            "overflow_preserved_initial_plan": (
                overflow_preserved_operator_plan
            ),
            "stale_initial_plan_rejected_before_mutation": (
                stale_plan_rejected and stale_output_preserved
            ),
            "alias_rejected_before_mutation": (
                alias_rejected and alias_input_preserved
            ),
            "numeric_version_mismatch_rejected_before_mutation": (
                numeric_version_mismatch_rejected
                and version_output_preserved
            ),
        },
        "native_operator": {
            "abi": (
                "i32_active_size_operator_data_ndarray_f32_input_f32_output"
            ),
            "snode_dependencies_allowed": False,
            "operator_data_snapshot_owned": True,
            "alias_rejected_before_mutation": (
                native_alias_rejected and native_alias_input_preserved
            ),
            "operator_data_source_released_before_second_apply": (
                operator_data_source_released
            ),
            "before_destroy_difference_l1": (
                native_before_destroy_difference
            ),
            "post_destroy_difference_l1": native_post_destroy_difference,
            "before_destroy_stats": native_before_destroy_stats,
            "post_destroy_stats": native_post_destroy_stats,
        },
        "native_graph_operator": {
            "abi": (
                "fixed_i32_plus_typed_topology_numeric_workspace_ndarrays_"
                "plus_f32_input_output"
            ),
            "snode_dependencies_allowed": False,
            "resource_roles_explicit": True,
            "source_snapshots_owned": True,
            "alias_rejected_before_mutation": (
                native_graph_alias_rejected
                and native_graph_alias_input_preserved
            ),
            "source_python_graph_plan_released": (
                native_graph_source_plan_released
            ),
            "before_destroy_difference_l1": (
                native_graph_before_destroy_difference
            ),
            "before_destroy_difference_linf": (
                native_graph_before_destroy_difference_linf
            ),
            "post_destroy_difference_l1": native_graph_post_destroy_difference,
            "post_destroy_difference_linf": (
                native_graph_post_destroy_difference_linf
            ),
            "before_destroy_stats": native_graph_before_destroy_stats,
            "post_destroy_stats": native_graph_post_destroy_stats,
            "solver_integrated": False,
        },
        "two_level_preconditioner": {
            "method": "two_level_additive_galerkin",
            "fine_level_storage": "compact_active_dof_ndarray",
            "coarse_level_storage": "one_dof_per_sorted_active_brick",
            "boundary_condition": "inactive_fine_neighbor_is_dirichlet_zero",
            "coarse_operator_storage_probe": {
                "representation": "directional_stencil_or_csr",
                "public_solver_baseline": "generic_csr_or_bsr",
                "directional_stencil_scope": (
                    "profile_private_axis_aligned_brick_specialization"
                ),
                "storage_complexity": "linear_in_coarse_dofs",
                "dense_inverse_complexity": "quadratic_in_coarse_dofs",
                "solve_not_selected_by_storage_probe": True,
                "latest": dict(refreshed_two_level["coarse_storage"]),
            },
            "checks": {
                "initial": initial_two_level,
                "migrated": migrated_two_level,
                "numeric_refreshed": refreshed_two_level,
                "after_transient_rebind": rebound_two_level,
                "post_tree_destroy": post_destroy_two_level,
            },
            "plans": {
                "initial": initial_two_level_plan_stats,
                "migrated": migrated_two_level_plan_stats,
                "numeric_refreshed": refreshed_two_level_plan_stats,
            },
            "lifecycle": {
                "built_after_topology_publish": True,
                "overflow_preserved_initial_plan": (
                    overflow_preserved_two_level_plan
                ),
                "stale_initial_plan_rejected_before_mutation": (
                    stale_two_level_plan_rejected
                    and stale_two_level_output_preserved
                ),
                "alias_rejected_before_mutation": (
                    two_level_alias_rejected
                    and two_level_alias_input_preserved
                ),
                "stale_numeric_plan_rejected_before_mutation": (
                    stale_two_level_numeric_plan_rejected
                    and stale_two_level_numeric_output_preserved
                ),
                "numeric_refresh_rebuilt_current_plan": (
                    refreshed_two_level_plan_stats["identity"][
                        "numeric_version"
                    ]
                    == level_numeric_version
                ),
                "plan_source_snapshots_released": (
                    initial_two_level_sources_released
                    and migrated_two_level_sources_released
                    and refreshed_two_level_sources_released
                ),
                "transient_input_output_wrappers_released": (
                    transient_level_wrappers_released
                ),
                "graph_cache_pinned_last_transient_native_views": (
                    graph_cache_pinned_transient_native_views
                ),
                "transient_native_views_released_after_rebind": (
                    transient_native_views_released_after_rebind
                ),
                "migration_kept_stale_and_current_plans_alive": True,
                "numeric_refresh_kept_stale_and_current_plans_alive": True,
                "current_plan_survives_snode_tree_destroy": (
                    post_destroy_two_level["output_difference_linf"] <= 5e-5
                ),
            },
            "execution": {
                "global_phases": [
                    "restrict",
                    "coarse_inverse_apply",
                    "prolongate_plus_fine_jacobi",
                ],
                "kernel_launches_per_apply": 3,
                "host_graph_submissions_per_apply": 1,
                "single_compiled_kernel_provider_sufficient": False,
                "compiled_graph_sequence_candidate": True,
                "compiled_graph_integrated": True,
                "graph_runtime_resource_lease_managed_by_program": True,
                "graph_fast_arg_cache_retains_last_native_binding": True,
                "solver_integrated": False,
            },
            "memory": {
                "steady_plan_reserved_bytes": (
                    refreshed_two_level_reserved_bytes
                ),
                "migration_overlap_peak_reserved_bytes": (
                    two_level_migration_overlap_reserved_bytes
                ),
                "numeric_refresh_overlap_peak_reserved_bytes": (
                    two_level_numeric_refresh_overlap_reserved_bytes
                ),
                "publish_overlap_peak_reserved_bytes": (
                    two_level_publish_overlap_peak_reserved_bytes
                ),
                "native_graph_provider_reserved_bytes": (
                    native_graph_provider_reserved_bytes
                ),
                "native_bridge_handoff_overlap_reserved_bytes": (
                    two_level_native_bridge_overlap_reserved_bytes
                ),
                "level_residency_peak_reserved_bytes": (
                    two_level_residency_peak_reserved_bytes
                ),
                "overlap_model": (
                    "old_stale_plan_plus_new_current_plan_per_publish"
                ),
                "host_reference_matrices_are_profile_diagnostics": True,
                "last_bound_caller_vector_bytes_not_plan_owned": (
                    2 * expected_dofs * scalar_bytes
                ),
                "last_bound_caller_vectors_may_be_pinned_until_rebind": True,
                "graph_persistent_argument_bytes_outside_plan_payload": (
                    refreshed_two_level_plan_stats["resources"][
                        "graph_persistent_argument_bytes_outside_plan_payload"
                    ]
                ),
                "graph_persistent_argument_memory_domain": (
                    "backend_host_runtime_argument_buffers_not_level_payload"
                ),
            },
        },
        "lifecycle": {
            "tree_identity": tree_identity,
            "tree_present_before_destroy": str(tree_identity[0])
            in before_destroy["lifecycle"]["snode_trees"],
            "tree_recovered_after_destroy": str(tree_identity[0])
            not in after_destroy["lifecycle"]["snode_trees"],
            "operator_survives_tree_destroy": (
                post_destroy_operator_difference == 0.0
            ),
            "post_destroy_operator_difference_l1": (
                post_destroy_operator_difference
            ),
            "native_operator_survives_tree_and_source_destroy": (
                operator_data_source_released
                and native_post_destroy_difference == 0.0
            ),
            "native_graph_operator_survives_tree_and_source_destroy": (
                native_graph_source_plan_released
                and native_graph_post_destroy_difference_linf <= 5e-5
            ),
            "two_level_preconditioner_survives_tree_destroy": (
                post_destroy_two_level["output_difference_linf"] <= 5e-5
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--dimensions", type=int, choices=(2, 3), default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import taichi_forge as ti

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    ti.init(
        arch=arch,
        enable_fallback=False,
        offline_cache=False,
        vulkan_sparse_experimental=True,
        cuda_sparse_pool_auto_size=True,
        cuda_sparse_per_snode_pool=True,
    )
    try:
        report = run_initialized(ti, dimensions=args.dimensions)
    finally:
        ti.reset()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
