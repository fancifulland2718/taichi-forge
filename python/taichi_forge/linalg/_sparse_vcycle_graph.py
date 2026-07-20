"""Private device resources for hierarchy-backed recursive V-cycles.

The module deliberately separates numeric setup from hierarchy construction
and Graph execution.  Callers select smoother coefficients and a bounded
bottom inverse, while this layer owns and validates device-visible snapshots.
Nothing here is exported through ``taichi.linalg``.
"""

import threading

import numpy as np

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError

from ._sparse_hierarchy_assembly import (
    _ensure_current_program,
    _positive_int,
    _positive_version,
)
from ._sparse_hierarchy_candidate import _SparseCsrHierarchySnapshot
from ._sparse_runtime_memory import _graph_cache_memory_attribution
from .sparse_matrix import _require_current_scalar_ndarray


_NUMERIC_STATUS = {
    1: "inverse diagonal is not finite and positive",
    2: "smoother damping is not finite and positive",
    3: "bottom inverse contains a non-finite value",
    4: "bottom inverse diagonal is not positive",
    5: "bottom inverse is not exactly symmetric",
}


@ti.kernel
def _copy_f32_prefix(
    source: ti.types.ndarray(dtype=ti.f32, ndim=1),
    destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
    count: ti.i32,
):
    for index in range(count):
        destination[index] = source[index]


@ti.kernel
def _validate_positive_f32(
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    count: ti.i32,
    status: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for index in range(count):
        value = values[index]
        if (
            value <= 0.0
            or ti.math.isnan(value)
            or ti.math.isinf(value)
        ):
            ti.atomic_max(control[0], status)


@ti.kernel
def _validate_dense_bottom_inverse(
    values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    size: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for index in range(size * size):
        value = values[index]
        row = index // size
        column = index - row * size
        if ti.math.isnan(value) or ti.math.isinf(value):
            ti.atomic_max(control[0], 3)
        if row == column and value <= 0.0:
            ti.atomic_max(control[0], 4)
        if value != values[column * size + row]:
            ti.atomic_max(control[0], 5)


@ti.kernel
def _vcycle_pre_smooth(
    active_size: ti.i32,
    inverse_diagonal: ti.types.ndarray(dtype=ti.f32, ndim=1),
    damping: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
    pre_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for row in range(active_size):
        pre_solution[row] = (
            damping[0] * inverse_diagonal[row] * level_rhs[row]
        )


@ti.kernel
def _vcycle_restrict_gather(
    coarse_size: ti.i32,
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    restriction_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    restriction_fine_rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    numeric_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
    pre_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
    coarse_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for coarse_row in range(coarse_size):
        total = ti.cast(0.0, ti.f32)
        for schedule_offset in range(
            restriction_offsets[coarse_row],
            restriction_offsets[coarse_row + 1],
        ):
            fine_row = restriction_fine_rows[schedule_offset]
            applied = ti.cast(0.0, ti.f32)
            for offset in range(
                row_offsets[fine_row], row_offsets[fine_row + 1]
            ):
                applied += (
                    numeric_values[offset]
                    * pre_solution[columns[offset]]
                )
            total += level_rhs[fine_row] - applied
        coarse_rhs[coarse_row] = total


@ti.kernel
def _vcycle_dense_bottom(
    active_size: ti.i32,
    inverse_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for row in range(active_size):
        total = ti.cast(0.0, ti.f32)
        for column in range(active_size):
            total += (
                inverse_values[row * active_size + column]
                * level_rhs[column]
            )
        level_solution[row] = total


@ti.kernel
def _vcycle_post_smooth(
    active_size: ti.i32,
    row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
    fine_to_coarse: ti.types.ndarray(dtype=ti.i32, ndim=1),
    numeric_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    inverse_diagonal: ti.types.ndarray(dtype=ti.f32, ndim=1),
    damping: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
    pre_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
    coarse_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
    level_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for row in range(active_size):
        corrected = (
            pre_solution[row] + coarse_solution[fine_to_coarse[row]]
        )
        applied = ti.cast(0.0, ti.f32)
        for offset in range(row_offsets[row], row_offsets[row + 1]):
            column = columns[offset]
            neighbor_corrected = (
                pre_solution[column]
                + coarse_solution[fine_to_coarse[column]]
            )
            applied += numeric_values[offset] * neighbor_corrected
        level_solution[row] = corrected + (
            damping[0]
            * inverse_diagonal[row]
            * (level_rhs[row] - applied)
        )


class _SparseVcycleNumericSnapshot:
    """Owned scalar-Jacobi and bounded dense-bottom numeric resources."""

    def __init__(
        self,
        *,
        program,
        backend,
        topology_version,
        numeric_version,
        level_sizes,
        level_nnz,
        inverse_diagonals,
        dampings,
        bottom_inverse,
        validation_control_readback_bytes,
        device_to_device_bytes,
    ):
        self._program = program
        self._backend = backend
        self.topology_version = int(topology_version)
        self.numeric_version = int(numeric_version)
        self._level_sizes = tuple(int(value) for value in level_sizes)
        self._level_nnz = tuple(int(value) for value in level_nnz)
        self._inverse_diagonals = tuple(inverse_diagonals)
        self._dampings = tuple(dampings)
        self._bottom_inverse = bottom_inverse
        self._validation_control_readback_bytes = int(
            validation_control_readback_bytes
        )
        self._device_to_device_bytes = int(device_to_device_bytes)

    @classmethod
    def copy_validated(
        cls,
        hierarchy,
        *,
        inverse_diagonals,
        dampings,
        bottom_inverse,
        topology_version,
        numeric_version,
    ):
        topology_version = _positive_version(
            topology_version, "V-cycle numeric topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "V-cycle numeric numeric_version"
        )
        if not isinstance(hierarchy, _SparseCsrHierarchySnapshot):
            raise TaichiRuntimeError(
                "V-cycle numeric setup requires a sparse hierarchy snapshot"
            )
        hierarchy._ensure_current()
        if (
            hierarchy.topology_version != topology_version
            or hierarchy.numeric_version != numeric_version
        ):
            raise TaichiRuntimeError(
                "V-cycle numeric versions must match the hierarchy snapshot"
            )
        try:
            inverse_diagonals = tuple(inverse_diagonals)
            dampings = tuple(dampings)
        except TypeError as exc:
            raise TaichiRuntimeError(
                "V-cycle inverse diagonals and dampings must be iterable"
            ) from exc
        nonbottom_levels = hierarchy.level_count - 1
        if (
            len(inverse_diagonals) != nonbottom_levels
            or len(dampings) != nonbottom_levels
        ):
            raise TaichiRuntimeError(
                "V-cycle numeric setup requires one inverse diagonal and "
                "damping per non-bottom level"
            )

        checked_inverse_diagonals = []
        checked_dampings = []
        for level_index in range(nonbottom_levels):
            level_size = hierarchy.level_sizes[level_index]
            inverse_diagonal = _require_current_scalar_ndarray(
                inverse_diagonals[level_index],
                f"V-cycle level {level_index} inverse diagonal",
                ti.f32,
                one_dimensional=True,
            )
            damping = _require_current_scalar_ndarray(
                dampings[level_index],
                f"V-cycle level {level_index} damping",
                ti.f32,
                one_dimensional=True,
            )
            if inverse_diagonal.shape != (level_size,):
                raise TaichiRuntimeError(
                    f"V-cycle level {level_index} inverse diagonal must "
                    f"have shape ({level_size},)"
                )
            if damping.shape != (1,):
                raise TaichiRuntimeError(
                    f"V-cycle level {level_index} damping must have shape "
                    "(1,)"
                )
            checked_inverse_diagonals.append(inverse_diagonal)
            checked_dampings.append(damping)

        bottom_size = hierarchy.level_sizes[-1]
        bottom_inverse = _require_current_scalar_ndarray(
            bottom_inverse,
            "V-cycle dense bottom inverse",
            ti.f32,
            one_dimensional=True,
        )
        if bottom_inverse.shape != (bottom_size * bottom_size,):
            raise TaichiRuntimeError(
                "V-cycle dense bottom inverse must contain bottom_size^2 "
                "entries"
            )

        owned_inverse_diagonals = []
        owned_dampings = []
        device_to_device_bytes = 0
        control = ti.ndarray(ti.i32, shape=2)
        control.fill(0)
        for level_index in range(nonbottom_levels):
            level_size = hierarchy.level_sizes[level_index]
            owned_inverse = ti.ndarray(ti.f32, shape=level_size)
            owned_damping = ti.ndarray(ti.f32, shape=1)
            _copy_f32_prefix(
                checked_inverse_diagonals[level_index],
                owned_inverse,
                level_size,
            )
            _copy_f32_prefix(
                checked_dampings[level_index], owned_damping, 1
            )
            _validate_positive_f32(owned_inverse, level_size, 1, control)
            _validate_positive_f32(owned_damping, 1, 2, control)
            owned_inverse_diagonals.append(owned_inverse)
            owned_dampings.append(owned_damping)
            device_to_device_bytes += 4 * (level_size + 1)

        owned_bottom_inverse = ti.ndarray(
            ti.f32, shape=bottom_size * bottom_size
        )
        _copy_f32_prefix(
            bottom_inverse,
            owned_bottom_inverse,
            bottom_size * bottom_size,
        )
        _validate_dense_bottom_inverse(
            owned_bottom_inverse, bottom_size, control
        )
        device_to_device_bytes += 4 * bottom_size * bottom_size
        control_host = control.to_numpy()
        hierarchy._ensure_current()
        status = int(control_host[0])
        if status != 0:
            reason = _NUMERIC_STATUS.get(status, "unknown validation failure")
            raise TaichiRuntimeError(
                "V-cycle numeric validation failed before publish: " + reason
            )
        return cls(
            program=hierarchy._program,
            backend=hierarchy._backend,
            topology_version=topology_version,
            numeric_version=numeric_version,
            level_sizes=hierarchy.level_sizes,
            level_nnz=hierarchy.level_nnz,
            inverse_diagonals=owned_inverse_diagonals,
            dampings=owned_dampings,
            bottom_inverse=owned_bottom_inverse,
            validation_control_readback_bytes=8,
            device_to_device_bytes=device_to_device_bytes,
        )

    def _ensure_current(self):
        _ensure_current_program(self._program, "V-cycle numeric snapshot")

    @property
    def inverse_diagonal_reserved_bytes(self):
        return sum(
            4 * int(value.shape[0]) for value in self._inverse_diagonals
        )

    @property
    def damping_reserved_bytes(self):
        return 4 * len(self._dampings)

    @property
    def bottom_inverse_reserved_bytes(self):
        return 4 * int(self._bottom_inverse.shape[0])

    @property
    def total_reserved_bytes(self):
        return (
            self.inverse_diagonal_reserved_bytes
            + self.damping_reserved_bytes
            + self.bottom_inverse_reserved_bytes
        )

    def debug_runtime_stats(self):
        self._ensure_current()
        return {
            "schema_version": 1,
            "identity": {
                "backend_family": self._backend,
                "method": "scalar_jacobi_dense_bottom_inverse",
                "topology_version": self.topology_version,
                "numeric_version": self.numeric_version,
                "level_sizes": self._level_sizes,
                "level_nnz": self._level_nnz,
                "nonbottom_level_count": len(self._inverse_diagonals),
                "bottom_size": self._level_sizes[-1],
            },
            "resources": {
                "inverse_diagonal_reserved_bytes": (
                    self.inverse_diagonal_reserved_bytes
                ),
                "damping_reserved_bytes": self.damping_reserved_bytes,
                "bottom_inverse_reserved_bytes": (
                    self.bottom_inverse_reserved_bytes
                ),
                "total_reserved_bytes": self.total_reserved_bytes,
            },
            "transfers": {
                "device_to_host_bytes": (
                    self._validation_control_readback_bytes
                ),
                "device_to_device_bytes": self._device_to_device_bytes,
                "device_payload_readback_bytes": 0,
            },
            "contract": {
                "caller_selects_smoother_coefficients": True,
                "caller_certifies_bottom_inverse_spd": True,
                "runtime_checks_finite_positive_and_symmetric": True,
                "all_numeric_arrays_owned": True,
                "host_matrix_payload_required": False,
                "coarsening_policy_selected": False,
                "graph_or_solver_constructed": False,
                "public_api": False,
            },
        }


def _ensure_hierarchy_numeric_pair(hierarchy, numeric, role):
    if not isinstance(hierarchy, _SparseCsrHierarchySnapshot):
        raise TaichiRuntimeError(
            f"{role} requires a sparse hierarchy snapshot"
        )
    if not isinstance(numeric, _SparseVcycleNumericSnapshot):
        raise TaichiRuntimeError(f"{role} requires a V-cycle numeric snapshot")
    hierarchy._ensure_current()
    numeric._ensure_current()
    if hierarchy._program is not numeric._program:
        raise TaichiRuntimeError(
            f"{role} hierarchy and numeric resources must share one Program"
        )
    if (
        hierarchy.topology_version != numeric.topology_version
        or hierarchy.numeric_version != numeric.numeric_version
    ):
        raise TaichiRuntimeError(
            f"{role} hierarchy and numeric versions must match"
        )
    if (
        hierarchy.level_sizes != numeric._level_sizes
        or hierarchy.level_nnz != numeric._level_nnz
    ):
        raise TaichiRuntimeError(
            f"{role} hierarchy and numeric level identities must match"
        )


def _vcycle_numeric_role_arrays(hierarchy, numeric):
    result = {}
    for level_index in range(hierarchy.level_count - 1):
        prefix = f"l{level_index}"
        result[f"{prefix}_values"] = hierarchy._levels[level_index]._values
        result[f"{prefix}_inverse_diagonal"] = (
            numeric._inverse_diagonals[level_index]
        )
        result[f"{prefix}_damping"] = numeric._dampings[level_index]
    result["bottom_inverse_values"] = numeric._bottom_inverse
    return result


class _SparseRecursiveVcycleNumericPublisher:
    """Version guard that binds existing device numeric arrays for refresh."""

    def __init__(self, hierarchy, numeric):
        _ensure_hierarchy_numeric_pair(
            hierarchy, numeric, "V-cycle numeric publisher"
        )
        self._program = hierarchy._program
        self._topology_version = hierarchy.topology_version
        self._level_sizes = np.asarray(hierarchy.level_sizes, dtype=np.int32)
        self._level_nnz = np.asarray(hierarchy.level_nnz, dtype=np.int32)
        self._role_shapes = {
            name: tuple(value.shape)
            for name, value in _vcycle_numeric_role_arrays(
                hierarchy, numeric
            ).items()
        }
        self._lock = threading.Lock()
        self._bind_calls = 0
        self._rejected_bind_calls = 0

    def _ensure_current(self):
        _ensure_current_program(self._program, "V-cycle numeric publisher")

    def _reject(self, message):
        self._rejected_bind_calls += 1
        raise TaichiRuntimeError(message)

    def bind_sources(
        self,
        hierarchy,
        numeric,
        *,
        expected_topology_version,
        expected_numeric_version,
    ):
        expected_topology_version = _positive_version(
            expected_topology_version,
            "V-cycle numeric publisher expected_topology_version",
        )
        expected_numeric_version = _positive_version(
            expected_numeric_version,
            "V-cycle numeric publisher expected_numeric_version",
        )
        with self._lock:
            self._ensure_current()
            try:
                _ensure_hierarchy_numeric_pair(
                    hierarchy, numeric, "V-cycle numeric refresh"
                )
            except TaichiRuntimeError as exc:
                self._reject(str(exc))
            if hierarchy._program is not self._program:
                self._reject(
                    "V-cycle numeric refresh cannot cross Program ownership"
                )
            if (
                expected_topology_version != self._topology_version
                or hierarchy.topology_version != self._topology_version
            ):
                self._reject(
                    "V-cycle numeric refresh topology version mismatch"
                )
            if hierarchy.numeric_version != expected_numeric_version + 1:
                self._reject(
                    "V-cycle numeric refresh must advance numeric version "
                    "exactly once"
                )
            if not np.array_equal(
                self._level_sizes,
                np.asarray(hierarchy.level_sizes, dtype=np.int32),
            ) or not np.array_equal(
                self._level_nnz,
                np.asarray(hierarchy.level_nnz, dtype=np.int32),
            ):
                self._reject(
                    "V-cycle numeric refresh changed level sizes or nnz"
                )
            sources = _vcycle_numeric_role_arrays(hierarchy, numeric)
            if set(sources) != set(self._role_shapes):
                self._reject("V-cycle numeric refresh role set changed")
            for name, value in sources.items():
                if tuple(value.shape) != self._role_shapes[name]:
                    self._reject(
                        f"V-cycle numeric refresh role {name!r} changed shape"
                    )
            self._bind_calls += 1
            return {name: value.arr for name, value in sources.items()}

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            return {
                "schema_version": 1,
                "identity": {
                    "topology_version": self._topology_version,
                    "level_sizes": tuple(int(v) for v in self._level_sizes),
                    "level_nnz": tuple(int(v) for v in self._level_nnz),
                },
                "operations": {
                    "bind_calls": self._bind_calls,
                    "rejected_bind_calls": self._rejected_bind_calls,
                },
                "host_topology_metadata_bytes": int(
                    self._level_sizes.nbytes + self._level_nnz.nbytes
                ),
                "device_reserved_bytes": 0,
                "numeric_role_count": len(self._role_shapes),
                "numeric_payload_bytes": sum(
                    int(np.prod(shape)) * 4
                    for shape in self._role_shapes.values()
                ),
                "contract": {
                    "topology_identity_uses_version_contract": True,
                    "host_pattern_payload_retained": False,
                    "source_arrays_copied_by_native_operator": True,
                    "host_metadata_total_bytes_reported": False,
                    "public_api": False,
                },
            }


class _SparseRecursiveVcycleGraphPlan:
    """Private symmetric V-cycle Graph backed by hierarchy snapshots."""

    def __init__(
        self,
        hierarchy,
        numeric,
        *,
        explicit_array_capacity_bytes,
    ):
        _ensure_hierarchy_numeric_pair(
            hierarchy, numeric, "recursive V-cycle Graph plan"
        )
        self._program = hierarchy._program
        self._backend = hierarchy._backend
        self._hierarchy = hierarchy
        self._numeric_setup = numeric
        self._topology_version = hierarchy.topology_version
        self._numeric_version = hierarchy.numeric_version
        self._size = hierarchy.level_sizes[0]
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "V-cycle Graph explicit_array_capacity_bytes",
        )
        self._topology = {}
        self._numeric = _vcycle_numeric_role_arrays(hierarchy, numeric)
        self._workspace = {}
        self._resource_types = {}
        self._scalars = {}
        self._lock = threading.Lock()
        self._apply_calls = 0
        self._rejected_apply_calls = 0
        self._native_operator_publishes = 0
        self._native_operator_publish_d2d_bytes = 0

        def add_array(role, name, value):
            getattr(self, f"_{role}")[name] = value
            self._resource_types[name] = (value.dtype, len(value.shape))

        for name, value in tuple(self._numeric.items()):
            self._resource_types[name] = (value.dtype, len(value.shape))

        for level_index in range(hierarchy.level_count - 1):
            prefix = f"l{level_index}"
            level = hierarchy._levels[level_index]
            schedule = hierarchy._restriction_schedules[level_index]
            self._scalars[f"{prefix}_size"] = level.rows
            self._scalars[f"{prefix}_coarse_size"] = (
                hierarchy.level_sizes[level_index + 1]
            )
            add_array(
                "topology",
                f"{prefix}_row_offsets",
                level._row_offsets,
            )
            add_array(
                "topology",
                f"{prefix}_columns",
                level._column_indices,
            )
            add_array(
                "topology",
                f"{prefix}_fine_to_coarse",
                hierarchy._aggregate_maps[level_index],
            )
            add_array(
                "topology",
                f"{prefix}_restriction_offsets",
                schedule._coarse_offsets,
            )
            add_array(
                "topology",
                f"{prefix}_restriction_fine_rows",
                schedule._ordered_fine_rows,
            )

        topology_bytes = sum(
            int(np.prod(value.shape)) * 4
            for value in self._topology.values()
        )
        numeric_bytes = sum(
            int(np.prod(value.shape)) * 4
            for value in self._numeric.values()
        )
        workspace_elements = sum(hierarchy.level_sizes[:-1]) + 2 * sum(
            hierarchy.level_sizes[1:]
        )
        workspace_bytes = 4 * workspace_elements
        native_operator_bytes = (
            topology_bytes + numeric_bytes + workspace_bytes
        )
        build_peak_bytes = (
            hierarchy.steady_reserved_bytes
            + numeric.total_reserved_bytes
            + workspace_bytes
            + native_operator_bytes
        )
        if build_peak_bytes > self._capacity_bytes:
            raise TaichiRuntimeError(
                "V-cycle Graph explicit-array capacity overflow before "
                "workspace or graph construction"
            )
        self._topology_argument_bytes = topology_bytes
        self._numeric_argument_bytes = numeric_bytes
        self._workspace_reserved_bytes = workspace_bytes
        self._native_operator_reserved_bytes = native_operator_bytes
        self._build_peak_explicit_array_bytes = build_peak_bytes

        for level_index in range(hierarchy.level_count):
            prefix = f"l{level_index}"
            level_size = hierarchy.level_sizes[level_index]
            if level_index < hierarchy.level_count - 1:
                value = ti.ndarray(ti.f32, shape=level_size)
                value.fill(0.0)
                add_array("workspace", f"{prefix}_pre_solution", value)
            if level_index > 0:
                rhs = ti.ndarray(ti.f32, shape=level_size)
                solution = ti.ndarray(ti.f32, shape=level_size)
                rhs.fill(0.0)
                solution.fill(0.0)
                add_array("workspace", f"{prefix}_rhs", rhs)
                add_array("workspace", f"{prefix}_solution", solution)

        bottom_index = hierarchy.level_count - 1
        bottom_prefix = f"l{bottom_index}"
        self._scalars[f"{bottom_prefix}_size"] = hierarchy.level_sizes[-1]

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
        for level_index in range(hierarchy.level_count - 1):
            prefix = f"l{level_index}"
            rhs = sym_input if level_index == 0 else symbols[f"{prefix}_rhs"]
            graph_builder.dispatch(
                _vcycle_pre_smooth,
                scalar_symbols[f"{prefix}_size"],
                symbols[f"{prefix}_inverse_diagonal"],
                symbols[f"{prefix}_damping"],
                rhs,
                symbols[f"{prefix}_pre_solution"],
            )
            graph_builder.dispatch(
                _vcycle_restrict_gather,
                scalar_symbols[f"{prefix}_coarse_size"],
                symbols[f"{prefix}_row_offsets"],
                symbols[f"{prefix}_columns"],
                symbols[f"{prefix}_restriction_offsets"],
                symbols[f"{prefix}_restriction_fine_rows"],
                symbols[f"{prefix}_values"],
                rhs,
                symbols[f"{prefix}_pre_solution"],
                symbols[f"l{level_index + 1}_rhs"],
            )
        graph_builder.dispatch(
            _vcycle_dense_bottom,
            scalar_symbols[f"{bottom_prefix}_size"],
            symbols["bottom_inverse_values"],
            symbols[f"{bottom_prefix}_rhs"],
            symbols[f"{bottom_prefix}_solution"],
        )
        for level_index in range(hierarchy.level_count - 2, -1, -1):
            prefix = f"l{level_index}"
            rhs = sym_input if level_index == 0 else symbols[f"{prefix}_rhs"]
            output = (
                sym_output
                if level_index == 0
                else symbols[f"{prefix}_solution"]
            )
            graph_builder.dispatch(
                _vcycle_post_smooth,
                scalar_symbols[f"{prefix}_size"],
                symbols[f"{prefix}_row_offsets"],
                symbols[f"{prefix}_columns"],
                symbols[f"{prefix}_fine_to_coarse"],
                symbols[f"{prefix}_values"],
                symbols[f"{prefix}_inverse_diagonal"],
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

    def _ensure_current(self):
        _ensure_current_program(self._program, "recursive V-cycle Graph plan")

    def _reject_apply(self, message):
        self._rejected_apply_calls += 1
        raise TaichiRuntimeError(message)

    def apply(self, input_array, output_array):
        with self._lock:
            self._ensure_current()
            input_array = _require_current_scalar_ndarray(
                input_array,
                "recursive V-cycle input",
                ti.f32,
                one_dimensional=True,
            )
            output_array = _require_current_scalar_ndarray(
                output_array,
                "recursive V-cycle output",
                ti.f32,
                one_dimensional=True,
            )
            if input_array.shape != (self._size,):
                self._reject_apply(
                    "recursive V-cycle input size does not match"
                )
            if output_array.shape != (self._size,):
                self._reject_apply(
                    "recursive V-cycle output size does not match"
                )
            if int(input_array.arr.device_allocation_ptr()) == int(
                output_array.arr.device_allocation_ptr()
            ):
                self._reject_apply(
                    "recursive V-cycle input/output alias is unsupported"
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
        with self._lock:
            self._ensure_current()
            if self._native_operator_publishes != 0:
                raise TaichiRuntimeError(
                    "recursive V-cycle plan publishes at most one native "
                    "operator"
                )
            operator = self._program._create_compiled_graph_linear_operator(
                self._graph._compiled_graph,
                self._size,
                self._topology_version,
                self._numeric_version,
                dict(self._scalars),
                {name: value.arr for name, value in self._topology.items()},
                {name: value.arr for name, value in self._numeric.items()},
                {name: value.arr for name, value in self._workspace.items()},
            )
            self._native_operator_publishes = 1
            self._native_operator_publish_d2d_bytes = (
                self._native_operator_reserved_bytes
            )
            return operator

    def create_numeric_publisher(self):
        with self._lock:
            self._ensure_current()
            return _SparseRecursiveVcycleNumericPublisher(
                self._hierarchy, self._numeric_setup
            )

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            graph_execution = self._graph.execution_stats()
            graph_cache = _graph_cache_memory_attribution(self._graph)
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": "recursive_symmetric_vcycle",
                    "size": self._size,
                    "level_count": self._hierarchy.level_count,
                    "topology_version": self._topology_version,
                    "numeric_version": self._numeric_version,
                },
                "operations": {
                    "apply_calls": self._apply_calls,
                    "rejected_apply_calls": self._rejected_apply_calls,
                    "graph_node_count": graph_execution.node_count,
                    "graph_dispatch_count": graph_execution.dispatch_count,
                    "kernel_dispatches_per_apply": (
                        1 + 3 * (self._hierarchy.level_count - 1)
                    ),
                    "host_graph_submissions_per_apply": 1,
                    "explicit_apply_host_synchronizations": 0,
                    "native_operator_publishes": (
                        self._native_operator_publishes
                    ),
                },
                "resources": {
                    "borrowed_hierarchy_reserved_bytes": (
                        self._hierarchy.steady_reserved_bytes
                    ),
                    "borrowed_numeric_setup_reserved_bytes": (
                        self._numeric_setup.total_reserved_bytes
                    ),
                    "topology_argument_reserved_bytes": (
                        self._topology_argument_bytes
                    ),
                    "numeric_argument_reserved_bytes": (
                        self._numeric_argument_bytes
                    ),
                    "plan_workspace_reserved_bytes": (
                        self._workspace_reserved_bytes
                    ),
                    "native_operator_reserved_bytes": (
                        self._native_operator_reserved_bytes
                    ),
                    "build_peak_explicit_array_bytes": (
                        self._build_peak_explicit_array_bytes
                    ),
                    "explicit_array_capacity_bytes": self._capacity_bytes,
                    **graph_cache["resources"],
                },
                "transfers": {
                    "device_to_host_bytes": 0,
                    "device_to_device_bytes": (
                        self._native_operator_publish_d2d_bytes
                    ),
                    "device_kernel_workspace_initialization_bytes": (
                        self._workspace_reserved_bytes
                    ),
                    "device_payload_readback_bytes": 0,
                },
                "contract": {
                    "deterministic_restriction_gather": True,
                    "floating_atomic_restriction_required": False,
                    "hierarchy_and_numeric_sources_borrowed": True,
                    "native_operator_owns_argument_snapshots": True,
                    "caller_selects_numeric_setup": True,
                    "no_host_matrix_payload": True,
                    "no_host_fallback": True,
                    **graph_cache["contract"],
                    "workspace_total_bytes_reported": False,
                    "coarsening_policy_selected": False,
                    "public_api": False,
                },
            }
