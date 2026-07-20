"""Private execution inputs for caller-provided varying-block hierarchies.

This layer binds an immutable varying-block hierarchy to deterministic
rectangular transfer Graph plans and caller-qualified packed smoother data.
It deliberately stops before constructing a recursive V-cycle or a solver.
"""

import threading

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError

from ._sparse_block_transfer_graph import _SparseBlockTransferGraphPlan
from ._sparse_block_vcycle_numeric import (
    _BLOCK_VCYCLE_NUMERIC_STATUS,
    _copy_f32_segment,
    _validate_block_vcycle_dampings,
    _validate_packed_block_inverses,
    _validate_packed_bottom_inverse,
)
from ._sparse_hierarchy_assembly import (
    _ensure_current_program,
    _positive_int,
    _positive_version,
)
from ._sparse_runtime_memory import (
    SORT_SCAN_WORKSPACE_FAMILIES,
    _graph_cache_memory_attribution,
    _program_workspace_attribution,
    _workspace_family_reserved_bytes,
)
from ._sparse_varying_bsr_hierarchy import _VaryingBsrHierarchySnapshot
from .sparse_matrix import _require_current_scalar_ndarray


@ti.kernel
def _apply_packed_damped_block_inverse(
    inverse_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    dampings: ti.types.ndarray(dtype=ti.f32, ndim=1),
    inverse_offset: ti.i32,
    damping_index: ti.i32,
    block_rows: ti.i32,
    block_size: ti.i32,
    source: ti.types.ndarray(dtype=ti.f32, ndim=1),
    destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for block_row, local_row in ti.ndrange(block_rows, block_size):
        total = 0.0
        inverse_base = (
            inverse_offset
            + block_row * block_size * block_size
            + local_row * block_size
        )
        source_base = block_row * block_size
        for local_column in range(block_size):
            total += (
                inverse_values[inverse_base + local_column]
                * source[source_base + local_column]
            )
        destination[source_base + local_row] = (
            dampings[damping_index] * total
        )


@ti.kernel
def _apply_packed_bottom_inverse(
    inverse_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    inverse_offset: ti.i32,
    size: ti.i32,
    source: ti.types.ndarray(dtype=ti.f32, ndim=1),
    destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for row in range(size):
        total = 0.0
        for column in range(size):
            total += (
                inverse_values[inverse_offset + row * size + column]
                * source[column]
            )
        destination[row] = total


class _SparseVaryingBlockVcycleNumericSnapshot:
    """Owned packed block inverses for independently sized BSR levels."""

    def __init__(
        self,
        *,
        hierarchy,
        level_block_inverse_offsets,
        bottom_inverse_offset,
        inverse_values,
        dampings,
        topology_version,
        numeric_version,
        validation_control_readback_bytes,
        device_to_device_bytes,
    ):
        self._program = hierarchy._program
        self._backend = hierarchy._backend
        self._hierarchy = hierarchy
        self._level_block_rows = tuple(
            level.block_rows for level in hierarchy._levels
        )
        self._level_block_sizes = tuple(
            level.block_size for level in hierarchy._levels
        )
        self._level_block_nnz = tuple(
            level.block_nnz for level in hierarchy._levels
        )
        self._level_block_inverse_offsets = tuple(
            int(value) for value in level_block_inverse_offsets
        )
        self._bottom_inverse_offset = int(bottom_inverse_offset)
        self.bottom_scalar_size = hierarchy._levels[-1].rows
        self._inverse_values = inverse_values
        self._dampings = dampings
        self.topology_version = int(topology_version)
        self.numeric_version = int(numeric_version)
        self._validation_control_readback_bytes = int(
            validation_control_readback_bytes
        )
        self._device_to_device_bytes = int(device_to_device_bytes)
        self._lock = threading.Lock()
        self._block_apply_calls = 0
        self._bottom_apply_calls = 0
        self._rejected_apply_calls = 0

    @classmethod
    def _preflight_sources(
        cls,
        hierarchy,
        *,
        block_inverses,
        dampings,
        bottom_inverse,
        topology_version,
        numeric_version,
    ):
        if not isinstance(hierarchy, _VaryingBsrHierarchySnapshot):
            raise TaichiRuntimeError(
                "varying block V-cycle numeric setup requires a varying "
                "BSR hierarchy snapshot"
            )
        hierarchy._ensure_current()
        topology_version = _positive_version(
            topology_version,
            "varying block V-cycle numeric topology_version",
        )
        numeric_version = _positive_version(
            numeric_version,
            "varying block V-cycle numeric numeric_version",
        )
        if (
            hierarchy.topology_version != topology_version
            or hierarchy.numeric_version != numeric_version
        ):
            raise TaichiRuntimeError(
                "varying block V-cycle numeric versions must match the "
                "hierarchy"
            )
        try:
            block_inverses = tuple(block_inverses)
        except TypeError as exc:
            raise TaichiRuntimeError(
                "varying block V-cycle inverse blocks must be iterable"
            ) from exc
        nonbottom_levels = hierarchy.level_count - 1
        if len(block_inverses) != nonbottom_levels:
            raise TaichiRuntimeError(
                "varying block V-cycle numeric setup requires one block "
                "inverse array per non-bottom level"
            )

        checked_inverses = []
        inverse_offsets = []
        inverse_count = 0
        for level_index, level in enumerate(hierarchy._levels[:-1]):
            expected_count = (
                level.block_rows * level.block_size * level.block_size
            )
            inverse = _require_current_scalar_ndarray(
                block_inverses[level_index],
                f"varying block V-cycle level {level_index} inverse blocks",
                ti.f32,
                one_dimensional=True,
            )
            if inverse.shape != (expected_count,):
                raise TaichiRuntimeError(
                    f"varying block V-cycle level {level_index} inverse "
                    "block shape does not match the hierarchy"
                )
            inverse_offsets.append(inverse_count)
            inverse_count += expected_count
            checked_inverses.append(inverse)

        dampings = _require_current_scalar_ndarray(
            dampings,
            "varying block V-cycle dampings",
            ti.f32,
            one_dimensional=True,
        )
        if dampings.shape != (nonbottom_levels,):
            raise TaichiRuntimeError(
                "varying block V-cycle dampings must contain one value per "
                "non-bottom level"
            )
        bottom_scalar_size = hierarchy._levels[-1].rows
        bottom_count = bottom_scalar_size * bottom_scalar_size
        bottom_inverse = _require_current_scalar_ndarray(
            bottom_inverse,
            "varying block V-cycle bottom inverse",
            ti.f32,
            one_dimensional=True,
        )
        if bottom_inverse.shape != (bottom_count,):
            raise TaichiRuntimeError(
                "varying block V-cycle bottom inverse shape does not match "
                "the bottom scalar size"
            )
        total_inverse_count = inverse_count + bottom_count
        if total_inverse_count >= 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "varying block V-cycle packed inverse count exceeds the i32 "
                "limit"
            )
        total_bytes = 4 * (total_inverse_count + nonbottom_levels)
        return {
            "topology_version": topology_version,
            "numeric_version": numeric_version,
            "checked_inverses": tuple(checked_inverses),
            "dampings": dampings,
            "bottom_inverse": bottom_inverse,
            "inverse_offsets": tuple(inverse_offsets),
            "bottom_inverse_offset": inverse_count,
            "bottom_count": bottom_count,
            "total_inverse_count": total_inverse_count,
            "total_bytes": total_bytes,
        }

    @classmethod
    def _copy_preflight(cls, hierarchy, preflight):
        owned_inverse_values = ti.ndarray(
            ti.f32, shape=preflight["total_inverse_count"]
        )
        nonbottom_levels = hierarchy.level_count - 1
        owned_dampings = ti.ndarray(ti.f32, shape=nonbottom_levels)
        control = ti.ndarray(ti.i32, shape=2)
        control.fill(0)
        for level_index, inverse in enumerate(
            preflight["checked_inverses"]
        ):
            level = hierarchy._levels[level_index]
            count = level.block_rows * level.block_size * level.block_size
            _copy_f32_segment(
                inverse,
                owned_inverse_values,
                preflight["inverse_offsets"][level_index],
                count,
            )
        _copy_f32_segment(
            preflight["bottom_inverse"],
            owned_inverse_values,
            preflight["bottom_inverse_offset"],
            preflight["bottom_count"],
        )
        _copy_f32_segment(
            preflight["dampings"], owned_dampings, 0, nonbottom_levels
        )
        for level_index, level in enumerate(hierarchy._levels[:-1]):
            _validate_packed_block_inverses(
                owned_inverse_values,
                preflight["inverse_offsets"][level_index],
                level.block_rows,
                level.block_size,
                control,
            )
        _validate_packed_bottom_inverse(
            owned_inverse_values,
            preflight["bottom_inverse_offset"],
            hierarchy._levels[-1].rows,
            control,
        )
        _validate_block_vcycle_dampings(
            owned_dampings, nonbottom_levels, control
        )
        control_host = control.to_numpy()
        _ensure_current_program(
            hierarchy._program,
            "varying block V-cycle numeric construction",
        )
        status = int(control_host[0])
        if status == 0 and int(control_host[1]) != nonbottom_levels:
            status = 8
        if status != 0:
            reason = _BLOCK_VCYCLE_NUMERIC_STATUS.get(
                status, "unknown validation failure"
            )
            raise TaichiRuntimeError(
                "varying block V-cycle numeric validation failed before "
                "publish: "
                + reason
            )
        return cls(
            hierarchy=hierarchy,
            level_block_inverse_offsets=preflight["inverse_offsets"],
            bottom_inverse_offset=preflight["bottom_inverse_offset"],
            inverse_values=owned_inverse_values,
            dampings=owned_dampings,
            topology_version=preflight["topology_version"],
            numeric_version=preflight["numeric_version"],
            validation_control_readback_bytes=8,
            device_to_device_bytes=preflight["total_bytes"],
        )

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "varying block V-cycle numeric snapshot"
        )

    @staticmethod
    def _aliases(left, right):
        return int(left.arr.device_allocation_ptr()) == int(
            right.arr.device_allocation_ptr()
        )

    def _checked_level_index(self, level_index):
        if isinstance(level_index, bool):
            self._rejected_apply_calls += 1
            raise TaichiRuntimeError(
                "varying block V-cycle numeric level index is invalid"
            )
        try:
            checked = int(level_index)
        except (TypeError, ValueError, OverflowError) as exc:
            self._rejected_apply_calls += 1
            raise TaichiRuntimeError(
                "varying block V-cycle numeric level index is invalid"
            ) from exc
        if checked != level_index or not (
            0 <= checked < self.nonbottom_level_count
        ):
            self._rejected_apply_calls += 1
            raise TaichiRuntimeError(
                "varying block V-cycle numeric level index is invalid"
            )
        return checked

    def _checked_vectors(self, source, destination, expected_size, role):
        source = _require_current_scalar_ndarray(
            source,
            f"varying block V-cycle {role} source",
            ti.f32,
            one_dimensional=True,
        )
        destination = _require_current_scalar_ndarray(
            destination,
            f"varying block V-cycle {role} destination",
            ti.f32,
            one_dimensional=True,
        )
        if source.shape != (expected_size,) or destination.shape != (
            expected_size,
        ):
            self._rejected_apply_calls += 1
            raise TaichiRuntimeError(
                f"varying block V-cycle {role} vector size mismatch"
            )
        if self._aliases(source, destination):
            self._rejected_apply_calls += 1
            raise TaichiRuntimeError(
                f"varying block V-cycle {role} input/output alias is "
                "unsupported"
            )
        return source, destination

    def apply_damped_block_inverse(
        self, level_index, source, destination
    ):
        with self._lock:
            self._ensure_current()
            level_index = self._checked_level_index(level_index)
            block_rows = self._level_block_rows[level_index]
            block_size = self._level_block_sizes[level_index]
            source, destination = self._checked_vectors(
                source,
                destination,
                block_rows * block_size,
                f"level {level_index} block inverse",
            )
            _apply_packed_damped_block_inverse(
                self._inverse_values,
                self._dampings,
                self._level_block_inverse_offsets[level_index],
                level_index,
                block_rows,
                block_size,
                source,
                destination,
            )
            self._block_apply_calls += 1

    def apply_bottom_inverse(self, source, destination):
        with self._lock:
            self._ensure_current()
            source, destination = self._checked_vectors(
                source,
                destination,
                self.bottom_scalar_size,
                "bottom inverse",
            )
            _apply_packed_bottom_inverse(
                self._inverse_values,
                self._bottom_inverse_offset,
                self.bottom_scalar_size,
                source,
                destination,
            )
            self._bottom_apply_calls += 1

    @property
    def nonbottom_level_count(self):
        return len(self._level_block_inverse_offsets)

    @property
    def block_inverse_reserved_bytes(self):
        return 4 * self._bottom_inverse_offset

    @property
    def bottom_inverse_reserved_bytes(self):
        return 4 * self.bottom_scalar_size * self.bottom_scalar_size

    @property
    def damping_reserved_bytes(self):
        return 4 * self.nonbottom_level_count

    @property
    def total_reserved_bytes(self):
        return (
            self.block_inverse_reserved_bytes
            + self.bottom_inverse_reserved_bytes
            + self.damping_reserved_bytes
        )

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": "packed_varying_block_jacobi_dense_bottom_inverse",
                    "level_block_rows": self._level_block_rows,
                    "level_block_sizes": self._level_block_sizes,
                    "level_block_nnz": self._level_block_nnz,
                    "nonbottom_level_count": self.nonbottom_level_count,
                    "level_block_inverse_offsets": (
                        self._level_block_inverse_offsets
                    ),
                    "bottom_inverse_offset": self._bottom_inverse_offset,
                    "bottom_scalar_size": self.bottom_scalar_size,
                    "topology_version": self.topology_version,
                    "numeric_version": self.numeric_version,
                },
                "operations": {
                    "damped_block_inverse_apply_calls": (
                        self._block_apply_calls
                    ),
                    "bottom_inverse_apply_calls": self._bottom_apply_calls,
                    "rejected_apply_calls": self._rejected_apply_calls,
                    "explicit_apply_host_synchronizations": 0,
                },
                "resources": {
                    "block_inverse_reserved_bytes": (
                        self.block_inverse_reserved_bytes
                    ),
                    "bottom_inverse_reserved_bytes": (
                        self.bottom_inverse_reserved_bytes
                    ),
                    "damping_reserved_bytes": self.damping_reserved_bytes,
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
                    "per_level_block_size_honored": True,
                    "one_packed_inverse_numeric_role": True,
                    "one_packed_damping_numeric_role": True,
                    "single_validation_control_readback": True,
                    "full_spd_qualification_is_caller_responsibility": True,
                    "host_inversion_performed": False,
                    "immutable_by_internal_contract": True,
                    "recursive_vcycle_constructed": False,
                    "public_api": False,
                },
            }


class _SparseVaryingBlockVcycleInputs:
    """One immutable hierarchy, transfer-plan, and numeric input generation."""

    def __init__(
        self,
        *,
        hierarchy,
        transfer_plans,
        numeric,
        explicit_array_capacity_bytes,
        build_stats,
    ):
        self._program = hierarchy._program
        self._backend = hierarchy._backend
        self._hierarchy = hierarchy
        self._transfer_plans = tuple(transfer_plans)
        self._numeric = numeric
        self.topology_version = hierarchy.topology_version
        self.numeric_version = hierarchy.numeric_version
        self._capacity_bytes = int(explicit_array_capacity_bytes)
        self._build_stats = dict(build_stats)

    @classmethod
    def build_validated(
        cls,
        hierarchy,
        *,
        block_inverses,
        dampings,
        bottom_inverse,
        topology_version,
        numeric_version,
        explicit_array_capacity_bytes,
    ):
        capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "varying block V-cycle inputs explicit_array_capacity_bytes",
        )
        numeric_preflight = (
            _SparseVaryingBlockVcycleNumericSnapshot._preflight_sources(
                hierarchy,
                block_inverses=block_inverses,
                dampings=dampings,
                bottom_inverse=bottom_inverse,
                topology_version=topology_version,
                numeric_version=numeric_version,
            )
        )
        hierarchy_bytes = hierarchy.steady_reserved_bytes
        source_numeric_bytes = numeric_preflight["total_bytes"]
        schedule_bytes_by_transition = tuple(
            4 * (transfer.coarse_block_rows + 1 + 2 * transfer.block_nnz)
            for transfer in hierarchy._transfers
        )
        schedule_staging_by_transition = tuple(
            12 * transfer.block_nnz for transfer in hierarchy._transfers
        )
        retained_schedule_bytes = 0
        phase_peak_bytes = []
        build_peak_bytes = hierarchy_bytes + source_numeric_bytes
        for schedule_bytes, staging_bytes in zip(
            schedule_bytes_by_transition,
            schedule_staging_by_transition,
        ):
            phase_peak = (
                hierarchy_bytes
                + source_numeric_bytes
                + retained_schedule_bytes
                + schedule_bytes
                + staging_bytes
            )
            phase_peak_bytes.append(phase_peak)
            build_peak_bytes = max(build_peak_bytes, phase_peak)
            retained_schedule_bytes += schedule_bytes
        schedule_bytes = sum(schedule_bytes_by_transition)
        numeric_phase_peak = (
            hierarchy_bytes
            + schedule_bytes
            + source_numeric_bytes
            + source_numeric_bytes
            + 8
        )
        build_peak_bytes = max(build_peak_bytes, numeric_phase_peak)
        steady_bytes = (
            hierarchy_bytes + schedule_bytes + source_numeric_bytes
        )
        if build_peak_bytes > capacity_bytes:
            raise TaichiRuntimeError(
                "varying block V-cycle inputs explicit-array capacity "
                "overflow during preflight"
            )

        transfer_plans = []
        for transfer, schedule_size, staging_size in zip(
            hierarchy._transfers,
            schedule_bytes_by_transition,
            schedule_staging_by_transition,
        ):
            local_capacity = (
                transfer.total_reserved_bytes
                + schedule_size
                + staging_size
            )
            transfer_plans.append(
                _SparseBlockTransferGraphPlan(
                    transfer,
                    explicit_array_capacity_bytes=local_capacity,
                )
            )
        numeric = _SparseVaryingBlockVcycleNumericSnapshot._copy_preflight(
            hierarchy, numeric_preflight
        )
        return cls(
            hierarchy=hierarchy,
            transfer_plans=transfer_plans,
            numeric=numeric,
            explicit_array_capacity_bytes=capacity_bytes,
            build_stats={
                "schedule_bytes_by_transition": (
                    schedule_bytes_by_transition
                ),
                "schedule_staging_by_transition": (
                    schedule_staging_by_transition
                ),
                "phase_peak_bytes": tuple(phase_peak_bytes),
                "numeric_phase_peak_bytes": numeric_phase_peak,
                "source_numeric_bytes": source_numeric_bytes,
                "steady_bytes": steady_bytes,
                "build_peak_bytes": build_peak_bytes,
            },
        )

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "varying block V-cycle input generation"
        )

    @property
    def level_count(self):
        return self._hierarchy.level_count

    @property
    def transition_count(self):
        return self._hierarchy.transition_count

    @property
    def transfer_schedule_reserved_bytes(self):
        return sum(self._build_stats["schedule_bytes_by_transition"])

    @property
    def steady_reserved_bytes(self):
        return self._build_stats["steady_bytes"]

    def debug_runtime_stats(self):
        self._ensure_current()
        plan_stats = tuple(
            plan.debug_runtime_stats() for plan in self._transfer_plans
        )
        numeric_stats = self._numeric.debug_runtime_stats()
        program_workspace = _program_workspace_attribution(self._program)
        shared_sort_scan_bytes = _workspace_family_reserved_bytes(
            program_workspace["groups"], SORT_SCAN_WORKSPACE_FAMILIES
        )
        graph_cache = _graph_cache_memory_attribution(
            *(
                graph
                for plan in self._transfer_plans
                for graph in (
                    plan._prolongate_graph,
                    plan._restrict_graph,
                )
            )
        )
        return {
            "schema_version": 1,
            "identity": {
                "backend_family": self._backend,
                "method": "varying_block_hierarchy_execution_inputs",
                "level_count": self.level_count,
                "transition_count": self.transition_count,
                "level_block_rows": tuple(
                    level.block_rows for level in self._hierarchy._levels
                ),
                "level_block_sizes": tuple(
                    level.block_size for level in self._hierarchy._levels
                ),
                "level_scalar_rows": tuple(
                    level.rows for level in self._hierarchy._levels
                ),
                "transfer_topology_versions": tuple(
                    transfer.topology_version
                    for transfer in self._hierarchy._transfers
                ),
                "transfer_numeric_versions": tuple(
                    transfer.numeric_version
                    for transfer in self._hierarchy._transfers
                ),
                "topology_version": self.topology_version,
                "numeric_version": self.numeric_version,
            },
            "operations": {
                "transfer_graph_plan_count": len(self._transfer_plans),
                "prolongate_calls": sum(
                    stats["operations"]["prolongate_calls"]
                    for stats in plan_stats
                ),
                "restrict_calls": sum(
                    stats["operations"]["restrict_calls"]
                    for stats in plan_stats
                ),
                "numeric_apply_calls": (
                    numeric_stats["operations"][
                        "damped_block_inverse_apply_calls"
                    ]
                    + numeric_stats["operations"][
                        "bottom_inverse_apply_calls"
                    ]
                ),
                "transfer_schedule_construction_host_synchronizations": sum(
                    stats["operations"][
                        "construction_host_synchronizations"
                    ]
                    for stats in plan_stats
                ),
                "numeric_validation_control_readbacks": 1,
            },
            "resources": {
                "borrowed_hierarchy_reserved_bytes": (
                    self._hierarchy.steady_reserved_bytes
                ),
                "transfer_schedule_reserved_bytes_by_transition": (
                    self._build_stats["schedule_bytes_by_transition"]
                ),
                "transfer_schedule_reserved_bytes": (
                    self.transfer_schedule_reserved_bytes
                ),
                "packed_numeric_reserved_bytes": (
                    self._numeric.total_reserved_bytes
                ),
                "additional_owned_reserved_bytes": (
                    self.transfer_schedule_reserved_bytes
                    + self._numeric.total_reserved_bytes
                ),
                "steady_reserved_bytes": self.steady_reserved_bytes,
                "caller_numeric_source_reserved_bytes_during_build": (
                    self._build_stats["source_numeric_bytes"]
                ),
                "validation_control_peak_bytes": 8,
                "retired_schedule_staging_reserved_bytes_by_transition": (
                    self._build_stats["schedule_staging_by_transition"]
                ),
                "transfer_plan_phase_peak_explicit_array_bytes": (
                    self._build_stats["phase_peak_bytes"]
                ),
                "numeric_phase_peak_explicit_array_bytes": (
                    self._build_stats["numeric_phase_peak_bytes"]
                ),
                "build_peak_explicit_array_bytes": (
                    self._build_stats["build_peak_bytes"]
                ),
                "explicit_array_capacity_bytes": self._capacity_bytes,
                "shared_sort_scan_workspace_bytes": shared_sort_scan_bytes,
                "shared_sort_scan_workspace_ownership_scope": (
                    "program_ordering_ordering_aux_scan_arena"
                    if program_workspace["available"]
                    else None
                ),
                "program_primitive_workspace_reserved_bytes": (
                    program_workspace["reserved_bytes"]
                ),
                "program_primitive_workspace_peak_reserved_bytes": (
                    program_workspace["peak_reserved_bytes"]
                ),
                "program_primitive_workspace_groups": (
                    program_workspace["groups"]
                ),
                **graph_cache["resources"],
            },
            "transfers": {
                "device_to_host_bytes": 8,
                "device_to_device_bytes": self._numeric.total_reserved_bytes,
                "device_kernel_publish_bytes": (
                    self.transfer_schedule_reserved_bytes
                ),
                "device_payload_readback_bytes": 0,
            },
            "per_transition": tuple(
                {
                    "transition_index": index,
                    "fine_block_size": transfer.fine_block_size,
                    "coarse_block_size": transfer.coarse_block_size,
                    "transfer_reserved_bytes": transfer.total_reserved_bytes,
                    "transpose_schedule_reserved_bytes": (
                        self._build_stats["schedule_bytes_by_transition"][
                            index
                        ]
                    ),
                    "retired_schedule_staging_reserved_bytes": (
                        self._build_stats[
                            "schedule_staging_by_transition"
                        ][index]
                    ),
                    "phase_peak_explicit_array_bytes": self._build_stats[
                        "phase_peak_bytes"
                    ][index],
                }
                for index, transfer in enumerate(
                    self._hierarchy._transfers
                )
            ),
            "numeric": numeric_stats,
            "contract": {
                "hierarchy_and_numeric_versions_match": True,
                "rectangular_transfer_graph_plans_owned": True,
                "deterministic_transpose_schedules_owned": True,
                "fine_and_coarse_block_sizes_may_differ": True,
                "caller_numeric_sources_not_retained": True,
                "failed_generation_never_publishes_partial_inputs": True,
                "full_spd_qualification_is_caller_responsibility": True,
                "program_shared_workspace_reported": (
                    program_workspace["available"]
                ),
                "program_shared_workspace_current_groups_exact": (
                    program_workspace["current_group_totals_match"]
                ),
                "program_shared_workspace_group_peak_reported": (
                    program_workspace[
                        "historical_peak_group_breakdown_available"
                    ]
                ),
                "program_shared_workspace_in_explicit_capacity": False,
                **graph_cache["contract"],
                "candidate_modes_or_coarsening_selected": False,
                "recursive_vcycle_constructed": False,
                "pcg_constructed": False,
                "public_api": False,
            },
        }
