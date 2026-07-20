"""Private caller-coarsened multilevel BSR hierarchy generations.

This module composes exact device-visible BSR levels, owned block aggregate
maps, and deterministic block-row restriction schedules.  It deliberately
does not select a coarsening policy, construct block smoothers, or publish a
V-cycle.  The explicit-array capacity excludes shared native sort/scan
workspace whose complete device ownership is not yet reported.
"""

import copy
import threading
import weakref

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime

from ._sparse_bsr_graph_operator import _DeviceBsrSnapshot
from ._sparse_bsr_hierarchy_assembly import (
    _SparseGalerkinBsrAssemblyPlan,
)
from ._sparse_hierarchy_assembly import (
    _copy_i32_prefix,
    _ensure_current_program,
    _positive_int,
    _positive_version,
)
from ._sparse_hierarchy_candidate import (
    _AggregateRestrictionSchedulePlan,
)
from .sparse_matrix import _require_current_scalar_ndarray


class _SparseBsrHierarchySnapshot:
    """Self-contained BSR levels and block aggregate maps for one generation."""

    def __init__(
        self,
        *,
        program,
        backend,
        topology_version,
        numeric_version,
        bottom_scalar_size_cap,
        levels,
        aggregate_maps,
        restriction_schedules,
        build_report,
    ):
        self._program = program
        self._backend = backend
        self.topology_version = int(topology_version)
        self.numeric_version = int(numeric_version)
        self.bottom_scalar_size_cap = int(bottom_scalar_size_cap)
        self._levels = tuple(levels)
        self._aggregate_maps = tuple(aggregate_maps)
        self._restriction_schedules = tuple(restriction_schedules)
        self._build_report = copy.deepcopy(build_report)
        if len(self._levels) != len(self._aggregate_maps) + 1:
            raise TaichiRuntimeError(
                "block hierarchy levels must contain one more entry than "
                "maps"
            )
        if len(self._restriction_schedules) != len(self._aggregate_maps):
            raise TaichiRuntimeError(
                "block hierarchy requires one restriction schedule per map"
            )
        if not self._levels:
            raise TaichiRuntimeError(
                "block hierarchy requires at least one level"
            )
        block_size = self._levels[0].block_size
        for level_index, level in enumerate(self._levels):
            if not isinstance(level, _DeviceBsrSnapshot):
                raise TaichiRuntimeError(
                    "block hierarchy levels must be owned BSR snapshots"
                )
            level._ensure_current()
            if level._program is not self._program:
                raise TaichiRuntimeError(
                    "block hierarchy levels must belong to one Program"
                )
            if level.block_rows != level.block_cols:
                raise TaichiRuntimeError(
                    "block hierarchy levels must remain square"
                )
            if level.block_size != block_size:
                raise TaichiRuntimeError(
                    "block hierarchy levels must preserve block_size"
                )
            if (
                level.topology_version != self.topology_version
                or level.numeric_version != self.numeric_version
            ):
                raise TaichiRuntimeError(
                    "block hierarchy level versions must match the generation"
                )
            if level_index + 1 < len(self._levels):
                mapping = self._aggregate_maps[level_index]
                schedule = self._restriction_schedules[level_index]
                if mapping.shape != (level.block_rows,):
                    raise TaichiRuntimeError(
                        "block hierarchy map shape does not match its fine level"
                    )
                if (
                    schedule.fine_rows != level.block_rows
                    or schedule.coarse_rows
                    != self._levels[level_index + 1].block_rows
                ):
                    raise TaichiRuntimeError(
                        "block hierarchy restriction schedule dimensions do "
                        "not match neighboring levels"
                    )

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "block sparse hierarchy snapshot"
        )

    @property
    def level_count(self):
        return len(self._levels)

    @property
    def block_size(self):
        return self._levels[0].block_size

    @property
    def level_block_rows(self):
        return tuple(level.block_rows for level in self._levels)

    @property
    def level_scalar_rows(self):
        return tuple(level.rows for level in self._levels)

    @property
    def level_block_nnz(self):
        return tuple(level.block_nnz for level in self._levels)

    @property
    def operator_reserved_bytes(self):
        return sum(level.total_reserved_bytes for level in self._levels)

    @property
    def aggregate_map_reserved_bytes(self):
        return sum(
            4 * int(mapping.shape[0]) for mapping in self._aggregate_maps
        )

    @property
    def restriction_schedule_reserved_bytes(self):
        return sum(
            schedule.total_reserved_bytes
            for schedule in self._restriction_schedules
        )

    @property
    def steady_reserved_bytes(self):
        return (
            self.operator_reserved_bytes
            + self.aggregate_map_reserved_bytes
            + self.restriction_schedule_reserved_bytes
        )

    def debug_runtime_stats(self):
        self._ensure_current()
        return {
            "schema_version": 1,
            "identity": {
                "backend_family": self._backend,
                "storage_format": "recursive_bsr",
                "dtype": "f32",
                "index_dtype": "i32",
                "block_size": self.block_size,
                "topology_version": self.topology_version,
                "numeric_version": self.numeric_version,
                "level_count": self.level_count,
                "level_block_rows": self.level_block_rows,
                "level_scalar_rows": self.level_scalar_rows,
                "level_block_nnz": self.level_block_nnz,
                "bottom_scalar_size_cap": self.bottom_scalar_size_cap,
            },
            "resources": {
                "operator_reserved_bytes": self.operator_reserved_bytes,
                "aggregate_map_reserved_bytes": (
                    self.aggregate_map_reserved_bytes
                ),
                "restriction_schedule_reserved_bytes": (
                    self.restriction_schedule_reserved_bytes
                ),
                "steady_reserved_bytes": self.steady_reserved_bytes,
                "level_operator_reserved_bytes": tuple(
                    level.total_reserved_bytes for level in self._levels
                ),
                "level_pattern_reserved_bytes": tuple(
                    level.pattern_reserved_bytes for level in self._levels
                ),
                "level_value_reserved_bytes": tuple(
                    level.value_reserved_bytes for level in self._levels
                ),
                "level_map_reserved_bytes": tuple(
                    4 * int(mapping.shape[0])
                    for mapping in self._aggregate_maps
                ),
                "level_restriction_schedule_reserved_bytes": tuple(
                    schedule.total_reserved_bytes
                    for schedule in self._restriction_schedules
                ),
            },
            "build": copy.deepcopy(self._build_report),
            "contract": {
                "caller_supplies_ordered_block_aggregate_maps": True,
                "all_levels_and_maps_owned": True,
                "deterministic_block_row_restriction_schedules_owned": True,
                "restriction_schedule_shared_across_block_components": True,
                "exact_sized_level_bsr": True,
                "block_size_preserved_across_levels": True,
                "no_partial_hierarchy_publish": True,
                "device_payload_readback_bytes": 0,
                "coarsening_policy_selected": False,
                "vcycle_or_solver_constructed": False,
                "public_api": False,
            },
        }


class _CallerCoarsenedBsrHierarchyBuilder:
    """Build bounded BSR levels from ``(coarse_block_rows, map)`` specs."""

    def __init__(
        self,
        *,
        explicit_array_capacity_bytes,
        bottom_scalar_size_cap,
    ):
        self._program = get_runtime().prog
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "block hierarchy explicit_array_capacity_bytes",
        )
        self._bottom_scalar_size_cap = _positive_int(
            bottom_scalar_size_cap,
            "block hierarchy bottom_scalar_size_cap",
        )
        self._lock = threading.Lock()
        self._snapshots = weakref.WeakSet()
        self._build_attempts = 0
        self._successful_builds = 0
        self._rejected_builds = 0
        self._failed_builds = 0
        self._host_control_readbacks = 0
        self._final_host_synchronizations = 0
        self._device_to_host_bytes = 0
        self._device_to_device_bytes = 0
        self._device_kernel_publish_bytes = 0
        self._last_status = "not_run"
        self._last_preflight = None
        self._last_success_report = None

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "block sparse hierarchy builder"
        )

    def _parse_specs(self, source, level_specs):
        try:
            specs = list(level_specs)
        except TypeError as exc:
            raise TaichiRuntimeError(
                "block hierarchy level_specs must be an iterable"
            ) from exc
        if not specs:
            raise TaichiRuntimeError(
                "block hierarchy requires at least one coarse level"
            )
        parsed = []
        fine_block_rows = source.block_rows
        for level_index, spec in enumerate(specs, start=1):
            if not isinstance(spec, (tuple, list)) or len(spec) != 2:
                raise TaichiRuntimeError(
                    "each block hierarchy level spec must be "
                    "(coarse_block_rows, map)"
                )
            coarse_block_rows = _positive_int(
                spec[0],
                f"block hierarchy level {level_index} coarse_block_rows",
            )
            if coarse_block_rows >= fine_block_rows:
                raise TaichiRuntimeError(
                    "block hierarchy level sizes must decrease strictly"
                )
            mapping = _require_current_scalar_ndarray(
                spec[1],
                f"block hierarchy level {level_index} aggregate map",
                ti.i32,
                one_dimensional=True,
            )
            if mapping.shape != (fine_block_rows,):
                raise TaichiRuntimeError(
                    f"block hierarchy level {level_index} map must have "
                    f"shape ({fine_block_rows},)"
                )
            parsed.append(
                (fine_block_rows, coarse_block_rows, mapping)
            )
            fine_block_rows = coarse_block_rows
        bottom_scalar_rows = fine_block_rows * source.block_size
        if bottom_scalar_rows > self._bottom_scalar_size_cap:
            raise TaichiRuntimeError(
                "block hierarchy final scalar size exceeds "
                "bottom_scalar_size_cap"
            )
        return parsed

    @staticmethod
    def _plan_staging_bytes(
        capacity, fine_block_rows, coarse_block_rows, block_elements
    ):
        return (
            (28 + 4 * block_elements) * capacity
            + 4 * fine_block_rows
            + 8 * coarse_block_rows
            + 16
        )

    def _preflight(self, source, parsed_specs):
        candidate_exact_upper = source.total_reserved_bytes
        previous_staging_upper = 0
        capacity_upper = source.block_nnz
        block_elements = source.block_size * source.block_size
        peak_upper = candidate_exact_upper
        level_upper = []
        for fine_block_rows, coarse_block_rows, _ in parsed_specs:
            map_bytes = 4 * fine_block_rows
            output_upper = (
                4 * (coarse_block_rows + 1)
                + 4 * capacity_upper * (1 + block_elements)
            )
            galerkin_staging_upper = self._plan_staging_bytes(
                capacity_upper,
                fine_block_rows,
                coarse_block_rows,
                block_elements,
            )
            schedule_bytes = 4 * (
                coarse_block_rows + 1 + fine_block_rows
            )
            schedule_staging_upper = 8 * fine_block_rows
            candidate_exact_upper += map_bytes + output_upper
            galerkin_phase_peak_upper = (
                candidate_exact_upper
                + previous_staging_upper
                + galerkin_staging_upper
            )
            candidate_exact_upper += schedule_bytes
            schedule_phase_peak_upper = (
                candidate_exact_upper
                + galerkin_staging_upper
                + schedule_staging_upper
            )
            step_peak_upper = max(
                galerkin_phase_peak_upper, schedule_phase_peak_upper
            )
            peak_upper = max(peak_upper, step_peak_upper)
            level_upper.append(
                {
                    "fine_block_rows": fine_block_rows,
                    "coarse_block_rows": coarse_block_rows,
                    "fine_scalar_rows": (
                        fine_block_rows * source.block_size
                    ),
                    "coarse_scalar_rows": (
                        coarse_block_rows * source.block_size
                    ),
                    "block_edge_capacity_upper": capacity_upper,
                    "map_bytes": map_bytes,
                    "output_upper_bytes": output_upper,
                    "restriction_schedule_bytes": schedule_bytes,
                    "galerkin_staging_upper_bytes": (
                        galerkin_staging_upper
                    ),
                    "restriction_schedule_staging_upper_bytes": (
                        schedule_staging_upper
                    ),
                    "galerkin_phase_peak_upper_bytes": (
                        galerkin_phase_peak_upper
                    ),
                    "schedule_phase_peak_upper_bytes": (
                        schedule_phase_peak_upper
                    ),
                    "step_peak_upper_bytes": step_peak_upper,
                }
            )
            previous_staging_upper = (
                galerkin_staging_upper + schedule_staging_upper
            )
            # Piecewise-constant block projection emits one item per source
            # block edge and cannot increase the unique block count. Keep the
            # fine capacity as a conservative upper bound until device RLE.
        return {
            "steady_exact_upper_bytes": candidate_exact_upper,
            "build_peak_excluding_workspace_upper_bytes": peak_upper,
            "levels": tuple(level_upper),
        }

    def build(
        self,
        source,
        level_specs,
        *,
        topology_version,
        numeric_version,
    ):
        topology_version = _positive_version(
            topology_version, "block hierarchy topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "block hierarchy numeric_version"
        )
        with self._lock:
            self._ensure_current()
            if not isinstance(source, _DeviceBsrSnapshot):
                raise TaichiRuntimeError(
                    "block hierarchy source must be an owned "
                    "_DeviceBsrSnapshot"
                )
            source._ensure_current()
            if source._program is not self._program:
                raise TaichiRuntimeError(
                    "block hierarchy source and builder must belong to one "
                    "Program"
                )
            if source.block_rows != source.block_cols:
                raise TaichiRuntimeError(
                    "block hierarchy source must be square"
                )
            if (
                topology_version != source.topology_version
                or numeric_version != source.numeric_version
            ):
                raise TaichiRuntimeError(
                    "block hierarchy versions must match the owned fine "
                    "snapshot"
                )
            parsed_specs = self._parse_specs(source, level_specs)
            preflight = self._preflight(source, parsed_specs)
            self._build_attempts += 1
            self._last_preflight = copy.deepcopy(preflight)
            if (
                preflight[
                    "build_peak_excluding_workspace_upper_bytes"
                ]
                > self._capacity_bytes
            ):
                self._rejected_builds += 1
                self._last_status = "capacity_overflow"
                raise TaichiRuntimeError(
                    "block hierarchy explicit-array capacity overflow "
                    "before build"
                )

            levels = [source]
            owned_maps = []
            restriction_schedules = []
            plan_refs = []
            schedule_plan_refs = []
            plan_records = []
            previous_plan = None
            previous_schedule_plan = None
            previous_staging_bytes = 0
            previous_known_workspace_bytes = 0
            actual_peak = source.total_reserved_bytes
            staging_overlap_peak = 0
            known_workspace_overlap_peak = 0
            candidate_exact_bytes = source.total_reserved_bytes
            build_d2h = 0
            build_d2d = 0
            device_kernel_publish = 0
            control_readbacks = 0
            current = source
            plan = None
            schedule_plan = None
            plan_accounted = False
            completion_synchronized = False
            try:
                for (
                    fine_block_rows,
                    coarse_block_rows,
                    mapping,
                ) in parsed_specs:
                    plan = None
                    schedule_plan = None
                    plan_accounted = False
                    owned_map = ti.ndarray(
                        ti.i32, shape=fine_block_rows
                    )
                    _copy_i32_prefix(
                        mapping, owned_map, fine_block_rows
                    )
                    map_bytes = 4 * fine_block_rows
                    build_d2d += map_bytes
                    candidate_exact_bytes += map_bytes
                    owned_maps.append(owned_map)

                    plan = _SparseGalerkinBsrAssemblyPlan(
                        fine_block_rows=fine_block_rows,
                        coarse_block_rows=coarse_block_rows,
                        block_size=source.block_size,
                        capacity=current.block_nnz,
                    )
                    plan_refs.append(weakref.ref(plan))
                    output = plan.build(
                        current,
                        owned_map,
                        topology_version=topology_version,
                        numeric_version=numeric_version,
                    )
                    plan_stats = plan.debug_runtime_stats()
                    plan_resources = plan_stats["resources"]
                    plan_transfers = plan_stats["transfers"]
                    current_staging_bytes = int(
                        plan_resources[
                            "persistent_staging_reserved_bytes"
                        ]
                    )
                    current_known_workspace_bytes = int(
                        plan_resources[
                            "known_workspace_reported_bytes"
                        ]
                    )
                    candidate_exact_bytes += output.total_reserved_bytes
                    step_peak = (
                        candidate_exact_bytes
                        + previous_staging_bytes
                        + current_staging_bytes
                    )
                    actual_peak = max(actual_peak, step_peak)
                    staging_overlap_peak = max(
                        staging_overlap_peak,
                        previous_staging_bytes + current_staging_bytes,
                    )
                    known_workspace_overlap_peak = max(
                        known_workspace_overlap_peak,
                        previous_known_workspace_bytes
                        + current_known_workspace_bytes,
                    )
                    build_d2h += int(
                        plan_transfers["device_to_host_bytes"]
                    )
                    build_d2d += int(
                        plan_transfers["device_to_device_bytes"]
                    )
                    control_readbacks += int(
                        plan_stats["operations"][
                            "host_control_readbacks"
                        ]
                    )
                    plan_accounted = True
                    # The current control readback completed all previous
                    # exact copies and schedule reads. Retire that neighboring
                    # staging generation before allocating this schedule.
                    previous_plan = None
                    previous_schedule_plan = None
                    previous_staging_bytes = 0
                    previous_known_workspace_bytes = 0

                    schedule_plan = _AggregateRestrictionSchedulePlan(
                        fine_rows=fine_block_rows,
                        coarse_rows=coarse_block_rows,
                    )
                    schedule_plan_refs.append(weakref.ref(schedule_plan))
                    schedule = schedule_plan.build(owned_map)
                    schedule_stats = schedule_plan.debug_runtime_stats()
                    schedule_resources = schedule_stats["resources"]
                    schedule_staging_bytes = int(
                        schedule_resources[
                            "persistent_staging_reserved_bytes"
                        ]
                    )
                    schedule_known_workspace_bytes = int(
                        schedule_resources[
                            "known_workspace_reported_bytes"
                        ]
                    )
                    schedule_bytes = schedule.total_reserved_bytes
                    candidate_exact_bytes += schedule_bytes
                    schedule_phase_peak = (
                        candidate_exact_bytes
                        + current_staging_bytes
                        + schedule_staging_bytes
                    )
                    actual_peak = max(actual_peak, schedule_phase_peak)
                    staging_overlap_peak = max(
                        staging_overlap_peak,
                        current_staging_bytes + schedule_staging_bytes,
                    )
                    known_workspace_overlap_peak = max(
                        known_workspace_overlap_peak,
                        current_known_workspace_bytes
                        + schedule_known_workspace_bytes,
                    )
                    device_kernel_publish += schedule_bytes
                    restriction_schedules.append(schedule)
                    plan_records.append(
                        {
                            "fine_block_rows": fine_block_rows,
                            "coarse_block_rows": coarse_block_rows,
                            "source_block_nnz": current.block_nnz,
                            "output_block_nnz": output.block_nnz,
                            "map_bytes": map_bytes,
                            "galerkin_staging_bytes": (
                                current_staging_bytes
                            ),
                            "restriction_schedule_staging_bytes": (
                                schedule_staging_bytes
                            ),
                            "known_workspace_bytes": (
                                current_known_workspace_bytes
                                + schedule_known_workspace_bytes
                            ),
                            "output_pattern_bytes": (
                                output.pattern_reserved_bytes
                            ),
                            "output_value_bytes": (
                                output.value_reserved_bytes
                            ),
                            "output_bytes": output.total_reserved_bytes,
                            "restriction_schedule_bytes": schedule_bytes,
                            "stable_key_sort_passes": 1,
                        }
                    )
                    levels.append(output)
                    current = output

                    # Current Galerkin and schedule staging remain until the
                    # next level's control readback or final completion sync.
                    previous_plan = plan
                    previous_schedule_plan = schedule_plan
                    previous_staging_bytes = (
                        current_staging_bytes + schedule_staging_bytes
                    )
                    previous_known_workspace_bytes = (
                        current_known_workspace_bytes
                        + schedule_known_workspace_bytes
                    )

                ti.sync()
                completion_synchronized = True
                self._final_host_synchronizations += 1
                previous_plan = None
                previous_schedule_plan = None
                plan = None
                schedule_plan = None
                all_plan_refs = plan_refs + schedule_plan_refs
                if any(reference() is not None for reference in all_plan_refs):
                    raise TaichiRuntimeError(
                        "completed block hierarchy staging did not retire"
                    )
                if actual_peak > self._capacity_bytes:
                    raise TaichiRuntimeError(
                        "block hierarchy actual explicit-array peak exceeded "
                        "capacity"
                    )
                report = {
                    "preflight": copy.deepcopy(preflight),
                    "levels": tuple(plan_records),
                    "steady_exact_bytes": candidate_exact_bytes,
                    "actual_build_peak_excluding_workspace_bytes": (
                        actual_peak
                    ),
                    "staging_overlap_peak_bytes": staging_overlap_peak,
                    "known_workspace_overlap_peak_bytes": (
                        known_workspace_overlap_peak
                    ),
                    "workspace_total_bytes_reported": False,
                    "control_readbacks": control_readbacks,
                    "final_completion_synchronizations": 1,
                    "device_to_host_bytes": build_d2h,
                    "device_to_device_bytes": build_d2d,
                    "device_kernel_publish_bytes": device_kernel_publish,
                    "device_payload_readback_bytes": 0,
                    "retired_plan_count": len(all_plan_refs),
                    "live_plan_count_after_publish": 0,
                }
                snapshot = _SparseBsrHierarchySnapshot(
                    program=self._program,
                    backend=source._backend,
                    topology_version=topology_version,
                    numeric_version=numeric_version,
                    bottom_scalar_size_cap=(
                        self._bottom_scalar_size_cap
                    ),
                    levels=levels,
                    aggregate_maps=owned_maps,
                    restriction_schedules=restriction_schedules,
                    build_report=report,
                )
                self._snapshots.add(snapshot)
                self._successful_builds += 1
                self._host_control_readbacks += control_readbacks
                self._device_to_host_bytes += build_d2h
                self._device_to_device_bytes += build_d2d
                self._device_kernel_publish_bytes += device_kernel_publish
                self._last_success_report = copy.deepcopy(report)
                self._last_status = "published"
                return snapshot
            except Exception:
                # A failed later level may follow asynchronous exact copies
                # from an earlier level. Synchronize once before local staging
                # owners unwind; no partial hierarchy is ever returned.
                if not completion_synchronized:
                    ti.sync()
                    completion_synchronized = True
                    self._final_host_synchronizations += 1
                if plan is not None and not plan_accounted:
                    failed_plan_stats = plan.debug_runtime_stats()
                    failed_plan_transfers = failed_plan_stats["transfers"]
                    control_readbacks += int(
                        failed_plan_stats["operations"][
                            "host_control_readbacks"
                        ]
                    )
                    build_d2h += int(
                        failed_plan_transfers["device_to_host_bytes"]
                    )
                    build_d2d += int(
                        failed_plan_transfers["device_to_device_bytes"]
                    )
                    plan_accounted = True
                previous_plan = None
                previous_schedule_plan = None
                plan = None
                schedule_plan = None
                self._failed_builds += 1
                self._host_control_readbacks += control_readbacks
                self._device_to_host_bytes += build_d2h
                self._device_to_device_bytes += build_d2d
                self._device_kernel_publish_bytes += device_kernel_publish
                self._last_status = "build_failed"
                raise

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            live_snapshots = list(self._snapshots)
            return {
                "schema_version": 1,
                "identity": {
                    "explicit_array_capacity_bytes": self._capacity_bytes,
                    "bottom_scalar_size_cap": (
                        self._bottom_scalar_size_cap
                    ),
                    "last_status": self._last_status,
                },
                "operations": {
                    "build_attempts": self._build_attempts,
                    "successful_builds": self._successful_builds,
                    "rejected_builds": self._rejected_builds,
                    "failed_builds": self._failed_builds,
                    "host_control_readbacks": (
                        self._host_control_readbacks
                    ),
                    "final_host_synchronizations": (
                        self._final_host_synchronizations
                    ),
                },
                "resources": {
                    "last_preflight": copy.deepcopy(self._last_preflight),
                    "last_success_report": copy.deepcopy(
                        self._last_success_report
                    ),
                    "live_snapshot_count": len(live_snapshots),
                    "live_snapshot_reserved_bytes": sum(
                        snapshot.steady_reserved_bytes
                        for snapshot in live_snapshots
                    ),
                },
                "transfers": {
                    "device_to_host_bytes": self._device_to_host_bytes,
                    "device_to_device_bytes": (
                        self._device_to_device_bytes
                    ),
                    "device_kernel_publish_bytes": (
                        self._device_kernel_publish_bytes
                    ),
                    "device_payload_readback_bytes": 0,
                },
                "contract": {
                    "capacity_checked_before_first_plan": True,
                    "capacity_covers_explicit_arrays_only": True,
                    "workspace_total_bytes_reported": False,
                    "at_most_two_neighbor_staging_generations": True,
                    "one_key_sort_per_level_independent_of_block_size": True,
                    "deterministic_block_restriction_without_float_atomics": True,
                    "restriction_schedule_shared_across_block_components": True,
                    "one_final_completion_sync": True,
                    "failed_build_publishes_no_partial_hierarchy": True,
                    "bottom_cap_counts_scalar_rows": True,
                    "coarsening_policy_selected": False,
                    "public_api": False,
                },
            }
