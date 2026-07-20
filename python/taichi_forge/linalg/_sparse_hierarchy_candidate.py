"""Private caller-coarsened sparse hierarchy candidates.

This module composes exact device-visible CSR snapshots without selecting an
aggregation policy or constructing a V-cycle.  It keeps only two neighboring
assembly plans live while building, then publishes one self-contained set of
level CSR arrays and aggregate maps.
"""

import copy
import threading
import weakref

import numpy as np

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime

from ._sparse_hierarchy_assembly import (
    _DeviceCsrSnapshot,
    _SparseGalerkinCsrAssemblyPlan,
    _backend_methods,
    _copy_i32_prefix,
    _ensure_current_program,
    _positive_int,
    _positive_version,
)
from ._sparse_runtime_memory import (
    SCAN_WORKSPACE_FAMILIES,
    _program_workspace_attribution,
    _workspace_family_reserved_bytes,
)
from .sparse_matrix import _require_current_scalar_ndarray


@ti.kernel
def _emit_restriction_schedule_keys(
    aggregate: ti.types.ndarray(dtype=ti.i32, ndim=1),
    keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
    fine_rows: ti.i32,
):
    for fine_row in range(fine_rows):
        keys[fine_row] = (
            ti.cast(aggregate[fine_row], ti.u64) << 32
        ) | ti.cast(fine_row, ti.u64)


@ti.kernel
def _decode_restriction_schedule(
    keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
    coarse_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ordered_fine_rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    fine_rows: ti.i32,
):
    for index in range(fine_rows):
        key = keys[index]
        coarse_row = ti.cast(key >> 32, ti.i32)
        fine_row = ti.cast(key & ti.u64(0xFFFFFFFF), ti.i32)
        ordered_fine_rows[index] = fine_row
        ti.atomic_add(coarse_offsets[coarse_row + 1], 1)


class _AggregateRestrictionSchedule:
    """Exact deterministic transpose schedule for one aggregate map."""

    def __init__(
        self,
        *,
        program,
        backend,
        fine_rows,
        coarse_rows,
        coarse_offsets,
        ordered_fine_rows,
    ):
        self._program = program
        self._backend = backend
        self.fine_rows = int(fine_rows)
        self.coarse_rows = int(coarse_rows)
        self._coarse_offsets = coarse_offsets
        self._ordered_fine_rows = ordered_fine_rows

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "aggregate restriction schedule"
        )

    @property
    def total_reserved_bytes(self):
        return 4 * (self.coarse_rows + 1 + self.fine_rows)

    def debug_runtime_stats(self):
        self._ensure_current()
        return {
            "schema_version": 1,
            "identity": {
                "backend_family": self._backend,
                "fine_rows": self.fine_rows,
                "coarse_rows": self.coarse_rows,
                "ordering": "coarse_row_then_fine_source_ordinal",
            },
            "resources": {
                "coarse_offsets_reserved_bytes": 4
                * (self.coarse_rows + 1),
                "ordered_fine_rows_reserved_bytes": 4 * self.fine_rows,
                "total_reserved_bytes": self.total_reserved_bytes,
            },
            "contract": {
                "floating_atomic_restriction_required": False,
                "deterministic_gather_within_each_aggregate": True,
                "arbitrary_aggregate_map_order_supported": True,
                "public_api": False,
            },
        }


class _AggregateRestrictionSchedulePlan:
    """One-level native-sort plan for deterministic restriction gathers."""

    def __init__(self, *, fine_rows, coarse_rows):
        self.fine_rows = _positive_int(
            fine_rows, "restriction schedule fine_rows"
        )
        self.coarse_rows = _positive_int(
            coarse_rows, "restriction schedule coarse_rows"
        )
        if self.coarse_rows >= self.fine_rows:
            raise TaichiRuntimeError(
                "restriction schedule coarse_rows must be smaller than "
                "fine_rows"
            )
        self._program = get_runtime().prog
        self._backend, self._sort_method, _ = _backend_methods(self._program)
        self._keys = ti.ndarray(ti.u64, shape=self.fine_rows)
        self._sort_workspace = ti.algorithms.SortWorkspace(
            max_items=self.fine_rows
        )
        self._offset_scan = ti.algorithms.PrefixSumExecutor(
            self.coarse_rows + 1
        )
        self._lock = threading.Lock()
        self._build_calls = 0
        self._persistent_staging_reserved_bytes = 8 * self.fine_rows

    def _ensure_current(self):
        _ensure_current_program(self._program, "restriction schedule plan")

    def build(self, aggregate):
        with self._lock:
            self._ensure_current()
            aggregate = _require_current_scalar_ndarray(
                aggregate,
                "restriction schedule aggregate map",
                ti.i32,
                one_dimensional=True,
            )
            if aggregate.shape != (self.fine_rows,):
                raise TaichiRuntimeError(
                    "restriction schedule aggregate map shape does not match"
                )
            coarse_offsets = ti.ndarray(
                ti.i32, shape=self.coarse_rows + 1
            )
            ordered_fine_rows = ti.ndarray(ti.i32, shape=self.fine_rows)
            coarse_offsets.fill(0)
            _emit_restriction_schedule_keys(
                aggregate, self._keys, self.fine_rows
            )
            ti.algorithms.sort(
                self._keys,
                method=self._sort_method,
                workspace=self._sort_workspace,
            )
            _decode_restriction_schedule(
                self._keys,
                coarse_offsets,
                ordered_fine_rows,
                self.fine_rows,
            )
            self._offset_scan.run(coarse_offsets)
            self._build_calls += 1
            return _AggregateRestrictionSchedule(
                program=self._program,
                backend=self._backend,
                fine_rows=self.fine_rows,
                coarse_rows=self.coarse_rows,
                coarse_offsets=coarse_offsets,
                ordered_fine_rows=ordered_fine_rows,
            )

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            program_workspace = _program_workspace_attribution(self._program)
            shared_scan_bytes = _workspace_family_reserved_bytes(
                program_workspace["groups"], SCAN_WORKSPACE_FAMILIES
            )
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": "native_sort_coarse_then_fine_ordinal",
                    "fine_rows": self.fine_rows,
                    "coarse_rows": self.coarse_rows,
                },
                "operations": {"build_calls": self._build_calls},
                "resources": {
                    "persistent_staging_reserved_bytes": (
                        self._persistent_staging_reserved_bytes
                    ),
                    "sort_workspace_reported_bytes": int(
                        self._sort_workspace.workspace_bytes_current
                    ),
                    "shared_scan_workspace_bytes": shared_scan_bytes,
                    "shared_scan_workspace_ownership_scope": (
                        "program_scan_arena"
                        if program_workspace["available"]
                        else None
                    ),
                    "known_workspace_reported_bytes": int(
                        self._sort_workspace.workspace_bytes_current
                    ),
                },
                "transfers": {
                    "device_to_host_bytes": 0,
                    "device_to_device_bytes": 0,
                    "device_kernel_publish_bytes": self._build_calls
                    * 4
                    * (self.coarse_rows + 1 + self.fine_rows),
                },
                "contract": {
                    "native_provider_required_without_host_fallback": True,
                    "stable_source_ordinal_order": True,
                    "workspace_total_bytes_reported": False,
                    "shared_scan_workspace_current_bytes_reported": (
                        program_workspace["available"]
                    ),
                    "shared_scan_workspace_in_plan_owned_bytes": False,
                    "public_api": False,
                },
            }


class _SparseCsrHierarchySnapshot:
    """Self-contained CSR levels and aggregate maps for one generation."""

    def __init__(
        self,
        *,
        program,
        backend,
        topology_version,
        numeric_version,
        bottom_size_cap,
        levels,
        aggregate_maps,
        restriction_schedules,
        build_report,
    ):
        self._program = program
        self._backend = backend
        self.topology_version = int(topology_version)
        self.numeric_version = int(numeric_version)
        self.bottom_size_cap = int(bottom_size_cap)
        self._levels = tuple(levels)
        self._aggregate_maps = tuple(aggregate_maps)
        self._restriction_schedules = tuple(restriction_schedules)
        self._build_report = copy.deepcopy(build_report)
        if len(self._levels) != len(self._aggregate_maps) + 1:
            raise TaichiRuntimeError(
                "hierarchy levels must contain one more entry than maps"
            )
        if len(self._restriction_schedules) != len(self._aggregate_maps):
            raise TaichiRuntimeError(
                "hierarchy requires one restriction schedule per map"
            )

    def _ensure_current(self):
        _ensure_current_program(self._program, "sparse hierarchy snapshot")

    @property
    def level_count(self):
        return len(self._levels)

    @property
    def level_sizes(self):
        return tuple(level.rows for level in self._levels)

    @property
    def level_nnz(self):
        return tuple(level.nnz for level in self._levels)

    @property
    def operator_reserved_bytes(self):
        return sum(level.total_reserved_bytes for level in self._levels)

    @property
    def aggregate_map_reserved_bytes(self):
        return sum(
            4 * int(mapping.shape[0]) for mapping in self._aggregate_maps
        )

    @property
    def steady_reserved_bytes(self):
        return (
            self.operator_reserved_bytes
            + self.aggregate_map_reserved_bytes
            + self.restriction_schedule_reserved_bytes
        )

    @property
    def restriction_schedule_reserved_bytes(self):
        return sum(
            schedule.total_reserved_bytes
            for schedule in self._restriction_schedules
        )

    def debug_runtime_stats(self):
        self._ensure_current()
        return {
            "schema_version": 1,
            "identity": {
                "backend_family": self._backend,
                "storage_format": "recursive_csr",
                "dtype": "f32",
                "index_dtype": "i32",
                "topology_version": self.topology_version,
                "numeric_version": self.numeric_version,
                "level_count": self.level_count,
                "level_sizes": self.level_sizes,
                "level_nnz": self.level_nnz,
                "bottom_size_cap": self.bottom_size_cap,
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
                "caller_supplies_ordered_aggregate_maps": True,
                "all_levels_and_maps_owned": True,
                "deterministic_restriction_schedules_owned": True,
                "exact_sized_level_csr": True,
                "no_partial_hierarchy_publish": True,
                "device_payload_readback_bytes": 0,
                "coarsening_policy_selected": False,
                "vcycle_or_solver_constructed": False,
                "public_api": False,
            },
        }


class _CallerCoarsenedSparseHierarchyBuilder:
    """Build a bounded hierarchy from explicit ``(coarse_rows, map)`` specs."""

    def __init__(self, *, explicit_array_capacity_bytes, bottom_size_cap):
        self._program = get_runtime().prog
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "hierarchy explicit_array_capacity_bytes",
        )
        self._bottom_size_cap = _positive_int(
            bottom_size_cap, "hierarchy bottom_size_cap"
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
        _ensure_current_program(self._program, "sparse hierarchy builder")

    def _parse_specs(self, source, level_specs):
        try:
            specs = list(level_specs)
        except TypeError as exc:
            raise TaichiRuntimeError(
                "hierarchy level_specs must be an iterable"
            ) from exc
        if not specs:
            raise TaichiRuntimeError(
                "hierarchy requires at least one coarse level"
            )
        parsed = []
        fine_rows = source.rows
        for level_index, spec in enumerate(specs, start=1):
            if not isinstance(spec, (tuple, list)) or len(spec) != 2:
                raise TaichiRuntimeError(
                    "each hierarchy level spec must be (coarse_rows, map)"
                )
            coarse_rows = _positive_int(
                spec[0], f"hierarchy level {level_index} coarse_rows"
            )
            if coarse_rows >= fine_rows:
                raise TaichiRuntimeError(
                    "hierarchy level sizes must decrease strictly"
                )
            mapping = _require_current_scalar_ndarray(
                spec[1],
                f"hierarchy level {level_index} aggregate map",
                ti.i32,
                one_dimensional=True,
            )
            if mapping.shape != (fine_rows,):
                raise TaichiRuntimeError(
                    f"hierarchy level {level_index} map must have shape "
                    f"({fine_rows},)"
                )
            parsed.append((fine_rows, coarse_rows, mapping))
            fine_rows = coarse_rows
        if fine_rows > self._bottom_size_cap:
            raise TaichiRuntimeError(
                "hierarchy final level exceeds bottom_size_cap"
            )
        return parsed

    @staticmethod
    def _plan_staging_bytes(capacity, fine_rows, coarse_rows):
        return 32 * capacity + 4 * fine_rows + 8 * coarse_rows + 16

    def _preflight(self, source, parsed_specs):
        candidate_exact_upper = source.total_reserved_bytes
        previous_staging_upper = 0
        capacity_upper = source.nnz
        peak_upper = candidate_exact_upper
        level_upper = []
        for fine_rows, coarse_rows, _ in parsed_specs:
            map_bytes = 4 * fine_rows
            output_upper = 4 * (coarse_rows + 1) + 8 * capacity_upper
            galerkin_staging_upper = self._plan_staging_bytes(
                capacity_upper, fine_rows, coarse_rows
            )
            schedule_bytes = 4 * (coarse_rows + 1 + fine_rows)
            schedule_staging_upper = 8 * fine_rows
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
                    "fine_rows": fine_rows,
                    "coarse_rows": coarse_rows,
                    "capacity_upper": capacity_upper,
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
            # Piecewise-constant projection emits one item per source nnz and
            # cannot increase unique nnz. Keep the same conservative upper
            # bound until a device count is available.
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
            topology_version, "hierarchy topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "hierarchy numeric_version"
        )
        with self._lock:
            self._ensure_current()
            if not isinstance(source, _DeviceCsrSnapshot):
                raise TaichiRuntimeError(
                    "hierarchy source must be an owned _DeviceCsrSnapshot"
                )
            source._ensure_current()
            if source._program is not self._program:
                raise TaichiRuntimeError(
                    "hierarchy source and builder must belong to one Program"
                )
            if source.rows != source.cols:
                raise TaichiRuntimeError("hierarchy source must be square")
            if (
                topology_version != source.topology_version
                or numeric_version != source.numeric_version
            ):
                raise TaichiRuntimeError(
                    "hierarchy versions must match the owned fine snapshot"
                )
            parsed_specs = self._parse_specs(source, level_specs)
            preflight = self._preflight(source, parsed_specs)
            self._build_attempts += 1
            self._last_preflight = copy.deepcopy(preflight)
            if (
                preflight["build_peak_excluding_workspace_upper_bytes"]
                > self._capacity_bytes
            ):
                self._rejected_builds += 1
                self._last_status = "capacity_overflow"
                raise TaichiRuntimeError(
                    "hierarchy explicit-array capacity overflow before build"
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
                for fine_rows, coarse_rows, mapping in parsed_specs:
                    plan = None
                    schedule_plan = None
                    plan_accounted = False
                    owned_map = ti.ndarray(ti.i32, shape=fine_rows)
                    _copy_i32_prefix(mapping, owned_map, fine_rows)
                    map_bytes = 4 * fine_rows
                    build_d2d += map_bytes
                    candidate_exact_bytes += map_bytes
                    owned_maps.append(owned_map)

                    plan = _SparseGalerkinCsrAssemblyPlan(
                        fine_rows=fine_rows,
                        coarse_rows=coarse_rows,
                        capacity=current.nnz,
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
                        plan_resources["persistent_staging_reserved_bytes"]
                    )
                    current_known_workspace_bytes = int(
                        plan_resources["known_workspace_reported_bytes"]
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
                        plan_stats["operations"]["host_control_readbacks"]
                    )
                    plan_accounted = True
                    # This control readback completed the previous level's
                    # exact copy, restriction schedule, and every current read
                    # from them. Retire both previous staging owners before
                    # allocating the current restriction schedule.
                    previous_plan = None
                    previous_schedule_plan = None
                    previous_staging_bytes = 0
                    previous_known_workspace_bytes = 0

                    schedule_plan = _AggregateRestrictionSchedulePlan(
                        fine_rows=fine_rows,
                        coarse_rows=coarse_rows,
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
                        schedule_resources["known_workspace_reported_bytes"]
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
                            "fine_rows": fine_rows,
                            "coarse_rows": coarse_rows,
                            "source_nnz": current.nnz,
                            "output_nnz": output.nnz,
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
                            "output_bytes": output.total_reserved_bytes,
                            "restriction_schedule_bytes": schedule_bytes,
                        }
                    )
                    levels.append(output)
                    current = output

                    # Current Galerkin and schedule staging remain until the
                    # next level's control readback (or final completion sync).
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
                        "completed hierarchy assembly staging did not retire"
                    )
                if actual_peak > self._capacity_bytes:
                    raise TaichiRuntimeError(
                        "hierarchy actual explicit-array peak exceeded "
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
                snapshot = _SparseCsrHierarchySnapshot(
                    program=self._program,
                    backend=source._backend,
                    topology_version=topology_version,
                    numeric_version=numeric_version,
                    bottom_size_cap=self._bottom_size_cap,
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
                # A failed later level may still follow an asynchronous exact
                # copy from an earlier level. One final synchronization is the
                # only safe point before local staging owners unwind.
                if not completion_synchronized:
                    ti.sync()
                    completion_synchronized = True
                    self._final_host_synchronizations += 1
                if plan is not None and not plan_accounted:
                    # A one-level plan records its 8-byte control readback and
                    # aggregate copy before raising validation or reduction
                    # failures. Preserve those real costs in hierarchy-level
                    # telemetry even though no partial hierarchy is published.
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
                    "bottom_size_cap": self._bottom_size_cap,
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
                    "at_most_two_neighbor_staging_plans": True,
                    "deterministic_restriction_without_float_atomics": True,
                    "one_final_completion_sync": True,
                    "failed_build_publishes_no_partial_hierarchy": True,
                    "coarsening_policy_selected": False,
                    "public_api": False,
                },
            }
