"""Private caller-provided varying-block BSR hierarchy generations.

The hierarchy owns immutable BSR level snapshots and rectangular block-CSR
transfers whose fine/coarse block sizes may differ. It composes the existing
exact rectangular ``P^T A P`` builder without choosing candidates,
aggregation, smoothing, or a solver policy.
"""

import copy
import gc
import threading
import weakref

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime

from ._sparse_block_transfer_graph import _DeviceBlockTransferSnapshot
from ._sparse_bsr_graph_operator import _DeviceBsrSnapshot
from ._sparse_hierarchy_assembly import (
    _backend_methods,
    _ensure_current_program,
    _positive_int,
    _positive_version,
)
from ._sparse_rectangular_galerkin_bsr import (
    _SparseRectangularGalerkinBsrBuilder,
)
from ._sparse_runtime_memory import (
    SORT_SCAN_WORKSPACE_FAMILIES,
    _program_workspace_attribution,
    _workspace_family_reserved_bytes,
)


class _VaryingBsrHierarchyLevelSpec:
    """One caller-qualified rectangular transfer and coarse-level identity."""

    def __init__(
        self,
        *,
        transfer,
        contribution_count,
        max_transfer_blocks_per_row,
        coarse_topology_version,
        coarse_numeric_version,
    ):
        if not isinstance(transfer, _DeviceBlockTransferSnapshot):
            raise TaichiRuntimeError(
                "varying BSR hierarchy spec requires an owned transfer snapshot"
            )
        transfer._ensure_current()
        self.transfer = transfer
        self.contribution_count = _positive_int(
            contribution_count,
            "varying BSR hierarchy contribution_count",
        )
        self.max_transfer_blocks_per_row = _positive_int(
            max_transfer_blocks_per_row,
            "varying BSR hierarchy max_transfer_blocks_per_row",
        )
        self.coarse_topology_version = _positive_version(
            coarse_topology_version,
            "varying BSR hierarchy coarse_topology_version",
        )
        self.coarse_numeric_version = _positive_version(
            coarse_numeric_version,
            "varying BSR hierarchy coarse_numeric_version",
        )


class _VaryingBsrHierarchySnapshot:
    """Self-contained immutable BSR levels and rectangular transfers."""

    def __init__(
        self,
        *,
        program,
        backend,
        levels,
        transfers,
        contribution_counts,
        max_transfer_row_degrees,
        topology_version,
        numeric_version,
        build_stats,
    ):
        self._program = program
        self._backend = backend
        self._levels = tuple(levels)
        self._transfers = tuple(transfers)
        self._contribution_counts = tuple(
            int(value) for value in contribution_counts
        )
        self._max_transfer_row_degrees = tuple(
            int(value) for value in max_transfer_row_degrees
        )
        self.topology_version = int(topology_version)
        self.numeric_version = int(numeric_version)
        self._build_stats = copy.deepcopy(build_stats)
        if len(self._levels) != len(self._transfers) + 1:
            raise TaichiRuntimeError(
                "varying BSR hierarchy requires one more level than transfer"
            )
        if len(self._contribution_counts) != len(self._transfers):
            raise TaichiRuntimeError(
                "varying BSR hierarchy contribution metadata is incomplete"
            )
        if len(self._max_transfer_row_degrees) != len(self._transfers):
            raise TaichiRuntimeError(
                "varying BSR hierarchy degree metadata is incomplete"
            )
        for index, transfer in enumerate(self._transfers):
            fine = self._levels[index]
            coarse = self._levels[index + 1]
            if transfer._program is not self._program:
                raise TaichiRuntimeError(
                    "varying BSR hierarchy transfer crossed Program"
                )
            if (
                transfer.fine_block_rows != fine.block_rows
                or transfer.fine_block_size != fine.block_size
                or transfer.coarse_block_rows != coarse.block_rows
                or transfer.coarse_block_size != coarse.block_size
            ):
                raise TaichiRuntimeError(
                    "varying BSR hierarchy level/transfer geometry mismatch"
                )

    def _ensure_current(self):
        _ensure_current_program(self._program, "varying BSR hierarchy")

    @property
    def level_count(self):
        return len(self._levels)

    @property
    def transition_count(self):
        return len(self._transfers)

    @property
    def operator_reserved_bytes(self):
        return sum(level.total_reserved_bytes for level in self._levels)

    @property
    def transfer_reserved_bytes(self):
        return sum(
            transfer.total_reserved_bytes for transfer in self._transfers
        )

    @property
    def steady_reserved_bytes(self):
        return self.operator_reserved_bytes + self.transfer_reserved_bytes

    def debug_runtime_stats(self):
        self._ensure_current()
        program_workspace = _program_workspace_attribution(self._program)
        shared_sort_scan_bytes = _workspace_family_reserved_bytes(
            program_workspace["groups"], SORT_SCAN_WORKSPACE_FAMILIES
        )
        return {
            "schema_version": 1,
            "identity": {
                "backend_family": self._backend,
                "level_count": self.level_count,
                "transition_count": self.transition_count,
                "level_block_rows": tuple(
                    level.block_rows for level in self._levels
                ),
                "level_block_sizes": tuple(
                    level.block_size for level in self._levels
                ),
                "level_block_nnz": tuple(
                    level.block_nnz for level in self._levels
                ),
                "level_scalar_rows": tuple(
                    level.rows for level in self._levels
                ),
                "transfer_block_nnz": tuple(
                    transfer.block_nnz for transfer in self._transfers
                ),
                "contribution_counts": self._contribution_counts,
                "max_transfer_row_degrees": (
                    self._max_transfer_row_degrees
                ),
                "topology_version": self.topology_version,
                "numeric_version": self.numeric_version,
            },
            "resources": {
                "level_operator_reserved_bytes": tuple(
                    level.total_reserved_bytes for level in self._levels
                ),
                "transfer_reserved_bytes_by_level": tuple(
                    transfer.total_reserved_bytes
                    for transfer in self._transfers
                ),
                "operator_reserved_bytes": self.operator_reserved_bytes,
                "transfer_reserved_bytes": self.transfer_reserved_bytes,
                "steady_reserved_bytes": self.steady_reserved_bytes,
                "preflight_steady_upper_bytes": self._build_stats[
                    "preflight_steady_upper_bytes"
                ],
                "preflight_build_peak_upper_bytes": self._build_stats[
                    "preflight_build_peak_upper_bytes"
                ],
                "actual_build_peak_excluding_workspace_bytes": (
                    self._build_stats[
                        "actual_build_peak_excluding_workspace_bytes"
                    ]
                ),
                "peak_retired_builder_staging_bytes": self._build_stats[
                    "peak_retired_builder_staging_bytes"
                ],
                "shared_sort_scan_workspace_bytes": shared_sort_scan_bytes,
                "shared_sort_scan_workspace_ownership_scope": (
                    "program_ordering_ordering_aux_scan_arena"
                    if program_workspace["available"]
                    else None
                ),
            },
            "transfers": {
                "device_to_host_bytes": self._build_stats[
                    "device_to_host_bytes"
                ],
                "device_to_device_bytes": 0,
                "device_kernel_publish_bytes": self._build_stats[
                    "device_kernel_publish_bytes"
                ],
                "device_payload_readback_bytes": 0,
            },
            "per_transition": copy.deepcopy(
                self._build_stats["per_transition"]
            ),
            "contract": {
                "source_and_transfers_are_owned_immutable_snapshots": True,
                "fine_and_coarse_block_sizes_may_differ": True,
                "all_levels_are_exact_owned_bsr_snapshots": True,
                "rectangular_transfers_replace_implicit_aggregate_maps": True,
                "failed_candidate_never_publishes_partial_hierarchy": True,
                "builder_staging_retired_before_publish": True,
                "shared_sort_scan_workspace_current_bytes_reported": (
                    program_workspace["available"]
                ),
                "shared_sort_scan_workspace_in_explicit_capacity": False,
                "workspace_total_bytes_reported": False,
                "candidate_modes_or_coarsening_selected": False,
                "transfer_graph_plans_constructed": False,
                "recursive_vcycle_constructed": False,
                "public_api": False,
            },
        }


class _CallerProvidedVaryingBsrHierarchyBuilder:
    """Builds one immutable varying-block hierarchy transactionally."""

    def __init__(
        self,
        *,
        explicit_array_capacity_bytes,
        bottom_scalar_size_cap,
    ):
        self._program = get_runtime().prog
        self._backend, _, _ = _backend_methods(
            self._program, "varying BSR hierarchy"
        )
        self._capacity_bytes = _positive_int(
            explicit_array_capacity_bytes,
            "varying BSR hierarchy explicit_array_capacity_bytes",
        )
        self._bottom_scalar_size_cap = _positive_int(
            bottom_scalar_size_cap,
            "varying BSR hierarchy bottom_scalar_size_cap",
        )
        self._lock = threading.Lock()
        self._snapshots = weakref.WeakSet()
        self._build_calls = 0
        self._successful_builds = 0
        self._failed_builds = 0
        self._last_preflight_steady_upper_bytes = 0
        self._last_preflight_build_peak_upper_bytes = 0
        self._last_actual_build_peak_bytes = 0
        self._last_steady_bytes = 0
        self._last_peak_staging_bytes = 0
        self._last_retired_builder_count = 0
        self._last_device_to_host_bytes = 0
        self._last_device_kernel_publish_bytes = 0
        self._last_per_transition = ()

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "varying BSR hierarchy builder"
        )

    def _parse_specs(self, source, level_specs):
        if not isinstance(source, _DeviceBsrSnapshot):
            raise TaichiRuntimeError(
                "varying BSR hierarchy source must be an owned BSR snapshot"
            )
        source._ensure_current()
        if source._program is not self._program:
            raise TaichiRuntimeError(
                "varying BSR hierarchy source and builder crossed Program"
            )
        specs = tuple(level_specs)
        if not specs:
            raise TaichiRuntimeError(
                "varying BSR hierarchy requires at least one transition"
            )
        seen_transfers = set()
        expected_rows = source.block_rows
        expected_block_size = source.block_size
        for index, spec in enumerate(specs):
            if not isinstance(spec, _VaryingBsrHierarchyLevelSpec):
                raise TaichiRuntimeError(
                    "varying BSR hierarchy level specs must be typed"
                )
            transfer = spec.transfer
            transfer._ensure_current()
            if transfer._program is not self._program:
                raise TaichiRuntimeError(
                    "varying BSR hierarchy transfer crossed Program"
                )
            if id(transfer) in seen_transfers:
                raise TaichiRuntimeError(
                    "varying BSR hierarchy cannot reuse one transfer snapshot"
                )
            seen_transfers.add(id(transfer))
            if (
                transfer.fine_block_rows != expected_rows
                or transfer.fine_block_size != expected_block_size
            ):
                raise TaichiRuntimeError(
                    f"varying BSR hierarchy transition {index} does not chain"
                )
            expected_rows = transfer.coarse_block_rows
            expected_block_size = transfer.coarse_block_size
        bottom_scalar_rows = expected_rows * expected_block_size
        if bottom_scalar_rows > self._bottom_scalar_size_cap:
            raise TaichiRuntimeError(
                "varying BSR hierarchy bottom scalar size exceeds cap"
            )
        return specs

    def _preflight(self, source, specs):
        all_transfer_bytes = sum(
            spec.transfer.total_reserved_bytes for spec in specs
        )
        level_bytes_upper = source.total_reserved_bytes
        current_block_nnz_upper = source.block_nnz
        build_peak_upper = all_transfer_bytes + level_bytes_upper
        per_transition = []
        for index, spec in enumerate(specs):
            transfer = spec.transfer
            unique_upper = min(
                spec.contribution_count,
                transfer.coarse_block_rows * transfer.coarse_block_rows,
            )
            output_value_count_upper = (
                unique_upper
                * transfer.coarse_block_size
                * transfer.coarse_block_size
            )
            if output_value_count_upper >= 0x7FFFFFFF:
                raise TaichiRuntimeError(
                    "varying BSR hierarchy output upper exceeds i32"
                )
            output_upper = 4 * (
                transfer.coarse_block_rows + 1 + unique_upper
            ) + 4 * output_value_count_upper
            staging_upper = (
                8 * current_block_nnz_upper
                + 24 * spec.contribution_count
                + 16
            )
            phase_peak_upper = (
                all_transfer_bytes
                + level_bytes_upper
                + staging_upper
                + output_upper
            )
            build_peak_upper = max(build_peak_upper, phase_peak_upper)
            per_transition.append(
                {
                    "transition_index": index,
                    "fine_block_rows": transfer.fine_block_rows,
                    "coarse_block_rows": transfer.coarse_block_rows,
                    "fine_block_size": transfer.fine_block_size,
                    "coarse_block_size": transfer.coarse_block_size,
                    "contribution_count": spec.contribution_count,
                    "max_transfer_blocks_per_row": (
                        spec.max_transfer_blocks_per_row
                    ),
                    "unique_block_nnz_upper": unique_upper,
                    "builder_staging_upper_bytes": staging_upper,
                    "output_upper_bytes": output_upper,
                    "phase_peak_upper_bytes": phase_peak_upper,
                }
            )
            level_bytes_upper += output_upper
            current_block_nnz_upper = unique_upper
        steady_upper = all_transfer_bytes + level_bytes_upper
        if build_peak_upper > self._capacity_bytes:
            raise TaichiRuntimeError(
                "varying BSR hierarchy capacity overflow during preflight"
            )
        return {
            "all_transfer_bytes": all_transfer_bytes,
            "steady_upper_bytes": steady_upper,
            "build_peak_upper_bytes": build_peak_upper,
            "per_transition": tuple(per_transition),
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
            topology_version, "varying BSR hierarchy topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "varying BSR hierarchy numeric_version"
        )
        with self._lock:
            self._ensure_current()
            self._build_calls += 1
            self._last_preflight_steady_upper_bytes = 0
            self._last_preflight_build_peak_upper_bytes = 0
            self._last_actual_build_peak_bytes = 0
            self._last_steady_bytes = 0
            self._last_peak_staging_bytes = 0
            self._last_retired_builder_count = 0
            self._last_device_to_host_bytes = 0
            self._last_device_kernel_publish_bytes = 0
            self._last_per_transition = ()
            try:
                specs = self._parse_specs(source, level_specs)
                preflight = self._preflight(source, specs)
                self._last_preflight_steady_upper_bytes = preflight[
                    "steady_upper_bytes"
                ]
                self._last_preflight_build_peak_upper_bytes = preflight[
                    "build_peak_upper_bytes"
                ]
                levels = [source]
                transfers = [spec.transfer for spec in specs]
                all_transfer_bytes = preflight["all_transfer_bytes"]
                actual_peak = all_transfer_bytes + source.total_reserved_bytes
                per_transition = []
                retired_refs = []

                for index, spec in enumerate(specs):
                    builder = _SparseRectangularGalerkinBsrBuilder(
                        explicit_array_capacity_bytes=self._capacity_bytes
                    )
                    coarse = builder.build(
                        levels[-1],
                        spec.transfer,
                        contribution_count=spec.contribution_count,
                        max_transfer_blocks_per_row=(
                            spec.max_transfer_blocks_per_row
                        ),
                        topology_version=spec.coarse_topology_version,
                        numeric_version=spec.coarse_numeric_version,
                    )
                    builder_stats = builder.debug_runtime_stats()
                    resources = builder_stats["resources"]
                    staging_bytes = int(
                        resources["retired_builder_staging_reserved_bytes"]
                    )
                    output_bytes = coarse.total_reserved_bytes
                    retained_level_bytes = sum(
                        level.total_reserved_bytes for level in levels
                    )
                    phase_peak = (
                        all_transfer_bytes
                        + retained_level_bytes
                        + staging_bytes
                        + output_bytes
                    )
                    actual_peak = max(actual_peak, phase_peak)
                    self._last_peak_staging_bytes = max(
                        self._last_peak_staging_bytes, staging_bytes
                    )
                    self._last_device_to_host_bytes += int(
                        builder_stats["transfers"]["device_to_host_bytes"]
                    )
                    self._last_device_kernel_publish_bytes += int(
                        builder_stats["transfers"][
                            "device_kernel_publish_bytes"
                        ]
                    )
                    per_transition.append(
                        {
                            "transition_index": index,
                            "fine_block_rows": spec.transfer.fine_block_rows,
                            "coarse_block_rows": (
                                spec.transfer.coarse_block_rows
                            ),
                            "fine_block_size": spec.transfer.fine_block_size,
                            "coarse_block_size": (
                                spec.transfer.coarse_block_size
                            ),
                            "contribution_count": spec.contribution_count,
                            "max_transfer_blocks_per_row": (
                                spec.max_transfer_blocks_per_row
                            ),
                            "unique_block_nnz": coarse.block_nnz,
                            "transfer_reserved_bytes": (
                                spec.transfer.total_reserved_bytes
                            ),
                            "builder_staging_bytes": staging_bytes,
                            "output_reserved_bytes": output_bytes,
                            "phase_peak_excluding_workspace_bytes": phase_peak,
                            "device_to_host_bytes": builder_stats["transfers"][
                                "device_to_host_bytes"
                            ],
                        }
                    )
                    levels.append(coarse)
                    retired_refs.append(weakref.ref(builder))
                    del builder
                    gc.collect()
                    self._last_actual_build_peak_bytes = actual_peak
                    self._last_steady_bytes = all_transfer_bytes + sum(
                        level.total_reserved_bytes for level in levels
                    )
                    self._last_retired_builder_count = len(retired_refs)
                    self._last_per_transition = tuple(per_transition)

                if any(reference() is not None for reference in retired_refs):
                    raise TaichiRuntimeError(
                        "varying BSR hierarchy builder staging did not retire"
                    )
                steady_bytes = all_transfer_bytes + sum(
                    level.total_reserved_bytes for level in levels
                )
                self._last_actual_build_peak_bytes = actual_peak
                self._last_steady_bytes = steady_bytes
                self._last_retired_builder_count = len(retired_refs)
                self._last_per_transition = tuple(per_transition)
                build_stats = {
                    "preflight_steady_upper_bytes": preflight[
                        "steady_upper_bytes"
                    ],
                    "preflight_build_peak_upper_bytes": preflight[
                        "build_peak_upper_bytes"
                    ],
                    "actual_build_peak_excluding_workspace_bytes": actual_peak,
                    "peak_retired_builder_staging_bytes": (
                        self._last_peak_staging_bytes
                    ),
                    "device_to_host_bytes": self._last_device_to_host_bytes,
                    "device_kernel_publish_bytes": (
                        self._last_device_kernel_publish_bytes
                    ),
                    "per_transition": tuple(per_transition),
                }
                snapshot = _VaryingBsrHierarchySnapshot(
                    program=self._program,
                    backend=self._backend,
                    levels=levels,
                    transfers=transfers,
                    contribution_counts=(
                        spec.contribution_count for spec in specs
                    ),
                    max_transfer_row_degrees=(
                        spec.max_transfer_blocks_per_row for spec in specs
                    ),
                    topology_version=topology_version,
                    numeric_version=numeric_version,
                    build_stats=build_stats,
                )
                self._snapshots.add(snapshot)
                self._successful_builds += 1
                return snapshot
            except Exception:
                self._failed_builds += 1
                raise

    def debug_runtime_stats(self):
        with self._lock:
            self._ensure_current()
            live_snapshots = list(self._snapshots)
            program_workspace = _program_workspace_attribution(self._program)
            shared_sort_scan_bytes = _workspace_family_reserved_bytes(
                program_workspace["groups"], SORT_SCAN_WORKSPACE_FAMILIES
            )
            return {
                "schema_version": 1,
                "identity": {
                    "backend_family": self._backend,
                    "method": "caller_provided_varying_block_galerkin_chain",
                    "bottom_scalar_size_cap": self._bottom_scalar_size_cap,
                },
                "operations": {
                    "build_calls": self._build_calls,
                    "successful_builds": self._successful_builds,
                    "failed_builds": self._failed_builds,
                    "last_retired_transition_builder_count": (
                        self._last_retired_builder_count
                    ),
                },
                "resources": {
                    "explicit_array_capacity_bytes": self._capacity_bytes,
                    "last_preflight_steady_upper_bytes": (
                        self._last_preflight_steady_upper_bytes
                    ),
                    "last_preflight_build_peak_upper_bytes": (
                        self._last_preflight_build_peak_upper_bytes
                    ),
                    "last_actual_build_peak_excluding_workspace_bytes": (
                        self._last_actual_build_peak_bytes
                    ),
                    "last_steady_reserved_bytes": self._last_steady_bytes,
                    "last_peak_retired_builder_staging_bytes": (
                        self._last_peak_staging_bytes
                    ),
                    "live_snapshot_count": len(live_snapshots),
                    "live_snapshot_reserved_bytes": sum(
                        snapshot.steady_reserved_bytes
                        for snapshot in live_snapshots
                    ),
                    "shared_sort_scan_workspace_bytes": shared_sort_scan_bytes,
                    "shared_sort_scan_workspace_ownership_scope": (
                        "program_ordering_ordering_aux_scan_arena"
                        if program_workspace["available"]
                        else None
                    ),
                },
                "transfers": {
                    "device_to_host_bytes": self._last_device_to_host_bytes,
                    "device_to_device_bytes": 0,
                    "device_kernel_publish_bytes": (
                        self._last_device_kernel_publish_bytes
                    ),
                    "device_payload_readback_bytes": 0,
                },
                "last_per_transition": copy.deepcopy(
                    self._last_per_transition
                ),
                "contract": {
                    "preflight_counts_all_transfer_snapshots_live": True,
                    "preflight_uses_block_nnz_upper_per_level": True,
                    "each_transition_uses_exact_rectangular_galerkin": True,
                    "transition_builders_retired_before_publish": True,
                    "failed_candidate_not_published": True,
                    "bottom_cap_counts_scalar_rows": True,
                    "shared_sort_scan_workspace_current_bytes_reported": (
                        program_workspace["available"]
                    ),
                    "shared_sort_scan_workspace_in_explicit_capacity": False,
                    "workspace_total_bytes_reported": False,
                    "candidate_modes_or_coarsening_selected": False,
                    "public_api": False,
                },
            }
