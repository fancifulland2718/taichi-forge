"""Read-only metadata for compiled offloaded tasks."""

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class OffloadedTaskManifest:
    """One backend task emitted for a specialized Taichi kernel.

    Geometry fields are split into compiler requests, backend selections, and
    values proven for an ordinary direct launch. ``None`` is intentional: for
    example, CPU execution uses a worker scheduler and is not reported as a
    fictional GPU grid.

    This object is observational. It cannot override launch geometry and
    querying it does not enqueue work or allocate device telemetry storage.
    """

    task_id: str
    logical_task_id: str
    optimization_spec_id: str
    task_name: str
    backend: str
    task_index: int
    task_type: str
    requested_grid_size: Optional[int]
    requested_block_size: Optional[int]
    source_block_size_explicit: bool
    requested_thread_local_mode: str
    requested_cuda_min_blocks_per_sm: int
    requested_cuda_max_registers: Optional[int]
    requested_grid_residency_waves: Optional[int]
    requested_range_work_per_thread_target: int
    requested_memory_strategy: str
    selected_grid_size: Optional[int]
    selected_block_size: Optional[int]
    actual_grid_size: Optional[int]
    actual_block_size: Optional[int]
    actual_geometry_kind: str
    actual_geometry_reason: str
    range_mapping: str
    constant_range_size: Optional[int]
    staged_external_arg_index: Optional[int]
    staged_halo_low: Optional[int]
    staged_halo_high: Optional[int]
    static_shared_bytes: int
    dynamic_shared_bytes: int
    thread_local_bytes: int
    staged_external_arg_indices: tuple
    staged_halo_lows: tuple
    staged_halo_highs: tuple
    staged_byte_offsets: tuple
    staged_element_bytes: tuple
    staged_scalar_bytes: tuple
    staged_element_shapes: tuple
    staged_iteration_shape: tuple
    staged_iteration_origin: tuple
    staged_tile_shape: tuple
    staged_halo_lows_nd: tuple
    staged_halo_highs_nd: tuple
    staged_access_offsets: tuple

    @classmethod
    def _from_core(cls, value: Mapping[str, object]):
        fields = cls.__dataclass_fields__
        payload = {name: value[name] for name in fields if name in value}
        for name in (
            "staged_external_arg_indices",
            "staged_halo_lows",
            "staged_halo_highs",
            "staged_byte_offsets",
            "staged_element_bytes",
            "staged_scalar_bytes",
            "staged_iteration_shape",
            "staged_iteration_origin",
            "staged_tile_shape",
        ):
            payload[name] = tuple(payload.get(name, ()))
        payload["staged_element_shapes"] = tuple(
            tuple(shape) for shape in payload.get("staged_element_shapes", ())
        )
        for name in ("staged_halo_lows_nd", "staged_halo_highs_nd"):
            payload[name] = tuple(tuple(axis) for axis in payload.get(name, ()))
        payload["staged_access_offsets"] = tuple(
            tuple(tuple(offset) for offset in source)
            for source in payload.get("staged_access_offsets", ())
        )
        return cls(**payload)


@dataclass(frozen=True)
class GraphTaskManifest(OffloadedTaskManifest):
    """One physical Graph dispatch task and its invocation label.

    ``dispatch_index`` identifies the physical dispatch after safe Graph
    composition. A non-empty label keeps its dispatch uncomposed so profiler
    events remain one-to-one with the labeled invocation.
    """

    dispatch_index: int
    kernel_name: str
    dispatch_label: str
    indirect: bool
    source_dispatch_count: int


__all__ = ["GraphTaskManifest", "OffloadedTaskManifest"]
