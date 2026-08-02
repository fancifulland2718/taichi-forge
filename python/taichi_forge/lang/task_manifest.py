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
    task_name: str
    backend: str
    task_index: int
    task_type: str
    requested_grid_size: Optional[int]
    requested_block_size: Optional[int]
    selected_grid_size: Optional[int]
    selected_block_size: Optional[int]
    actual_grid_size: Optional[int]
    actual_block_size: Optional[int]
    actual_geometry_kind: str
    actual_geometry_reason: str
    static_shared_bytes: int
    dynamic_shared_bytes: int

    @classmethod
    def _from_core(cls, value: Mapping[str, object]):
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


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
