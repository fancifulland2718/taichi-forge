"""Constrained launch policy for direct JIT range kernels."""

from dataclasses import dataclass
from typing import Optional, Tuple

from taichi_forge.lang.task_manifest import OffloadedTaskManifest


@dataclass(frozen=True)
class TaskLaunchPolicy:
    """An immutable request for a kernel's physical block size.

    ``hint`` may yield to an explicit ``ti.loop_config(block_dim=...)`` in the
    kernel. ``require`` fails before device submission unless the selected
    CUDA/Vulkan block size exactly matches the request. CPU has no equivalent
    GPU block contract: a hint falls back to ``auto`` and a requirement fails.
    """

    block_dim: Optional[int] = None
    mode: str = "auto"

    def __post_init__(self):
        if self.mode not in ("auto", "hint", "require"):
            raise ValueError(
                "TaskLaunchPolicy mode must be 'auto', 'hint', or 'require'"
            )
        if self.mode == "auto":
            if self.block_dim is not None:
                raise ValueError(
                    "TaskLaunchPolicy mode='auto' does not accept block_dim"
                )
            return
        if isinstance(self.block_dim, bool) or not isinstance(self.block_dim, int):
            raise TypeError("TaskLaunchPolicy block_dim must be an integer")
        if not 1 <= self.block_dim <= 1024:
            raise ValueError("TaskLaunchPolicy block_dim must be in [1, 1024]")
        if self.block_dim % 32 != 0 and self.block_dim & (self.block_dim - 1):
            raise ValueError(
                "TaskLaunchPolicy block_dim must be a power of two or a multiple of 32"
            )

    @classmethod
    def auto(cls):
        """Use the compiler/backend default."""

        return cls()

    @classmethod
    def block(cls, block_dim, *, mode="hint"):
        """Request a block size as a hint or strict requirement."""

        return cls(block_dim=block_dim, mode=mode)

    @property
    def _specialization_key(self):
        return (self.mode, self.block_dim)


@dataclass(frozen=True)
class TaskLaunchCandidateRejection:
    """Why one requested block candidate was not selected."""

    block_dim: int
    reason: str


@dataclass(frozen=True)
class TaskLaunchResourceReport:
    """No-submit resource context for one compiled backend task.

    ``representative_legal_block_sizes`` are geometry-valid probes, not
    performance recommendations.  Register and local-memory information stay
    unavailable when obtaining them would require native module
    materialization or a profiler launch.
    """

    task_id: str
    task_type: str
    observation_kind: str
    selected_block_size: Optional[int]
    static_shared_bytes: int
    dynamic_shared_bytes: int
    max_threads_per_block: Optional[int]
    max_threads_reason: str
    registers_per_thread: Optional[int]
    registers_reason: str
    local_memory_bytes_per_thread: Optional[int]
    local_memory_reason: str
    representative_legal_block_sizes: Tuple[int, ...]
    rejected_candidates: Tuple[TaskLaunchCandidateRejection, ...]
    selection_reason: str


@dataclass(frozen=True)
class TaskLaunchReport:
    """Read-only result of resolving a policy for one kernel specialization."""

    policy: TaskLaunchPolicy
    backend: str
    status: str
    reason: str
    tasks: Tuple[OffloadedTaskManifest, ...]
    resources: Tuple[TaskLaunchResourceReport, ...] = ()


def _task_launch_resource_reports(tasks, policy, status, config):
    reports = []
    canonical_candidates = (32, 64, 128, 256, 512, 1024)
    backend_max = int(config.max_block_dim)
    for task in tasks:
        is_cpu = task.backend in ("x64", "arm64")
        is_range = task.task_type == "range_for"
        max_threads = None
        if not is_cpu and backend_max > 0:
            max_threads = backend_max
            max_threads_reason = "backend device limit from CompileConfig"
        elif is_cpu:
            max_threads_reason = "CPU uses a worker scheduler, not GPU blocks"
        else:
            max_threads_reason = "backend device limit is not exposed to read-only reports"

        candidates = ()
        if is_range and not is_cpu:
            values = set(canonical_candidates)
            if task.selected_block_size is not None:
                values.add(task.selected_block_size)
            if policy.block_dim is not None:
                values.add(policy.block_dim)
            limit = min(1024, max_threads) if max_threads is not None else 1024
            candidates = tuple(sorted(value for value in values if 1 <= value <= limit))

        rejected = ()
        if policy.block_dim is not None and status in (
            "fallback_auto",
            "hint_not_applied",
        ):
            if status == "fallback_auto":
                rejection_reason = "CPU worker scheduling has no GPU block contract"
            else:
                rejection_reason = (
                    "explicit source loop_config or a backend constraint selected "
                    "another block size"
                )
            rejected = (
                TaskLaunchCandidateRejection(
                    block_dim=policy.block_dim, reason=rejection_reason
                ),
            )

        if not is_range:
            selection_reason = "task type is not a tunable parallel range"
        elif is_cpu:
            selection_reason = "CPU runtime worker scheduler"
        elif policy.mode == "auto":
            selection_reason = "compiler/backend default"
        elif status == "applied":
            selection_reason = "requested block selected"
        else:
            selection_reason = "source or backend constraint selected the block"

        if is_cpu:
            registers_reason = "not applicable to CPU worker scheduling"
            local_memory_reason = "not applicable to CPU worker scheduling"
        elif task.backend == "cuda":
            registers_reason = (
                "CUDA register allocation requires a materialized native function; "
                "read-only report did not materialize or launch it"
            )
            local_memory_reason = (
                "CUDA local-memory allocation requires a materialized native function; "
                "read-only report did not materialize or launch it"
            )
        else:
            registers_reason = "SPIR-V register allocation is backend/driver owned"
            local_memory_reason = "SPIR-V local-memory allocation is backend/driver owned"

        reports.append(
            TaskLaunchResourceReport(
                task_id=task.task_id,
                task_type=task.task_type,
                observation_kind="compile_time_no_submit",
                selected_block_size=task.selected_block_size,
                static_shared_bytes=task.static_shared_bytes,
                dynamic_shared_bytes=task.dynamic_shared_bytes,
                max_threads_per_block=max_threads,
                max_threads_reason=max_threads_reason,
                registers_per_thread=None,
                registers_reason=registers_reason,
                local_memory_bytes_per_thread=None,
                local_memory_reason=local_memory_reason,
                representative_legal_block_sizes=candidates,
                rejected_candidates=rejected,
                selection_reason=selection_reason,
            )
        )
    return tuple(reports)


__all__ = [
    "TaskLaunchCandidateRejection",
    "TaskLaunchPolicy",
    "TaskLaunchReport",
    "TaskLaunchResourceReport",
]
