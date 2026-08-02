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
class TaskLaunchReport:
    """Read-only result of resolving a policy for one kernel specialization."""

    policy: TaskLaunchPolicy
    backend: str
    status: str
    reason: str
    tasks: Tuple[OffloadedTaskManifest, ...]


__all__ = ["TaskLaunchPolicy", "TaskLaunchReport"]
