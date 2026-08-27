"""Internal immutable optimization identity for kernel specializations."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Optional


@dataclass(frozen=True)
class _IrOptimizationOptions:
    thread_local: str = "auto"

    def __post_init__(self):
        if self.thread_local not in ("auto", "on", "off"):
            raise ValueError("thread_local must be 'auto', 'on', or 'off'")


@dataclass(frozen=True)
class _BackendCodegenOptions:
    workgroup_size: Optional[int] = None

    def __post_init__(self):
        value = self.workgroup_size
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("workgroup_size must be an integer")
        if not 1 <= value <= 1024:
            raise ValueError("workgroup_size must be in [1, 1024]")


@dataclass(frozen=True)
class _ArtifactOptions:
    provider_mode: str = "baseline"
    max_registers: Optional[int] = None

    def __post_init__(self):
        if self.provider_mode not in (
            "baseline",
            "apply_explicit_acf",
            "request_tuning",
        ):
            raise ValueError(
                "provider_mode must be 'baseline', 'apply_explicit_acf', or "
                "'request_tuning'"
            )
        value = self.max_registers
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 255
        ):
            raise ValueError("max_registers must be an integer in [1, 255]")


@dataclass(frozen=True)
class _LaunchOptions:
    block_mode: str = "auto"
    grid_residency_waves: Optional[int] = None

    def __post_init__(self):
        if self.block_mode not in ("auto", "hint", "require"):
            raise ValueError("block_mode must be 'auto', 'hint', or 'require'")
        if self.grid_residency_waves not in (None, 1, 2, 4):
            raise ValueError("grid_residency_waves must be None, 1, 2, or 4")


@dataclass(frozen=True)
class _KernelOptimizationSpec:
    """Complete internal identity for a late kernel optimization variant."""

    ir: _IrOptimizationOptions = field(default_factory=_IrOptimizationOptions)
    backend: _BackendCodegenOptions = field(default_factory=_BackendCodegenOptions)
    artifact: _ArtifactOptions = field(default_factory=_ArtifactOptions)
    launch: _LaunchOptions = field(default_factory=_LaunchOptions)

    def __post_init__(self):
        has_block = self.backend.workgroup_size is not None
        if (self.launch.block_mode == "auto") != (not has_block):
            raise ValueError(
                "block_mode='auto' requires no workgroup_size; hint/require "
                "requires one"
            )

    @classmethod
    def from_task_launch_policy(cls, policy):
        if policy.mode == "auto":
            return cls()
        return cls(
            backend=_BackendCodegenOptions(workgroup_size=policy.block_dim),
            launch=_LaunchOptions(block_mode=policy.mode),
        )

    @property
    def is_baseline(self):
        return self == type(self)()

    @property
    def stable_payload(self):
        return json.dumps(
            {"schema_version": 1, **asdict(self)},
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def identity(self):
        if self.is_baseline:
            return ""
        digest = hashlib.sha256(self.stable_payload.encode("utf-8")).hexdigest()
        return f"kos1:{digest}"

    @property
    def specialization_key(self):
        return self.stable_payload


__all__ = [
    "_ArtifactOptions",
    "_BackendCodegenOptions",
    "_IrOptimizationOptions",
    "_KernelOptimizationSpec",
    "_LaunchOptions",
]
