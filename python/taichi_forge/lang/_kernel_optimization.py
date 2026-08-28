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


@dataclass(frozen=True)
class _LaunchOptions:
    block_mode: str = "auto"
    grid_residency_waves: Optional[int] = None
    range_work_per_thread_target: int = 1

    def __post_init__(self):
        if self.block_mode not in ("auto", "hint", "require"):
            raise ValueError("block_mode must be 'auto', 'hint', or 'require'")
        if self.grid_residency_waves not in (None, 1, 2, 4):
            raise ValueError("grid_residency_waves must be None, 1, 2, or 4")
        value = self.range_work_per_thread_target
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("range_work_per_thread_target must be an integer")
        if value not in (1, 2, 4, 8):
            raise ValueError(
                "range_work_per_thread_target must be 1, 2, 4, or 8"
            )


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
    def compilation_payload(self):
        payload = asdict(self)
        # Grid residency is resolved from the materialized CUfunction at
        # launch registration. It must not manufacture a second IR/PTX cache
        # entry for the same block/TLS/artifact variant.
        payload["launch"]["grid_residency_waves"] = None
        payload["launch"]["range_work_per_thread_target"] = 1
        return json.dumps(
            {"schema_version": 1, **payload},
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
    def compilation_identity(self):
        baseline = type(self)(
            ir=self.ir,
            backend=self.backend,
            artifact=self.artifact,
            launch=_LaunchOptions(block_mode=self.launch.block_mode),
        )
        if baseline.is_baseline:
            return ""
        digest = hashlib.sha256(self.compilation_payload.encode("utf-8")).hexdigest()
        return f"kos1:{digest}"

    @property
    def specialization_key(self):
        return self.stable_payload

    @property
    def compilation_specialization_key(self):
        return self.compilation_payload


def _bind_kernel_optimization_spec(kernel, spec):
    """Bind one private P1 spec without expanding the public kernel API."""

    if not isinstance(spec, _KernelOptimizationSpec):
        raise TypeError("spec must be a _KernelOptimizationSpec")
    if spec.artifact.provider_mode == "request_tuning":
        raise ValueError(
            "an outer optimization spec cannot recursively request provider tuning"
        )
    if spec.backend.workgroup_size is None:
        raise ValueError("P1 optimization specs require an explicit workgroup_size")
    from taichi_forge.lang.task_launch import TaskLaunchPolicy

    policy = TaskLaunchPolicy.block(
        spec.backend.workgroup_size, mode=spec.launch.block_mode
    )
    return kernel.with_launch_policy(policy)._with_optimization_spec(spec)


__all__ = [
    "_ArtifactOptions",
    "_BackendCodegenOptions",
    "_IrOptimizationOptions",
    "_KernelOptimizationSpec",
    "_LaunchOptions",
    "_bind_kernel_optimization_spec",
]
