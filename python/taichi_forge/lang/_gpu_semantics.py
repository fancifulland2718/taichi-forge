"""Internal, GPU-only execution semantics contracts.

This module intentionally depends only on the Python standard library.  It is
loaded by explicit introspection/tuning paths, never by ordinary kernel launch.
The schema models physical CUDA/Vulkan dispatches; Taichi functions are source
composition details and are not represented as GPU execution objects.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
from typing import Any, Optional, Tuple


_GPU_SEMANTICS_SCHEMA_NAME = "taichi-forge-gpu-semantics"
_GPU_SEMANTICS_SCHEMA_VERSION = 1


class _GpuBackend(str, Enum):
    CUDA = "cuda"
    VULKAN = "vulkan"


class _GpuAvailability(str, Enum):
    PROVEN = "proven"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class _GpuBindingTime(str, Enum):
    UNKNOWN = "unknown"
    LOGICAL = "logical"
    CODEGEN = "codegen"
    ARTIFACT = "artifact"
    LAUNCH = "launch"
    REPLAY = "replay"
    OBSERVATION = "observation"


class _GpuOwnership(str, Enum):
    UNKNOWN = "unknown"
    USER = "user"
    COMPILER = "compiler"
    ARTIFACT = "artifact"
    HOST_LAUNCH = "host_launch"
    DEVICE_INDIRECT = "device_indirect"
    REPLAY_PLAN = "replay_plan"
    DRIVER = "driver"
    PROFILER = "profiler"


class _GpuAutodiffRole(str, Enum):
    NONE = "none"
    PRIMAL = "primal"
    FORWARD = "forward"
    ADJOINT = "adjoint"


class _GpuLaunchKind(str, Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    RETAINED_REPLAY = "retained_replay"


class _GpuResourceKind(str, Enum):
    SCALAR = "scalar"
    STORAGE_BUFFER = "storage_buffer"
    UNIFORM_OR_PUSH = "uniform_or_push"
    SAMPLED_IMAGE = "sampled_image"
    STORAGE_IMAGE = "storage_image"
    SAMPLER = "sampler"
    ACCELERATION_STRUCTURE = "acceleration_structure"
    OPAQUE = "opaque"


class _GpuResourceAccess(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    ATOMIC = "atomic"
    REDUCTION = "reduction"
    OPAQUE = "opaque"


class _GpuAccessPattern(str, Enum):
    EXACT_POINTWISE = "exact_pointwise"
    AFFINE = "affine"
    STENCIL = "stencil"
    GATHER = "gather"
    SCATTER = "scatter"
    OPAQUE = "opaque"


class _GpuSynchronizationScope(str, Enum):
    NONE = "none"
    WORKGROUP = "workgroup"
    DISPATCH_BOUNDARY = "dispatch_boundary"
    DEVICE = "device"
    HOST = "host"
    OPAQUE = "opaque"


class _GpuMemoryVisibility(str, Enum):
    NONE = "none"
    WORKGROUP = "workgroup"
    DEVICE = "device"
    HOST = "host"
    OPAQUE = "opaque"


class _GpuTuningLocus(str, Enum):
    LOGICAL_TRANSFORM = "logical_transform"
    ARTIFACT_CODEGEN = "artifact_codegen"
    LAUNCH = "launch"
    EXECUTABLE_PLAN = "executable_plan"


class _GpuPhysicalEffect(str, Enum):
    ARTIFACT = "artifact"
    LAUNCH = "launch"
    PLAN = "plan"
    NONE = "none"


class _GpuBottleneckClass(str, Enum):
    DISPATCH = "dispatch"
    OCCUPANCY = "occupancy"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    MEMORY_LATENCY = "memory_latency"
    REDUCTION_ATOMIC = "reduction_atomic"
    COMPUTE = "compute"
    WORKGROUP_STORAGE = "workgroup_storage"


class _GpuTuningAutodiffPolicy(str, Enum):
    PRESERVED = "preserved"
    PRIMAL_ONLY = "primal_only"
    INDEPENDENT_ORACLE = "independent_oracle"
    UNSUPPORTED = "unsupported"


_ENUM_TYPES = {
    enum_type.__name__: enum_type
    for enum_type in (
        _GpuBackend,
        _GpuAvailability,
        _GpuBindingTime,
        _GpuOwnership,
        _GpuAutodiffRole,
        _GpuLaunchKind,
        _GpuResourceKind,
        _GpuResourceAccess,
        _GpuAccessPattern,
        _GpuSynchronizationScope,
        _GpuMemoryVisibility,
        _GpuTuningLocus,
        _GpuPhysicalEffect,
        _GpuBottleneckClass,
        _GpuTuningAutodiffPolicy,
    )
}
_SCHEMA_TYPES = {}


def _schema_type(cls):
    if cls.__name__ in _SCHEMA_TYPES:
        raise RuntimeError(f"duplicate GPU semantics schema type {cls.__name__}")
    _SCHEMA_TYPES[cls.__name__] = cls
    return cls


def _require_text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_backend(value):
    if not isinstance(value, _GpuBackend):
        raise TypeError("GPU semantics backend must be _GpuBackend.CUDA or VULKAN")


def _require_tuple_members(values, expected_type, name):
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not all(isinstance(value, expected_type) for value in values):
        raise TypeError(f"{name} contains an invalid member")


@_schema_type
@dataclass(frozen=True)
class _GpuExtent3:
    x: int
    y: int = 1
    z: int = 1

    def __post_init__(self):
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


def _validate_fact_value(value):
    if value is None or isinstance(value, (str, int, float, bool, Enum)):
        return
    if is_dataclass(value) and type(value).__name__ in _SCHEMA_TYPES:
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_fact_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_fact_value(item)
        return
    raise TypeError(
        f"GPU facts cannot retain runtime/native object {type(value).__name__}"
    )


@_schema_type
@dataclass(frozen=True)
class _GpuFact:
    availability: _GpuAvailability
    value: Any = None
    binding_time: _GpuBindingTime = _GpuBindingTime.UNKNOWN
    ownership: _GpuOwnership = _GpuOwnership.UNKNOWN
    provenance: str = ""
    reason: str = ""
    qualifiers: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self):
        if not isinstance(self.availability, _GpuAvailability):
            raise TypeError("availability must be a _GpuAvailability")
        if not isinstance(self.binding_time, _GpuBindingTime):
            raise TypeError("binding_time must be a _GpuBindingTime")
        if not isinstance(self.ownership, _GpuOwnership):
            raise TypeError("ownership must be a _GpuOwnership")
        if self.availability == _GpuAvailability.PROVEN:
            if self.value is None:
                raise ValueError("a proven GPU fact must carry a value")
            if not self.provenance:
                raise ValueError("a proven GPU fact must carry provenance")
            _validate_fact_value(self.value)
        else:
            if self.value is not None:
                raise ValueError("unknown/unsupported GPU facts cannot carry values")
            if not self.reason:
                raise ValueError("unknown/unsupported GPU facts must carry a reason")
        if tuple(sorted(self.qualifiers)) != self.qualifiers:
            raise ValueError("GPU fact qualifiers must be sorted")
        if len({name for name, _ in self.qualifiers}) != len(self.qualifiers):
            raise ValueError("GPU fact qualifier names must be unique")


def _gpu_fact_proven(
    value,
    *,
    binding_time,
    ownership,
    provenance,
    qualifiers=(),
):
    return _GpuFact(
        availability=_GpuAvailability.PROVEN,
        value=value,
        binding_time=binding_time,
        ownership=ownership,
        provenance=provenance,
        qualifiers=tuple(sorted(qualifiers)),
    )


def _gpu_fact_unknown(reason, *, binding_time=_GpuBindingTime.UNKNOWN):
    return _GpuFact(
        availability=_GpuAvailability.UNKNOWN,
        binding_time=binding_time,
        reason=reason,
    )


def _gpu_fact_unsupported(reason, *, binding_time=_GpuBindingTime.UNKNOWN):
    return _GpuFact(
        availability=_GpuAvailability.UNSUPPORTED,
        binding_time=binding_time,
        reason=reason,
    )


def _default_unknown_fact():
    return _gpu_fact_unknown("not_resolved")


@_schema_type
@dataclass(frozen=True)
class _GpuResolvedValue:
    requested: _GpuFact = field(default_factory=_default_unknown_fact)
    selected: _GpuFact = field(default_factory=_default_unknown_fact)
    materialized: _GpuFact = field(default_factory=_default_unknown_fact)
    actual: _GpuFact = field(default_factory=_default_unknown_fact)
    observed: _GpuFact = field(default_factory=_default_unknown_fact)

    def __post_init__(self):
        for value in (
            self.requested,
            self.selected,
            self.materialized,
            self.actual,
            self.observed,
        ):
            if not isinstance(value, _GpuFact):
                raise TypeError("_GpuResolvedValue members must be _GpuFact")


@_schema_type
@dataclass(frozen=True)
class _GpuNamedFact:
    name: str
    fact: _GpuFact

    def __post_init__(self):
        _require_text(self.name, "GPU fact name")
        if not isinstance(self.fact, _GpuFact):
            raise TypeError("fact must be a _GpuFact")


@_schema_type
@dataclass(frozen=True)
class _GpuTargetSemantics:
    target_id: str
    backend: _GpuBackend
    vendor: str = ""
    device: str = ""
    architecture: str = ""
    driver_identity: str = ""
    runtime_identity: str = ""
    limits: Tuple[_GpuNamedFact, ...] = ()
    capabilities: Tuple[_GpuNamedFact, ...] = ()

    def __post_init__(self):
        _require_text(self.target_id, "target_id")
        _require_backend(self.backend)
        _require_tuple_members(self.limits, _GpuNamedFact, "target limits")
        _require_tuple_members(
            self.capabilities, _GpuNamedFact, "target capabilities"
        )


@_schema_type
@dataclass(frozen=True)
class _GpuAccessFootprint:
    """A proven logical access map, independent of backend pointer objects."""

    pattern: _GpuAccessPattern
    iteration_rank: int
    affine_coefficients: Tuple[Tuple[int, ...], ...] = ()
    affine_offsets: Tuple[int, ...] = ()
    halo: Tuple[Tuple[int, int], ...] = ()
    contiguous_axis: Optional[int] = None
    byte_alignment: Optional[int] = None
    reuse_class: str = "unknown"
    layout_fingerprint: str = ""
    block_uniform_control: _GpuFact = field(default_factory=_default_unknown_fact)
    provenance: str = ""

    def __post_init__(self):
        if not isinstance(self.pattern, _GpuAccessPattern):
            raise TypeError("access footprint pattern must be _GpuAccessPattern")
        if (
            isinstance(self.iteration_rank, bool)
            or not isinstance(self.iteration_rank, int)
            or self.iteration_rank < 0
        ):
            raise ValueError("access footprint iteration_rank must be non-negative")
        if not isinstance(self.affine_coefficients, tuple) or any(
            not isinstance(row, tuple)
            or len(row) != self.iteration_rank
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in row
            )
            for row in self.affine_coefficients
        ):
            raise TypeError(
                "affine coefficients must be integer tuples of iteration rank"
            )
        if not isinstance(self.affine_offsets, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.affine_offsets
        ):
            raise TypeError("affine offsets must be an integer tuple")
        if len(self.affine_offsets) != len(self.affine_coefficients):
            raise ValueError("affine coefficient and offset ranks must match")
        if not isinstance(self.halo, tuple) or any(
            not isinstance(bounds, tuple)
            or len(bounds) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in bounds
            )
            or bounds[0] > bounds[1]
            for bounds in self.halo
        ):
            raise TypeError("halo must contain ordered integer bound pairs")
        if self.halo and len(self.halo) != len(self.affine_coefficients):
            raise ValueError("halo and affine access ranks must match")
        if self.pattern == _GpuAccessPattern.EXACT_POINTWISE:
            identity = tuple(
                tuple(int(row == column) for column in range(self.iteration_rank))
                for row in range(self.iteration_rank)
            )
            if (
                self.affine_coefficients != identity
                or self.affine_offsets != (0,) * self.iteration_rank
                or any(bounds != (0, 0) for bounds in self.halo)
            ):
                raise ValueError(
                    "exact-pointwise footprints require identity affine maps "
                    "and a zero halo"
                )
        if self.contiguous_axis is not None and (
            isinstance(self.contiguous_axis, bool)
            or not isinstance(self.contiguous_axis, int)
            or self.contiguous_axis < 0
        ):
            raise ValueError("contiguous_axis must be non-negative or None")
        if (
            self.contiguous_axis is not None
            and self.contiguous_axis >= len(self.affine_coefficients)
        ):
            raise ValueError("contiguous_axis exceeds the access rank")
        if self.byte_alignment is not None and (
            isinstance(self.byte_alignment, bool)
            or not isinstance(self.byte_alignment, int)
            or self.byte_alignment <= 0
            or self.byte_alignment & (self.byte_alignment - 1)
        ):
            raise ValueError("byte_alignment must be a positive power of two")
        if self.reuse_class not in (
            "none",
            "neighbor",
            "tile",
            "broadcast",
            "unknown",
        ):
            raise ValueError("invalid access footprint reuse class")
        if not isinstance(self.block_uniform_control, _GpuFact):
            raise TypeError("block_uniform_control must be a _GpuFact")
        _require_text(self.provenance, "access footprint provenance")


@_schema_type
@dataclass(frozen=True)
class _GpuResourceEffect:
    resource_id: str
    access: _GpuResourceAccess
    is_gradient: bool = False
    provenance: str = ""
    footprint: Optional[_GpuAccessFootprint] = None

    def __post_init__(self):
        _require_text(self.resource_id, "resource_id")
        if not isinstance(self.access, _GpuResourceAccess):
            raise TypeError("access must be a _GpuResourceAccess")
        if self.footprint is not None and not isinstance(
            self.footprint, _GpuAccessFootprint
        ):
            raise TypeError("resource effect footprint must be _GpuAccessFootprint")


@_schema_type
@dataclass(frozen=True)
class _GpuBinding:
    logical_path: Tuple[int, ...]
    kind: _GpuResourceKind
    backend_slot: str
    dtype: str = ""
    ndim: Optional[int] = None
    access: _GpuResourceAccess = _GpuResourceAccess.OPAQUE
    alias_group: str = ""
    required: bool = True
    is_gradient: bool = False
    replay_mutable: bool = True

    def __post_init__(self):
        if not isinstance(self.logical_path, tuple) or not all(
            isinstance(index, int) and not isinstance(index, bool) and index >= 0
            for index in self.logical_path
        ):
            raise TypeError("logical_path must contain non-negative integers")
        if not isinstance(self.kind, _GpuResourceKind):
            raise TypeError("kind must be a _GpuResourceKind")
        if not isinstance(self.access, _GpuResourceAccess):
            raise TypeError("access must be a _GpuResourceAccess")
        _require_text(self.backend_slot, "backend_slot")
        if self.ndim is not None and self.ndim < 0:
            raise ValueError("ndim must be non-negative")


@_schema_type
@dataclass(frozen=True)
class _GpuBindingSchema:
    schema_id: str
    bindings: Tuple[_GpuBinding, ...] = ()
    provenance: str = ""

    def __post_init__(self):
        _require_text(self.schema_id, "binding schema_id")
        _require_tuple_members(self.bindings, _GpuBinding, "bindings")


@_schema_type
@dataclass(frozen=True)
class _GpuIntrinsicRequirement:
    name: str
    capability: _GpuFact
    constraints: Tuple[_GpuNamedFact, ...] = ()
    lowering_route: str = ""
    differentiation_policy: str = "unknown"

    def __post_init__(self):
        _require_text(self.name, "intrinsic requirement name")
        if not isinstance(self.capability, _GpuFact):
            raise TypeError("capability must be a _GpuFact")
        _require_tuple_members(
            self.constraints, _GpuNamedFact, "intrinsic constraints"
        )
        if self.differentiation_policy not in (
            "differentiable",
            "custom_backward",
            "unsupported",
            "unknown",
        ):
            raise ValueError("invalid intrinsic differentiation policy")


@_schema_type
@dataclass(frozen=True)
class _CudaArtifactExtension:
    function_identity: _GpuFact = field(default_factory=_default_unknown_fact)
    max_threads_per_block: _GpuFact = field(default_factory=_default_unknown_fact)
    static_shared_memory_bytes: _GpuFact = field(default_factory=_default_unknown_fact)
    registers_per_thread: _GpuFact = field(default_factory=_default_unknown_fact)
    constant_memory_bytes: _GpuFact = field(default_factory=_default_unknown_fact)
    local_memory_bytes_per_thread: _GpuFact = field(default_factory=_default_unknown_fact)
    ptx_version: _GpuFact = field(default_factory=_default_unknown_fact)
    binary_version: _GpuFact = field(default_factory=_default_unknown_fact)
    cache_mode_ca: _GpuFact = field(default_factory=_default_unknown_fact)
    max_dynamic_shared_bytes: _GpuFact = field(default_factory=_default_unknown_fact)
    preferred_shared_carveout: _GpuFact = field(default_factory=_default_unknown_fact)


@_schema_type
@dataclass(frozen=True)
class _VulkanArtifactExtension:
    spirv_entry_point: _GpuFact = field(default_factory=_default_unknown_fact)
    local_size: _GpuFact = field(default_factory=_default_unknown_fact)
    descriptor_layout_identity: _GpuFact = field(default_factory=_default_unknown_fact)
    pipeline_layout_identity: _GpuFact = field(default_factory=_default_unknown_fact)
    specialization_constants: Tuple[_GpuNamedFact, ...] = ()
    required_subgroup_size: _GpuFact = field(default_factory=_default_unknown_fact)
    pipeline_identity: _GpuFact = field(default_factory=_default_unknown_fact)
    pipeline_executable_statistics: _GpuFact = field(default_factory=_default_unknown_fact)


@_schema_type
@dataclass(frozen=True)
class _CudaLaunchExtension:
    stream_class: _GpuFact = field(default_factory=_default_unknown_fact)
    cooperative: _GpuFact = field(default_factory=_default_unknown_fact)
    cluster_shape: _GpuFact = field(default_factory=_default_unknown_fact)
    residency_waves: _GpuFact = field(default_factory=_default_unknown_fact)


@_schema_type
@dataclass(frozen=True)
class _VulkanLaunchExtension:
    queue_class: _GpuFact = field(default_factory=_default_unknown_fact)
    indirect_packet: _GpuFact = field(default_factory=_default_unknown_fact)
    pipeline_binding: _GpuFact = field(default_factory=_default_unknown_fact)
    retained_command_owner: _GpuFact = field(default_factory=_default_unknown_fact)


@_schema_type
@dataclass(frozen=True)
class _GpuWorkgroupResourceEnvelope:
    """Proven resources for one already-materialized workgroup shape.

    This deliberately does not claim that another workgroup shape is
    equivalent.  Shared-memory indexing and block collectives may make the
    exact shape part of the kernel semantics even when another shape fits the
    device resource limits.
    """

    selected_workgroup_shape: _GpuFact
    max_threads_per_block: _GpuFact
    static_workgroup_memory_bytes: _GpuFact
    dynamic_workgroup_memory_bytes: _GpuFact
    registers_per_thread: _GpuFact = field(default_factory=_default_unknown_fact)
    local_memory_bytes_per_thread: _GpuFact = field(
        default_factory=_default_unknown_fact
    )
    shape_scope: str = "exact_materialized"
    provenance: str = ""

    def __post_init__(self):
        for name, value in (
            ("selected_workgroup_shape", self.selected_workgroup_shape),
            ("max_threads_per_block", self.max_threads_per_block),
            (
                "static_workgroup_memory_bytes",
                self.static_workgroup_memory_bytes,
            ),
            (
                "dynamic_workgroup_memory_bytes",
                self.dynamic_workgroup_memory_bytes,
            ),
            ("registers_per_thread", self.registers_per_thread),
            (
                "local_memory_bytes_per_thread",
                self.local_memory_bytes_per_thread,
            ),
        ):
            if not isinstance(value, _GpuFact):
                raise TypeError(f"{name} must be a _GpuFact")
        if self.shape_scope not in ("exact_materialized", "rematerializable"):
            raise ValueError("invalid workgroup resource shape scope")
        _require_text(self.provenance, "workgroup resource provenance")


@_schema_type
@dataclass(frozen=True)
class _GpuArtifactSemantics:
    artifact_id: str
    entry_point_id: str
    backend: _GpuBackend
    target_id: str
    codegen_identity: str
    workgroup_shape: _GpuResolvedValue
    static_workgroup_memory_bytes: _GpuFact
    compiler_thread_local_scratch_bytes: _GpuFact
    binding_schema_id: str
    required_capabilities: Tuple[_GpuNamedFact, ...] = ()
    cache_provenance: str = ""
    extension: Any = None

    def __post_init__(self):
        _require_text(self.artifact_id, "artifact_id")
        _require_text(self.entry_point_id, "entry_point_id")
        _require_backend(self.backend)
        _require_text(self.target_id, "artifact target_id")
        _require_text(self.binding_schema_id, "artifact binding_schema_id")
        if not isinstance(self.workgroup_shape, _GpuResolvedValue):
            raise TypeError("workgroup_shape must be a _GpuResolvedValue")
        if not isinstance(self.static_workgroup_memory_bytes, _GpuFact):
            raise TypeError("static_workgroup_memory_bytes must be a _GpuFact")
        if not isinstance(self.compiler_thread_local_scratch_bytes, _GpuFact):
            raise TypeError(
                "compiler_thread_local_scratch_bytes must be a _GpuFact"
            )
        _require_tuple_members(
            self.required_capabilities,
            _GpuNamedFact,
            "artifact required_capabilities",
        )
        if self.backend == _GpuBackend.CUDA:
            if self.extension is not None and not isinstance(
                self.extension, _CudaArtifactExtension
            ):
                raise TypeError("CUDA artifact requires _CudaArtifactExtension")
        elif self.extension is not None and not isinstance(
            self.extension, _VulkanArtifactExtension
        ):
            raise TypeError("Vulkan artifact requires _VulkanArtifactExtension")


@_schema_type
@dataclass(frozen=True)
class _GpuLaunchSemantics:
    launch_id: str
    backend: _GpuBackend
    kind: _GpuLaunchKind
    dispatch_group_count: _GpuResolvedValue
    workgroup_shape: _GpuResolvedValue
    dynamic_workgroup_memory_bytes: _GpuResolvedValue
    ordering_scope: str = ""
    actual_geometry_blocker: str = ""
    extension: Any = None

    def __post_init__(self):
        _require_text(self.launch_id, "launch_id")
        _require_backend(self.backend)
        if not isinstance(self.kind, _GpuLaunchKind):
            raise TypeError("kind must be a _GpuLaunchKind")
        for name, value in (
            ("dispatch_group_count", self.dispatch_group_count),
            ("workgroup_shape", self.workgroup_shape),
            (
                "dynamic_workgroup_memory_bytes",
                self.dynamic_workgroup_memory_bytes,
            ),
        ):
            if not isinstance(value, _GpuResolvedValue):
                raise TypeError(f"{name} must be a _GpuResolvedValue")
        if self.backend == _GpuBackend.CUDA:
            if self.extension is not None and not isinstance(
                self.extension, _CudaLaunchExtension
            ):
                raise TypeError("CUDA launch requires _CudaLaunchExtension")
        elif self.extension is not None and not isinstance(
            self.extension, _VulkanLaunchExtension
        ):
            raise TypeError("Vulkan launch requires _VulkanLaunchExtension")


@_schema_type
@dataclass(frozen=True)
class _GpuDispatchSemantics:
    logical_task_id: str
    physical_dispatch_id: str
    ordinal: int
    task_kind: str
    backend: _GpuBackend
    artifact_id: str
    launch_id: str
    binding_schema_id: str
    dimension_rank: int
    logical_work_extent: _GpuFact
    dispatch_group_count: _GpuResolvedValue
    workgroup_shape: _GpuResolvedValue
    range_mapping: _GpuFact
    effects: Tuple[_GpuResourceEffect, ...] = ()
    intrinsic_requirements: Tuple[_GpuIntrinsicRequirement, ...] = ()
    effects_blocker: str = ""
    provenance: str = ""

    def __post_init__(self):
        _require_text(self.logical_task_id, "logical_task_id")
        _require_text(self.physical_dispatch_id, "physical_dispatch_id")
        _require_text(self.task_kind, "task_kind")
        _require_backend(self.backend)
        _require_text(self.artifact_id, "dispatch artifact_id")
        _require_text(self.launch_id, "dispatch launch_id")
        _require_text(self.binding_schema_id, "dispatch binding_schema_id")
        if self.ordinal < 0:
            raise ValueError("dispatch ordinal must be non-negative")
        if self.dimension_rank not in (1, 2, 3):
            raise ValueError("dimension_rank must be 1, 2, or 3")
        if not isinstance(self.logical_work_extent, _GpuFact):
            raise TypeError("logical_work_extent must be a _GpuFact")
        if not isinstance(self.dispatch_group_count, _GpuResolvedValue):
            raise TypeError("dispatch_group_count must be a _GpuResolvedValue")
        if not isinstance(self.workgroup_shape, _GpuResolvedValue):
            raise TypeError("workgroup_shape must be a _GpuResolvedValue")
        if not isinstance(self.range_mapping, _GpuFact):
            raise TypeError("range_mapping must be a _GpuFact")
        _require_tuple_members(self.effects, _GpuResourceEffect, "dispatch effects")
        _require_tuple_members(
            self.intrinsic_requirements,
            _GpuIntrinsicRequirement,
            "dispatch intrinsic requirements",
        )


@_schema_type
@dataclass(frozen=True)
class _GpuProgramSemantics:
    logical_program_id: str
    specialization_id: str
    backend: _GpuBackend
    target_id: str
    autodiff_role: _GpuAutodiffRole
    primal_program_id: str = ""
    differentiation_relation: _GpuFact = field(default_factory=_default_unknown_fact)
    iteration_domain: _GpuFact = field(default_factory=_default_unknown_fact)
    effects: Tuple[_GpuResourceEffect, ...] = ()
    side_effects: Tuple[str, ...] = ()
    synchronization: _GpuFact = field(default_factory=_default_unknown_fact)
    dispatch_ids: Tuple[str, ...] = ()
    graph_eligibility: _GpuFact = field(default_factory=_default_unknown_fact)
    provenance: str = ""

    def __post_init__(self):
        _require_text(self.logical_program_id, "logical_program_id")
        _require_text(self.specialization_id, "specialization_id")
        _require_backend(self.backend)
        _require_text(self.target_id, "program target_id")
        if not isinstance(self.autodiff_role, _GpuAutodiffRole):
            raise TypeError("autodiff_role must be a _GpuAutodiffRole")
        if self.autodiff_role in (_GpuAutodiffRole.FORWARD, _GpuAutodiffRole.ADJOINT):
            _require_text(self.primal_program_id, "derivative primal_program_id")
        if not isinstance(self.differentiation_relation, _GpuFact):
            raise TypeError("differentiation_relation must be a _GpuFact")
        if not isinstance(self.iteration_domain, _GpuFact):
            raise TypeError("iteration_domain must be a _GpuFact")
        if not isinstance(self.synchronization, _GpuFact):
            raise TypeError("synchronization must be a _GpuFact")
        if not isinstance(self.graph_eligibility, _GpuFact):
            raise TypeError("graph_eligibility must be a _GpuFact")
        _require_tuple_members(self.effects, _GpuResourceEffect, "program effects")
        _require_tuple_members(self.side_effects, str, "program side_effects")
        _require_tuple_members(self.dispatch_ids, str, "program dispatch_ids")


@_schema_type
@dataclass(frozen=True)
class _GpuRuntimeObservation:
    observation_id: str
    target_id: str
    artifact_id: str
    launch_id: str
    workload_profile_id: str
    provider: str
    provider_version: str = ""
    metrics: Tuple[_GpuNamedFact, ...] = ()
    sample_count: int = 0
    coefficient_of_variation: Optional[float] = None
    qualification_status: str = "diagnostic"
    fixed_cost_seconds: Optional[float] = None
    scale_dependent_cost_seconds: Optional[float] = None

    def __post_init__(self):
        _require_text(self.observation_id, "observation_id")
        _require_text(self.target_id, "observation target_id")
        _require_text(self.artifact_id, "observation artifact_id")
        _require_text(self.launch_id, "observation launch_id")
        _require_text(self.workload_profile_id, "workload_profile_id")
        _require_text(self.provider, "observation provider")
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        _require_tuple_members(self.metrics, _GpuNamedFact, "observation metrics")


@_schema_type
@dataclass(frozen=True)
class _GpuPlanDependency:
    source_node_id: str
    target_node_id: str
    kind: str
    execution_scope: _GpuSynchronizationScope = (
        _GpuSynchronizationScope.DEVICE
    )
    memory_visibility: _GpuMemoryVisibility = _GpuMemoryVisibility.DEVICE
    resource_ids: Tuple[str, ...] = ()
    provenance: str = "ordered_executable_plan"

    def __post_init__(self):
        _require_text(self.source_node_id, "dependency source_node_id")
        _require_text(self.target_node_id, "dependency target_node_id")
        _require_text(self.kind, "dependency kind")
        if not isinstance(self.execution_scope, _GpuSynchronizationScope):
            raise TypeError("dependency execution_scope is invalid")
        if not isinstance(self.memory_visibility, _GpuMemoryVisibility):
            raise TypeError("dependency memory_visibility is invalid")
        _require_tuple_members(self.resource_ids, str, "dependency resource_ids")
        _require_text(self.provenance, "dependency provenance")


@_schema_type
@dataclass(frozen=True)
class _GpuExecutablePlanSemantics:
    plan_id: str
    backend: _GpuBackend
    target_id: str
    ordered_node_ids: Tuple[str, ...]
    semantic_plan_id: str = ""
    optimization_spec_id: str = ""
    fusion_recipe_ids: Tuple[str, ...] = ()
    optimization_status: str = ""
    dispatch_ids: Tuple[str, ...] = ()
    native_action_ids: Tuple[str, ...] = ()
    dependencies: Tuple[_GpuPlanDependency, ...] = ()
    binding_schema_ids: Tuple[str, ...] = ()
    workload_profile_id: str = ""
    retained_replay: _GpuFact = field(default_factory=_default_unknown_fact)
    lifecycle: Tuple[_GpuNamedFact, ...] = ()
    provenance: str = ""

    def __post_init__(self):
        _require_text(self.plan_id, "plan_id")
        _require_backend(self.backend)
        _require_text(self.target_id, "plan target_id")
        if self.optimization_spec_id and not self.semantic_plan_id:
            raise ValueError(
                "optimized executable plan requires semantic plan identity"
            )
        _require_tuple_members(
            self.fusion_recipe_ids, str, "plan fusion recipe IDs"
        )
        known_nodes = set(self.dispatch_ids) | set(self.native_action_ids)
        if set(self.ordered_node_ids) != known_nodes:
            raise ValueError("ordered plan nodes must match dispatch/native action ids")
        if len(self.ordered_node_ids) != len(known_nodes):
            raise ValueError("ordered plan nodes must be unique")
        _require_tuple_members(
            self.dependencies, _GpuPlanDependency, "plan dependencies"
        )
        _require_tuple_members(
            self.lifecycle, _GpuNamedFact, "plan lifecycle facts"
        )


@_schema_type
@dataclass(frozen=True)
class _GpuSemanticSnapshot:
    target: _GpuTargetSemantics
    program: _GpuProgramSemantics
    binding_schemas: Tuple[_GpuBindingSchema, ...]
    artifacts: Tuple[_GpuArtifactSemantics, ...]
    launches: Tuple[_GpuLaunchSemantics, ...]
    dispatches: Tuple[_GpuDispatchSemantics, ...]
    executable_plan: Optional[_GpuExecutablePlanSemantics] = None
    resident_only: bool = True

    def __post_init__(self):
        if not isinstance(self.target, _GpuTargetSemantics):
            raise TypeError("snapshot target must be _GpuTargetSemantics")
        if not isinstance(self.program, _GpuProgramSemantics):
            raise TypeError("snapshot program must be _GpuProgramSemantics")
        _require_tuple_members(
            self.binding_schemas, _GpuBindingSchema, "snapshot binding_schemas"
        )
        _require_tuple_members(
            self.artifacts, _GpuArtifactSemantics, "snapshot artifacts"
        )
        _require_tuple_members(
            self.launches, _GpuLaunchSemantics, "snapshot launches"
        )
        _require_tuple_members(
            self.dispatches, _GpuDispatchSemantics, "snapshot dispatches"
        )
        if self.executable_plan is not None and not isinstance(
            self.executable_plan, _GpuExecutablePlanSemantics
        ):
            raise TypeError(
                "snapshot executable_plan must be _GpuExecutablePlanSemantics"
            )
        backend = self.target.backend
        if self.program.backend != backend:
            raise ValueError("snapshot target/program backend mismatch")
        if any(item.backend != backend for item in self.artifacts):
            raise ValueError("snapshot artifact backend mismatch")
        if any(item.backend != backend for item in self.launches):
            raise ValueError("snapshot launch backend mismatch")
        if any(item.backend != backend for item in self.dispatches):
            raise ValueError("snapshot dispatch backend mismatch")
        dispatch_ids = tuple(item.physical_dispatch_id for item in self.dispatches)
        if self.program.dispatch_ids != dispatch_ids:
            raise ValueError("snapshot program dispatch order mismatch")
        if {item.artifact_id for item in self.artifacts} != {
            item.artifact_id for item in self.dispatches
        }:
            raise ValueError("snapshot artifact coverage mismatch")
        if {item.launch_id for item in self.launches} != {
            item.launch_id for item in self.dispatches
        }:
            raise ValueError("snapshot launch coverage mismatch")
        if {item.schema_id for item in self.binding_schemas} != {
            item.binding_schema_id for item in self.dispatches
        }:
            raise ValueError("snapshot binding schema coverage mismatch")


@_schema_type
@dataclass(frozen=True)
class _GpuArtifactQualificationSnapshot:
    semantics: _GpuSemanticSnapshot
    observations: Tuple[_GpuRuntimeObservation, ...]
    provider: str
    provider_version: str
    fixed_cost_seconds: float
    scale_dependent_cost_seconds: float
    qualified_artifact_count: int
    registration_materialization_count: int

    def __post_init__(self):
        if not isinstance(self.semantics, _GpuSemanticSnapshot):
            raise TypeError("qualification semantics must be _GpuSemanticSnapshot")
        _require_tuple_members(self.observations, _GpuRuntimeObservation, "qualification observations")
        _require_text(self.provider, "qualification provider")
        _require_text(self.provider_version, "qualification provider_version")
        for name, value in (
            ("fixed_cost_seconds", self.fixed_cost_seconds),
            ("scale_dependent_cost_seconds", self.scale_dependent_cost_seconds),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in (
            ("qualified_artifact_count", self.qualified_artifact_count),
            ("registration_materialization_count", self.registration_materialization_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        target_id = self.semantics.target.target_id
        artifact_ids = {item.artifact_id for item in self.semantics.artifacts}
        launch_ids = {item.launch_id for item in self.semantics.launches}
        for observation in self.observations:
            if observation.target_id != target_id:
                raise ValueError("qualification observation target mismatch")
            if observation.artifact_id not in artifact_ids:
                raise ValueError("qualification observation artifact mismatch")
            if observation.launch_id not in launch_ids:
                raise ValueError("qualification observation launch mismatch")


@_schema_type
@dataclass(frozen=True)
class _GpuExecutablePlanSnapshot:
    target: _GpuTargetSemantics
    programs: Tuple[_GpuProgramSemantics, ...]
    binding_schemas: Tuple[_GpuBindingSchema, ...]
    artifacts: Tuple[_GpuArtifactSemantics, ...]
    launches: Tuple[_GpuLaunchSemantics, ...]
    dispatches: Tuple[_GpuDispatchSemantics, ...]
    executable_plan: _GpuExecutablePlanSemantics
    resident_only: bool = True

    def __post_init__(self):
        if not isinstance(self.target, _GpuTargetSemantics):
            raise TypeError("plan snapshot target must be _GpuTargetSemantics")
        _require_tuple_members(
            self.programs, _GpuProgramSemantics, "plan snapshot programs"
        )
        _require_tuple_members(
            self.binding_schemas,
            _GpuBindingSchema,
            "plan snapshot binding_schemas",
        )
        _require_tuple_members(
            self.artifacts, _GpuArtifactSemantics, "plan snapshot artifacts"
        )
        _require_tuple_members(
            self.launches, _GpuLaunchSemantics, "plan snapshot launches"
        )
        _require_tuple_members(
            self.dispatches, _GpuDispatchSemantics, "plan snapshot dispatches"
        )
        if not isinstance(self.executable_plan, _GpuExecutablePlanSemantics):
            raise TypeError(
                "plan snapshot executable_plan must be "
                "_GpuExecutablePlanSemantics"
            )
        backend = self.target.backend
        for collection in (
            self.programs,
            self.artifacts,
            self.launches,
            self.dispatches,
        ):
            if any(item.backend != backend for item in collection):
                raise ValueError("plan snapshot backend mismatch")
        if self.executable_plan.backend != backend:
            raise ValueError("plan snapshot executable backend mismatch")
        dispatch_ids = {
            dispatch.physical_dispatch_id for dispatch in self.dispatches
        }
        if set(self.executable_plan.dispatch_ids) != dispatch_ids:
            raise ValueError("plan snapshot dispatch coverage mismatch")
        if len(dispatch_ids) != len(self.dispatches):
            raise ValueError("plan snapshot dispatch identities must be unique")
        if {launch.launch_id for launch in self.launches} != {
            dispatch.launch_id for dispatch in self.dispatches
        }:
            raise ValueError("plan snapshot launch coverage mismatch")
        if {artifact.artifact_id for artifact in self.artifacts} != {
            dispatch.artifact_id for dispatch in self.dispatches
        }:
            raise ValueError("plan snapshot artifact coverage mismatch")
        schema_ids = {schema.schema_id for schema in self.binding_schemas}
        if not {
            dispatch.binding_schema_id for dispatch in self.dispatches
        } <= schema_ids:
            raise ValueError("plan snapshot binding coverage mismatch")
        if set(self.executable_plan.binding_schema_ids) != schema_ids:
            raise ValueError("plan snapshot executable binding coverage mismatch")


@_schema_type
@dataclass(frozen=True)
class _GpuTuningDimension:
    name: str
    locus: _GpuTuningLocus
    backend_applicability: Tuple[_GpuBackend, ...]
    legal_values: Tuple[Any, ...]
    required_capabilities: Tuple[str, ...]
    controller: str
    binding_time: _GpuBindingTime
    physical_effect: _GpuPhysicalEffect
    equivalence_key: str
    dependencies: Tuple[str, ...] = ()
    bottleneck_classes: Tuple[_GpuBottleneckClass, ...] = ()
    autodiff_policy: _GpuTuningAutodiffPolicy = (
        _GpuTuningAutodiffPolicy.INDEPENDENT_ORACLE
    )
    status: _GpuFact = field(default_factory=_default_unknown_fact)

    def __post_init__(self):
        _require_text(self.name, "tuning dimension name")
        if not isinstance(self.locus, _GpuTuningLocus):
            raise TypeError("locus must be a _GpuTuningLocus")
        if not self.backend_applicability:
            raise ValueError("backend_applicability must not be empty")
        for backend in self.backend_applicability:
            _require_backend(backend)
        if not isinstance(self.legal_values, tuple):
            raise TypeError("legal_values must be a tuple")
        for value in self.legal_values:
            _validate_fact_value(value)
        _require_tuple_members(
            self.required_capabilities, str, "required_capabilities"
        )
        _require_tuple_members(self.dependencies, str, "dimension dependencies")
        _require_tuple_members(
            self.bottleneck_classes,
            _GpuBottleneckClass,
            "dimension bottleneck_classes",
        )
        if not isinstance(self.autodiff_policy, _GpuTuningAutodiffPolicy):
            raise TypeError("autodiff_policy must be a _GpuTuningAutodiffPolicy")
        _require_text(self.controller, "tuning dimension controller")
        if not isinstance(self.binding_time, _GpuBindingTime):
            raise TypeError("binding_time must be a _GpuBindingTime")
        if not isinstance(self.physical_effect, _GpuPhysicalEffect):
            raise TypeError("physical_effect must be a _GpuPhysicalEffect")
        _require_text(self.equivalence_key, "tuning dimension equivalence_key")


def _encode_gpu_semantics(value):
    if isinstance(value, Enum):
        return {"$enum": type(value).__name__, "value": value.value}
    if is_dataclass(value):
        type_name = type(value).__name__
        if type_name not in _SCHEMA_TYPES:
            raise TypeError(f"unregistered GPU semantics dataclass {type_name}")
        return {
            "$type": type_name,
            **{
                item.name: _encode_gpu_semantics(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, tuple):
        return {"$tuple": [_encode_gpu_semantics(item) for item in value]}
    if isinstance(value, list):
        return [_encode_gpu_semantics(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("GPU semantics mappings require string keys")
        return {
            key: _encode_gpu_semantics(item)
            for key, item in sorted(value.items())
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"GPU semantics cannot serialize runtime/native object {type(value).__name__}"
    )


def _decode_gpu_semantics(value):
    if isinstance(value, list):
        return [_decode_gpu_semantics(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$enum" in value:
        enum_type = _ENUM_TYPES.get(value["$enum"])
        if enum_type is None:
            raise ValueError(f"unknown GPU semantics enum {value['$enum']}")
        return enum_type(value["value"])
    if "$tuple" in value:
        return tuple(_decode_gpu_semantics(item) for item in value["$tuple"])
    if "$type" in value:
        cls = _SCHEMA_TYPES.get(value["$type"])
        if cls is None:
            raise ValueError(f"unknown GPU semantics type {value['$type']}")
        return cls(
            **{
                name: _decode_gpu_semantics(item)
                for name, item in value.items()
                if name != "$type"
            }
        )
    return {key: _decode_gpu_semantics(item) for key, item in value.items()}


def _dumps_gpu_semantics(value):
    payload = {
        "schema": _GPU_SEMANTICS_SCHEMA_NAME,
        "version": _GPU_SEMANTICS_SCHEMA_VERSION,
        "payload": _encode_gpu_semantics(value),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _loads_gpu_semantics(payload):
    root = json.loads(payload)
    if root.get("schema") != _GPU_SEMANTICS_SCHEMA_NAME:
        raise ValueError("not a Taichi Forge GPU semantics payload")
    if root.get("version") != _GPU_SEMANTICS_SCHEMA_VERSION:
        raise ValueError("unsupported Taichi Forge GPU semantics schema version")
    return _decode_gpu_semantics(root["payload"])


__all__ = [
    "_CudaArtifactExtension",
    "_CudaLaunchExtension",
    "_GPU_SEMANTICS_SCHEMA_VERSION",
    "_GpuArtifactQualificationSnapshot",
    "_GpuAccessFootprint",
    "_GpuAccessPattern",
    "_GpuArtifactSemantics",
    "_GpuAutodiffRole",
    "_GpuAvailability",
    "_GpuBackend",
    "_GpuBottleneckClass",
    "_GpuBinding",
    "_GpuBindingSchema",
    "_GpuBindingTime",
    "_GpuDispatchSemantics",
    "_GpuExecutablePlanSemantics",
    "_GpuExtent3",
    "_GpuFact",
    "_GpuIntrinsicRequirement",
    "_GpuLaunchKind",
    "_GpuLaunchSemantics",
    "_GpuMemoryVisibility",
    "_GpuNamedFact",
    "_GpuOwnership",
    "_GpuPhysicalEffect",
    "_GpuPlanDependency",
    "_GpuProgramSemantics",
    "_GpuResolvedValue",
    "_GpuResourceAccess",
    "_GpuResourceEffect",
    "_GpuResourceKind",
    "_GpuRuntimeObservation",
    "_GpuSemanticSnapshot",
    "_GpuSynchronizationScope",
    "_GpuTargetSemantics",
    "_GpuTuningDimension",
    "_GpuTuningAutodiffPolicy",
    "_GpuTuningLocus",
    "_GpuWorkgroupResourceEnvelope",
    "_VulkanArtifactExtension",
    "_VulkanLaunchExtension",
    "_dumps_gpu_semantics",
    "_gpu_fact_proven",
    "_gpu_fact_unknown",
    "_gpu_fact_unsupported",
    "_loads_gpu_semantics",
]
