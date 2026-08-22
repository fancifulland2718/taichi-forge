"""Static hardware-operation contracts and side-effect-free capability reports."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple


HARDWARE_CAPABILITY_SCHEMA_VERSION = 1

DEPENDENCY_TIERS = (
    "core",
    "lazy_external",
    "build_external",
)

LOAD_MODES = (
    "built_in",
    "runtime_lazy",
    "build_only",
)

PROVIDER_CLASSES = (
    "hardware_intrinsic",
    "fixed_function",
    "vendor_hardware_runtime",
    "vendor_algorithm",
    "compute_native",
    "compute_fallback",
    "runtime_interop",
)

EXECUTION_CLASSES = (
    "hardware_instruction",
    "native_shader_operation",
    "fixed_function",
    "vendor_hardware_runtime",
    "native_command",
    "vendor_library",
    "compute_kernel",
)

HARDWARE_ACCELERATION_LEVELS = (
    "guaranteed",
    "qualified",
    "implementation_defined",
    "none",
)

EXECUTION_KINDS = (
    "kernel_intrinsic",
    "native_command",
    "external_library",
    "compute_kernel",
)

GRAPH_SUPPORT_MODES = (
    "inline",
    "recordable",
    "stream_capture",
    "opaque",
    "unsupported",
)

STREAM_BINDINGS = (
    "runtime_ordered",
    "current",
    "explicit",
)

WORKSPACE_OWNERSHIP = (
    "none",
    "caller_owned",
    "provider_owned",
    "graph_owned",
)

LIFETIME_POLICIES = (
    "call",
    "runtime_generation",
    "resource_generation",
    "graph_generation",
    "provider_plan",
    "implementation_defined",
)

UPDATE_POLICIES = (
    "immutable",
    "per_dispatch",
    "rebind",
    "rebuild",
    "refit",
    "implementation_defined",
)

OPERATION_SCOPES = (
    "kernel",
    "python",
    "graph",
    "internal",
)

IMPLEMENTATION_STATUSES = (
    "existing_public",
    "internal_foundation",
    "qualification_required",
    "planned",
    "reference_only",
)

DISCOVERY_STATES = (
    "missing",
    "present",
    "incompatible",
    "available",
)

ENABLEMENT_STATES = (
    "disabled",
    "enabled",
)

SELECTION_STATES = (
    "not_considered",
    "eligible",
    "selected",
    "rejected",
)

FAILURE_SCOPES = (
    "invocation",
    "plan",
    "provider",
    "runtime",
)

_BACKENDS = ("cuda", "vulkan")
_LOAD_MODE_BY_DEPENDENCY_TIER = {
    "core": "built_in",
    "lazy_external": "runtime_lazy",
    "build_external": "build_only",
}


def _validate_member(field_name, value, allowed):
    if value not in allowed:
        raise ValueError(f"unsupported {field_name}: {value!r}")


def _unique_strings(field_name, values, *, nonempty=False):
    values = tuple(values)
    if nonempty and not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise TypeError(f"{field_name} must contain nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


@dataclass(frozen=True)
class HardwareOperationDescriptor:
    """Immutable deployment and execution contract for one provider route."""

    operation_id: str
    semantic_family: str
    provider_id: str
    backends: Tuple[str, ...]
    dependency_tier: str
    provider_class: str
    execution_class: str
    hardware_acceleration: str
    scopes: Tuple[str, ...]
    execution_kind: str
    graph_support: str
    stream_binding: str
    workspace_ownership: str
    implementation_status: str
    dependency_name: Optional[str] = None
    load_mode: Optional[str] = None
    resource_effects: Tuple[str, ...] = ()
    lifetime_policy: str = "implementation_defined"
    update_policy: str = "implementation_defined"
    dtypes: Tuple[str, ...] = ()
    shapes_or_tiles: Tuple[str, ...] = ()
    layouts: Tuple[str, ...] = ()
    numeric_contracts: Tuple[str, ...] = ()
    deterministic: Optional[bool] = None
    fallback_provider: Optional[str] = None
    fallback_equivalent: Optional[bool] = None
    requirements: Tuple[str, ...] = ()
    public_api: Optional[str] = None
    notes: Tuple[str, ...] = ()
    schema_version: int = HARDWARE_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self):
        for field_name in ("operation_id", "semantic_family", "provider_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{field_name} must be a nonempty string")
        if self.schema_version != HARDWARE_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("hardware operation descriptor schema version mismatch")
        backends = _unique_strings("backends", self.backends, nonempty=True)
        scopes = _unique_strings("scopes", self.scopes, nonempty=True)
        resource_effects = _unique_strings("resource_effects", self.resource_effects)
        dtypes = _unique_strings("dtypes", self.dtypes)
        shapes_or_tiles = _unique_strings("shapes_or_tiles", self.shapes_or_tiles)
        layouts = _unique_strings("layouts", self.layouts)
        numeric_contracts = _unique_strings("numeric_contracts", self.numeric_contracts)
        requirements = _unique_strings("requirements", self.requirements)
        notes = _unique_strings("notes", self.notes)
        for backend in backends:
            _validate_member("backend", backend, _BACKENDS)
        for scope in scopes:
            _validate_member("operation scope", scope, OPERATION_SCOPES)
        _validate_member("dependency tier", self.dependency_tier, DEPENDENCY_TIERS)
        load_mode = self.load_mode or _LOAD_MODE_BY_DEPENDENCY_TIER[self.dependency_tier]
        _validate_member("load mode", load_mode, LOAD_MODES)
        if load_mode != _LOAD_MODE_BY_DEPENDENCY_TIER[self.dependency_tier]:
            raise ValueError("load_mode must match dependency_tier")
        if self.dependency_name is not None and (not isinstance(self.dependency_name, str) or not self.dependency_name):
            raise TypeError("dependency_name must be None or a nonempty string")
        if self.dependency_tier == "core" and self.dependency_name is not None:
            raise ValueError("core operations must not name an external dependency")
        if self.dependency_tier != "core" and self.dependency_name is None:
            raise ValueError("external operations must name their dependency")
        _validate_member("provider class", self.provider_class, PROVIDER_CLASSES)
        _validate_member("execution class", self.execution_class, EXECUTION_CLASSES)
        _validate_member(
            "hardware acceleration level",
            self.hardware_acceleration,
            HARDWARE_ACCELERATION_LEVELS,
        )
        _validate_member("execution kind", self.execution_kind, EXECUTION_KINDS)
        _validate_member("Graph support", self.graph_support, GRAPH_SUPPORT_MODES)
        _validate_member("stream binding", self.stream_binding, STREAM_BINDINGS)
        _validate_member("workspace ownership", self.workspace_ownership, WORKSPACE_OWNERSHIP)
        _validate_member("lifetime policy", self.lifetime_policy, LIFETIME_POLICIES)
        _validate_member("update policy", self.update_policy, UPDATE_POLICIES)
        _validate_member(
            "implementation status",
            self.implementation_status,
            IMPLEMENTATION_STATUSES,
        )
        if self.implementation_status in (
            "qualification_required",
            "planned",
            "reference_only",
        ) and self.hardware_acceleration in ("guaranteed", "qualified"):
            raise ValueError(
                "unqualified implementations cannot claim guaranteed or qualified " "hardware acceleration"
            )
        if self.public_api is not None and (not isinstance(self.public_api, str) or not self.public_api):
            raise TypeError("public_api must be None or a nonempty string")
        if self.deterministic is not None and not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be None or bool")
        if self.fallback_provider is not None and (
            not isinstance(self.fallback_provider, str) or not self.fallback_provider
        ):
            raise TypeError("fallback_provider must be None or a nonempty string")
        if self.fallback_equivalent is not None and not isinstance(self.fallback_equivalent, bool):
            raise TypeError("fallback_equivalent must be None or bool")
        if self.fallback_provider is None and self.fallback_equivalent is not None:
            raise ValueError("fallback_equivalent requires an explicit fallback_provider")
        if "internal" in scopes and len(scopes) != 1:
            raise ValueError("internal scope cannot be combined with a public scope")
        if "kernel" in scopes and self.execution_kind != "kernel_intrinsic":
            raise ValueError("kernel-scoped operations must be kernel intrinsics")
        if self.execution_kind == "kernel_intrinsic" and self.graph_support != "inline":
            raise ValueError("kernel intrinsics must report graph_support='inline'")
        object.__setattr__(self, "backends", backends)
        object.__setattr__(self, "scopes", scopes)
        object.__setattr__(self, "load_mode", load_mode)
        object.__setattr__(self, "resource_effects", resource_effects)
        object.__setattr__(self, "dtypes", dtypes)
        object.__setattr__(self, "shapes_or_tiles", shapes_or_tiles)
        object.__setattr__(self, "layouts", layouts)
        object.__setattr__(self, "numeric_contracts", numeric_contracts)
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "notes", notes)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "semantic_family": self.semantic_family,
            "provider_id": self.provider_id,
            "backends": self.backends,
            "dependency_tier": self.dependency_tier,
            "dependency_name": self.dependency_name,
            "load_mode": self.load_mode,
            "provider_class": self.provider_class,
            "execution_class": self.execution_class,
            "hardware_acceleration": self.hardware_acceleration,
            "scopes": self.scopes,
            "execution_kind": self.execution_kind,
            "graph_support": self.graph_support,
            "stream_binding": self.stream_binding,
            "workspace_ownership": self.workspace_ownership,
            "resource_effects": self.resource_effects,
            "lifetime_policy": self.lifetime_policy,
            "update_policy": self.update_policy,
            "implementation_status": self.implementation_status,
            "dtypes": self.dtypes,
            "shapes_or_tiles": self.shapes_or_tiles,
            "layouts": self.layouts,
            "numeric_contracts": self.numeric_contracts,
            "deterministic": self.deterministic,
            "fallback_provider": self.fallback_provider,
            "fallback_equivalent": self.fallback_equivalent,
            "requirements": self.requirements,
            "public_api": self.public_api,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class HardwareProviderDescriptor:
    """Immutable grouping for provider routes with one deployment contract."""

    provider_id: str
    dependency_tier: str
    dependency_name: Optional[str]
    load_mode: str
    provider_class: str
    operation_ids: Tuple[str, ...]
    schema_version: int = HARDWARE_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise TypeError("provider_id must be a nonempty string")
        if self.schema_version != HARDWARE_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("hardware provider descriptor schema version mismatch")
        _validate_member("dependency tier", self.dependency_tier, DEPENDENCY_TIERS)
        _validate_member("load mode", self.load_mode, LOAD_MODES)
        if self.load_mode != _LOAD_MODE_BY_DEPENDENCY_TIER[self.dependency_tier]:
            raise ValueError("load_mode must match dependency_tier")
        if self.dependency_tier == "core" and self.dependency_name is not None:
            raise ValueError("core providers must not name an external dependency")
        if self.dependency_tier != "core" and (not isinstance(self.dependency_name, str) or not self.dependency_name):
            raise ValueError("external providers must name their dependency")
        _validate_member("provider class", self.provider_class, PROVIDER_CLASSES)
        operation_ids = _unique_strings("operation_ids", self.operation_ids, nonempty=True)
        object.__setattr__(self, "operation_ids", operation_ids)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "dependency_tier": self.dependency_tier,
            "dependency_name": self.dependency_name,
            "load_mode": self.load_mode,
            "provider_class": self.provider_class,
            "operation_ids": self.operation_ids,
        }


@dataclass(frozen=True)
class ResolvedHardwareOperation:
    """One passive resolution against the active Forge runtime, if any."""

    descriptor: HardwareOperationDescriptor
    backend: Optional[str]
    runtime_initialized: bool
    discovery: Optional[str]
    enablement: str
    selection: str
    unavailable_reason: str
    native_facts: Mapping[str, object] = field(default_factory=dict)
    provider_abi: Optional[str] = None
    provider_version: Optional[str] = None
    last_error: Optional[str] = None
    failure_scope: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.descriptor, HardwareOperationDescriptor):
            raise TypeError("descriptor must be a HardwareOperationDescriptor")
        if self.backend is not None and (not isinstance(self.backend, str) or not self.backend):
            raise TypeError("backend must be None or a nonempty string")
        if self.discovery is not None:
            _validate_member("discovery state", self.discovery, DISCOVERY_STATES)
        _validate_member("enablement state", self.enablement, ENABLEMENT_STATES)
        _validate_member("selection state", self.selection, SELECTION_STATES)
        if not isinstance(self.unavailable_reason, str) or not self.unavailable_reason:
            raise TypeError("unavailable_reason must be a nonempty string")
        if self.last_error is not None and (not isinstance(self.last_error, str) or not self.last_error):
            raise TypeError("last_error must be None or a nonempty string")
        if self.failure_scope is not None:
            _validate_member("failure scope", self.failure_scope, FAILURE_SCOPES)
        for field_name in ("provider_abi", "provider_version"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"{field_name} must be None or a nonempty string")
        facts = dict(self.native_facts)
        if any(not isinstance(name, str) or not name for name in facts):
            raise TypeError("native fact names must be nonempty strings")
        object.__setattr__(self, "native_facts", MappingProxyType(facts))

    def to_dict(self):
        result = self.descriptor.to_dict()
        result.update(
            {
                "backend": self.backend,
                "runtime_initialized": self.runtime_initialized,
                "discovery": self.discovery,
                "enablement": self.enablement,
                "selection": self.selection,
                "unavailable_reason": self.unavailable_reason,
                "provider_abi": self.provider_abi,
                "provider_version": self.provider_version,
                "native_facts": dict(self.native_facts),
                "last_error": self.last_error,
                "failure_scope": self.failure_scope,
            }
        )
        return result


@dataclass(frozen=True)
class HardwareCapabilityReport:
    """Side-effect-free snapshot of static contracts and passive runtime facts."""

    runtime_initialized: bool
    backend: Optional[str]
    compiled_backends: Mapping[str, bool]
    operations: Tuple[ResolvedHardwareOperation, ...]
    external_components_probed: bool = False
    schema_version: int = HARDWARE_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != HARDWARE_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("hardware report schema version mismatch")
        compiled_backends = dict(self.compiled_backends)
        if set(compiled_backends) != set(_BACKENDS):
            raise ValueError("compiled_backends must report CUDA and Vulkan")
        if any(not isinstance(value, bool) for value in compiled_backends.values()):
            raise TypeError("compiled backend values must be bool")
        operations = tuple(self.operations)
        if not all(isinstance(operation, ResolvedHardwareOperation) for operation in operations):
            raise TypeError("operations must contain resolved hardware operations")
        object.__setattr__(self, "compiled_backends", MappingProxyType(compiled_backends))
        object.__setattr__(self, "operations", operations)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "runtime_initialized": self.runtime_initialized,
            "backend": self.backend,
            "compiled_backends": dict(self.compiled_backends),
            "external_components_probed": self.external_components_probed,
            "operations": tuple(operation.to_dict() for operation in self.operations),
        }


def _operation(
    operation_id,
    semantic_family,
    provider_id,
    backends,
    dependency_tier,
    provider_class,
    execution_class,
    hardware_acceleration,
    scopes,
    execution_kind,
    graph_support,
    stream_binding,
    workspace_ownership,
    implementation_status,
    *,
    dependency_name=None,
    resource_effects=(),
    lifetime_policy="implementation_defined",
    update_policy="implementation_defined",
    dtypes=(),
    shapes_or_tiles=(),
    layouts=(),
    numeric_contracts=(),
    deterministic=None,
    fallback_provider=None,
    fallback_equivalent=None,
    requirements=(),
    public_api=None,
    notes=(),
):
    return HardwareOperationDescriptor(
        operation_id=operation_id,
        semantic_family=semantic_family,
        provider_id=provider_id,
        backends=tuple(backends),
        dependency_tier=dependency_tier,
        provider_class=provider_class,
        execution_class=execution_class,
        hardware_acceleration=hardware_acceleration,
        scopes=tuple(scopes),
        execution_kind=execution_kind,
        graph_support=graph_support,
        stream_binding=stream_binding,
        workspace_ownership=workspace_ownership,
        implementation_status=implementation_status,
        dependency_name=dependency_name,
        resource_effects=tuple(resource_effects),
        lifetime_policy=lifetime_policy,
        update_policy=update_policy,
        dtypes=tuple(dtypes),
        shapes_or_tiles=tuple(shapes_or_tiles),
        layouts=tuple(layouts),
        numeric_contracts=tuple(numeric_contracts),
        deterministic=deterministic,
        fallback_provider=fallback_provider,
        fallback_equivalent=fallback_equivalent,
        requirements=tuple(requirements),
        public_api=public_api,
        notes=tuple(notes),
    )


_OPERATIONS = (
    _operation(
        "raster.draw.vulkan",
        "raster.draw",
        "vulkan_raster",
        ("vulkan",),
        "core",
        "fixed_function",
        "fixed_function",
        "qualified",
        ("python", "graph"),
        "native_command",
        "recordable",
        "runtime_ordered",
        "graph_owned",
        "internal_foundation",
        resource_effects=("read:geometry", "write:color", "write:depth"),
        lifetime_policy="graph_generation",
        update_policy="rebind",
        requirements=("vulkan_graphics_pipeline",),
        public_api="ti.hardware.raster",
    ),
    _operation(
        "ray.as_build.vulkan",
        "ray.acceleration_structure",
        "vulkan_ray",
        ("vulkan",),
        "core",
        "fixed_function",
        "native_command",
        "implementation_defined",
        ("python", "graph"),
        "native_command",
        "recordable",
        "runtime_ordered",
        "graph_owned",
        "planned",
        resource_effects=(
            "read:geometry",
            "write:acceleration_structure",
            "write:scratch",
        ),
        lifetime_policy="resource_generation",
        update_policy="rebuild",
        requirements=("VK_KHR_acceleration_structure",),
        public_api="ti.hardware.ray",
    ),
    _operation(
        "ray.as_refit.vulkan",
        "ray.acceleration_structure",
        "vulkan_ray",
        ("vulkan",),
        "core",
        "fixed_function",
        "native_command",
        "implementation_defined",
        ("python", "graph"),
        "native_command",
        "recordable",
        "runtime_ordered",
        "graph_owned",
        "planned",
        resource_effects=(
            "read:geometry",
            "read_write:acceleration_structure",
            "write:scratch",
        ),
        lifetime_policy="resource_generation",
        update_policy="refit",
        requirements=("VK_KHR_acceleration_structure",),
        public_api="ti.hardware.ray",
    ),
    _operation(
        "ray.query.batch.vulkan",
        "ray.query",
        "vulkan_ray",
        ("vulkan",),
        "core",
        "fixed_function",
        "native_shader_operation",
        "implementation_defined",
        ("python", "graph"),
        "native_command",
        "recordable",
        "runtime_ordered",
        "caller_owned",
        "planned",
        resource_effects=(
            "read:acceleration_structure",
            "read:rays",
            "write:hits",
        ),
        lifetime_policy="resource_generation",
        update_policy="rebind",
        requirements=("VK_KHR_acceleration_structure", "VK_KHR_ray_query"),
        public_api="ti.hardware.ray",
    ),
    _operation(
        "ray.query.inline.vulkan",
        "ray.query",
        "vulkan_ray",
        ("vulkan",),
        "core",
        "fixed_function",
        "native_shader_operation",
        "implementation_defined",
        ("kernel",),
        "kernel_intrinsic",
        "inline",
        "runtime_ordered",
        "none",
        "planned",
        resource_effects=("read:acceleration_structure",),
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        requirements=("SPV_KHR_ray_query", "VK_KHR_ray_query"),
        public_api="ti.hardware.ray",
    ),
    _operation(
        "sampling.texture.vulkan",
        "sampling.texture",
        "vulkan_texture",
        ("vulkan",),
        "core",
        "fixed_function",
        "native_shader_operation",
        "qualified",
        ("kernel",),
        "kernel_intrinsic",
        "inline",
        "runtime_ordered",
        "none",
        "existing_public",
        resource_effects=("read:texture",),
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        requirements=("SPIR-V image and sampler operations",),
        public_api="ti.Texture",
    ),
    _operation(
        "sampling.texture.cuda",
        "sampling.texture",
        "cuda_texture",
        ("cuda",),
        "core",
        "hardware_intrinsic",
        "native_shader_operation",
        "implementation_defined",
        ("kernel",),
        "kernel_intrinsic",
        "inline",
        "current",
        "none",
        "qualification_required",
        resource_effects=("read:texture",),
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        requirements=("CUDA texture-object lowering",),
        public_api="ti.Texture",
    ),
    _operation(
        "matrix.mma.cuda",
        "matrix.mma",
        "cuda_matrix",
        ("cuda",),
        "core",
        "hardware_intrinsic",
        "hardware_instruction",
        "implementation_defined",
        ("kernel",),
        "kernel_intrinsic",
        "inline",
        "current",
        "none",
        "planned",
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        requirements=("CUDA compute capability", "CUDA PTX version"),
        public_api="ti.hardware.matrix",
    ),
    _operation(
        "matrix.mma.vulkan",
        "matrix.mma",
        "vulkan_matrix",
        ("vulkan",),
        "core",
        "hardware_intrinsic",
        "native_shader_operation",
        "implementation_defined",
        ("kernel",),
        "kernel_intrinsic",
        "inline",
        "runtime_ordered",
        "none",
        "planned",
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        requirements=("VK_KHR_cooperative_matrix",),
        public_api="ti.hardware.matrix",
    ),
    _operation(
        "interop.external_buffer.cuda_vulkan",
        "runtime.interop",
        "cuda_vulkan_interop",
        ("cuda", "vulkan"),
        "core",
        "runtime_interop",
        "native_command",
        "implementation_defined",
        ("python", "graph"),
        "native_command",
        "recordable",
        "explicit",
        "caller_owned",
        "existing_public",
        resource_effects=("read_write:external_buffer",),
        lifetime_policy="resource_generation",
        update_policy="rebind",
        requirements=(
            "matching device UUID",
            "external memory",
            "external semaphore",
        ),
        public_api="ti.interop",
    ),
    _operation(
        "linalg.gemm.cublas",
        "linalg.gemm",
        "cublas",
        ("cuda",),
        "lazy_external",
        "vendor_algorithm",
        "vendor_library",
        "implementation_defined",
        ("python", "graph"),
        "external_library",
        "opaque",
        "explicit",
        "caller_owned",
        "internal_foundation",
        dependency_name="cuBLAS",
        resource_effects=("read:inputs", "write:output"),
        lifetime_policy="call",
        update_policy="per_dispatch",
        requirements=("compatible cuBLAS shared library",),
        public_api="ti.linalg",
    ),
    _operation(
        "linalg.spmv.cusparse",
        "linalg.spmv",
        "cusparse",
        ("cuda",),
        "lazy_external",
        "vendor_algorithm",
        "vendor_library",
        "implementation_defined",
        ("python", "graph"),
        "external_library",
        "opaque",
        "explicit",
        "caller_owned",
        "internal_foundation",
        dependency_name="cuSPARSE",
        resource_effects=("read:sparse_matrix", "read:input", "write:output"),
        lifetime_policy="call",
        update_policy="per_dispatch",
        requirements=("compatible cuSPARSE shared library",),
        public_api="ti.linalg",
    ),
    _operation(
        "linalg.solve.cusolver",
        "linalg.solve",
        "cusolver",
        ("cuda",),
        "lazy_external",
        "vendor_algorithm",
        "vendor_library",
        "implementation_defined",
        ("python", "graph"),
        "external_library",
        "opaque",
        "explicit",
        "provider_owned",
        "internal_foundation",
        dependency_name="cuSOLVER",
        resource_effects=("read:system", "write:solution", "write:workspace"),
        lifetime_policy="provider_plan",
        update_policy="rebind",
        requirements=("compatible cuSOLVER shared library",),
        public_api="ti.linalg",
    ),
    _operation(
        "fft.transform.cufft",
        "fft.transform",
        "cufft",
        ("cuda",),
        "lazy_external",
        "vendor_algorithm",
        "vendor_library",
        "implementation_defined",
        ("python", "graph"),
        "external_library",
        "stream_capture",
        "explicit",
        "provider_owned",
        "planned",
        dependency_name="cuFFT",
        resource_effects=("read:input", "write:output", "write:workspace"),
        lifetime_policy="provider_plan",
        update_policy="rebuild",
        requirements=("compatible cuFFT shared library",),
        public_api="ti.algorithms",
        notes=("first version excludes callbacks, LTO, and multi-GPU plans",),
    ),
    _operation(
        "ray.query.batch.optix",
        "ray.query",
        "optix",
        ("cuda",),
        "lazy_external",
        "vendor_hardware_runtime",
        "vendor_hardware_runtime",
        "implementation_defined",
        ("python", "graph"),
        "external_library",
        "opaque",
        "explicit",
        "provider_owned",
        "planned",
        dependency_name="OptiX",
        resource_effects=("read:scene", "read:rays", "write:hits"),
        lifetime_policy="provider_plan",
        update_policy="rebuild",
        requirements=("qualified OPTIX_ABI_VERSION", "OptiX license gate"),
        public_api="ti.hardware.ray",
    ),
    _operation(
        "algorithms.primitives.cub",
        "algorithms.primitives",
        "cub_reference",
        ("cuda",),
        "build_external",
        "vendor_algorithm",
        "vendor_library",
        "none",
        ("python", "graph"),
        "external_library",
        "recordable",
        "explicit",
        "caller_owned",
        "reference_only",
        dependency_name="CCCL/CUB",
        resource_effects=("read:input", "write:output", "write:workspace"),
        lifetime_policy="call",
        update_policy="per_dispatch",
        requirements=("CUB/CCCL headers", "CUDA device compiler", "CUDART"),
        public_api="ti.algorithms",
        notes=("disabled in official wheel builds",),
    ),
    _operation(
        "internal.tile.async.cuda",
        "tile.async",
        "cuda_async_tile",
        ("cuda",),
        "core",
        "hardware_intrinsic",
        "hardware_instruction",
        "implementation_defined",
        ("internal",),
        "kernel_intrinsic",
        "inline",
        "current",
        "none",
        "planned",
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        requirements=("admitted PTX ISA", "admitted CUDA compute capability"),
        notes=("provider implementation detail; no public TMA API",),
    ),
    _operation(
        "internal.raster.mesh_shader.vulkan",
        "raster.geometry_frontend",
        "vulkan_mesh_shader",
        ("vulkan",),
        "core",
        "compute_native",
        "native_shader_operation",
        "none",
        ("internal",),
        "native_command",
        "recordable",
        "runtime_ordered",
        "graph_owned",
        "planned",
        resource_effects=("read:geometry", "write:raster_primitives"),
        lifetime_policy="graph_generation",
        update_policy="rebind",
        requirements=("VK_EXT_mesh_shader",),
        notes=("Raster provider specialization; not a public shader model",),
    ),
)

_OPERATIONS_BY_ID = MappingProxyType({operation.operation_id: operation for operation in _OPERATIONS})
if len(_OPERATIONS_BY_ID) != len(_OPERATIONS):
    raise RuntimeError("hardware operation IDs must be unique")


def _build_provider_catalog():
    grouped = {}
    for operation in _OPERATIONS:
        provider = grouped.setdefault(
            operation.provider_id,
            {
                "dependency_tier": operation.dependency_tier,
                "dependency_name": operation.dependency_name,
                "load_mode": operation.load_mode,
                "provider_class": operation.provider_class,
                "operation_ids": [],
            },
        )
        if provider["dependency_tier"] != operation.dependency_tier:
            raise RuntimeError("provider dependency tiers must be consistent")
        if provider["dependency_name"] != operation.dependency_name:
            raise RuntimeError("provider dependency names must be consistent")
        if provider["load_mode"] != operation.load_mode:
            raise RuntimeError("provider load modes must be consistent")
        if provider["provider_class"] != operation.provider_class:
            raise RuntimeError("provider classes must be consistent")
        provider["operation_ids"].append(operation.operation_id)
    return tuple(
        HardwareProviderDescriptor(
            provider_id=provider_id,
            dependency_tier=values["dependency_tier"],
            dependency_name=values["dependency_name"],
            load_mode=values["load_mode"],
            provider_class=values["provider_class"],
            operation_ids=tuple(values["operation_ids"]),
        )
        for provider_id, values in sorted(grouped.items())
    )


_PROVIDERS = _build_provider_catalog()
_PROVIDERS_BY_ID = MappingProxyType({provider.provider_id: provider for provider in _PROVIDERS})
_TRANSIENT_NATIVE_PROVIDERS = frozenset(("cublas", "cusparse", "cusolver"))


def operations():
    """Return the immutable, backend-independent operation catalog."""

    return _OPERATIONS


def capability(operation_id):
    """Return one static operation contract by its stable ID."""

    if not isinstance(operation_id, str) or not operation_id:
        raise TypeError("operation_id must be a nonempty string")
    try:
        return _OPERATIONS_BY_ID[operation_id]
    except KeyError as exc:
        raise KeyError(f"unknown hardware operation: {operation_id}") from exc


def providers():
    """Return immutable provider group descriptors without probing libraries."""

    return _PROVIDERS


def _provider(provider_id):
    if not isinstance(provider_id, str) or not provider_id:
        raise TypeError("provider_id must be a nonempty string")
    try:
        return _PROVIDERS_BY_ID[provider_id]
    except KeyError as exc:
        raise KeyError(f"unknown hardware provider: {provider_id}") from exc


def _runtime_facts():
    from taichi_forge._lib import core as _ti_core  # pylint: disable=C0415
    from taichi_forge.lang import impl  # pylint: disable=C0415

    runtime = impl.get_runtime()
    program = runtime.prog
    backend = None
    if program is not None:
        backend = _ti_core.arch_name(impl.current_cfg().arch)
    return (
        program is not None,
        backend,
        {
            "cuda": bool(_ti_core.with_cuda()),
            "vulkan": bool(_ti_core.with_vulkan()),
        },
    )


def _passive_resolution(descriptor, *, runtime_initialized, backend, compiled_backends):
    compiled = all(compiled_backends[item] for item in descriptor.backends)
    facts = {
        "probe_policy": "passive",
        "provider_backends_compiled": tuple(item for item in descriptor.backends if compiled_backends[item]),
        "external_component_probed": False,
    }

    if not compiled:
        return ResolvedHardwareOperation(
            descriptor=descriptor,
            backend=backend,
            runtime_initialized=runtime_initialized,
            discovery="missing",
            enablement=("enabled" if descriptor.dependency_tier == "core" else "disabled"),
            selection="rejected",
            unavailable_reason="backend_not_compiled",
            native_facts=facts,
        )

    if descriptor.dependency_tier == "lazy_external":
        return ResolvedHardwareOperation(
            descriptor=descriptor,
            backend=backend,
            runtime_initialized=runtime_initialized,
            discovery=None,
            enablement="disabled",
            selection="not_considered",
            unavailable_reason="external_probe_not_requested",
            native_facts=facts,
        )

    if descriptor.dependency_tier == "build_external":
        return ResolvedHardwareOperation(
            descriptor=descriptor,
            backend=backend,
            runtime_initialized=runtime_initialized,
            discovery=None,
            enablement="disabled",
            selection="not_considered",
            unavailable_reason="build_external_not_probed",
            native_facts=facts,
        )

    if not runtime_initialized:
        return ResolvedHardwareOperation(
            descriptor=descriptor,
            backend=None,
            runtime_initialized=False,
            discovery="present",
            enablement="enabled",
            selection="not_considered",
            unavailable_reason="runtime_not_initialized",
            native_facts=facts,
        )

    if backend not in descriptor.backends:
        return ResolvedHardwareOperation(
            descriptor=descriptor,
            backend=backend,
            runtime_initialized=True,
            discovery="present",
            enablement="enabled",
            selection="rejected",
            unavailable_reason="backend_not_active",
            native_facts=facts,
        )

    if descriptor.implementation_status == "existing_public":
        return ResolvedHardwareOperation(
            descriptor=descriptor,
            backend=backend,
            runtime_initialized=True,
            discovery="available",
            enablement="enabled",
            selection="eligible",
            unavailable_reason="none",
            native_facts=facts,
        )

    reason_by_status = {
        "internal_foundation": "public_operation_not_implemented",
        "qualification_required": "qualification_required",
        "planned": "implementation_planned",
        "reference_only": "reference_only",
    }
    return ResolvedHardwareOperation(
        descriptor=descriptor,
        backend=backend,
        runtime_initialized=True,
        discovery="present",
        enablement="enabled",
        selection="rejected",
        unavailable_reason=reason_by_status[descriptor.implementation_status],
        native_facts=facts,
    )


def _unimplemented_external_probe_resolution(descriptor, *, runtime_initialized, backend):
    return ResolvedHardwareOperation(
        descriptor=descriptor,
        backend=backend,
        runtime_initialized=runtime_initialized,
        discovery=None,
        enablement="disabled",
        selection="not_considered",
        unavailable_reason="native_probe_not_implemented",
        native_facts={
            "probe_policy": "explicit",
            "external_component_probed": False,
            "provider_enablement_changed": False,
            "provider_selection_changed": False,
        },
    )


def _failed_external_probe_resolution(descriptor, *, runtime_initialized, backend, error):
    return ResolvedHardwareOperation(
        descriptor=descriptor,
        backend=backend,
        runtime_initialized=runtime_initialized,
        discovery="incompatible",
        enablement="disabled",
        selection="not_considered",
        unavailable_reason="native_probe_failed",
        native_facts={
            "probe_policy": "explicit_transient_load",
            "external_component_probed": True,
            "provider_enablement_changed": False,
            "provider_selection_changed": False,
        },
        last_error=str(error) or type(error).__name__,
        failure_scope="provider",
    )


def _native_external_probe(provider_id):
    from taichi_forge._lib import core as _ti_core  # pylint: disable=C0415

    return dict(_ti_core.probe_cuda_external_library(provider_id))


def _explicit_external_probe_resolution(descriptor, *, runtime_initialized, backend, native_result):
    try:
        if native_result.get("provider_id") != descriptor.provider_id:
            raise ValueError("native probe returned a mismatched provider_id")
        native_facts = dict(native_result["native_facts"])
        external_component_probed = bool(native_result["external_component_probed"])
        native_facts["external_component_probed"] = external_component_probed
        return ResolvedHardwareOperation(
            descriptor=descriptor,
            backend=backend,
            runtime_initialized=runtime_initialized,
            discovery=native_result["discovery"],
            enablement="disabled",
            selection="not_considered",
            unavailable_reason=native_result["unavailable_reason"],
            native_facts=native_facts,
            provider_abi=native_result.get("provider_abi"),
            provider_version=native_result.get("provider_version"),
            last_error=native_result.get("last_error"),
            failure_scope=native_result.get("failure_scope"),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return _failed_external_probe_resolution(
            descriptor,
            runtime_initialized=runtime_initialized,
            backend=backend,
            error=exc,
        )


def report():
    """Return a passive report without loading or enabling optional providers."""

    runtime_initialized, backend, compiled_backends = _runtime_facts()
    return HardwareCapabilityReport(
        runtime_initialized=runtime_initialized,
        backend=backend,
        compiled_backends=compiled_backends,
        operations=tuple(
            _passive_resolution(
                descriptor,
                runtime_initialized=runtime_initialized,
                backend=backend,
                compiled_backends=compiled_backends,
            )
            for descriptor in _OPERATIONS
        ),
        external_components_probed=False,
    )


def probe(provider_id):
    """Explicitly probe one D1 provider without enabling or selecting it.

    Existing CUDA library probes use a transient native library handle and do
    not mutate the runtime provider singleton. Planned providers fail closed
    until they acquire an equally side-effect-free native probe.
    """

    provider = _provider(provider_id)
    if provider.dependency_tier != "lazy_external":
        raise ValueError("only lazy_external providers support runtime probing")

    runtime_initialized, backend, compiled_backends = _runtime_facts()
    passive = tuple(
        _passive_resolution(
            descriptor,
            runtime_initialized=runtime_initialized,
            backend=backend,
            compiled_backends=compiled_backends,
        )
        for descriptor in _OPERATIONS
    )
    provider_operations = tuple(operation for operation in _OPERATIONS if operation.provider_id == provider_id)
    provider_backends_compiled = all(
        compiled_backends[backend_name] for operation in provider_operations for backend_name in operation.backends
    )
    if not provider_backends_compiled:
        resolved_provider_operations = {
            operation.descriptor.operation_id: operation
            for operation in passive
            if operation.descriptor.provider_id == provider_id
        }
    elif provider_id not in _TRANSIENT_NATIVE_PROVIDERS:
        resolved_provider_operations = {
            descriptor.operation_id: _unimplemented_external_probe_resolution(
                descriptor,
                runtime_initialized=runtime_initialized,
                backend=backend,
            )
            for descriptor in provider_operations
        }
    else:
        try:
            native_result = _native_external_probe(provider_id)
        except Exception as exc:  # Native loader failures must remain inspectable.
            resolved_provider_operations = {
                descriptor.operation_id: _failed_external_probe_resolution(
                    descriptor,
                    runtime_initialized=runtime_initialized,
                    backend=backend,
                    error=exc,
                )
                for descriptor in provider_operations
            }
        else:
            resolved_provider_operations = {
                descriptor.operation_id: _explicit_external_probe_resolution(
                    descriptor,
                    runtime_initialized=runtime_initialized,
                    backend=backend,
                    native_result=native_result,
                )
                for descriptor in provider_operations
            }

    operations = tuple(
        resolved_provider_operations.get(operation.descriptor.operation_id, operation) for operation in passive
    )
    return HardwareCapabilityReport(
        runtime_initialized=runtime_initialized,
        backend=backend,
        compiled_backends=compiled_backends,
        operations=operations,
        external_components_probed=any(
            operation.native_facts.get("external_component_probed", False) for operation in operations
        ),
    )


__all__ = [
    "DEPENDENCY_TIERS",
    "DISCOVERY_STATES",
    "ENABLEMENT_STATES",
    "EXECUTION_CLASSES",
    "EXECUTION_KINDS",
    "FAILURE_SCOPES",
    "GRAPH_SUPPORT_MODES",
    "HARDWARE_ACCELERATION_LEVELS",
    "HARDWARE_CAPABILITY_SCHEMA_VERSION",
    "HardwareCapabilityReport",
    "HardwareOperationDescriptor",
    "HardwareProviderDescriptor",
    "IMPLEMENTATION_STATUSES",
    "LIFETIME_POLICIES",
    "LOAD_MODES",
    "OPERATION_SCOPES",
    "PROVIDER_CLASSES",
    "ResolvedHardwareOperation",
    "SELECTION_STATES",
    "STREAM_BINDINGS",
    "UPDATE_POLICIES",
    "WORKSPACE_OWNERSHIP",
    "capability",
    "operations",
    "providers",
    "probe",
    "report",
]
