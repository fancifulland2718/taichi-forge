"""Static hardware-operation contracts and side-effect-free capability reports."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple


HARDWARE_CAPABILITY_SCHEMA_VERSION = 3

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

HARDWARE_ROUTE_LEVELS = (
    "qualified",
    "implementation_defined",
    "none",
)

PERFORMANCE_STATES = (
    "stable_positive",
    "stable_negative",
    "unstable",
    "not_measured",
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

ACTIVATION_MODES = (
    "explicit_hardware_api",
    "explicit_kernel_intrinsic",
    "domain_api_auto_provider",
    "compiler_automatic",
)

IMPLEMENTATION_STATUSES = (
    "existing_public",
    "existing_internal",
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
    activation_mode: str
    hardware_route: Optional[str] = None
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
        hardware_route = self.hardware_route
        if hardware_route is None:
            hardware_route = (
                "qualified"
                if self.hardware_acceleration in ("guaranteed", "qualified")
                else self.hardware_acceleration
            )
        _validate_member("hardware route level", hardware_route, HARDWARE_ROUTE_LEVELS)
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
        _validate_member("activation mode", self.activation_mode, ACTIVATION_MODES)
        if self.implementation_status in (
            "qualification_required",
            "planned",
            "reference_only",
        ) and (
            self.hardware_acceleration in ("guaranteed", "qualified")
            or hardware_route == "qualified"
        ):
            raise ValueError(
                "unqualified implementations cannot claim guaranteed or qualified " "hardware acceleration"
            )
        legacy_route = (
            "qualified"
            if self.hardware_acceleration in ("guaranteed", "qualified")
            else self.hardware_acceleration
        )
        if hardware_route != legacy_route:
            raise ValueError(
                "hardware_route must match the legacy hardware_acceleration field"
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
        object.__setattr__(self, "hardware_route", hardware_route)
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
            "hardware_route": self.hardware_route,
            "scopes": self.scopes,
            "execution_kind": self.execution_kind,
            "graph_support": self.graph_support,
            "stream_binding": self.stream_binding,
            "workspace_ownership": self.workspace_ownership,
            "resource_effects": self.resource_effects,
            "lifetime_policy": self.lifetime_policy,
            "update_policy": self.update_policy,
            "implementation_status": self.implementation_status,
            "activation_mode": self.activation_mode,
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
    performance_state: str = "not_measured"
    performance_scope: Mapping[str, object] = field(default_factory=dict)
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
        _validate_member("performance state", self.performance_state, PERFORMANCE_STATES)
        for field_name in ("provider_abi", "provider_version"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"{field_name} must be None or a nonempty string")
        facts = dict(self.native_facts)
        if any(not isinstance(name, str) or not name for name in facts):
            raise TypeError("native fact names must be nonempty strings")
        object.__setattr__(self, "native_facts", MappingProxyType(facts))
        performance_scope = dict(self.performance_scope)
        if any(not isinstance(name, str) or not name for name in performance_scope):
            raise TypeError("performance scope names must be nonempty strings")
        if self.performance_state == "not_measured" and performance_scope:
            raise ValueError("not_measured operations cannot carry a performance scope")
        if self.performance_state != "not_measured" and not performance_scope:
            raise ValueError("measured performance states require a performance scope")
        object.__setattr__(
            self, "performance_scope", MappingProxyType(performance_scope)
        )

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
                "performance_state": self.performance_state,
                "performance_scope": dict(self.performance_scope),
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
    activation_mode,
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
        activation_mode=activation_mode,
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
        "runtime.buffer_commands.vulkan",
        "runtime.buffer_commands",
        "vulkan_rhi",
        ("vulkan",),
        "core",
        "compute_native",
        "native_command",
        "qualified",
        ("python", "graph"),
        "native_command",
        "recordable",
        "runtime_ordered",
        "none",
        "existing_public",
        activation_mode="explicit_hardware_api",
        resource_effects=("read:source_buffers", "write:destination_buffers"),
        lifetime_policy="runtime_generation",
        update_policy="rebind",
        requirements=("Vulkan RHI command list",),
        public_api="ti.graph.VulkanBufferCommandRecording",
        notes=("low-level substrate; not a RasterPass or AS provider",),
    ),
    _operation(
        "image.copy.vulkan",
        "image.copy",
        "vulkan_rhi",
        ("vulkan",),
        "core",
        "compute_native",
        "native_command",
        "implementation_defined",
        ("python", "graph"),
        "native_command",
        "recordable",
        "runtime_ordered",
        "none",
        "existing_public",
        activation_mode="explicit_hardware_api",
        resource_effects=("read:source_image", "write:destination_image"),
        lifetime_policy="runtime_generation",
        update_policy="rebind",
        layouts=("whole color image",),
        requirements=("Vulkan RHI vkCmdCopyImage path",),
        public_api="ti.hardware.image.VulkanImageCopyRecording",
        notes=(
            "This is a native device image command, but Vulkan does not promise "
            "which physical engine executes the copy; it is not labeled "
            "dedicated-hardware acceleration.",
            "The current slice requires matching format and extent, rejects "
            "aliases and depth/stencil, and restores both images to "
            "shader-read layout.",
        ),
    ),
    _operation(
        "raster.draw.vulkan",
        "raster.draw",
        "vulkan_graphics",
        ("vulkan",),
        "core",
        "fixed_function",
        "fixed_function",
        "qualified",
        ("python", "graph"),
        "native_command",
        "recordable",
        "runtime_ordered",
        "provider_owned",
        "existing_public",
        activation_mode="explicit_hardware_api",
        resource_effects=(
            "read:geometry_and_uniform_buffers",
            "read_write:storage_buffers",
            "write_or_read_write:color_depth_attachments",
        ),
        lifetime_policy="resource_generation",
        update_policy="immutable",
        dtypes=(
            "vertex:declared BufferFormat",
            "index:u32",
            "shader:uniform_or_storage_ndarray",
            "color:attachment format",
            "depth:depth32f",
        ),
        shapes_or_tiles=("2D offscreen target",),
        layouts=(
            "declared vertex bindings and attributes",
            "optional uint32 index buffer",
            "explicit uniform/storage descriptor buffers",
            "one color and optional depth attachment with clear/load and store",
            "one or more draws recorded into one render pass",
        ),
        numeric_contracts=(
            "depth:test_and_write",
            "color:rgba_offscreen",
        ),
        deterministic=False,
        requirements=(
            "Vulkan RHI graphics pipeline and graphics queue",
            "caller-provided SPIR-V vertex and fragment shaders",
        ),
        public_api="ti.hardware.graphics.VulkanGraphicsPipeline",
        notes=(
            "Explicit direct or root-Graph native command; kernel calls are impossible.",
            "This interface owns no renderer semantics: the caller supplies shaders, raw vertex/index buffers, attachments, draw ranges, and scheduling policy.",
            "VulkanGraphicsPassRecording batches N draws into one backend pass action; the legacy single-draw API is a compatibility wrapper.",
            "Compute-to-graphics-to-compute ordering uses backend semaphores and does not add a host wait.",
        ),
    ),
    _operation(
        "raster.adapter.ggui.vulkan",
        "raster.adapter",
        "vulkan_raster",
        ("vulkan",),
        "core",
        "fixed_function",
        "fixed_function",
        "qualified",
        ("python", "graph"),
        "native_command",
        "opaque",
        "runtime_ordered",
        "provider_owned",
        "existing_public",
        activation_mode="explicit_hardware_api",
        resource_effects=("read:geometry", "write:hidden_color_depth"),
        lifetime_policy="resource_generation",
        update_policy="immutable",
        dtypes=("vertex:f32", "index:i32", "color:f32"),
        shapes_or_tiles=("2D offscreen target",),
        layouts=("mesh", "mesh_instance", "particles", "lines"),
        deterministic=False,
        requirements=(
            "Vulkan graphics pipeline",
            "GGUI built-in raster shaders",
        ),
        public_api="ti.hardware.raster.RasterPass",
        notes=(
            "Compatibility and qualification adapter only; direct Python and explicit segmented root-Graph execution are supported.",
            "Its GGUI scene semantics and hidden attachments are not the Forge low-level graphics abstraction.",
        ),
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
        ("python",),
        "native_command",
        "unsupported",
        "runtime_ordered",
        "provider_owned",
        "existing_public",
        activation_mode="explicit_hardware_api",
        resource_effects=(
            "read:geometry",
            "write:acceleration_structure",
            "write:scratch",
        ),
        lifetime_policy="resource_generation",
        update_policy="rebuild",
        requirements=("VK_KHR_acceleration_structure",),
        public_api="ti.hardware.ray.TriangleScene",
        dtypes=("vertex:f32", "index:i32"),
        layouts=("scalar (N,3)", "AOS vector-3 (N,)"),
        notes=(
            "Indexed triangles and one identity TLAS instance only.",
            "Construction records an update-enabled build; vertex-only refit is a separate public route.",
        ),
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
        "provider_owned",
        "existing_public",
        activation_mode="explicit_hardware_api",
        resource_effects=(
            "read:geometry",
            "read_write:acceleration_structure",
            "write:scratch",
        ),
        lifetime_policy="resource_generation",
        update_policy="refit",
        requirements=("VK_KHR_acceleration_structure",),
        public_api="ti.hardware.ray.TriangleScene.refit",
        dtypes=("vertex:f32",),
        layouts=("scalar (N,3)", "AOS vector-3 (N,)"),
        notes=(
            "Explicit Python or Graph native command; never selected by an ordinary kernel.",
            "Vertex-only BLAS update; vertex count and index topology remain fixed.",
            "The identity TLAS retains the stable BLAS device address.",
        ),
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
        "provider_owned",
        "existing_public",
        activation_mode="explicit_hardware_api",
        resource_effects=(
            "read:acceleration_structure",
            "read:rays",
            "write:hits",
        ),
        lifetime_policy="resource_generation",
        update_policy="rebind",
        requirements=("VK_KHR_acceleration_structure", "VK_KHR_ray_query"),
        public_api="ti.hardware.ray.TriangleScene.trace",
        dtypes=("ray:f32", "hit:f32"),
        shapes_or_tiles=("rays:(N,8)", "hits:(N,4)", "workgroup:128"),
        layouts=("scalar 2D", "AOS vector"),
        numeric_contracts=(
            "ray:[origin.xyz,tmin,direction.xyz,tmax]",
            "hit:[t,primitive_id,instance_id,hit_flag]",
            "miss:[-1,-1,-1,0]",
        ),
        notes=(
            "Explicit Python or Graph native command; never selected by an ordinary kernel.",
        ),
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
        activation_mode="explicit_kernel_intrinsic",
        resource_effects=("read:acceleration_structure",),
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        requirements=(
            "acceleration-structure kernel argument and lifetime/effect contract",
            "typed RayQuery IR and structured query control",
            "SPV_KHR_ray_query type/instruction lowering and descriptor binding",
            "VK_KHR_ray_query device feature",
        ),
        public_api="ti.hardware.ray",
        notes=(
            "The existing TriangleScene batch provider uses a separate embedded SPIR-V shader; it does not supply a kernel-visible AS value or RayQuery IR.",
            "Batch direct/root-Graph query remains the qualified route until the complete inline compiler and resource-binding chain exists.",
        ),
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
        activation_mode="explicit_kernel_intrinsic",
        resource_effects=("read:texture",),
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        dtypes=(
            "sampled:f32",
            "storage:f32",
            "storage:i32",
            "storage:u32",
        ),
        shapes_or_tiles=("1D", "2D", "3D"),
        layouts=("sampled_image", "storage_image"),
        numeric_contracts=(
            "sample_lod:explicit_lod_vec4_f32",
            "fetch:texel_vec4_f32",
            "storage_load_store:format_sampled_type",
        ),
        deterministic=False,
        requirements=(
            "SPIR-V OpImageSampleExplicitLod and OpImageFetch",
            "Vulkan combined image sampler",
        ),
        public_api="ti.Texture(..., sampler=ti.hardware.sampling.SamplerConfig(...))",
        notes=(
            "sample_lod uses immutable per-texture min/mag filter and U/V/W "
            "address modes; Vulkan sampler objects are cached per device.",
            "The current one-mip contract uses normalized coordinates and exposes no anisotropy or comparison mode.",
            "fetch uses integer texel coordinates and ignores sampler configuration.",
        ),
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
        "planned",
        activation_mode="explicit_kernel_intrinsic",
        resource_effects=("read:texture",),
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        requirements=(
            "LLVM/CUDA Program texture resource allocation and lifetime",
            "CUDA array plus texture-object creation and upload",
            "CUDA texture kernel-argument ABI",
            "LLVM TextureOpStmt lowering",
        ),
        public_api="ti.Texture",
        notes=(
            "Texture resources are currently allocated only by the GFX Program; the LLVM/CUDA Program returns a null texture allocation.",
            "LLVM CUDA TextureOp lowering is not implemented, so hardware presence alone cannot admit this route.",
        ),
    ),
    _operation(
        "kernel.atomic.cuda",
        "kernel.atomic",
        "cuda_atomic",
        ("cuda",),
        "core",
        "hardware_intrinsic",
        "hardware_instruction",
        "implementation_defined",
        ("kernel",),
        "kernel_intrinsic",
        "inline",
        "runtime_ordered",
        "none",
        "existing_public",
        activation_mode="explicit_kernel_intrinsic",
        resource_effects=("read_write:target",),
        lifetime_policy="runtime_generation",
        update_policy="per_dispatch",
        dtypes=("integer operation-dependent", "f16/f32/f64 operation-dependent"),
        numeric_contracts=("returns old value and atomically publishes the update",),
        requirements=("supported CUDA dtype/operation/device combination",),
        public_api="ti.atomic_*",
        notes=(
            "An explicit atomic call or parallel global augmented assignment is lowered inside the kernel; it is not a Python hardware command.",
            "The backend may use a native atomic instruction or an atomic-CAS implementation depending on dtype, operation, and target.",
        ),
    ),
    _operation(
        "kernel.atomic.vulkan",
        "kernel.atomic",
        "vulkan_atomic",
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
        "existing_public",
        activation_mode="explicit_kernel_intrinsic",
        resource_effects=("read_write:target",),
        lifetime_policy="runtime_generation",
        update_policy="per_dispatch",
        dtypes=("integer operation-dependent", "f32/f64 capability-dependent"),
        numeric_contracts=("returns old value and atomically publishes the update",),
        requirements=("supported SPIR-V atomic dtype/operation/device capability",),
        public_api="ti.atomic_*",
        notes=(
            "Explicit atomic semantics are lowered to SPIR-V atomic operations or capability-gated CAS implementations.",
            "This is kernel-inline lowering, not a separately dispatched native executable.",
        ),
    ),
    _operation(
        "kernel.simt.warp.cuda",
        "kernel.simt.warp",
        "cuda_warp",
        ("cuda",),
        "core",
        "hardware_intrinsic",
        "hardware_instruction",
        "implementation_defined",
        ("kernel",),
        "kernel_intrinsic",
        "inline",
        "runtime_ordered",
        "none",
        "existing_public",
        activation_mode="explicit_kernel_intrinsic",
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        dtypes=("i32", "u32 mask", "f32 shuffle"),
        shapes_or_tiles=("warp:32 lanes",),
        requirements=("CUDA warp execution", "match operations require compute capability >= 7.0"),
        public_api="ti.simt.warp",
        notes=(
            "User-written kernel intrinsics expose vote, ballot, shuffle, active-mask, match, and warp synchronization operations.",
        ),
    ),
    _operation(
        "kernel.simt.subgroup.vulkan",
        "kernel.simt.subgroup",
        "vulkan_subgroup",
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
        "existing_public",
        activation_mode="explicit_kernel_intrinsic",
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        dtypes=("i32/u32", "f32 capability-dependent"),
        shapes_or_tiles=("device subgroup size",),
        requirements=("matching Vulkan subgroup capability for each operation",),
        public_api="ti.simt.subgroup",
        notes=(
            "Qualified operations include barrier/elect, broadcast, size/id, reductions, inclusive scans, and shuffle/up/down.",
            "all/any/equal, broadcast-first, exclusive scans, and shuffle-xor fail closed because no lowering is registered.",
        ),
    ),
    _operation(
        "kernel.shared_memory.cuda_vulkan",
        "kernel.shared_memory",
        "gpu_shared_memory",
        ("cuda", "vulkan"),
        "core",
        "hardware_intrinsic",
        "hardware_instruction",
        "implementation_defined",
        ("kernel",),
        "kernel_intrinsic",
        "inline",
        "runtime_ordered",
        "none",
        "existing_public",
        activation_mode="explicit_kernel_intrinsic",
        resource_effects=("read_write:block_local_storage",),
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        dtypes=("primitive", "fixed-shape tensor element"),
        layouts=("compile-time fixed per-block SharedArray",),
        requirements=("parallel range-for scope", "device shared-memory capacity"),
        public_api="ti.simt.block.SharedArray",
        notes=(
            "The user explicitly declares block-local storage and synchronization inside a kernel; allocation is per workgroup and cannot escape its offloaded loop.",
        ),
    ),
    _operation(
        "kernel.block_local.cuda",
        "kernel.block_local_cache",
        "cuda_bls",
        ("cuda",),
        "core",
        "hardware_intrinsic",
        "compute_kernel",
        "implementation_defined",
        ("kernel",),
        "kernel_intrinsic",
        "inline",
        "runtime_ordered",
        "none",
        "existing_public",
        activation_mode="explicit_kernel_intrinsic",
        resource_effects=("read_write:block_local_cache",),
        lifetime_policy="runtime_generation",
        update_policy="per_dispatch",
        requirements=("ti.extension.bls", "supported SNode access pattern"),
        public_api="ti.block_local",
        notes=(
            "This explicit compiler hint lowers admitted field tiles into block-local storage; the current qualified slice covers supported gather/read-cache access patterns.",
            "Sparse SNode scatter/write-back automatically retains the ordinary global store/atomic path because a whole-pad BLS epilogue is not semantics-preserving.",
            "The separate internal.tile.async.cuda descriptor reports when an admitted read-only copy is further specialized to cp.async.",
        ),
    ),
    _operation(
        "internal.reduction.grouped.cuda_vulkan",
        "kernel.reduction",
        "gpu_grouped_reduction",
        ("cuda", "vulkan"),
        "core",
        "compute_native",
        "compute_kernel",
        "implementation_defined",
        ("internal",),
        "kernel_intrinsic",
        "inline",
        "runtime_ordered",
        "none",
        "existing_internal",
        activation_mode="compiler_automatic",
        resource_effects=("read:lane_values", "atomic:destination",),
        lifetime_policy="runtime_generation",
        update_policy="per_dispatch",
        requirements=("compiler-recognized reduction", "matching device group capability"),
        notes=(
            "Automatic compiler selection aggregates recognized reductions within a CUDA block or Vulkan subgroup before publishing fewer global atomics.",
            "User atomic semantics remain unchanged, and unsupported patterns retain the ordinary atomic path.",
        ),
    ),
    _operation(
        "internal.listgen.subgroup_ballot.vulkan",
        "runtime.list_generation",
        "vulkan_subgroup_listgen",
        ("vulkan",),
        "core",
        "hardware_intrinsic",
        "native_shader_operation",
        "implementation_defined",
        ("internal",),
        "kernel_intrinsic",
        "inline",
        "runtime_ordered",
        "none",
        "existing_internal",
        activation_mode="compiler_automatic",
        resource_effects=("read:active_lanes", "atomic:list_counter", "write:list_entries"),
        lifetime_policy="runtime_generation",
        update_policy="per_dispatch",
        requirements=("spirv_has_subgroup_ballot", "spirv_listgen_subgroup_ballot enabled"),
        notes=(
            "Automatic internal lowering lets one elected lane reserve a contiguous list range per subgroup and broadcasts the base to active lanes.",
            "The legacy per-active-lane atomic path remains the fail-closed fallback.",
        ),
    ),
    _operation(
        "matrix.mma.cuda",
        "matrix.mma",
        "cuda_matrix",
        ("cuda",),
        "core",
        "hardware_intrinsic",
        "hardware_instruction",
        "qualified",
        ("python", "graph"),
        "native_command",
        "recordable",
        "runtime_ordered",
        "none",
        "existing_public",
        activation_mode="explicit_hardware_api",
        resource_effects=("read:a", "read:b", "write:output"),
        lifetime_policy="runtime_generation",
        update_policy="rebind",
        dtypes=("a:f16", "b:f16", "accumulator:f32", "output:f32"),
        shapes_or_tiles=("m16n16k16", "compact batch"),
        layouts=("row_major_a", "row_major_b", "row_major_output"),
        numeric_contracts=("output=A*B", "f16_inputs:f32_accumulate"),
        deterministic=False,
        requirements=(
            "NVIDIA compute capability >= 7.0",
            "PTX ISA >= 6.3",
            "warp size 32",
            "32-byte aligned compact buffers",
        ),
        public_api="ti.hardware.matrix.mma_f16_f32",
        notes=(
            "Explicit Driver/PTX native command; ordinary ti.Matrix matmul is "
            "not rewritten and kernel calls remain unsupported.",
            "No CUDA Toolkit runtime or vendor algorithm package is required.",
        ),
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
        activation_mode="explicit_kernel_intrinsic",
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        requirements=(
            "VK_KHR_cooperative_matrix feature query and device enablement",
            "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR tuple enumeration",
            "opaque cooperative-matrix tile type and typed kernel IR",
            "SPV_KHR_cooperative_matrix load/mul-add/store lowering",
        ),
        public_api="ti.hardware.matrix",
        notes=(
            "The repository has no Vulkan cooperative-matrix feature, type, IR, or code-generation chain yet.",
            "Supported M/N/K, component types, scope, layout, and saturation must come from device properties rather than copying the CUDA m16n16k16 contract.",
        ),
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
        activation_mode="explicit_hardware_api",
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
        "recordable",
        "explicit",
        "none",
        "existing_public",
        activation_mode="explicit_hardware_api",
        dependency_name="cuBLAS",
        resource_effects=("read:inputs", "read_write:output"),
        lifetime_policy="runtime_generation",
        update_policy="per_dispatch",
        requirements=("compatible cuBLAS shared library",),
        public_api="ti.hardware.linalg.gemm_f32",
        dtypes=("input:f32", "output:f32"),
        layouts=("compact row-major 2D",),
        numeric_contracts=("C = alpha * A @ B + beta * C",),
        notes=(
            "Explicit Python or root-Graph vendor-library command.",
            "The first execution lazy-loads the user's compatible cuBLAS shared library and retains one handle per Program.",
            "Ordinary matrix multiplication and Taichi kernels are never rewritten to call this provider.",
        ),
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
        ("python",),
        "external_library",
        "unsupported",
        "runtime_ordered",
        "provider_owned",
        "existing_public",
        activation_mode="domain_api_auto_provider",
        dependency_name="cuSPARSE",
        resource_effects=("read:sparse_matrix", "read:input", "write:output"),
        lifetime_policy="resource_generation",
        update_policy="rebind",
        requirements=("compatible cuSPARSE shared library",),
        public_api="ti.linalg.SparseMatrix.__matmul__",
        dtypes=("matrix:f32", "input:f32", "output:f32"),
        layouts=("scalar CSR", "fixed-block BSR when provider supports it"),
        numeric_contracts=("output = sparse_matrix @ input",),
        notes=(
            "The public sparse-matrix domain API selects cuSPARSE automatically on CUDA; users do not call a ti.hardware command.",
            "CSR SpMV caches matrix/vector descriptors, workspace, and optional provider preprocessing for repeated calls.",
            "The current stored SparseMatrix route is direct Python execution and is not a recordable Graph action.",
        ),
    ),
    _operation(
        "linalg.spmv.cusparse_explicit",
        "linalg.spmv",
        "cusparse",
        ("cuda",),
        "lazy_external",
        "vendor_algorithm",
        "vendor_library",
        "implementation_defined",
        ("python", "graph"),
        "external_library",
        "recordable",
        "runtime_ordered",
        "provider_owned",
        "existing_public",
        activation_mode="explicit_hardware_api",
        dependency_name="cuSPARSE",
        resource_effects=("read:sparse_matrix", "read:input", "write:output"),
        lifetime_policy="resource_generation",
        update_policy="rebind",
        requirements=("compatible cuSPARSE shared library",),
        public_api="ti.hardware.linalg.spmv_f32",
        dtypes=("matrix:f32", "input:f32", "output:f32"),
        layouts=("scalar CSR", "fixed-block BSR when provider supports it"),
        numeric_contracts=("output = sparse_matrix @ input",),
        notes=(
            "Explicit Python or root-Graph command over an existing CUDA SparseMatrix.",
            "The recording holds the matrix generation alive and reuses its cuSPARSE handle, descriptors, workspace, and optional preprocessing.",
            "This manual hardware interface is separate from the automatic SparseMatrix @ ndarray domain route.",
        ),
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
        ("python",),
        "external_library",
        "unsupported",
        "runtime_ordered",
        "provider_owned",
        "existing_public",
        activation_mode="domain_api_auto_provider",
        dependency_name="cuSOLVER",
        resource_effects=("read:system", "write:solution", "write:workspace"),
        lifetime_policy="provider_plan",
        update_policy="rebind",
        requirements=(
            "compatible cuSOLVER shared library",
            "compatible cuSPARSE shared library",
        ),
        public_api="ti.linalg.SparseSolver",
        dtypes=("matrix:f32", "rhs:f32", "solution:f32"),
        layouts=("scalar CSR",),
        numeric_contracts=(
            "LLT/LDLT use the cuSOLVER sparse Cholesky path",
            "LU uses the host-assisted cuSOLVER sparse LU path",
        ),
        notes=(
            "SparseSolver selects this provider automatically when constructed on CUDA; it is not a ti.hardware command.",
            "Pattern analysis, numeric factorization, and solve are explicit domain-API stages and are not Graph-recordable.",
            "CUDA LU includes device-host transfers; the Cholesky solve remains device-resident after setup.",
        ),
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
        "recordable",
        "runtime_ordered",
        "provider_owned",
        "existing_public",
        activation_mode="explicit_hardware_api",
        dependency_name="cuFFT",
        resource_effects=("read:input", "write:output", "write:workspace"),
        lifetime_policy="provider_plan",
        update_policy="rebind",
        requirements=("compatible cuFFT shared library",),
        public_api="ti.hardware.fft.CufftPlan1D / CufftPlanND",
        dtypes=("real:f32", "complex-pair:f32"),
        shapes_or_tiles=(
            "C2C:(length,2) or (batch,length,2)",
            "R2C:real length to Hermitian length//2+1",
            "C2R:Hermitian length//2+1 to real length",
            "CufftPlanND:batched rank-2/rank-3",
        ),
        layouts=(
            "compact out-of-place C2C/R2C/C2R",
            "explicit embed/element-stride/batch-distance",
        ),
        numeric_contracts=(
            "forward sign:-1",
            "inverse sign:+1 unnormalized",
            "inverse_scale:1/length",
        ),
        notes=(
            "Callbacks, LTO, multi-GPU, and independently arbitrary per-axis strides are excluded.",
            "Identical plan descriptors reuse a runtime-generation cuFFT plan; workspace bytes are queried from cuFFT.",
        ),
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
        activation_mode="explicit_hardware_api",
        dependency_name="OptiX",
        resource_effects=("read:scene", "read:rays", "write:hits"),
        lifetime_policy="provider_plan",
        update_policy="rebuild",
        requirements=(
            "user-provided licensed OptiX SDK headers for a qualified OPTIX_ABI_VERSION",
            "lazy optixQueryFunctionTable loader with ABI isolation",
            "OptiX module/program-group/pipeline/SBT and GAS/IAS resource contracts",
            "qualified device-program build or artifact strategy",
        ),
        public_api="ti.hardware.ray",
        notes=(
            "OptiX function-table layout and initialization are SDK-header/ABI defined; a shared library name alone is not a safe provider contract.",
            "Keep this as a user-built plugin/source-build candidate until header licensing, ABI coverage, device programs, and pipeline lifetime are closed.",
        ),
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
        activation_mode="explicit_hardware_api",
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
        "qualified",
        ("internal",),
        "kernel_intrinsic",
        "inline",
        "current",
        "none",
        "existing_internal",
        activation_mode="compiler_automatic",
        dtypes=("i32", "u32", "f32", "i64", "u64", "f64"),
        shapes_or_tiles=("compiler-generated struct-for BLS >= 8192 bytes",),
        lifetime_policy="runtime_generation",
        update_policy="immutable",
        requirements=(
            "PTX ISA >= 7.0",
            "CUDA compute capability >= 8.0",
            "direct 4/8/16-byte global-to-BLS compiler copy",
            "read-only BLS with no write-back epilogue",
        ),
        notes=(
            "transparent compiler specialization; no public cp.async or TMA API",
            "smaller or non-admitted prologues retain synchronous lowering",
        ),
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
        activation_mode="compiler_automatic",
        resource_effects=("read:geometry", "write:raster_primitives"),
        lifetime_policy="graph_generation",
        update_policy="rebind",
        requirements=(
            "VK_EXT_mesh_shader feature query and device enablement",
            "SPV_EXT_mesh_shader code generation",
            "mesh pipeline and vkCmdDrawMeshTasksEXT recording",
        ),
        notes=(
            "Raster provider specialization; not a public shader model",
            "Vulkan headers alone do not constitute an implemented provider",
        ),
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
_TRANSIENT_NATIVE_PROVIDERS = frozenset(
    ("cublas", "cusparse", "cusolver", "cufft")
)


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


def _passive_core_statuses(runtime_initialized, backend):
    if not runtime_initialized or backend not in ("cuda", "vulkan"):
        return {}
    from taichi_forge.lang import impl  # pylint: disable=C0415

    program = impl.get_runtime().prog
    if backend == "cuda":
        matrix_available = bool(
            program is not None
            and program.cuda_matrix_mma_f16_f32_available()
        )
        async_tile_status = (
            dict(program._cuda_async_tile_status())
            if program is not None
            else {}
        )
        async_tile_available = bool(
            async_tile_status.get("provider_available", False)
        )
        lowered_specializations = int(
            async_tile_status.get("lowered_specializations", 0)
        )
        return {
            "matrix.mma.cuda": {
                "available": matrix_available,
                "native_facts": {
                    "provider_available": matrix_available,
                    "capability_query": "cuda_driver_compute_capability",
                    "capability_query_loads_ptx": False,
                },
            },
            "internal.tile.async.cuda": {
                "available": async_tile_available,
                "selected": lowered_specializations > 0,
                "native_facts": {
                    "provider_available": async_tile_available,
                    "capability_query": "active_cuda_codegen_target",
                    "capability_query_compiles_kernel": False,
                    "device_compute_capability": async_tile_status.get(
                        "device_compute_capability"
                    ),
                    "codegen_compute_capability": async_tile_status.get(
                        "codegen_compute_capability"
                    ),
                    "ptx_version": async_tile_status.get("ptx_version"),
                    "minimum_bls_bytes": async_tile_status.get(
                        "minimum_bls_bytes", 8192
                    ),
                    "lowered_specializations": lowered_specializations,
                    "copy_sites": int(
                        async_tile_status.get("copy_sites", 0)
                    ),
                    **{
                        name: int(async_tile_status.get(name, 0))
                        for name in (
                            "candidates",
                            "admitted",
                            "lowered",
                            "fallback",
                            "rejected",
                            "below_size",
                            "read_write_bls",
                            "unsupported_width",
                            "non_direct_address",
                            "alias_unknown",
                            "shared_memory_pressure",
                            "target_capability",
                            "cost_gate",
                        )
                    },
                    "selection_scope": "current_program_codegen",
                },
            },
        }
    graphics_available = bool(
        program is not None and program.vulkan_graphics_pipeline_available()
    )
    ray_available = bool(
        program is not None and program.vulkan_ray_query_available()
    )
    ray_facts = {
        "provider_available": ray_available,
        "capability_query": "active_vulkan_feature_chain",
        "required_features": (
            "bufferDeviceAddress",
            "accelerationStructure",
            "rayQuery",
        ),
        "capability_query_builds_acceleration_structure": False,
    }
    return {
        "runtime.buffer_commands.vulkan": {
            "available": True,
            "native_facts": {
                "provider_available": True,
                "capability_query": "active_vulkan_runtime",
                "admission_scope": "provider_route",
            },
        },
        "image.copy.vulkan": {
            "available": True,
            "native_facts": {
                "provider_available": True,
                "capability_query": "active_vulkan_runtime",
                "admission_scope": "provider_route",
                "resource_requirements_evaluated": False,
            },
        },
        "raster.draw.vulkan": {
            "available": graphics_available,
            "native_facts": {
                "provider_available": graphics_available,
                "capability_query": "active_vulkan_graphics_pipeline",
                "admission_scope": "provider_route",
                "resource_requirements_evaluated": False,
            },
        },
        "ray.as_build.vulkan": {
            "available": ray_available,
            "native_facts": ray_facts,
        },
        "ray.as_refit.vulkan": {
            "available": ray_available,
            "native_facts": ray_facts,
        },
        "ray.query.batch.vulkan": {
            "available": ray_available,
            "native_facts": ray_facts,
        },
    }


def _passive_resolution(
    descriptor,
    *,
    runtime_initialized,
    backend,
    compiled_backends,
    external_status=None,
    core_status=None,
):
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

    if (
        descriptor.dependency_tier == "lazy_external"
        and external_status is not None
        and external_status.get("library_loaded", False)
    ):
        facts.update(dict(external_status.get("native_facts", {})))
        provider_abi = external_status.get("provider_abi")
        provider_version = external_status.get("provider_version")
        if not runtime_initialized:
            return ResolvedHardwareOperation(
                descriptor=descriptor,
                backend=None,
                runtime_initialized=False,
                discovery="available",
                enablement="enabled",
                selection="not_considered",
                unavailable_reason="runtime_not_initialized",
                native_facts=facts,
                provider_abi=provider_abi,
                provider_version=provider_version,
            )
        if backend not in descriptor.backends:
            return ResolvedHardwareOperation(
                descriptor=descriptor,
                backend=backend,
                runtime_initialized=True,
                discovery="available",
                enablement="enabled",
                selection="rejected",
                unavailable_reason="backend_not_active",
                native_facts=facts,
                provider_abi=provider_abi,
                provider_version=provider_version,
            )
        return ResolvedHardwareOperation(
            descriptor=descriptor,
            backend=backend,
            runtime_initialized=True,
            discovery="available",
            enablement="enabled",
            selection="eligible",
            unavailable_reason="none",
            native_facts=facts,
            provider_abi=provider_abi,
            provider_version=provider_version,
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

    if core_status is not None:
        facts.update(dict(core_status.get("native_facts", {})))
        if not core_status.get("available", False):
            return ResolvedHardwareOperation(
                descriptor=descriptor,
                backend=backend,
                runtime_initialized=True,
                discovery="incompatible",
                enablement="enabled",
                selection="rejected",
                unavailable_reason="hardware_requirement_not_met",
                native_facts=facts,
            )

    if core_status is not None and core_status.get("selected", False):
        return ResolvedHardwareOperation(
            descriptor=descriptor,
            backend=backend,
            runtime_initialized=True,
            discovery="available",
            enablement="enabled",
            selection="selected",
            unavailable_reason="none",
            native_facts=facts,
        )

    if core_status is None and descriptor.implementation_status in (
        "existing_public",
        "existing_internal",
    ):
        facts["operation_requirements_evaluated"] = False
        return ResolvedHardwareOperation(
            descriptor=descriptor,
            backend=backend,
            runtime_initialized=True,
            discovery="present",
            enablement="enabled",
            selection="not_considered",
            unavailable_reason="operation_requirements_not_evaluated",
            native_facts=facts,
        )

    if descriptor.implementation_status in (
        "existing_public",
        "existing_internal",
    ):
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


def _native_external_status(provider_id):
    from taichi_forge._lib import core as _ti_core  # pylint: disable=C0415

    status = getattr(_ti_core, "cuda_external_library_status", None)
    if status is None:
        return {
            "provider_id": provider_id,
            "library_loaded": False,
            "provider_abi": None,
            "provider_version": None,
            "native_facts": {
                "status_policy": "passive_status_unavailable",
                "external_component_probed": False,
                "provider_enablement_changed": False,
                "provider_selection_changed": False,
            },
        }
    return dict(status(provider_id))


def _passive_external_statuses():
    return {
        provider_id: _native_external_status(provider_id)
        for provider_id in _TRANSIENT_NATIVE_PROVIDERS
    }


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
    external_statuses = _passive_external_statuses()
    core_statuses = _passive_core_statuses(runtime_initialized, backend)
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
                external_status=external_statuses.get(
                    descriptor.provider_id
                ),
                core_status=core_statuses.get(descriptor.operation_id),
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
    external_statuses = _passive_external_statuses()
    passive = tuple(
        _passive_resolution(
            descriptor,
            runtime_initialized=runtime_initialized,
            backend=backend,
            compiled_backends=compiled_backends,
            external_status=external_statuses.get(descriptor.provider_id),
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
    "HARDWARE_ROUTE_LEVELS",
    "HardwareCapabilityReport",
    "HardwareOperationDescriptor",
    "HardwareProviderDescriptor",
    "IMPLEMENTATION_STATUSES",
    "LIFETIME_POLICIES",
    "LOAD_MODES",
    "OPERATION_SCOPES",
    "PERFORMANCE_STATES",
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
