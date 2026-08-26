"""Static hardware-operation contracts and side-effect-free capability reports."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from taichi_forge.hardware._capability_catalog_kernel import (
    kernel_internal_operations,
    kernel_intrinsic_operations,
)
from taichi_forge.hardware._capability_catalog_providers import (
    d1_provider_operations,
    reference_provider_operations,
)
from taichi_forge.hardware._capability_catalog_ray import (
    ray_command_operations,
    ray_optional_operations,
)
from taichi_forge.hardware._capability_catalog_vulkan import (
    vulkan_command_operations,
    vulkan_future_operations,
    vulkan_interop_operations,
)
from taichi_forge.hardware._external_providers import (
    external_provider_ids,
    external_provider_spec,
    passive_external_provider_status,
    probe_external_provider,
)


HARDWARE_CAPABILITY_SCHEMA_VERSION = 4

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

GRAPH_INTEGRATION_MODES = (
    "inline",
    "root_ordered",
    "backend_recorded",
    "stream_captured",
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
    """Diagnostic deployment and execution contract for one provider route.

    This expert-facing descriptor intentionally carries planner and provider
    details.  Ordinary availability checks should use ``HardwareCapability``.
    """

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
    graph_integration: str
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
        load_mode = (
            self.load_mode or _LOAD_MODE_BY_DEPENDENCY_TIER[self.dependency_tier]
        )
        _validate_member("load mode", load_mode, LOAD_MODES)
        if load_mode != _LOAD_MODE_BY_DEPENDENCY_TIER[self.dependency_tier]:
            raise ValueError("load_mode must match dependency_tier")
        if self.dependency_name is not None and (
            not isinstance(self.dependency_name, str) or not self.dependency_name
        ):
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
        _validate_member(
            "Graph integration", self.graph_integration, GRAPH_INTEGRATION_MODES
        )
        _validate_member("stream binding", self.stream_binding, STREAM_BINDINGS)
        _validate_member(
            "workspace ownership", self.workspace_ownership, WORKSPACE_OWNERSHIP
        )
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
                "unqualified implementations cannot claim guaranteed or qualified "
                "hardware acceleration"
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
        if self.public_api is not None and (
            not isinstance(self.public_api, str) or not self.public_api
        ):
            raise TypeError("public_api must be None or a nonempty string")
        if self.deterministic is not None and not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be None or bool")
        if self.fallback_provider is not None and (
            not isinstance(self.fallback_provider, str) or not self.fallback_provider
        ):
            raise TypeError("fallback_provider must be None or a nonempty string")
        if self.fallback_equivalent is not None and not isinstance(
            self.fallback_equivalent, bool
        ):
            raise TypeError("fallback_equivalent must be None or bool")
        if self.fallback_provider is None and self.fallback_equivalent is not None:
            raise ValueError(
                "fallback_equivalent requires an explicit fallback_provider"
            )
        if "internal" in scopes and len(scopes) != 1:
            raise ValueError("internal scope cannot be combined with a public scope")
        if "kernel" in scopes and self.execution_kind != "kernel_intrinsic":
            raise ValueError("kernel-scoped operations must be kernel intrinsics")
        if (
            self.execution_kind == "kernel_intrinsic"
            and self.graph_integration != "inline"
        ):
            raise ValueError("kernel intrinsics must report graph_integration='inline'")
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
            "graph_integration": self.graph_integration,
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
        if self.dependency_tier != "core" and (
            not isinstance(self.dependency_name, str) or not self.dependency_name
        ):
            raise ValueError("external providers must name their dependency")
        _validate_member("provider class", self.provider_class, PROVIDER_CLASSES)
        operation_ids = _unique_strings(
            "operation_ids", self.operation_ids, nonempty=True
        )
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
    """Diagnostic passive resolution against the active Forge runtime, if any."""

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
        if self.backend is not None and (
            not isinstance(self.backend, str) or not self.backend
        ):
            raise TypeError("backend must be None or a nonempty string")
        if self.discovery is not None:
            _validate_member("discovery state", self.discovery, DISCOVERY_STATES)
        _validate_member("enablement state", self.enablement, ENABLEMENT_STATES)
        _validate_member("selection state", self.selection, SELECTION_STATES)
        if not isinstance(self.unavailable_reason, str) or not self.unavailable_reason:
            raise TypeError("unavailable_reason must be a nonempty string")
        if self.last_error is not None and (
            not isinstance(self.last_error, str) or not self.last_error
        ):
            raise TypeError("last_error must be None or a nonempty string")
        if self.failure_scope is not None:
            _validate_member("failure scope", self.failure_scope, FAILURE_SCOPES)
        _validate_member(
            "performance state", self.performance_state, PERFORMANCE_STATES
        )
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

    def to_capability(self):
        """Return the small stable status view for ordinary callers."""

        available = (
            self.discovery == "available"
            and self.enablement == "enabled"
            and self.selection != "rejected"
        )
        return HardwareCapability(
            operation_id=self.descriptor.operation_id,
            available=available,
            backend=self.backend,
            selected_provider=(
                self.descriptor.provider_id if self.selection == "selected" else None
            ),
            route=self.descriptor.hardware_route,
            reason="none" if available else self.unavailable_reason,
        )


@dataclass(frozen=True)
class HardwareCapability:
    """Stable, compact availability and route status for one operation."""

    operation_id: str
    available: bool
    backend: Optional[str]
    selected_provider: Optional[str]
    route: str
    reason: str

    def __post_init__(self):
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise TypeError("operation_id must be a nonempty string")
        if not isinstance(self.available, bool):
            raise TypeError("available must be bool")
        if self.backend is not None and (
            not isinstance(self.backend, str) or not self.backend
        ):
            raise TypeError("backend must be None or a nonempty string")
        if self.selected_provider is not None and (
            not isinstance(self.selected_provider, str) or not self.selected_provider
        ):
            raise TypeError("selected_provider must be None or a nonempty string")
        _validate_member("hardware route level", self.route, HARDWARE_ROUTE_LEVELS)
        if not isinstance(self.reason, str) or not self.reason:
            raise TypeError("reason must be a nonempty string")
        if self.available and self.reason != "none":
            raise ValueError("available capabilities must report reason='none'")

    def to_dict(self):
        return {
            "operation_id": self.operation_id,
            "available": self.available,
            "backend": self.backend,
            "selected_provider": self.selected_provider,
            "route": self.route,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HardwareProviderStatus:
    """Stable, compact status for one provider family."""

    provider_id: str
    available: bool
    backend: Optional[str]
    selected: bool
    reason: str

    def __post_init__(self):
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise TypeError("provider_id must be a nonempty string")
        if not isinstance(self.available, bool) or not isinstance(self.selected, bool):
            raise TypeError("available and selected must be bool")
        if self.selected and not self.available:
            raise ValueError("selected providers must be available")
        if self.backend is not None and (
            not isinstance(self.backend, str) or not self.backend
        ):
            raise TypeError("backend must be None or a nonempty string")
        if not isinstance(self.reason, str) or not self.reason:
            raise TypeError("reason must be a nonempty string")
        if self.available and self.reason != "none":
            raise ValueError("available providers must report reason='none'")

    def to_dict(self):
        return {
            "provider_id": self.provider_id,
            "available": self.available,
            "backend": self.backend,
            "selected": self.selected,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HardwareExecutionReport:
    """Stable runtime snapshot without diagnostic planner internals."""

    runtime_initialized: bool
    backend: Optional[str]
    capabilities: Tuple[HardwareCapability, ...]
    providers: Tuple[HardwareProviderStatus, ...]

    def __post_init__(self):
        if not isinstance(self.runtime_initialized, bool):
            raise TypeError("runtime_initialized must be bool")
        if self.backend is not None and (
            not isinstance(self.backend, str) or not self.backend
        ):
            raise TypeError("backend must be None or a nonempty string")
        capabilities = tuple(self.capabilities)
        providers = tuple(self.providers)
        if not all(isinstance(item, HardwareCapability) for item in capabilities):
            raise TypeError("capabilities must contain HardwareCapability values")
        if not all(isinstance(item, HardwareProviderStatus) for item in providers):
            raise TypeError("providers must contain HardwareProviderStatus values")
        if len({item.operation_id for item in capabilities}) != len(capabilities):
            raise ValueError("capability operation IDs must be unique")
        if len({item.provider_id for item in providers}) != len(providers):
            raise ValueError("provider status IDs must be unique")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "providers", providers)

    def capability(self, operation_id):
        if not isinstance(operation_id, str) or not operation_id:
            raise TypeError("operation_id must be a nonempty string")
        for item in self.capabilities:
            if item.operation_id == operation_id:
                return item
        raise KeyError(f"unknown hardware operation: {operation_id}")

    def provider(self, provider_id):
        if not isinstance(provider_id, str) or not provider_id:
            raise TypeError("provider_id must be a nonempty string")
        for item in self.providers:
            if item.provider_id == provider_id:
                return item
        raise KeyError(f"unknown hardware provider: {provider_id}")

    def to_dict(self):
        return {
            "runtime_initialized": self.runtime_initialized,
            "backend": self.backend,
            "capabilities": tuple(item.to_dict() for item in self.capabilities),
            "providers": tuple(item.to_dict() for item in self.providers),
        }


@dataclass(frozen=True)
class HardwareCapabilityReport:
    """Diagnostic snapshot of static contracts and passive runtime facts."""

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
        if not all(
            isinstance(operation, ResolvedHardwareOperation) for operation in operations
        ):
            raise TypeError("operations must contain resolved hardware operations")
        object.__setattr__(
            self, "compiled_backends", MappingProxyType(compiled_backends)
        )
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

    def to_execution_report(self):
        """Project diagnostic operations into the stable user-facing layer."""

        capabilities = tuple(operation.to_capability() for operation in self.operations)
        grouped = {}
        for operation, capability in zip(self.operations, capabilities):
            grouped.setdefault(operation.descriptor.provider_id, []).append(capability)
        provider_statuses = []
        for provider_id, provider_capabilities in sorted(grouped.items()):
            available = any(item.available for item in provider_capabilities)
            selected = any(
                item.selected_provider == provider_id for item in provider_capabilities
            )
            if available:
                reason = "none"
            else:
                reasons = tuple(
                    dict.fromkeys(item.reason for item in provider_capabilities)
                )
                reason = reasons[0] if len(reasons) == 1 else "no_operation_available"
            provider_statuses.append(
                HardwareProviderStatus(
                    provider_id=provider_id,
                    available=available,
                    backend=self.backend,
                    selected=selected,
                    reason=reason,
                )
            )
        return HardwareExecutionReport(
            runtime_initialized=self.runtime_initialized,
            backend=self.backend,
            capabilities=capabilities,
            providers=tuple(provider_statuses),
        )


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
    graph_integration,
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
        graph_integration=graph_integration,
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
    *vulkan_command_operations(_operation),
    *ray_command_operations(_operation),
    *kernel_intrinsic_operations(_operation),
    *vulkan_interop_operations(_operation),
    *d1_provider_operations(_operation),
    *ray_optional_operations(_operation),
    *reference_provider_operations(_operation),
    *kernel_internal_operations(_operation),
    *vulkan_future_operations(_operation),
)

_OPERATIONS_BY_ID = MappingProxyType(
    {operation.operation_id: operation for operation in _OPERATIONS}
)
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
_PROVIDERS_BY_ID = MappingProxyType(
    {provider.provider_id: provider for provider in _PROVIDERS}
)
_TRANSIENT_NATIVE_PROVIDERS = frozenset(external_provider_ids())
_CATALOG_EXTERNAL_PROVIDERS = frozenset(
    provider.provider_id
    for provider in _PROVIDERS
    if provider.dependency_tier == "lazy_external"
)
if _CATALOG_EXTERNAL_PROVIDERS != _TRANSIENT_NATIVE_PROVIDERS:
    raise RuntimeError(
        "external provider registry and capability catalog must describe "
        "the same providers"
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
            program is not None and program.cuda_matrix_mma_f16_f32_available()
        )
        async_tile_status = (
            dict(program._cuda_async_tile_status()) if program is not None else {}
        )
        async_tile_available = bool(async_tile_status.get("provider_available", False))
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
                    "copy_sites": int(async_tile_status.get("copy_sites", 0)),
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
    graphics_indirect = (
        dict(program.vulkan_graphics_indirect_capabilities())
        if graphics_available
        else {
            "fixed_count": 0,
            "multi_draw": 0,
            "first_instance": 0,
            "count_buffer": 0,
            "max_draw_count": 0,
        }
    )
    graphics_bindless_buffers = (
        dict(program.vulkan_bindless_buffer_capabilities())
        if graphics_available
        else {
            "descriptor_indexing": 0,
            "storage_buffer_non_uniform_indexing": 0,
            "fixed_count": 0,
            "partially_bound": 0,
            "update_after_bind": 0,
            "variable_count": 0,
            "runtime_array": 0,
            "update_unused_while_pending": 0,
            "max_fixed_count": 0,
            "max_update_after_bind_descriptors_in_all_pools": 0,
            "max_per_stage_update_after_bind_storage_buffers": 0,
            "max_descriptor_set_update_after_bind_storage_buffers": 0,
        }
    )
    graphics_mesh_shader = (
        dict(program.vulkan_mesh_shader_capabilities())
        if graphics_available
        else {
            "mesh_shader": 0,
            "task_shader": 0,
            "max_task_group_count_x": 0,
            "max_task_group_count_y": 0,
            "max_task_group_count_z": 0,
            "max_task_group_total_count": 0,
            "max_task_group_invocations": 0,
            "max_mesh_group_count_x": 0,
            "max_mesh_group_count_y": 0,
            "max_mesh_group_count_z": 0,
            "max_mesh_group_total_count": 0,
            "max_mesh_group_invocations": 0,
            "max_mesh_output_vertices": 0,
            "max_mesh_output_primitives": 0,
        }
    )
    ray_available = bool(program is not None and program.vulkan_ray_query_available())
    cooperative_matrix_available = bool(
        program is not None and program.vulkan_cooperative_matrix_available()
    )
    cooperative_matrix_properties = (
        tuple(dict(item) for item in program._vulkan_cooperative_matrix_properties())
        if cooperative_matrix_available
        else ()
    )
    cooperative_matrix_executable = any(
        (
            item["a_type"],
            item["b_type"],
            item["c_type"],
            item["result_type"],
        )
        == (0, 0, 1, 1)
        and item["scope"] == 3
        and not item["saturating_accumulation"]
        and item["subgroup_size"] > 0
        for item in cooperative_matrix_properties
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
                "indirect_fixed_count": bool(graphics_indirect["fixed_count"]),
                "indirect_multi_draw": bool(graphics_indirect["multi_draw"]),
                "indirect_first_instance": bool(graphics_indirect["first_instance"]),
                "indirect_count_buffer": bool(graphics_indirect["count_buffer"]),
                "max_draw_indirect_count": int(graphics_indirect["max_draw_count"]),
                "bindless_storage_buffer_fixed_count": bool(
                    graphics_bindless_buffers["fixed_count"]
                ),
                "bindless_storage_buffer_non_uniform_indexing": bool(
                    graphics_bindless_buffers["storage_buffer_non_uniform_indexing"]
                ),
                "bindless_storage_buffer_partially_bound": bool(
                    graphics_bindless_buffers["partially_bound"]
                ),
                "bindless_storage_buffer_update_after_bind": bool(
                    graphics_bindless_buffers["update_after_bind"]
                ),
                "bindless_storage_buffer_variable_count": bool(
                    graphics_bindless_buffers["variable_count"]
                ),
                "bindless_storage_buffer_runtime_array": bool(
                    graphics_bindless_buffers["runtime_array"]
                ),
                "max_bindless_storage_buffer_fixed_count": min(
                    int(graphics_bindless_buffers["max_fixed_count"]), 64
                ),
            },
        },
        "raster.mesh_tasks.vulkan": {
            "available": bool(graphics_mesh_shader["mesh_shader"]),
            "native_facts": {
                "provider_available": bool(graphics_mesh_shader["mesh_shader"]),
                "capability_query": "active_vulkan_mesh_shader_feature_chain",
                "admission_scope": "provider_route",
                "mesh_shader": bool(graphics_mesh_shader["mesh_shader"]),
                "task_shader": bool(graphics_mesh_shader["task_shader"]),
                **{
                    name: int(graphics_mesh_shader[name])
                    for name in graphics_mesh_shader
                    if name not in {"mesh_shader", "task_shader"}
                },
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
        "ray.query.inline.vulkan": {
            "available": ray_available,
            "native_facts": {
                **ray_facts,
                "scope": "jit_kernel_intrinsic",
                "aot_supported": False,
                "graph_resource_argument_supported": False,
            },
        },
        "matrix.mma.vulkan": {
            "available": cooperative_matrix_executable,
            "native_facts": {
                "provider_available": cooperative_matrix_available,
                "capability_query": (
                    "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"
                ),
                "operation_requirements_evaluated": True,
                "executable_f16_f32_tuple_count": sum(
                    (
                        item["a_type"],
                        item["b_type"],
                        item["c_type"],
                        item["result_type"],
                    )
                    == (0, 0, 1, 1)
                    and item["scope"] == 3
                    and not item["saturating_accumulation"]
                    and item["subgroup_size"] > 0
                    for item in cooperative_matrix_properties
                ),
                "supported_tuples": cooperative_matrix_properties,
            },
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
        "provider_backends_compiled": tuple(
            item for item in descriptor.backends if compiled_backends[item]
        ),
        "external_component_probed": False,
    }

    if not compiled:
        return ResolvedHardwareOperation(
            descriptor=descriptor,
            backend=backend,
            runtime_initialized=runtime_initialized,
            discovery="missing",
            enablement=(
                "enabled" if descriptor.dependency_tier == "core" else "disabled"
            ),
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


def _unimplemented_external_probe_resolution(
    descriptor, *, runtime_initialized, backend
):
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


def _failed_external_probe_resolution(
    descriptor, *, runtime_initialized, backend, error
):
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


def _native_external_probe(provider_id, library_path=None):
    return probe_external_provider(provider_id, library_path)


def _native_external_status(provider_id):
    return passive_external_provider_status(provider_id)


def _passive_external_statuses():
    return {
        provider_id: _native_external_status(provider_id)
        for provider_id in _TRANSIENT_NATIVE_PROVIDERS
    }


def _explicit_external_probe_resolution(
    descriptor, *, runtime_initialized, backend, native_result
):
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
                external_status=external_statuses.get(descriptor.provider_id),
                core_status=core_statuses.get(descriptor.operation_id),
            )
            for descriptor in _OPERATIONS
        ),
        external_components_probed=False,
    )


def execution_report():
    """Return the compact stable runtime status without diagnostic details."""

    return report().to_execution_report()


def status(operation_id):
    """Return one compact stable operation status without loading providers."""

    return execution_report().capability(operation_id)


def provider_status(provider_id):
    """Return one compact stable provider status without loading providers."""

    return execution_report().provider(provider_id)


def probe(provider_id, *, library_path=None):
    """Explicitly probe one D1 provider without enabling or selecting it.

    Existing CUDA library probes use a transient native library handle and do
    not mutate the runtime provider singleton. Planned providers fail closed
    until they acquire an equally side-effect-free native probe.
    """

    provider = _provider(provider_id)
    if provider.dependency_tier != "lazy_external":
        raise ValueError("only lazy_external providers support runtime probing")
    if library_path is not None and not external_provider_spec(
        provider_id
    ).supports_library_path:
        raise ValueError(
            "library_path is supported for cuDSS probes only among native-symbol "
            "providers; the OptiX vendor-runtime probe also accepts a path"
        )

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
    provider_operations = tuple(
        operation for operation in _OPERATIONS if operation.provider_id == provider_id
    )
    provider_backends_compiled = all(
        compiled_backends[backend_name]
        for operation in provider_operations
        for backend_name in operation.backends
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
            native_result = (
                _native_external_probe(provider_id)
                if library_path is None
                else _native_external_probe(provider_id, library_path)
            )
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
        resolved_provider_operations.get(operation.descriptor.operation_id, operation)
        for operation in passive
    )
    return HardwareCapabilityReport(
        runtime_initialized=runtime_initialized,
        backend=backend,
        compiled_backends=compiled_backends,
        operations=operations,
        external_components_probed=any(
            operation.native_facts.get("external_component_probed", False)
            for operation in operations
        ),
    )


__all__ = [
    "DEPENDENCY_TIERS",
    "DISCOVERY_STATES",
    "ENABLEMENT_STATES",
    "EXECUTION_CLASSES",
    "EXECUTION_KINDS",
    "FAILURE_SCOPES",
    "GRAPH_INTEGRATION_MODES",
    "HARDWARE_ACCELERATION_LEVELS",
    "HARDWARE_CAPABILITY_SCHEMA_VERSION",
    "HARDWARE_ROUTE_LEVELS",
    "HardwareCapability",
    "HardwareCapabilityReport",
    "HardwareExecutionReport",
    "HardwareOperationDescriptor",
    "HardwareProviderDescriptor",
    "HardwareProviderStatus",
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
    "execution_report",
    "operations",
    "providers",
    "probe",
    "provider_status",
    "report",
    "status",
]
