"""Internal identity and cost contracts for retained hardware execution.

These values deliberately remain outside the public hardware and Graph
manifests.  They are an implementation contract for provider plans and the
qualification harness, not a new user-facing operation surface.
"""

import math
from dataclasses import dataclass
from typing import Mapping, Optional

from taichi_forge.lang import impl


_FIXED_COST_SCOPES = (
    "process",
    "runtime_generation",
    "provider_generation",
    "graph_instance",
    "first_execution",
    "invocation",
)


def _nonempty_string(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _freeze(value, name):
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze(item, f"{name}.{key}"))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, name) for item in value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"{name} contains an unsupported identity value")


def _thaw(value):
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {key: _thaw(item) for key, item in value}
        return tuple(_thaw(item) for item in value)
    return value


def _identity_complete(value):
    if value is None or value == "":
        return False
    if isinstance(value, Mapping):
        return bool(value) and all(_identity_complete(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return bool(value) and all(_identity_complete(item) for item in value)
    return True


@dataclass(frozen=True)
class HardwareCostComponent:
    """One fixed or scale-dependent contributor to an operation's cost."""

    name: str
    kind: str
    amortization_scope: str
    dimensions: tuple = ()

    def __post_init__(self):
        _nonempty_string(self.name, "hardware cost component name")
        if self.kind not in ("fixed", "scale"):
            raise ValueError("hardware cost component kind must be fixed or scale")
        dimensions = tuple(self.dimensions)
        if any(not isinstance(item, str) or not item for item in dimensions):
            raise ValueError("hardware cost dimensions must be nonempty strings")
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("hardware cost dimensions must be unique")
        if self.kind == "fixed":
            if self.amortization_scope not in _FIXED_COST_SCOPES:
                raise ValueError("unsupported fixed-cost amortization scope")
            if dimensions:
                raise ValueError("fixed hardware costs cannot declare dimensions")
        else:
            if self.amortization_scope != "invocation":
                raise ValueError("scale hardware costs must be invocation scoped")
            if not dimensions:
                raise ValueError("scale hardware costs require dimensions")
        object.__setattr__(self, "dimensions", dimensions)

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "amortization_scope": self.amortization_scope,
            "dimensions": self.dimensions,
        }


@dataclass(frozen=True)
class HardwareExecutionCostModel:
    """Structural cost model; it intentionally contains no timing claims."""

    components: tuple

    def __post_init__(self):
        components = tuple(self.components)
        if not components or not all(
            isinstance(item, HardwareCostComponent) for item in components
        ):
            raise ValueError("hardware execution cost model requires cost components")
        names = tuple(item.name for item in components)
        if len(names) != len(set(names)):
            raise ValueError("hardware execution cost component names must be unique")
        if not any(item.kind == "scale" for item in components):
            raise ValueError("hardware execution cost model requires a scale cost")
        object.__setattr__(self, "components", components)

    @property
    def fixed_costs(self):
        return tuple(item for item in self.components if item.kind == "fixed")

    @property
    def scale_costs(self):
        return tuple(item for item in self.components if item.kind == "scale")

    def to_dict(self):
        return {
            "fixed_costs": tuple(item.to_dict() for item in self.fixed_costs),
            "scale_costs": tuple(item.to_dict() for item in self.scale_costs),
        }


@dataclass(frozen=True)
class RetainedPlanIdentity:
    """Complete in-memory identity for a retained provider plan or descriptor."""

    operation_id: str
    provider_id: str
    backend: str
    provider_scope: tuple
    device_scope: tuple
    problem_scope: tuple
    runtime_scope: tuple
    execution_scope: tuple
    numeric_contract_revision: int
    persistent_cache_safe: bool

    def __post_init__(self):
        _nonempty_string(self.operation_id, "retained operation_id")
        _nonempty_string(self.provider_id, "retained provider_id")
        if self.backend not in ("cpu", "cuda", "vulkan"):
            raise ValueError("unsupported retained plan backend")
        for field_name in (
            "provider_scope",
            "device_scope",
            "problem_scope",
            "runtime_scope",
            "execution_scope",
        ):
            value = tuple(getattr(self, field_name))
            if not value:
                raise ValueError(f"retained {field_name} must be nonempty")
            object.__setattr__(self, field_name, value)
        if (
            isinstance(self.numeric_contract_revision, bool)
            or not isinstance(self.numeric_contract_revision, int)
            or self.numeric_contract_revision <= 0
        ):
            raise ValueError("numeric contract revision must be positive")
        if not isinstance(self.persistent_cache_safe, bool):
            raise TypeError("persistent_cache_safe must be bool")

    @property
    def cache_key(self):
        return (
            self.operation_id,
            self.provider_id,
            self.backend,
            self.provider_scope,
            self.device_scope,
            self.problem_scope,
            self.runtime_scope,
            self.execution_scope,
            self.numeric_contract_revision,
        )

    def to_dict(self):
        return {
            "operation_id": self.operation_id,
            "provider_id": self.provider_id,
            "backend": self.backend,
            "provider_scope": _thaw(self.provider_scope),
            "device_scope": _thaw(self.device_scope),
            "problem_scope": _thaw(self.problem_scope),
            "runtime_scope": _thaw(self.runtime_scope),
            "execution_scope": _thaw(self.execution_scope),
            "numeric_contract_revision": self.numeric_contract_revision,
            "persistent_cache_safe": self.persistent_cache_safe,
        }


@dataclass(frozen=True)
class RetainedExecutionContract:
    identity: Optional[RetainedPlanIdentity]
    cost_model: HardwareExecutionCostModel
    workspace_ownership: str
    concurrency_policy: str

    def __post_init__(self):
        if self.identity is not None and not isinstance(
            self.identity, RetainedPlanIdentity
        ):
            raise TypeError("retained execution identity has the wrong type")
        if not isinstance(self.cost_model, HardwareExecutionCostModel):
            raise TypeError("retained execution cost model has the wrong type")
        if self.workspace_ownership not in (
            "none",
            "graph_temporary",
            "provider_generation",
        ):
            raise ValueError("unsupported retained workspace ownership")
        if self.concurrency_policy not in (
            "stateless",
            "independent_invocations",
            "runtime_ordered",
            "single_inflight",
        ):
            raise ValueError("unsupported retained concurrency policy")
        if self.workspace_ownership == "provider_generation" and self.identity is None:
            raise ValueError("provider-owned workspace requires a retained identity")

    def to_dict(self):
        return {
            "identity": None if self.identity is None else self.identity.to_dict(),
            "cost_model": self.cost_model.to_dict(),
            "workspace_ownership": self.workspace_ownership,
            "concurrency_policy": self.concurrency_policy,
        }


def fixed_cost(name, amortization_scope):
    return HardwareCostComponent(name, "fixed", amortization_scope)


def scale_cost(name, *dimensions):
    return HardwareCostComponent(name, "scale", "invocation", dimensions)


def make_retained_plan_identity(
    operation_id,
    provider_id,
    backend,
    *,
    provider_scope,
    problem_scope,
    execution_scope,
    numeric_contract_revision=1,
):
    """Build an identity without probing or loading an optional provider."""

    from taichi_forge._lib import core as _ti_core  # pylint: disable=C0415
    from taichi_forge._lib import utils as runtime_utils  # pylint: disable=C0415
    from taichi_forge.hardware._admission import (  # pylint: disable=C0415
        _current_cuda_device_scope,
    )

    if impl.get_runtime().prog is None:
        raise RuntimeError("retained plan identity requires an active runtime")
    device_scope = {
        "runtime_generation": int(impl.runtime_generation()),
        "backend": backend,
    }
    if backend == "cuda":
        device_scope.update(_current_cuda_device_scope())
    runtime_scope = {
        "forge_version": _ti_core.get_version_string(),
        "forge_commit": _ti_core.get_commit_hash(),
        "native_runtime_binary_candidate": getattr(
            runtime_utils, "_loaded_native_runtime_path", None
        ),
        # Content hashing belongs to explicit persistent-artifact admission.
        # Plan construction must not read a large extension/runtime binary.
        "native_runtime_binary_identity": None,
        "retained_contract_revision": 1,
    }
    frozen_provider = _freeze(provider_scope, "provider_scope")
    frozen_device = _freeze(device_scope, "device_scope")
    frozen_problem = _freeze(problem_scope, "problem_scope")
    frozen_runtime = _freeze(runtime_scope, "runtime_scope")
    frozen_execution = _freeze(execution_scope, "execution_scope")
    provider_version = provider_scope.get("provider_version")
    provider_binary_identity = provider_scope.get("provider_binary_identity")
    device_uuid = device_scope.get("cuda_device_uuid") if backend == "cuda" else None
    persistent_cache_safe = bool(
        provider_version
        and _identity_complete(provider_binary_identity)
        and runtime_scope.get("native_runtime_binary_identity")
        and (backend != "cuda" or device_uuid)
    )
    return RetainedPlanIdentity(
        operation_id=operation_id,
        provider_id=provider_id,
        backend=backend,
        provider_scope=frozen_provider,
        device_scope=frozen_device,
        problem_scope=frozen_problem,
        runtime_scope=frozen_runtime,
        execution_scope=frozen_execution,
        numeric_contract_revision=numeric_contract_revision,
        persistent_cache_safe=persistent_cache_safe,
    )


def passive_dynamic_provider_scope(provider_id, provider_abi, *, version=None):
    """Read an already-loaded dynamic provider without triggering discovery."""

    from taichi_forge.hardware._external_providers import (  # pylint: disable=C0415
        passive_external_provider_status,
    )

    status = passive_external_provider_status(provider_id)
    observed_abi = status.get("provider_abi")
    if observed_abi is not None and observed_abi != provider_abi:
        raise RuntimeError(
            f"loaded {provider_id} provider ABI does not match the recording"
        )
    native_facts = status.get("native_facts") or {}
    return {
        "provider_abi": provider_abi,
        "provider_version": status.get("provider_version") or version,
        "library_candidate": native_facts.get("library_candidate"),
        # A DLL/SONAME and reported version are sufficient for process-local
        # reuse, but not for a persistent artifact cache.
        "provider_binary_identity": None,
    }


def attach_retained_execution_contract(recording, contract):
    if not isinstance(contract, RetainedExecutionContract):
        raise TypeError("retained execution contract has the wrong type")
    if recording.workspace_ownership != contract.workspace_ownership:
        raise ValueError("recording and retained workspace ownership disagree")
    identity = contract.identity
    if identity is not None and recording.backend != identity.backend:
        raise ValueError("recording and retained plan backends disagree")
    object.__setattr__(recording, "_retained_execution_contract", contract)
    return recording


def retained_execution_contract(recording):
    """Private diagnostic accessor used by tests and qualification tooling."""

    return getattr(recording, "_retained_execution_contract", None)


def validate_retained_execution_contract(recording, lifetime_leases):
    contract = retained_execution_contract(recording)
    if contract is None:
        return
    if recording.workspace_ownership == "provider_generation" and not tuple(
        lifetime_leases
    ):
        raise ValueError("provider-owned retained workspace requires a lifetime lease")
    if (
        contract.concurrency_policy == "runtime_ordered"
        and recording.stream_binding != "runtime_ordered"
    ):
        raise ValueError("runtime-ordered retained execution requires runtime stream binding")
    identity = contract.identity
    if identity is not None:
        current = dict(_thaw(identity.device_scope)).get("runtime_generation")
        if current != int(impl.runtime_generation()):
            raise RuntimeError(
                "retained execution belongs to another runtime generation"
            )


__all__ = []
