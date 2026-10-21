"""Cold definitions shared by native creation, recording and reconstruction.

Providers normalize their typed parameters before constructing these values.
This adapter only connects ports and resource references. It neither prepares
resources nor replaces a provider's direct execution/capture/lifetime policy.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from ._ir import ResourceEffect


def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Native definition parameter keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("Native definitions require finite, pointer-free parameter values")


def _parameters(value):
    if isinstance(value, Mapping):
        return {key: _parameters(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_parameters(item) for item in value]
    return value


@dataclass(frozen=True)
class NativeDefinition:
    """A normalized provider/operation/version definition, with no live state."""

    provider: str
    operation: str
    parameters: Mapping
    version: int = 1

    def __post_init__(self):
        if any(not isinstance(name, str) or not name or "\0" in name for name in (self.provider, self.operation)):
            raise ValueError("Native definitions require nonempty provider and operation names")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("Native definition versions must be positive integers")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("Native definition parameters must be a mapping")
        object.__setattr__(self, "parameters", _freeze(self.parameters))

    def parameter_dict(self):
        return _parameters(self.parameters)


@dataclass(frozen=True)
class NativeArrayBinding:
    port: str
    value: object
    access: object

    @property
    def runtime_bound(self):
        return isinstance(self.value, str)


@dataclass(frozen=True)
class NativeResourceReference:
    """An already resolved resource symbol, owned by the lowering context.

    Snapshot/asset adapters may need to publish data before they can provide a
    resource. A reference connects that result without repeating its factory.
    An optional port effect retains the recording's logical resource name and
    precise subresource scope even when lowering relocates the resource symbol.
    """

    name: str
    effect: ResourceEffect | None = None
    arrays = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or "\0" in self.name:
            raise ValueError("Native resource references require a nonempty symbol without NUL")
        if self.effect is not None and not isinstance(self.effect, ResourceEffect):
            raise TypeError("Native resource port effects require a ResourceEffect")


@dataclass(frozen=True)
class NativeResourceBinding:
    """A live owner and its reconstruction node in the context's resource DAG.

    Child resources are resolved references. Their snapshot/asset policy stays
    with the provider adapter; lowering never recursively rebuilds a child.
    """

    owner: object
    definition: NativeDefinition
    arrays: tuple = ()
    resources: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "arrays", tuple(self.arrays))
        object.__setattr__(self, "resources", tuple(self.resources))
        ports = [item.port for item in self.arrays] + [port for port, _ in self.resources]
        if len(set(ports)) != len(ports):
            raise ValueError("Native resource input ports must be distinct")
        if any(not isinstance(resource, NativeResourceReference) for _, resource in self.resources):
            raise TypeError("Native resource dependencies require resolved context references")

    def lower(self, context, *, requirements=None):
        definition = self.definition
        parameters = definition.parameter_dict()
        extra = requirements or {}
        if parameters.keys() & extra.keys():
            raise ValueError("Preparation requirements cannot replace a frozen native parameter")
        return context.resource(
            self.owner,
            definition.provider,
            definition.operation,
            parameters=dict(parameters, **extra),
            bindings=dict(_array_symbols(self.arrays, context), **{p: r.name for p, r in self.resources}),
            version=definition.version,
        )


def _array_symbols(arrays, context):
    return {
        item.port: item.value if item.runtime_bound else context.retained_arrays.require(item.value) for item in arrays
    }


@dataclass(frozen=True)
class NativeOperationBinding:
    """Default recording adapter; resolve symbols once during cold lowering."""

    definition: NativeDefinition
    arrays: tuple
    resources: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "arrays", tuple(self.arrays))
        object.__setattr__(self, "resources", tuple(self.resources))
        ports = [item.port for item in self.arrays] + [port for port, _ in self.resources]
        if len(set(ports)) != len(ports):
            raise ValueError("Native definition ports must be distinct")

    @property
    def binding_names(self):
        return tuple(item.value for item in self._array_bindings if item.runtime_bound) + tuple(
            effect.resource for effect in self._resource_effects if effect.runtime_bound
        )

    @property
    def _resource_effects(self):
        return tuple(
            resource.effect
            for _, resource in self.resources
            if isinstance(resource, NativeResourceReference) and resource.effect is not None
        )

    @property
    def _array_bindings(self):
        return self.arrays + tuple(item for _, resource in self.resources for item in resource.arrays)

    @property
    def resource_effects(self):
        return (
            tuple(
                ResourceEffect(item.value, item.access, runtime_bound=item.runtime_bound)
                for item in self._array_bindings
            )
            + self._resource_effects
        )

    def lower(self, context, *, resource_requirements=None):
        # The context decides how fixed arrays and resource owners are referenced;
        # this adapter has no AOT Module, Program or backend executor dependency.
        requirements = resource_requirements or {}
        if requirements.keys() - {port for port, _ in self.resources}:
            raise ValueError("Native resource requirements reference an unknown port")
        bindings = _array_symbols(self.arrays, context)
        for port, resource in self.resources:
            if isinstance(resource, NativeResourceReference):
                if requirements.get(port):
                    raise ValueError("Resolved resources cannot accept new preparation requirements")
                bindings[port] = resource.name
                continue
            bindings[port] = resource.lower(context, requirements=requirements.get(port))
        definition = self.definition
        context.native(
            definition.provider,
            definition.operation,
            bindings=bindings,
            parameters=definition.parameter_dict(),
            version=definition.version,
        )
