"""Versioned, session-scoped providers for complete Graph recipes."""

from dataclasses import dataclass
from typing import Protocol

from taichi_forge.graph._recipes.definition import _canonical_json, _digest

RUNTIME_GRAPH_ASSEMBLY_V1 = "taichi_forge.runtime_graph_recipe.v1"
PROVIDER_OWNED_WHOLE_GRAPH_V1 = "provider_owned_whole_graph.v1"

_KNOWN_ASSEMBLY_PROTOCOLS = frozenset(
    (RUNTIME_GRAPH_ASSEMBLY_V1, PROVIDER_OWNED_WHOLE_GRAPH_V1)
)
_PROVIDER_DESCRIPTOR_SCHEMA = "taichi_forge.graph_recipe_provider.v2"
_PROVIDER_REGISTRY_SCHEMA = "taichi_forge.graph_recipe_provider_registry.v1"
_GENERATION_DOMAIN_SCHEMA = "taichi_forge.graph_recipe_generation_domain.v1"


def _required_text(value, role):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role} must be a non-empty string")
    return value


def _normalized_strings(values, role):
    values = tuple(values)
    for value in values:
        _required_text(value, role)
    if len(values) != len(set(values)):
        raise ValueError(f"{role} values must be unique")
    return tuple(sorted(values))


class GraphRecipeProviderError(ValueError):
    """Structured provider contract failure outside the replay hot path."""

    def __init__(
        self,
        message,
        *,
        error_key,
        provider_namespace="",
        fragment_key="",
    ):
        super().__init__(message)
        self.error_key = _required_text(error_key, "Graph provider error key")
        self.provider_namespace = str(provider_namespace)
        self.fragment_key = str(fragment_key)

    def to_dict(self):
        return {
            "error_key": self.error_key,
            "message": str(self),
            "provider_namespace": self.provider_namespace,
            "fragment_key": self.fragment_key,
        }


@dataclass(frozen=True)
class GraphRecipeProviderDescriptor:
    """Serializable identity and capability contract for one provider."""

    namespace: str
    provider_version: str
    domain_version: str
    semantic_fingerprint: str
    assembly_protocols: tuple[str, ...] = (RUNTIME_GRAPH_ASSEMBLY_V1,)
    capabilities: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    backend_requirements: tuple[str, ...] = ()
    owned_fragment_namespaces: tuple[str, ...] = ()
    fragment_key_schema: str = "opaque-string.v1"

    def __post_init__(self):
        for name in (
            "namespace",
            "provider_version",
            "domain_version",
            "semantic_fingerprint",
            "fragment_key_schema",
        ):
            _required_text(getattr(self, name), f"Graph provider {name}")
        for name in (
            "assembly_protocols",
            "capabilities",
            "required_capabilities",
            "backend_requirements",
            "owned_fragment_namespaces",
        ):
            object.__setattr__(
                self,
                name,
                _normalized_strings(
                    getattr(self, name),
                    f"Graph provider {name}",
                ),
            )
        if not self.assembly_protocols:
            raise ValueError("Graph provider must support an assembly protocol")
        unknown = set(self.assembly_protocols).difference(_KNOWN_ASSEMBLY_PROTOCOLS)
        if unknown:
            raise ValueError(
                "Graph provider declares unknown assembly protocols: "
                + ", ".join(sorted(unknown))
            )
        if self.namespace in self.owned_fragment_namespaces:
            raise ValueError(
                "Graph provider owned_fragment_namespaces must omit its namespace"
            )

    @property
    def provider_id(self):
        return f"graph-recipe-provider:{_digest(self.to_dict())}"

    @property
    def fragment_namespaces(self):
        return (self.namespace,) + self.owned_fragment_namespaces

    def owns_fragment_namespace(self, namespace):
        return namespace in self.fragment_namespaces

    def to_dict(self):
        return {
            "schema": _PROVIDER_DESCRIPTOR_SCHEMA,
            "namespace": self.namespace,
            "provider_version": self.provider_version,
            "domain_version": self.domain_version,
            "semantic_fingerprint": self.semantic_fingerprint,
            "assembly_protocols": self.assembly_protocols,
            "capabilities": self.capabilities,
            "required_capabilities": self.required_capabilities,
            "backend_requirements": self.backend_requirements,
            "owned_fragment_namespaces": self.owned_fragment_namespaces,
            "fragment_key_schema": self.fragment_key_schema,
        }


class GraphRecipeProvider(Protocol):
    """Runtime half of a provider; never serialized into a recipe artifact."""

    @property
    def descriptor(self) -> GraphRecipeProviderDescriptor: ...

    def discover(self, definition): ...

    def resolve(self, definition, fragment_key): ...

    def expand(self, definition, fragment_key): ...

    def materialize(self, scope, fragment): ...

    def assemble(self, scope, definition, recipe, materialized_fragments): ...

    def describe(self, definition, fragment_key): ...


class GraphRecipeProviderSet:
    """Immutable provider namespace map owned by one Graph search session."""

    def __init__(self, definition, providers, *, available_capabilities=()):
        self.definition = definition
        self.available_capabilities = frozenset(available_capabilities)
        provider_entries = []
        for provider in tuple(providers):
            descriptor = getattr(provider, "descriptor", None)
            if callable(descriptor):
                descriptor = descriptor()
            if not isinstance(descriptor, GraphRecipeProviderDescriptor):
                raise GraphRecipeProviderError(
                    "Graph recipe provider must expose a versioned descriptor",
                    error_key="provider_descriptor_missing",
                )
            provider_entries.append((descriptor, provider))

        provider_entries.sort(key=lambda item: item[0].namespace)
        namespaces = tuple(item[0].namespace for item in provider_entries)
        if len(namespaces) != len(set(namespaces)):
            duplicate = next(
                namespace
                for namespace in namespaces
                if namespaces.count(namespace) > 1
            )
            raise GraphRecipeProviderError(
                f"Graph recipe provider namespace is duplicated: {duplicate}",
                error_key="provider_namespace_duplicate",
                provider_namespace=duplicate,
            )

        fragment_owners = {}
        for descriptor, provider in provider_entries:
            if (
                descriptor.backend_requirements
                and definition.backend not in descriptor.backend_requirements
            ):
                raise GraphRecipeProviderError(
                    "Graph recipe provider is unavailable on backend "
                    + str(definition.backend),
                    error_key="provider_backend_unavailable",
                    provider_namespace=descriptor.namespace,
                )
            missing = set(descriptor.required_capabilities).difference(
                self.available_capabilities
            )
            if missing:
                raise GraphRecipeProviderError(
                    "Graph recipe provider requires unavailable capabilities: "
                    + ", ".join(sorted(missing)),
                    error_key="provider_capability_unavailable",
                    provider_namespace=descriptor.namespace,
                )
            for namespace in descriptor.fragment_namespaces:
                previous = fragment_owners.get(namespace)
                if previous is not None:
                    raise GraphRecipeProviderError(
                        "Graph fragment namespace has multiple owners: " + namespace,
                        error_key="fragment_namespace_duplicate",
                        provider_namespace=descriptor.namespace,
                    )
                fragment_owners[namespace] = (descriptor, provider)

        self._entries = tuple(provider_entries)
        self._by_provider_namespace = {
            descriptor.namespace: (descriptor, provider)
            for descriptor, provider in self._entries
        }
        self._by_fragment_namespace = fragment_owners
        registry_payload = {
            "schema": _PROVIDER_REGISTRY_SCHEMA,
            "providers": tuple(
                descriptor.to_dict() for descriptor, _provider in self._entries
            ),
        }
        self.provider_registry_id = (
            f"graph-provider-registry:{_digest(registry_payload)}"
        )
        generation_payload = {
            "schema": _GENERATION_DOMAIN_SCHEMA,
            "semantic_graph_id": definition.semantic_graph_id,
            "provider_registry_id": self.provider_registry_id,
            "backend": definition.backend,
            "available_capabilities": tuple(sorted(self.available_capabilities)),
            "composer_protocol": "complete-graph-recipe.v2",
        }
        self.generation_domain_id = (
            f"graph-generation-domain:{_digest(generation_payload)}"
        )

    @property
    def descriptors(self):
        return tuple(descriptor for descriptor, _provider in self._entries)

    @property
    def providers(self):
        return tuple(provider for _descriptor, provider in self._entries)

    def descriptor(self, namespace):
        try:
            return self._by_provider_namespace[namespace][0]
        except KeyError as error:
            raise GraphRecipeProviderError(
                "Graph recipe provider is not present in this session: " + namespace,
                error_key="provider_unavailable",
                provider_namespace=namespace,
            ) from error

    def provider_for_fragment_namespace(self, namespace):
        try:
            return self._by_fragment_namespace[namespace][1]
        except KeyError as error:
            raise GraphRecipeProviderError(
                "Graph fragment namespace is not owned by this provider set: "
                + namespace,
                error_key="fragment_provider_unavailable",
                provider_namespace=namespace,
            ) from error

    def _validate_fragment(self, descriptor, fragment):
        from taichi_forge.graph._recipes.fragments import GraphRecipeFragment

        if not isinstance(fragment, GraphRecipeFragment):
            raise GraphRecipeProviderError(
                "Graph recipe provider returned a non-fragment value",
                error_key="provider_fragment_type",
                provider_namespace=descriptor.namespace,
            )
        if not descriptor.owns_fragment_namespace(fragment.provider_namespace):
            raise GraphRecipeProviderError(
                "Graph recipe provider returned a fragment outside its namespace",
                error_key="provider_fragment_namespace",
                provider_namespace=descriptor.namespace,
            )
        if fragment.provider_version != descriptor.provider_version:
            raise GraphRecipeProviderError(
                "Graph recipe fragment provider version does not match its descriptor",
                error_key="provider_version_drift",
                provider_namespace=descriptor.namespace,
                fragment_key=fragment.fragment_key,
            )
        if fragment.provider_domain_version != descriptor.domain_version:
            raise GraphRecipeProviderError(
                "Graph recipe fragment domain version does not match its descriptor",
                error_key="provider_domain_version_drift",
                provider_namespace=descriptor.namespace,
                fragment_key=fragment.fragment_key,
            )
        if fragment.assembly_protocol not in descriptor.assembly_protocols:
            raise GraphRecipeProviderError(
                "Graph recipe fragment uses an undeclared assembly protocol",
                error_key="provider_assembly_protocol_drift",
                provider_namespace=descriptor.namespace,
                fragment_key=fragment.fragment_key,
            )
        assembly_descriptor = self.descriptor(
            fragment.assembly_provider_namespace
        )
        if fragment.assembly_protocol not in assembly_descriptor.assembly_protocols:
            raise GraphRecipeProviderError(
                "Graph recipe assembly owner does not support the protocol",
                error_key="assembly_provider_protocol_unavailable",
                provider_namespace=fragment.assembly_provider_namespace,
                fragment_key=fragment.fragment_key,
            )
        return fragment

    def validate_fragment(self, fragment):
        try:
            descriptor, _provider = self._by_fragment_namespace[
                fragment.provider_namespace
            ]
        except (AttributeError, KeyError) as error:
            namespace = getattr(fragment, "provider_namespace", "")
            raise GraphRecipeProviderError(
                "Graph fragment namespace is not owned by this provider set: "
                + namespace,
                error_key="fragment_provider_unavailable",
                provider_namespace=namespace,
                fragment_key=getattr(fragment, "fragment_key", ""),
            ) from error
        return self._validate_fragment(descriptor, fragment)

    def discover(self):
        discovered = []
        for descriptor, provider in self._entries:
            method = getattr(provider, "discover", None)
            if not callable(method):
                method = getattr(provider, "fragments", None)
            if not callable(method):
                raise GraphRecipeProviderError(
                    "Graph recipe provider has no discovery method",
                    error_key="provider_discovery_missing",
                    provider_namespace=descriptor.namespace,
                )
            fragments = tuple(method(self.definition))
            discovered.extend(
                self._validate_fragment(descriptor, fragment)
                for fragment in fragments
            )
        return tuple(discovered)

    def resolve(self, fragment):
        fragment = self.validate_fragment(fragment)
        resolved = self.resolve_key(
            fragment.provider_namespace,
            fragment.fragment_key,
        )
        if resolved.to_dict() != fragment.to_dict():
            raise GraphRecipeProviderError(
                "Graph recipe provider resolved different fragment facts",
                error_key="provider_fragment_drift",
                provider_namespace=fragment.provider_namespace,
                fragment_key=fragment.fragment_key,
            )
        return resolved

    def resolve_key(self, fragment_namespace, fragment_key):
        """Resolve one fragment from portable provider namespace/key facts."""

        descriptor, provider = self._by_fragment_namespace.get(
            fragment_namespace,
            (None, None),
        )
        if descriptor is None:
            raise GraphRecipeProviderError(
                "Graph fragment namespace is not owned by this provider set: "
                + str(fragment_namespace),
                error_key="fragment_provider_unavailable",
                provider_namespace=str(fragment_namespace),
                fragment_key=str(fragment_key),
            )
        method = getattr(provider, "resolve", None)
        if not callable(method):
            raise GraphRecipeProviderError(
                "Graph recipe provider has no stable-key resolver",
                error_key="provider_resolver_missing",
                provider_namespace=descriptor.namespace,
                fragment_key=str(fragment_key),
            )
        try:
            resolved = method(self.definition, fragment_key)
        except KeyError as error:
            raise GraphRecipeProviderError(
                "Graph recipe fragment is unavailable by its stable key",
                error_key="recipe_fragment_unavailable",
                provider_namespace=descriptor.namespace,
                fragment_key=str(fragment_key),
            ) from error
        resolved = self.validate_fragment(resolved)
        if resolved.fragment_key != fragment_key:
            raise GraphRecipeProviderError(
                "Graph recipe provider resolved a different stable fragment key",
                error_key="provider_fragment_key_drift",
                provider_namespace=descriptor.namespace,
                fragment_key=str(fragment_key),
            )
        return resolved

    def expand(self, fragment):
        fragment = self.resolve(fragment)
        provider = self.provider_for_fragment_namespace(
            fragment.provider_namespace
        )
        method = getattr(provider, "expand", None)
        if not callable(method):
            return ()
        neighbors = tuple(method(self.definition, fragment.fragment_key))
        return tuple(self.validate_fragment(item) for item in neighbors)

    def materialize(self, scope, fragment):
        fragment = self.resolve(fragment)
        provider = self.provider_for_fragment_namespace(
            fragment.provider_namespace
        )
        method = getattr(provider, "materialize", None)
        if not callable(method):
            raise GraphRecipeProviderError(
                "Graph recipe provider has no fragment materializer",
                error_key="provider_materializer_missing",
                provider_namespace=fragment.provider_namespace,
                fragment_key=fragment.fragment_key,
            )
        return method(scope, fragment)

    def assemble(self, scope, recipe, materialized_fragments):
        namespace = recipe.assembly_provider_namespace
        try:
            descriptor, provider = self._by_provider_namespace[namespace]
        except KeyError as error:
            raise GraphRecipeProviderError(
                "Graph recipe assembly provider is unavailable: " + namespace,
                error_key="assembly_provider_unavailable",
                provider_namespace=namespace,
            ) from error
        if recipe.assembly_protocol not in descriptor.assembly_protocols:
            raise GraphRecipeProviderError(
                "Graph recipe assembly provider does not support the recipe protocol",
                error_key="assembly_provider_protocol_unavailable",
                provider_namespace=namespace,
            )
        method = getattr(provider, "assemble", None)
        if not callable(method):
            raise GraphRecipeProviderError(
                "Graph recipe provider has no whole-Graph assembler",
                error_key="provider_assembler_missing",
                provider_namespace=namespace,
            )
        return method(
            scope,
            self.definition,
            recipe,
            tuple(materialized_fragments),
        )

    def describe(self, fragment):
        fragment = self.resolve(fragment)
        provider = self.provider_for_fragment_namespace(
            fragment.provider_namespace
        )
        method = getattr(provider, "describe", None)
        if not callable(method):
            return fragment.provider_metadata
        description = method(self.definition, fragment.fragment_key)
        _canonical_json(description)
        return description

    def to_dict(self):
        return {
            "schema": _PROVIDER_REGISTRY_SCHEMA,
            "provider_registry_id": self.provider_registry_id,
            "generation_domain_id": self.generation_domain_id,
            "providers": tuple(
                descriptor.to_dict() for descriptor in self.descriptors
            ),
            "available_capabilities": tuple(sorted(self.available_capabilities)),
        }


__all__ = [
    "GraphRecipeProvider",
    "GraphRecipeProviderDescriptor",
    "GraphRecipeProviderError",
    "GraphRecipeProviderSet",
    "PROVIDER_OWNED_WHOLE_GRAPH_V1",
    "RUNTIME_GRAPH_ASSEMBLY_V1",
]
