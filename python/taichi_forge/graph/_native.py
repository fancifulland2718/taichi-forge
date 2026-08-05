from dataclasses import dataclass
from types import MappingProxyType

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.graph._ir import (
    NativeCallNode,
    ResourceEffect,
    RuntimeBinding,
    TemporaryRequirement,
)


class ProviderOwnedNdarrayBinding:
    """A non-owning Graph binding backed by a provider generation lease."""

    def __init__(self, native_array, owner):
        if native_array is None or owner is None:
            raise ValueError(
                "Provider-owned ndarray bindings require storage and an owner"
            )
        self.arr = native_array
        self._owner = owner
        self._runtime_prog = impl.get_runtime().prog
        self._runtime_allocation_identity = native_array.device_allocation().alloc_id
        self._runtime_storage_arguments = {}

    def _runtime_storage_argument(self, consumer, mode):
        program = impl.get_runtime().prog
        if program is None or program is not self._runtime_prog:
            raise TaichiRuntimeError(
                "Provider-owned ndarray belongs to another Taichi runtime"
            )
        key = (consumer, mode)
        cached = self._runtime_storage_arguments.get(key)
        if cached is not None:
            return cached
        described = _ti_core._describe_ndarray_storage(self.arr, "readwrite")
        if not described.ok:
            raise TaichiRuntimeError(
                "Cannot describe provider-owned ndarray storage: " f"{described.reason}"
            )
        argument = _ti_core._make_runtime_storage_argument(
            program, described.descriptor, consumer, mode
        )
        qualification = dict(argument.qualification)
        if not qualification["bindable"] or not qualification["replayable"]:
            raise TaichiRuntimeError(
                "Provider-owned ndarray storage is not Graph eligible: "
                f"{qualification['reason']}"
            )
        if mode == "capture" and not qualification["capturable"]:
            raise TaichiRuntimeError(
                "Provider-owned ndarray storage is not Graph-capturable: "
                f"{qualification['reason']}"
            )
        self._runtime_storage_arguments[key] = argument
        return argument


@dataclass(frozen=True)
class GraphTemporaryBuffer:
    """One named byte slice in a Graph-owned runtime allocation."""

    storage: object
    offset: int
    bytes: int
    alignment: int
    slot: int


@dataclass(frozen=True)
class RecordableActionCapabilities:
    """Backend-neutral guarantees for one recordable runtime action."""

    backends: tuple = ("cpu", "cuda", "vulkan")
    conditional_body_safe: bool = False
    address_stable: bool = True
    update_policy: str = "rebind"
    synchronization_domain: str = "runtime_ordered"

    def __post_init__(self):
        backends = tuple(self.backends)
        if len(backends) != len(set(backends)):
            raise ValueError("Recordable action backends must be unique")
        if any(backend not in ("cpu", "cuda", "vulkan") for backend in backends):
            raise ValueError("Unsupported recordable action backend")
        if self.update_policy not in ("immutable", "rebind", "rebuild"):
            raise ValueError("Unsupported recordable action update policy")
        if self.synchronization_domain not in (
            "runtime_ordered",
            "explicit_stream",
        ):
            raise ValueError("Unsupported recordable action synchronization domain")
        object.__setattr__(self, "backends", backends)

    def to_dict(self):
        return {
            "backends": self.backends,
            "conditional_body_safe": self.conditional_body_safe,
            "address_stable": self.address_stable,
            "update_policy": self.update_policy,
            "synchronization_domain": self.synchronization_domain,
        }


@dataclass(frozen=True)
class NativeActionManifest:
    """Immutable effect and lowering contract for one native Graph action.

    The manifest freezes provider declarations when a Graph is compiled. It
    intentionally contains symbolic names and counts, never provider-owned
    storage objects or runtime addresses.
    """

    schema_version: int
    name: str
    recordable: bool
    opaque: bool
    synchronization: bool
    dispatch_count: int
    backends: tuple
    conditional_body_safe: bool
    address_stable: bool
    update_policy: str
    synchronization_domain: str
    runtime_bindings: tuple
    effects: tuple
    temporaries: tuple
    fixed_binding_names: tuple
    temporary_bindings: tuple
    lifetime_lease_count: int

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "recordable": self.recordable,
            "opaque": self.opaque,
            "synchronization": self.synchronization,
            "dispatch_count": self.dispatch_count,
            "backends": self.backends,
            "conditional_body_safe": self.conditional_body_safe,
            "address_stable": self.address_stable,
            "update_policy": self.update_policy,
            "synchronization_domain": self.synchronization_domain,
            "runtime_bindings": tuple(
                binding.to_dict() for binding in self.runtime_bindings
            ),
            "effects": tuple(effect.to_dict() for effect in self.effects),
            "temporaries": tuple(
                temporary.to_dict() for temporary in self.temporaries
            ),
            "fixed_binding_names": self.fixed_binding_names,
            "temporary_bindings": self.temporary_bindings,
            "lifetime_lease_count": self.lifetime_lease_count,
        }


@dataclass(frozen=True)
class BoundedPublicationTarget:
    """Graph-owned physical target for an optional producer specialization.

    The extent name and bounds describe semantic data already owned by the
    provider.  ``packet_storage`` is backend launch scratch owned by the
    enclosing Graph instance; accepting this target never transfers that
    ownership to the provider.
    """

    backend: str
    extent_name: str
    capacity: int
    block_dim: int
    packet_binding: object
    packet_storage: object
    packet_layout: str = "dispatch_indirect_u32x4"

    def __post_init__(self):
        if self.backend not in ("cpu", "cuda", "vulkan"):
            raise ValueError("Unsupported bounded publication backend")
        if not isinstance(self.extent_name, str) or not self.extent_name:
            raise ValueError("Bounded publication extent name must be nonempty")
        if not 1 <= int(self.capacity) <= 0x7FFFFFFF:
            raise ValueError("Bounded publication capacity is out of range")
        if not 1 <= int(self.block_dim) <= 1024:
            raise ValueError("Bounded publication block dimension is out of range")
        if self.packet_binding is None or self.packet_storage is None:
            raise ValueError("Bounded publication requires Graph-owned packet state")


class RecordableGraphAction:
    """Optional provider-neutral lowering contract for a native node.

    An action describes dispatches whose semantics are identical to
    ``NativeGraphExecutable.run()``. The Graph runtime may concatenate them
    with adjacent CGraph dispatches or place them in a structured region only
    when the advertised capabilities permit it. Provider-owned fixed bindings
    are injected by the runtime and never become public Graph arguments.
    """

    @property
    def capabilities(self):
        return RecordableActionCapabilities(backends=())

    def supports_backend(self, backend):
        return backend in self.capabilities.backends

    def supports_region(self, region_kind):
        if region_kind == "sequential":
            return True
        if region_kind in ("while_body", "if_branch", "switch_branch"):
            return self.capabilities.conditional_body_safe
        return False

    @property
    def dispatches(self):
        return ()

    @property
    def fixed_bindings(self):
        return MappingProxyType({})

    @property
    def temporary_bindings(self):
        """Map private symbolic argument names to temporary requirement names."""
        return MappingProxyType({})

    @property
    def allows_unused_public_bindings(self):
        """Whether the physical recipe may consume a subset of public inputs.

        Fixed and temporary bindings must always be consumed. This opt-in is
        intended for a narrow transition action whose stable native interface
        is broader than the selected operation, not for silently incomplete
        provider lowering.
        """
        return False

    def bind_graph_temporaries(self, temporaries):
        """Resolve one arena slot into private fixed runtime bindings.

        Providers with nonempty ``temporary_bindings`` must override this and
        return a mapping whose keys exactly match the declared private symbols.
        Returning ``None`` fails closed before backend submission.
        """
        return MappingProxyType({}) if not temporaries else None


class DispatchGraphAction(RecordableGraphAction):
    """Recordable action backed by precompiled Taichi dispatches."""

    def __init__(
        self,
        dispatches,
        *,
        backends=("cpu", "cuda", "vulkan"),
        conditional_body_safe=True,
        fixed_bindings=None,
        allow_unused_public_bindings=False,
        update_policy="rebind",
        synchronization_domain="runtime_ordered",
    ):
        self._dispatches = tuple((kernel, tuple(args)) for kernel, args in dispatches)
        if not self._dispatches:
            raise ValueError("Recordable Graph action requires a dispatch")
        self._capabilities = RecordableActionCapabilities(
            backends=tuple(backends),
            conditional_body_safe=bool(conditional_body_safe),
            address_stable=True,
            update_policy=update_policy,
            synchronization_domain=synchronization_domain,
        )
        bindings = {} if fixed_bindings is None else dict(fixed_bindings)
        if any(not isinstance(name, str) or not name for name in bindings):
            raise ValueError("Recordable action fixed binding names must be nonempty")
        self._fixed_bindings = MappingProxyType(bindings)
        self._allows_unused_public_bindings = bool(
            allow_unused_public_bindings
        )

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def dispatches(self):
        return self._dispatches

    @property
    def fixed_bindings(self):
        return self._fixed_bindings

    @property
    def allows_unused_public_bindings(self):
        return self._allows_unused_public_bindings


class NativeGraphExecutable:
    """Compiled native graph node with an optional recordable action."""

    def prewarm(self):
        return self

    def run(self, runtime_args=None):
        raise NotImplementedError

    def run_with_graph_temporaries(self, temporaries, runtime_args=None):
        """Run once with named slices owned by the enclosing Graph."""
        if runtime_args is None:
            return self.run()
        return self.run(runtime_args)

    @property
    def debug_info(self):
        return {}

    @property
    def runtime_arg_schema(self):
        return ()

    @property
    def resource_effects(self):
        return ()

    @property
    def temporary_requirements(self):
        return ()

    @property
    def lifetime_leases(self):
        return ()

    @property
    def recordable_action(self):
        return None

    @property
    def recordable_sequence(self):
        """Optional structured sequence equivalent to :meth:`run`.

        Flat providers should continue to expose ``recordable_action``. A
        provider whose semantics include structured control may instead
        return a Graph ``Sequential`` definition. The Graph frontend validates
        its public bindings and inlines it at compile time; implementations
        must never submit or run a second Graph from this property.
        """
        return None

    def recordable_bounded_publication(self, target):
        """Optionally publish a semantic extent into Graph-owned launch state.

        Implementations must return an action equivalent to
        :attr:`recordable_action` for public state while additionally writing
        the requested physical packet.  Returning ``None`` keeps the normal
        producer action and lets the Graph insert a separate preparation
        dispatch.
        """
        return None

    @property
    def graph_ir_node(self):
        info = self.debug_info
        name = (
            info.get("kind", type(self).__name__)
            if isinstance(info, dict)
            else type(self).__name__
        )
        return NativeCallNode(
            name=name,
            effects=tuple(self.resource_effects),
            bindings=tuple(self.runtime_arg_schema),
            temporaries=tuple(self.temporary_requirements),
            opaque=self.recordable_action is None,
        )


_UNSET_RECORDABLE_ACTION = object()


def native_action_manifest(
    executable, recordable_action=_UNSET_RECORDABLE_ACTION
):
    """Freeze and validate one executable's symbolic native-action contract."""
    if not isinstance(executable, NativeGraphExecutable):
        raise TaichiRuntimeError(
            "Native action manifests require a NativeGraphExecutable"
        )
    action = (
        executable.recordable_action
        if recordable_action is _UNSET_RECORDABLE_ACTION
        else recordable_action
    )
    if action is not None and not isinstance(action, RecordableGraphAction):
        raise TaichiRuntimeError(
            "Native Graph recordable_action must implement RecordableGraphAction"
        )

    runtime_bindings = tuple(executable.runtime_arg_schema)
    effects = tuple(executable.resource_effects)
    temporaries = tuple(executable.temporary_requirements)
    if not all(isinstance(binding, RuntimeBinding) for binding in runtime_bindings):
        raise TaichiRuntimeError(
            "Native action runtime bindings must contain RuntimeBinding values"
        )
    if not all(isinstance(effect, ResourceEffect) for effect in effects):
        raise TaichiRuntimeError(
            "Native action effects must contain ResourceEffect values"
        )
    if not all(
        isinstance(temporary, TemporaryRequirement) for temporary in temporaries
    ):
        raise TaichiRuntimeError(
            "Native action temporaries must contain TemporaryRequirement values"
        )

    binding_names = tuple(binding.name for binding in runtime_bindings)
    temporary_names = tuple(temporary.name for temporary in temporaries)
    if len(binding_names) != len(set(binding_names)):
        raise TaichiRuntimeError("Native action runtime binding names must be unique")
    if len(temporary_names) != len(set(temporary_names)):
        raise TaichiRuntimeError("Native action temporary names must be unique")

    ir_node = executable.graph_ir_node
    name = getattr(ir_node, "name", type(executable).__name__)
    if not isinstance(name, str) or not name:
        raise TaichiRuntimeError("Native action manifest name must be non-empty")
    ir_effects = tuple(getattr(ir_node, "effects", effects))
    if not all(isinstance(effect, ResourceEffect) for effect in ir_effects):
        raise TaichiRuntimeError(
            "Native action IR effects must contain ResourceEffect values"
        )

    if action is None:
        capabilities = RecordableActionCapabilities(backends=())
        fixed_binding_names = ()
        temporary_bindings = ()
        dispatch_count = 0
        update_policy = "opaque"
        synchronization_domain = "opaque"
    else:
        capabilities = action.capabilities
        if not isinstance(capabilities, RecordableActionCapabilities):
            raise TaichiRuntimeError(
                "Recordable action capabilities must be a "
                "RecordableActionCapabilities value"
            )
        fixed_binding_names = tuple(sorted(action.fixed_bindings))
        temporary_bindings = tuple(sorted(action.temporary_bindings.items()))
        dispatch_count = len(tuple(action.dispatches))
        update_policy = capabilities.update_policy
        synchronization_domain = capabilities.synchronization_domain

    return NativeActionManifest(
        schema_version=1,
        name=name,
        recordable=action is not None,
        opaque=bool(getattr(ir_node, "opaque", action is None)),
        synchronization=bool(getattr(ir_node, "synchronization", False)),
        dispatch_count=dispatch_count,
        backends=tuple(capabilities.backends),
        conditional_body_safe=bool(capabilities.conditional_body_safe),
        address_stable=bool(capabilities.address_stable) if action else False,
        update_policy=update_policy,
        synchronization_domain=synchronization_domain,
        runtime_bindings=runtime_bindings,
        effects=ir_effects,
        temporaries=temporaries,
        fixed_binding_names=fixed_binding_names,
        temporary_bindings=temporary_bindings,
        lifetime_lease_count=len(tuple(executable.lifetime_leases)),
    )


class NativeGraphNode:
    """Definition-time native graph node."""

    dsl_defined = True

    def compile(self):
        raise NotImplementedError


def compile_native_graph_node(node):
    if isinstance(node, NativeGraphNode):
        return node.compile()
    to_graph_node = getattr(node, "_as_graph_native_node", None)
    if to_graph_node is not None:
        graph_node = to_graph_node()
        if isinstance(graph_node, NativeGraphNode):
            return graph_node.compile()
    raise TaichiRuntimeError(
        "Only DSL-defined native graph nodes are supported by GraphBuilder"
    )
