from dataclasses import dataclass
from types import MappingProxyType

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.graph._ir import NativeCallNode


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

    def bind_graph_temporaries(self, temporaries):
        """Return an action bound to one arena slot, or None to fail closed."""
        return self if not temporaries else None


class DispatchGraphAction(RecordableGraphAction):
    """Recordable action backed by precompiled Taichi dispatches."""

    def __init__(
        self,
        dispatches,
        *,
        backends=("cpu", "cuda", "vulkan"),
        conditional_body_safe=True,
        fixed_bindings=None,
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

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def dispatches(self):
        return self._dispatches

    @property
    def fixed_bindings(self):
        return self._fixed_bindings


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
