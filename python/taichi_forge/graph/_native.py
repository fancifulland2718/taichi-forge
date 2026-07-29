from dataclasses import dataclass

from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.graph._ir import NativeCallNode


@dataclass(frozen=True)
class GraphTemporaryBuffer:
    """One named byte slice in a Graph-owned runtime allocation."""

    storage: object
    offset: int
    bytes: int
    alignment: int
    slot: int


class NativeGraphBackendRecorder:
    """Optional lowering contract for a native node.

    A recorder must describe only dispatches whose semantics are identical to
    ``NativeGraphExecutable.run()``. Compatible recorders can be concatenated
    with adjacent CGraph dispatches into one backend-owned CGraph region.
    """

    def supports_backend(self, backend):
        return False

    @property
    def dispatches(self):
        return ()

    def bind_graph_temporaries(self, temporaries):
        """Return a recorder bound to one arena slot, or None to fail closed.

        Existing recorders do not consume scratch and therefore need no bind.
        """
        return self if not temporaries else None


class DispatchNativeGraphRecorder(NativeGraphBackendRecorder):
    """Recorder backed by precompiled Taichi dispatches."""

    def __init__(self, dispatches, *, backends=("cpu", "cuda", "vulkan")):
        self._dispatches = tuple(
            (kernel, tuple(args)) for kernel, args in dispatches
        )
        if not self._dispatches:
            raise ValueError("Native Graph recorder requires a dispatch")
        self._backends = frozenset(backends)

    def supports_backend(self, backend):
        return backend in self._backends

    @property
    def dispatches(self):
        return self._dispatches


class NativeGraphExecutable:
    """Compiled native graph node.

    This is the Python-side contract for native code that can be scheduled by
    ``Graph.run()``. Current producers are limited to DSL-defined nodes. A
    future general native path should lower into this same executable shape.
    """

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
    def backend_recorder(self):
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
            opaque=self.backend_recorder is None,
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
