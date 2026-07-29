from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.graph._ir import NativeCallNode


class NativeGraphExecutable:
    """Compiled native graph node.

    This is the Python-side contract for native code that can be scheduled by
    ``Graph.run()``. Current producers are limited to DSL-defined nodes. A
    future general native path should lower into this same executable shape.
    """

    def prewarm(self):
        return self

    def run(self):
        raise NotImplementedError

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
