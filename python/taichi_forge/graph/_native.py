from taichi_forge.lang.exception import TaichiRuntimeError


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
