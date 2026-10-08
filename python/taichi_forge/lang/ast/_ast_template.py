"""Reusable copy layout for pristine trees returned by ast.parse()."""

import ast


class ASTTemplate:
    """Create independent ASTs without repeatedly inspecting every attribute.

    Parsing produces a graph of AST nodes and lists, with immutable scalar
    leaves. Record its edges once; subsequent expansions only allocate the
    containers and connect them. Shared nodes (e.g. operator singletons) remain
    shared within a copy, but no mutable objects are shared between copies.
    The layout never holds lowered expressions or specialization values.
    """

    def __init__(self, tree):
        entries = []
        indices = {}
        pending = [tree]
        indices[id(tree)] = 0
        # Iterative construction also avoids adding a Python recursion limit to
        # the depth already accepted by the parser.
        for value in pending:
            node_type = type(value) if isinstance(value, ast.AST) else None
            attributes = vars(value).copy() if node_type is not None else value.copy()
            edges = []
            items = attributes.items() if node_type is not None else enumerate(attributes)
            for key, child in items:
                if isinstance(child, (ast.AST, list)):
                    identity = id(child)
                    index = indices.get(identity)
                    if index is None:
                        index = len(pending)
                        indices[identity] = index
                        pending.append(child)
                    edges.append((key, index))
                    attributes[key] = None
            entries.append((node_type, attributes, edges))
        self._entries = entries

    def instantiate(self):
        nodes = []
        containers = []
        for node_type, attributes, _ in self._entries:
            values = attributes.copy()
            if node_type is None:
                node = values
            else:
                # Fill the complete parsed state below, without invoking AST
                # constructors that require fields on newer Python versions.
                node = ast.AST.__new__(node_type)
                node.__dict__ = values
            nodes.append(node)
            containers.append(values)
        for values, (_, _, edges) in zip(containers, self._entries):
            for key, index in edges:
                values[key] = nodes[index]
        return nodes[0]
