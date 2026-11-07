"""Normalize fixed alias declarations while retaining source-scope ownership."""

import ast
import copy
from types import MappingProxyType

from ._package_factories import _source_value


def expand_alias_loop(node, scope_names, constructor, namespace, fail):
    if not (len(node.body) == 1 and isinstance(node.body[0], ast.Expr)):
        return None
    call = node.body[0].value
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in scope_names
        and call.func.attr == "borrow"
    ):
        return None
    iterator = node.iter
    if not (
        not node.orelse
        and isinstance(iterator, ast.Call)
        and not iterator.args
        and not iterator.keywords
        and isinstance(iterator.func, ast.Attribute)
        and iterator.func.attr == "items"
    ):
        fail(node, "Alias declarations require a fixed mapping items loop")
    source = iterator.func.value
    root = source
    while isinstance(root, ast.Attribute):
        root = root.value
    if not isinstance(root, ast.Name) or any(
        isinstance(n, ast.Name)
        and n.id == root.id
        and not isinstance(n.ctx, ast.Load)
        or isinstance(n, ast.arg)
        and n.arg == root.id
        or isinstance(n, ast.alias)
        and (n.asname or n.name.split(".")[0]) == root.id
        for n in ast.walk(constructor)
    ):
        fail(node, "Alias mapping must be an immutable definition, not a model input or local binding")
    mapping = _source_value(source, namespace)
    if type(mapping) is not MappingProxyType:
        fail(node, "Alias mapping requires an immutable definition mapping")
    items = tuple(mapping.items())
    if not items or any(
        type(name) is not str
        or not name.isidentifier()
        or type(path) is not tuple
        or not path
        or any(type(part) is not str or not part.isidentifier() or part.startswith("__") for part in path)
        for name, path in items
    ):
        fail(node, "Alias mapping requires fixed names and immutable attribute paths")
    if namespace.get("getattr", getattr) is not getattr or any(
        isinstance(n, ast.Name)
        and n.id == "getattr"
        and not isinstance(n.ctx, ast.Load)
        or isinstance(n, ast.arg)
        and n.arg == "getattr"
        or isinstance(n, ast.alias)
        and (n.asname or n.name.split(".")[0]) == "getattr"
        for n in ast.walk(constructor)
    ):
        fail(node, "Alias attribute access requires the unshadowed builtin")

    def bind(target, value, bindings):
        if isinstance(target, ast.Name):
            if target.id in bindings or target.id in scope_names or target.id in ("self", root.id):
                fail(target, "Alias loop cannot replace its scope, source or another loop binding")
            bindings[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and type(value) is tuple and len(target.elts) == len(value):
            for element, item in zip(target.elts, value):
                bind(element, item, bindings)
        else:
            fail(target, "Alias loop requires fixed tuple unpacking")

    guard = ast.copy_location(ast.Expr(ast.Constant(None)), node)
    guard._package_alias_mapping = (copy.deepcopy(source), items)
    result = [guard]
    for index, item in enumerate(items):
        bindings = {}
        bind(node.target, item, bindings)

        class Substitute(ast.NodeTransformer):
            def visit_Name(self, value):
                return ast.copy_location(ast.Constant(bindings[value.id]), value) if value.id in bindings else value

            def visit_Call(self, value):
                value = self.generic_visit(value)
                if (
                    isinstance(value.func, ast.Name)
                    and value.func.id == "getattr"
                    and len(value.args) == 2
                    and not value.keywords
                    and isinstance(value.args[1], ast.Constant)
                    and type(value.args[1].value) is str
                    and value.args[1].value.isidentifier()
                    and not value.args[1].value.startswith("__")
                ):
                    return ast.copy_location(ast.Attribute(value.args[0], value.args[1].value, ast.Load()), value)
                return value

        statement = Substitute().visit(copy.deepcopy(node.body[0]))
        # Declarations at a shared source line still need distinct binding sites.
        for part in ast.walk(statement):
            if isinstance(part, (ast.expr, ast.stmt)):
                part.lineno = part.end_lineno = node.lineno
                part.col_offset = node.col_offset + index + 1
                part.end_col_offset = part.col_offset + 1
        result.append(ast.copy_location(ast.Assign([copy.deepcopy(node.target)], ast.Constant(item)), node))
        result.append(statement)
    return result
