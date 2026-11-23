"""Discover an attached data-oriented child with one proven construction site."""
import ast
import copy
import inspect

from ._package_factories import _source_value


def unconditional_scope_statements(statements, contexts):
    for statement in statements:
        if id(statement) in contexts:
            yield from unconditional_scope_statements(statement.body, contexts)
        else:
            yield statement


def child_contract(module, constructor, call, member, namespace, fail):
    """Keep the owned class and its normal constructor, never execute it."""
    value = call.args[1]
    writes = [
        target for target in ast.walk(constructor)
        if isinstance(target, ast.Attribute) and not isinstance(target.ctx, ast.Load)
        and isinstance(target.value, ast.Name) and target.value.id == "self"
        and target.attr == member
    ]
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "self" and value.attr == member:
        construction = next((
            statement.value for statement in constructor.body
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and len(writes) == 1 and statement.targets[0] is writes[0] and statement.lineno < call.lineno
        ), None)
    else:
        construction = value if not writes else None
    if not isinstance(construction, ast.Call):
        fail(module, call, "Child scope requires a new class construction or its unique earlier owner assignment")
    child_class = _source_value(construction.func, namespace)
    root = construction.func
    while isinstance(root, ast.Attribute):
        root = root.value
    if not inspect.isclass(child_class) or not isinstance(root, ast.Name) or any(
        isinstance(node, ast.Name) and node.id == root.id and not isinstance(node.ctx, ast.Load)
        or isinstance(node, ast.arg) and node.arg == root.id
        for node in ast.walk(constructor)
    ):
        fail(module, call, "Child scope requires an unshadowed source class")
    return dict(kind="owner", owner_type=child_class.__module__ + "." + child_class.__qualname__,
                constructor_call=copy.deepcopy(construction))
