"""Resolve declared definition constants without executing model constructors."""

import ast
import inspect

from taichi_forge_runtime._static_values import static_key
from ._package_factories import _literal, _source_value


def definition_names(expression, namespace, constructor):
    """Read a fixed declaration-name tuple from its unshadowed module source."""
    if isinstance(expression, ast.Subscript) and isinstance(expression.slice, ast.Slice):
        names = definition_names(expression.value, namespace, constructor)
        if type(names) is not tuple:
            return None
        bounds = []
        for bound in (expression.slice.lower, expression.slice.upper, expression.slice.step):
            try:
                value = None if bound is None else ast.literal_eval(bound)
            except (ValueError, TypeError, SyntaxError):
                return None
            if value is not None and type(value) is not int:
                return None
            bounds.append(value)
        if bounds[2] == 0:
            return None
        return names[slice(*bounds)]
    if isinstance(expression, (ast.Tuple, ast.List)):
        names = []
        for item in expression.elts:
            if isinstance(item, ast.Starred):
                values = definition_names(item.value, namespace, constructor)
                if type(values) is not tuple:
                    return None
                names.extend(values)
            elif isinstance(item, ast.Constant) and type(item.value) is str:
                names.append(item.value)
            else:
                return None
        return tuple(names)
    root = expression
    while isinstance(root, ast.Attribute):
        root = root.value
    if not isinstance(root, ast.Name):
        return None
    for node in ast.walk(constructor):
        if (
            isinstance(node, ast.Name)
            and node.id == root.id
            and not isinstance(node.ctx, ast.Load)
            or isinstance(node, ast.arg)
            and node.arg == root.id
            or isinstance(node, ast.alias)
            and (node.asname or node.name.split(".")[0]) == root.id
            or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler))
            and node is not constructor
            and node.name == root.id
            or isinstance(node, (ast.Global, ast.Nonlocal))
            and root.id in node.names
        ):
            return None
    names = _source_value(expression, namespace)
    # Unlike a literal list written at this declaration, a global mutable list
    # can be populated by model preparation. It is not a definition contract.
    return names if type(names) is tuple else None


def class_constant(cls, name):
    value = inspect.getattr_static(cls, name)
    if not _literal(value):
        raise ValueError("Definition constant requires an immutable value, not a descriptor or object")
    return static_key(value)


def check_constant_slot(cls, name):
    descriptor = inspect.getattr_static(cls, name, None)
    if inspect.isdatadescriptor(descriptor) or inspect.getattr_static(cls, "__setattr__") is not object.__setattr__:
        raise ValueError("Definition constant cannot execute an owner descriptor or custom assignment")


def resolve_constants(module, node, schema, namespace, fail, *, constructor=None, owner_class=None):
    constants = {name: item for name, item in schema.items() if item["kind"] == "constant"}
    if not constants:
        return
    if constructor is None:
        constructor = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    parameters = {n.arg for n in ast.walk(constructor.args) if isinstance(n, ast.arg)}
    locals_ = parameters | {
        n.id for n in ast.walk(constructor) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
    }
    writes = {}
    for n in ast.walk(constructor):
        if isinstance(n, (ast.Name, ast.Attribute)) and not isinstance(n.ctx, ast.Load):
            key = n.id if isinstance(n, ast.Name) else "." + n.attr
            writes[key] = writes.get(key, 0) + 1
    fixed = {}

    def value(expression):
        try:
            result = ast.literal_eval(expression)
        except (ValueError, TypeError, SyntaxError):
            if isinstance(expression, ast.Name) and expression.id in fixed:
                return fixed[expression.id]
            root = expression
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name) or root.id in locals_:
                raise ValueError("Definition constant depends on model inputs or an unproven expression")
            # Only module dictionaries, never application properties or calls.
            result = _source_value(expression, namespace)
            if result is None and not (isinstance(expression, ast.Name) and expression.id in namespace):
                raise ValueError("Definition constant has an unresolved source")
        if not _literal(result):
            raise ValueError("Definition constant requires an immutable value")
        return result

    assigned = {}
    for statement in constructor.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)) or statement.value is None:
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if len(targets) != 1:
            continue
        target = targets[0]
        if isinstance(target, ast.Name) and target.id not in parameters and writes[target.id] == 1:
            try:
                fixed[target.id] = value(statement.value)
            except ValueError:
                pass
        elif isinstance(target, ast.Attribute) and target.attr in constants and writes["." + target.attr] == 1:
            item = constants[target.attr]
            if not (isinstance(target.value, ast.Name) and target.value.id == "self"):
                continue
            if (statement.lineno, statement.col_offset) >= tuple(item["constant_declaration"]):
                fail(module, statement, "Definition constant must be assigned before its declaration")
            try:
                assigned[target.attr] = static_key(value(statement.value))
            except ValueError as error:
                fail(module, statement, str(error))
    for name, item in constants.items():
        try:
            check_constant_slot(owner_class or namespace[node.name], name)
        except ValueError as error:
            fail(module, constructor, str(error))
        if name in assigned:
            item["value"] = assigned[name]
        elif writes.get("." + name, 0):
            fail(module, constructor, "Definition constant requires one unconditional fixed assignment: " + name)
        else:
            try:
                item["value"] = class_constant(owner_class or namespace[node.name], name)
            except (AttributeError, ValueError) as error:
                fail(module, constructor, f"Definition constant {name!r} has no fixed class value: {error}")
            item["constant_from_class"] = True
