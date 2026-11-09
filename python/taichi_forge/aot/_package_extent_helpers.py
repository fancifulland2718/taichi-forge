"""Prove small Host extent helpers through the existing dimension algebra."""

import ast
import inspect
import textwrap

from taichi_forge_runtime._dimensions import Extent

from ._package_factories import _source_value


def project_extent_helper(call, namespace, dimension, *, shadowed=()):
    """Never invoke application code or infer a relation from sampled counts."""
    root = call.func
    while isinstance(root, ast.Attribute):
        root = root.value
    if not isinstance(root, ast.Name) or root.id in shadowed:
        return None
    function = _source_value(call.func, namespace)
    if (
        not inspect.isfunction(function)
        or function.__closure__
        or any(name in vars(function) for name in ("__wrapped__", "__signature__"))
    ):
        return None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        return None
    local_names = {
        node.id for node in ast.walk(tree.body[0]) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    } | {node.arg for node in ast.walk(tree.body[0].args) if isinstance(node, ast.arg)}
    arguments = [dimension(node) for node in call.args]
    keywords = {keyword.arg: dimension(keyword.value) for keyword in call.keywords}
    if (
        None in keywords
        or len(keywords) != len(call.keywords)
        or not all(isinstance(value, Extent) for value in (*arguments, *keywords.values()))
    ):
        return None
    try:
        bound = inspect.signature(function).bind(*arguments, **keywords)
    except TypeError:
        return None
    bound.apply_defaults()
    values = {}
    for name, value in bound.arguments.items():
        if type(value) is int and 0 <= value <= 2**31 - 1:
            value = Extent(("const", value))
        if not isinstance(value, Extent):
            return None
        values[name] = value

    def expression(node):
        if isinstance(node, ast.Name):
            return node.id in values
        if isinstance(node, ast.Constant):
            return type(node.value) is int and 0 <= node.value <= 2**31 - 1
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
            return expression(node.left) and expression(node.right)
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("minimum", "maximum")
            and len(node.args) == 1
            and not node.keywords
            and expression(node.func.value)
            and expression(node.args[0])
        )

    def condition(node):
        # Compatibility helpers can choose the symbolic path by the actual
        # Extent protocol. No arbitrary Python attribute is looked up/executed.
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hasattr"
            and function.__globals__.get("hasattr", hasattr) is hasattr
            and "hasattr" not in local_names
            and len(node.args) == 2
            and not node.keywords
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in ("minimum", "maximum")
            and expression(node.args[0])
        ):
            return None
        return True

    def statements(body):
        for node in body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and type(node.value.value) is str:
                continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in values or not expression(node.value):
                    return None
                values[node.targets[0].id] = dimension(node.value, values)
                if not isinstance(values[node.targets[0].id], Extent):
                    return None
            elif isinstance(node, ast.Return):
                return dimension(node.value, values) if expression(node.value) else None
            elif isinstance(node, ast.If):
                selected = condition(node.test)
                if selected is None:
                    return None
                # Only an immediate returning protocol arm; no branch-local
                # mutable state or hidden continuation is reconstructed.
                arm = node.body if selected else node.orelse
                if len(arm) != 1 or not isinstance(arm[0], ast.Return):
                    return None
                return statements(arm)
            else:
                return None
        return None

    return statements(tree.body[0].body)
