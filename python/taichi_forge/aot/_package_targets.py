"""Select only constructor branches proved by the package's chosen backend."""

import ast
import copy

from ._package_types import qualified_kind

BACKEND_NAMES = frozenset(("cpu", "x64", "arm64", "cuda", "vulkan"))


def backend_constant(node, kind):
    if not isinstance(node, (ast.Name, ast.Attribute)):
        return None
    role = qualified_kind(node, kind)
    if role in BACKEND_NAMES:
        import taichi_forge as ti

        return getattr(ti, role).name
    return None


def backend_query(node, kind):
    if isinstance(node, ast.Attribute) and qualified_kind(node, kind) == "cfg.arch":
        return True
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "arch"
        and isinstance(node.value, ast.Call)
        and not node.value.args
        and not node.value.keywords
        and qualified_kind(node.value, kind) == "lang.impl.current_cfg"
    )


def select_target_branches(tree, kind, target, module, fail):
    selections = []

    def key(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            return "self." + node.attr
        return None

    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for function in cls.body:
            if not isinstance(function, ast.FunctionDef) or not (
                function.name == "__init__"
                or getattr(function, "_package_construction_method", False)
                or getattr(function, "_package_preparation_method", False)
            ):
                continue
            writes = {}
            locals_ = {n.arg for n in ast.walk(function.args) if isinstance(n, ast.arg)}
            for node in ast.walk(function):
                name = key(node)
                if name is not None and isinstance(getattr(node, "ctx", None), ast.Store):
                    writes[name] = writes.get(name, 0) + 1
                    if isinstance(node, ast.Name):
                        locals_.add(name)
            for node in ast.walk(function):
                if backend_query(node, kind) or backend_constant(node, kind) is not None:
                    root = node
                    while isinstance(root, (ast.Call, ast.Attribute)):
                        root = root.func if isinstance(root, ast.Call) else root.value
                    if isinstance(root, ast.Name) and root.id in locals_:
                        fail(module, node, "Backend selection cannot use a shadowed frontend name")
            aliases = set()

            def value(node):
                constant = backend_constant(node, kind)
                if constant is not None:
                    return constant
                if backend_query(node, kind) or key(node) in aliases:
                    return target
                return None

            def condition(node):
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                    result = condition(node.operand)
                    return None if result is None else not result
                if isinstance(node, ast.BoolOp):
                    items = [condition(item) for item in node.values]
                    if any(item is None for item in items):
                        return None
                    return all(items) if isinstance(node.op, ast.And) else any(items)
                if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                    return None
                left = value(node.left)
                right = node.comparators[0]
                op = node.ops[0]
                if left is None:
                    return None
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.Tuple, ast.List)):
                    choices = [value(item) for item in right.elts]
                    if any(item is None for item in choices):
                        return None
                    result = left in choices
                    return result if isinstance(op, ast.In) else not result
                right = value(right)
                if right is None or not isinstance(op, (ast.Eq, ast.NotEq)):
                    return None
                return (left == right) if isinstance(op, ast.Eq) else (left != right)

            def statements(body):
                output = []
                for node in body:
                    if isinstance(node, (ast.Assign, ast.AnnAssign)):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        if len(targets) == 1 and backend_query(node.value, kind):
                            name = key(targets[0])
                            if name is not None and writes.get(name) == 1:
                                aliases.add(name)
                    selected = condition(node.test) if isinstance(node, ast.If) else None
                    if selected is None:
                        output.append(node)
                        continue
                    location = f"{module}:{node.lineno}"
                    selections.append(dict(owner=cls.name, line=node.lineno, taken=selected))
                    # Keep the original test as a consumption-time guard. A Host
                    # call/property may have changed an owner alias after the
                    # assignment; choosing that different layout is not legal.
                    guard = ast.Expr(
                        ast.Call(
                            ast.Attribute(
                                ast.Name("_forge_package_runtime", ast.Load()), "check_target_branch", ast.Load()
                            ),
                            [copy.deepcopy(node.test), ast.Constant(selected), ast.Constant(location)],
                            [],
                        )
                    )
                    output.append(ast.copy_location(guard, node))
                    output.extend(statements(node.body if selected else node.orelse))
                return output

            function.body = statements(function.body)
    return selections
