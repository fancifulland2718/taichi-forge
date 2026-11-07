"""Project existing ResourceScope value declarations without running constructors."""

import ast

from ._package_types import qualified_kind


def scope_values(module, constructor, kind, fail):
    if constructor is None:
        return {}
    scopes, declarations = {}, {}
    for statement in ast.walk(constructor):
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if qualified_kind(call, kind) != "experimental.ResourceScope":
            continue
        if not (
            statement in constructor.body
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "self"
        ):
            fail(module, statement, "Package ResourceScope requires one unconditional local scope for self")
        scope = statement.targets[0].id
        stores = [
            n
            for n in ast.walk(constructor)
            if isinstance(n, ast.Name) and n.id == scope and isinstance(n.ctx, ast.Store)
        ]
        if len(stores) != 1:
            fail(module, statement, "Package ResourceScope aliases must not be reassigned")
        scopes[scope] = statement
    for statement in ast.walk(constructor):
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if not (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)):
            continue
        scope = call.func.value.id
        if scope not in scopes or call.func.attr != "value":
            continue
        if statement not in constructor.body or statement.lineno < scopes[scope].lineno:
            fail(module, statement, "Package ResourceScope values must be unconditional declarations after their scope")
        args = dict(zip(("name", "dtype"), call.args))
        if len(call.args) > 2 or any(k.arg not in ("name", "dtype") or k.arg in args for k in call.keywords):
            fail(module, statement, "ResourceScope.value requires explicit name and dtype")
        args.update((k.arg, k.value) for k in call.keywords)
        name = args.get("name")
        if not (
            set(args) == {"name", "dtype"}
            and isinstance(name, ast.Constant)
            and isinstance(name.value, str)
            and name.value.isidentifier()
        ):
            fail(module, statement, "ResourceScope.value requires a literal attribute name and fixed type")
        if name.value in declarations:
            fail(module, statement, "Repeated package ResourceScope value declarations require further support")
        declarations[name.value] = dict(
            kind="value",
            type_node=args["dtype"],
            value_declaration=[statement.lineno, statement.col_offset],
            scope_allocation=[scopes[scope].value.lineno, scopes[scope].value.col_offset],
        )
    parents = {id(child): parent for parent in ast.walk(constructor) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(constructor):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in scopes):
            continue
        parent = parents[id(node)]
        call = parents.get(id(parent))
        if not (
            isinstance(parent, ast.Attribute)
            and parent.attr == "value"
            and isinstance(call, ast.Call)
            and call.func is parent
            and any(item["value_declaration"] == [call.lineno, call.col_offset] for item in declarations.values())
        ):
            fail(
                module, node, "ResourceScope operations beyond value declarations require a package resource projection"
            )
    return declarations
