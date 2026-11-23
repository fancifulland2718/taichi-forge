"""Lower fixed rows of Field declarations in their original allocation order."""

import ast
import copy

from ._package_constants import definition_names, locally_bound


def expand_field_rows(node, scopes, constructor, namespace, fail):
    if not isinstance(node.target, (ast.Tuple, ast.List)) or len(node.body) != 1:
        return None
    statement = node.body[0]
    call = statement.value if isinstance(statement, ast.Expr) else None
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "setattr"
        and len(call.args) == 3
        and not call.keywords
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id in scopes
        and isinstance(call.args[1], ast.Name)
    ):
        return None
    receiver, name, allocation = call.args
    if node.orelse or namespace.get("setattr", setattr) is not setattr or locally_bound(constructor, "setattr"):
        fail(node, "Fixed Field rows require an ordinary loop and unshadowed setattr")
    columns = [item.id for item in node.target.elts if isinstance(item, ast.Name)]
    if len(set(columns)) != len(node.target.elts) or name.id not in columns or set(columns) & (scopes | {"self"}):
        fail(node.target, "Fixed Field rows require distinct column names")
    rows = definition_names(node.iter, namespace, constructor)
    if type(rows) is not tuple or not rows:
        fail(node.iter, "Fixed Field rows require a nonempty immutable definition table")
    seen = set()
    result = []
    has_getattr = namespace.get("getattr", getattr) is getattr and not locally_bound(constructor, "getattr")
    for index, row in enumerate(rows):
        if type(row) is not tuple or len(row) != len(columns) or any(type(v) not in (str, int, bool) for v in row):
            fail(node.iter, "Fixed Field rows require literal columns of the declared width")
        bindings = dict(zip(columns, row))
        member = bindings[name.id]
        if type(member) is not str or not member.isidentifier() or member.startswith("__") or member in seen:
            fail(name, "Fixed Field rows require distinct non-mangled member names")
        seen.add(member)

        class Bind(ast.NodeTransformer):
            def visit_Name(self, expr):
                if expr.id in bindings:
                    return ast.copy_location(ast.Constant(bindings[expr.id]), expr)
                return expr

            def visit_IfExp(self, expr):
                test = self.visit(expr.test)
                if isinstance(test, ast.Constant) and type(test.value) in (bool, int):
                    return self.visit(expr.body if test.value else expr.orelse)
                return self.generic_visit(expr)

            def visit_Call(self, expr):
                expr = self.generic_visit(expr)
                if (
                    isinstance(expr.func, ast.Name)
                    and expr.func.id == "getattr"
                    and len(expr.args) == 2
                    and not expr.keywords
                    and isinstance(expr.args[1], ast.Constant)
                ):
                    attr = expr.args[1].value
                    if type(attr) is not str or not attr.isidentifier() or attr.startswith("__") or not has_getattr:
                        fail(expr, "Fixed Field rows require an unshadowed, fixed attribute access")
                    return ast.copy_location(ast.Attribute(expr.args[0], attr, ast.Load()), expr)
                return expr

        value = Bind().visit(copy.deepcopy(allocation))
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == receiver.id
            and value.func.attr in ("field", "vector_field", "matrix_field", "struct_field")
        ):
            fail(allocation, "Fixed Field rows must declare a Field on the same scope")
        bound = ast.Assign([ast.Attribute(copy.deepcopy(receiver), member, ast.Store())], value)
        for part in ast.walk(bound):
            if isinstance(part, (ast.expr, ast.stmt)):
                part.lineno = part.end_lineno = node.lineno
                part.col_offset = node.col_offset + index + 1
                part.end_col_offset = part.col_offset + 1
        result.append(ast.copy_location(ast.Assign([copy.deepcopy(node.target)], ast.Constant(row)), node))
        result.append(bound)
    return result
