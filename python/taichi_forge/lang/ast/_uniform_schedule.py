"""Syntax proof shared by package classification and ordinary AST lowering.

Only range schedules with side-effect-free bounds and proven inline helpers
can be flattened into separately guarded loops. Device reads, assignments,
sparse iteration and control transfers retain their original boundaries.
"""

import ast


def _bound(node):
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.Attribute):
        return _bound(node.value)
    if isinstance(node, ast.Tuple):
        return all(_bound(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
        return _bound(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        return _bound(node.left) and _bound(node.right)
    if isinstance(node, ast.BoolOp):
        return all(_bound(value) for value in node.values)
    if isinstance(node, ast.Compare):
        return (all(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)) for op in node.ops)
                and _bound(node.left) and all(_bound(value) for value in node.comparators))
    return False


def helper_call_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"):
        return call.func.attr
    return None


def uniform_range_statements(statements, kind, helpers=()):
    pending_config = False
    for statement in statements:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            if isinstance(statement.value.value, str):  # helper docstring
                continue
        if isinstance(statement, ast.If):
            condition = statement.test
            if isinstance(condition, ast.Call) and kind(condition.func) == "static":
                if len(condition.args) != 1 or condition.keywords:
                    return False
                condition = condition.args[0]
            if pending_config or not _bound(condition) or not uniform_range_schedule(statement, kind, helpers):
                return False
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if helper_call_name(call) in helpers:
                if pending_config or not all(_bound(arg) for arg in call.args) or any(
                    kw.arg is None or not _bound(kw.value) for kw in call.keywords
                ):
                    return False
                continue
            # Policy expressions can use template parameters of a helper. The
            # static-use analysis separately keeps those parameters at build
            # time; a scheduling predicate must never become a block-size phi.
            if (kind(call.func) != "loop_config" or call.args or not call.keywords or
                    any(kw.arg not in ("block_dim", "serialize") or not _bound(kw.value)
                        for kw in call.keywords)):
                return False
            pending_config = True
            continue
        if not isinstance(statement, ast.For) or statement.orelse:
            return False
        iterator = statement.iter
        if isinstance(iterator, ast.Call) and kind(iterator.func) == "grouped":
            if len(iterator.args) != 1 or iterator.keywords:
                return False
            iterator = iterator.args[0]
        if not isinstance(iterator, ast.Call) or iterator.keywords or not iterator.args:
            return False
        if kind(iterator.func) != "ndrange" and not (
            isinstance(iterator.func, ast.Name) and iterator.func.id == "range"
        ):
            return False
        if not all(_bound(arg) for arg in iterator.args):
            return False
        pending_config = False
    return not pending_config


def uniform_range_schedule(node, kind, helpers=()):
    """Prove both arms without evaluating source or mutating compiler state."""
    if not isinstance(node, ast.If):
        return False
    return bool(node.body) and all(
        uniform_range_statements(statements, kind, helpers) for statements in (node.body, node.orelse)
    )
