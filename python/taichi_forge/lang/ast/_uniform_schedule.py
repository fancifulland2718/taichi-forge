"""Syntax proof shared by package classification and ordinary AST lowering.

Only range schedules with side-effect-free bounds can be flattened into
separately guarded loops. This does not classify arbitrary helper calls or
move device reads, local assignments, sparse iteration, or control transfers.
"""

import ast


def _bound(node):
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.Attribute):
        return _bound(node.value)
    if isinstance(node, ast.Tuple):
        return all(_bound(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _bound(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        return _bound(node.left) and _bound(node.right)
    return False


def uniform_range_schedule(node, kind):
    """Prove both arms without evaluating source or mutating compiler state."""
    if not isinstance(node, ast.If):
        return False
    for statements in (node.body, node.orelse):
        pending_config = False
        for statement in statements:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                # A literal launch policy belongs to the immediately following
                # loop, including when that branch is disabled at prepare.
                if (kind(call.func) != "loop_config" or call.args or not call.keywords or
                        any(kw.arg not in ("block_dim", "serialize") or
                            not isinstance(kw.value, ast.Constant) or
                            type(kw.value.value) not in (bool, int) for kw in call.keywords)):
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
        if pending_config:
            return False
    return bool(node.body)
