"""Syntax proof shared by package classification and ordinary AST lowering.

Range schedules and proven inline helpers can be flattened into independently
guarded loops. Device-read bounds are evaluated inside their original guard;
scalar reads and writes remain ordered between loops. Python aliases, sparse
iteration and unproven control transfers retain their original boundaries.
"""

import ast
import copy
from itertools import chain


def _bound(node, descriptor_bound=None):
    if isinstance(node, ast.Subscript):
        return descriptor_bound is not None and descriptor_bound(node)
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.Attribute):
        return _bound(node.value, descriptor_bound)
    if isinstance(node, ast.Tuple):
        return all(_bound(item, descriptor_bound) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
        return _bound(node.operand, descriptor_bound)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        return _bound(node.left, descriptor_bound) and _bound(node.right, descriptor_bound)
    if isinstance(node, ast.BoolOp):
        return all(_bound(value, descriptor_bound) for value in node.values)
    if isinstance(node, ast.Compare):
        return (all(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)) for op in node.ops)
                and _bound(node.left, descriptor_bound)
                and all(_bound(value, descriptor_bound) for value in node.comparators))
    return False


def helper_call_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"):
        return call.func.attr
    return None


def schedule_value_parameters(function, kind):
    """Keys used by callers for arguments with ordinary scalar/value semantics."""
    positional = function.args.posonlyargs + function.args.args
    keys = set()
    for index, arg in enumerate(positional + function.args.kwonlyargs):
        if arg.annotation is not None and kind(arg.annotation) not in (
                'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'f16', 'f32', 'f64'):
            continue
        if index < len(positional):
            keys.add(index)
        keys.add(arg.arg)
    return keys


def schedule_argument_reads(call, helpers):
    """Find direct reads of an owner's scalar slot passed by value to an inline schedule."""
    parameters = helpers.get(helper_call_name(call))
    if not parameters:
        return
    positional = enumerate(call.args, start=int(isinstance(call.func, ast.Attribute)))
    keywords = ((kw.arg, kw.value) for kw in call.keywords)
    for key, arg in chain(positional, keywords):
        if (key not in parameters or not isinstance(arg, ast.Subscript) or not isinstance(arg.slice, ast.Constant)
                or arg.slice.value is not None or not isinstance(arg.value, ast.Attribute)):
            continue
        owner = arg.value.value
        while isinstance(owner, ast.Attribute):
            owner = owner.value
        if isinstance(owner, ast.Name) and owner.id == 'self':
            yield arg


def snapshot_schedule_arguments(statements, helpers, scalar_helpers):
    """Bind device values once inside the model guard, before entering the helper's loops."""
    for index, statement in enumerate(statements):
        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                and helper_call_name(statement.value) not in scalar_helpers):
            call = statement.value
            replacements = {}
            for slot, arg in enumerate(schedule_argument_reads(call, helpers)):
                # Not a Python identifier, so this cannot shadow a source local.
                name = f'<schedule argument {index}.{slot}>'
                target = ast.copy_location(ast.Name(id=name, ctx=ast.Store()), arg)
                yield ast.copy_location(ast.Assign(targets=[target], value=arg), arg)
                replacements[arg] = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), arg)
            if replacements:
                call = copy.copy(call)
                call.args = [replacements.get(arg, arg) for arg in call.args]
                call.keywords = [ast.copy_location(ast.keyword(arg=kw.arg, value=replacements[kw.value]), kw)
                                 if kw.value in replacements else kw for kw in call.keywords]
                statement = ast.copy_location(ast.Expr(value=call), statement)
        yield statement


def guarded_range_bound(node, kind):
    """Read-only bounds whose evaluation must stay behind the schedule guard."""
    if _bound(node):
        return True
    if isinstance(node, ast.Subscript):
        return _bound(node.value) and guarded_range_bound(node.slice, kind)
    if isinstance(node, ast.Tuple):
        return all(guarded_range_bound(item, kind) for item in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
        return guarded_range_bound(node.left, kind) and guarded_range_bound(node.right, kind)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return guarded_range_bound(node.operand, kind)
    if (isinstance(node, ast.Call) and kind(node.func) in ('min', 'max')
            and not node.keywords and node.args):
        return all(guarded_range_bound(arg, kind) for arg in node.args)
    return False


def scalar_expression(node, kind, helpers=()):
    if _bound(node):
        return True
    if isinstance(node, ast.Subscript):
        return _bound(node.value) and scalar_expression(node.slice, kind, helpers)
    if isinstance(node, ast.Tuple):
        return all(scalar_expression(item, kind, helpers) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not, ast.Invert)):
        return scalar_expression(node.operand, kind, helpers)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
        return scalar_expression(node.left, kind, helpers) and scalar_expression(node.right, kind, helpers)
    if isinstance(node, ast.BoolOp):
        return all(scalar_expression(item, kind, helpers) for item in node.values)
    if isinstance(node, ast.Compare):
        return (all(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)) for op in node.ops)
                and all(scalar_expression(item, kind, helpers) for item in (node.left, *node.comparators)))
    if isinstance(node, ast.Call) and (kind(node.func) in ('min', 'max', 'cast', 'abs')
                                       or helper_call_name(node) in helpers):
        return (all(scalar_expression(arg, kind, helpers) for arg in node.args)
                and all(kw.arg is not None and scalar_expression(kw.value, kind, helpers)
                        for kw in node.keywords))
    return False


def scalar_statement(statement, kind, helpers=(), *, returns=False):
    if isinstance(statement, ast.Expr):
        return (isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)) or (
            isinstance(statement.value, ast.Call) and helper_call_name(statement.value) in helpers
            and scalar_expression(statement.value, kind, helpers))
    if isinstance(statement, ast.Return):
        return returns and (statement.value is None or scalar_expression(statement.value, kind, helpers))
    if isinstance(statement, ast.If):
        condition = statement.test
        if isinstance(condition, ast.Call) and kind(condition.func) == 'static':
            if len(condition.args) != 1 or condition.keywords:
                return False
            condition = condition.args[0]
        return scalar_expression(condition, kind, helpers) and all(
            scalar_statement(item, kind, helpers, returns=returns) for item in statement.body + statement.orelse)
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target, value = statement.targets[0], statement.value
        # Bare Python aliases may denote resources or types. They need their
        # original static scope rather than a numeric branch-local phi.
        if isinstance(target, ast.Name) and isinstance(value, (ast.Name, ast.Attribute, ast.Tuple)):
            return False
    elif isinstance(statement, ast.AugAssign) and isinstance(statement.op, (ast.Add, ast.Sub, ast.Mult)):
        target, value = statement.target, statement.value
    else:
        return False
    return ((isinstance(target, ast.Name) or isinstance(target, ast.Subscript)
             and _bound(target.value) and scalar_expression(target.slice, kind, helpers))
            and scalar_expression(value, kind, helpers))


def uniform_range_statements(statements, kind, helpers=(), scalar_helpers=(), *, select_static=None):
    def selected(body):
        for statement in body:
            test = statement.test if isinstance(statement, ast.If) else None
            fixed = (select_static(test.args[0]) if select_static is not None
                     and isinstance(test, ast.Call) and kind(test.func) == 'static'
                     and len(test.args) == 1 and not test.keywords else None)
            if fixed is None:
                yield statement
            else:
                yield from selected(statement.body if fixed else statement.orelse)

    pending_config = False
    for statement in selected(statements):
        if scalar_statement(statement, kind, scalar_helpers):
            if pending_config:
                return False
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            if isinstance(statement.value.value, str):  # helper docstring
                continue
        if isinstance(statement, ast.If):
            condition = statement.test
            if isinstance(condition, ast.Call) and kind(condition.func) == "static":
                if len(condition.args) != 1 or condition.keywords:
                    return False
                condition = condition.args[0]
            if pending_config or not _bound(condition) or not uniform_range_schedule(
                    statement, kind, helpers, scalar_helpers, select_static=select_static):
                return False
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if helper_call_name(call) in helpers:
                reads = set(schedule_argument_reads(call, helpers))
                if pending_config or not all(arg in reads or _bound(arg) for arg in call.args) or any(
                    kw.arg is None or (kw.value not in reads and not _bound(kw.value)) for kw in call.keywords
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
        range_loop = isinstance(iterator.func, ast.Name) and iterator.func.id == "range"
        if kind(iterator.func) != "ndrange" and not range_loop:
            return False
        if not all(guarded_range_bound(arg, kind) if range_loop else _bound(arg) for arg in iterator.args):
            return False
        pending_config = False
    return not pending_config


def uniform_range_schedule(node, kind, helpers=(), scalar_helpers=(), *, select_static=None):
    """Prove both arms without evaluating source or mutating compiler state."""
    if not isinstance(node, ast.If):
        return False
    return bool(node.body) and all(
        uniform_range_statements(statements, kind, helpers, scalar_helpers, select_static=select_static)
        for statements in (node.body, node.orelse)
    )
