"""Expand proven declaration helpers, not numerical bodies or model execution."""

import ast
import copy
import hashlib
import inspect
import textwrap

from ._package_factories import _Unproven, _literal


def expand_declarations(function, statement, scopes):
    """Inline lexical declaration forwarding with literal control and Host paths.

    Application functions are never called. Only ResourceScope operations are
    admitted; normal dimension/type analyzers still decide their contracts.
    Hygienic local names keep repeated declaration helpers and fixed loops
    independent without creating numerical specializations.
    """
    call = statement.value
    if not inspect.isfunction(function) or any(k in vars(function) for k in ("__wrapped__", "__signature__")):
        return None
    if not any(isinstance(n, ast.Name) and n.id in scopes for n in (*call.args, *(k.value for k in call.keywords))):
        return None
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return None
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        return None
    signature = inspect.signature(function, follow_wrapped=False)
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values()):
        raise _Unproven("declaration helper requires explicit parameters")
    if any(k.arg is None for k in call.keywords):
        raise _Unproven("declaration helper cannot infer expanded arguments")
    inputs = []

    def input_node(node):
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name):
            if isinstance(node, ast.Name) and node.id in scopes:
                return copy.deepcopy(node)
            name = f"_forge_package_decl_{call.lineno}_{call.col_offset}_input_{len(inputs)}"
            inputs.append(ast.Assign([ast.Name(name, ast.Store())], copy.deepcopy(node)))
            reference = ast.Name(name, ast.Load())
            # Preserve attribute/type provenance without evaluating descriptors
            # during export. Host evaluates the complete path once, in normal
            # argument order, including unused inputs and any getter exception.
            reference._package_forwarded_argument = copy.deepcopy(node)
            return reference
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            raise _Unproven("declaration helper inputs require names, attribute paths or immutable literals") from None
        if not _literal(value):
            raise _Unproven("declaration helper inputs require immutable literals")
        return ast.parse(repr(value), mode="eval").body

    bound = signature.bind(*(input_node(n) for n in call.args), **{k.arg: input_node(k.value) for k in call.keywords})
    for name, parameter in signature.parameters.items():
        if name not in bound.arguments:
            if not _literal(parameter.default):
                raise _Unproven("declaration helper defaults require immutable literals")
            bound.arguments[name] = ast.parse(repr(parameter.default), mode="eval").body
    environment = dict(bound.arguments)
    known_scopes = set(scopes)
    emitted = []
    serial = 0
    declaration_calls = 0
    operations = frozenset(
        ("record", "dim", "parent_dim", "derived_dim", "power_of_two_dim", "static", "constant", "value", "require_le")
    )

    def literal(node):
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            raise _Unproven("declaration helper control requires literal build inputs, not model values") from None
        if not _literal(value):
            raise _Unproven("declaration helper control requires immutable literals")
        return value

    def expression(node):
        nonlocal declaration_calls
        if isinstance(node, ast.Constant) and _literal(node.value):
            return copy.deepcopy(node)
        if isinstance(node, ast.Name):
            if node.id not in environment:
                raise _Unproven(f"declaration helper references an unbound name {node.id!r}")
            return copy.deepcopy(environment[node.id])
        if isinstance(node, ast.Attribute):
            return ast.Attribute(expression(node.value), node.attr, ast.Load())
        if isinstance(node, (ast.Tuple, ast.List)):
            return type(node)([expression(n) for n in node.elts], ast.Load())
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
            left, right = expression(node.left), expression(node.right)
            if isinstance(node.op, ast.Add) and all(
                isinstance(n, ast.Constant) and type(n.value) is str for n in (left, right)
            ):
                return ast.Constant(left.value + right.value)
            return ast.BinOp(left, node.op, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ast.Constant(not bool(literal(expression(node.operand))))
        if isinstance(node, ast.BoolOp):
            for term in node.values:
                value = expression(term)
                truth = bool(literal(value))
                if isinstance(node.op, ast.And) and not truth or isinstance(node.op, ast.Or) and truth:
                    break
            return value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = literal(expression(node.left)), literal(expression(node.comparators[0]))
            op = node.ops[0]
            if isinstance(op, (ast.Eq, ast.NotEq)):
                return ast.Constant(left == right if isinstance(op, ast.Eq) else left != right)
            if isinstance(op, (ast.Is, ast.IsNot)) and right is None:
                return ast.Constant(left is None if isinstance(op, ast.Is) else left is not None)
        if isinstance(node, ast.Call):
            function_node = expression(node.func)
            if not (
                isinstance(function_node, ast.Attribute)
                and isinstance(function_node.value, ast.Name)
                and function_node.value.id in known_scopes
                and function_node.attr in operations
                and all(k.arg is not None for k in node.keywords)
            ):
                raise _Unproven("declaration helpers may only call supported ResourceScope declaration methods")
            declaration_calls += 1
            return ast.Call(
                function_node,
                [expression(n) for n in node.args],
                [ast.keyword(k.arg, expression(k.value)) for k in node.keywords],
            )
        raise _Unproven(f"unsupported declaration helper expression {type(node).__name__}")

    def emit(node):
        # Sites stay attached to the original call, with distinct columns for
        # individual declarations. No fabricated file or model execution trace.
        for child in ast.walk(node):
            if isinstance(child, (ast.expr, ast.stmt)):
                child.lineno = child.end_lineno = call.lineno
                child.col_offset = call.col_offset + len(emitted) + 1
                child.end_col_offset = child.col_offset + 1
        emitted.append(node)

    for node in inputs:
        emit(node)

    def local(name):
        nonlocal serial
        serial += 1
        return f"_forge_package_decl_{call.lineno}_{call.col_offset}_{serial}_{name}"

    def assign(target, value, *, fixed=False):
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (ast.Tuple, ast.List)) or len(target.elts) != len(value.elts):
                raise _Unproven("declaration helper unpacking requires a fixed matching sequence")
            for member, item in zip(target.elts, value.elts):
                assign(member, item, fixed=fixed)
        elif isinstance(target, ast.Name):
            if fixed:
                literal(value)
                environment[target.id] = copy.deepcopy(value)
            else:
                name = local(target.id)
                environment[target.id] = ast.Name(name, ast.Load())
                emit(ast.Assign([ast.Name(name, ast.Store())], value))
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "record"
                ):
                    known_scopes.add(name)
        elif isinstance(target, ast.Attribute) and not fixed:
            receiver = expression(target.value)
            if not isinstance(receiver, ast.Name) or receiver.id not in known_scopes:
                raise _Unproven("declaration helper may only assign declared scope members")
            emit(ast.Assign([ast.Attribute(receiver, target.attr, ast.Store())], value))
        else:
            raise _Unproven("unsupported declaration helper assignment")

    fallthrough, continued = object(), object()

    def statements(body, *, in_loop=False):
        for node in body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and type(node.value.value) is str:
                continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                assign(node.targets[0], expression(node.value))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                emit(ast.Expr(expression(node.value)))
            elif isinstance(node, ast.If):
                result = statements(node.body if bool(literal(expression(node.test))) else node.orelse, in_loop=in_loop)
                if result is not fallthrough:
                    return result
            elif isinstance(node, ast.For) and not node.orelse:
                values = expression(node.iter)
                if not isinstance(values, (ast.Tuple, ast.List)):
                    raise _Unproven("declaration helper loops require a fixed literal sequence")
                literal(values)
                for value in values.elts:
                    assign(node.target, value, fixed=True)
                    result = statements(node.body, in_loop=True)
                    if result is not fallthrough and result is not continued:
                        return result
            elif isinstance(node, ast.Continue) and in_loop:
                return continued
            elif isinstance(node, ast.Return):
                return expression(node.value) if node.value is not None else ast.Constant(None)
            else:
                raise _Unproven(f"declaration helper has an unproven statement or side effect at line {node.lineno}")
        return fallthrough

    returned = statements(tree.body[0].body)
    if not declaration_calls:
        raise _Unproven("declaration helper contains no ResourceScope declaration")
    if returned is fallthrough:
        returned = ast.Constant(None)
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        emit(ast.Assign(copy.deepcopy(targets), returned))
    elif not isinstance(returned, ast.Constant) or returned.value is not None:
        emit(ast.Expr(returned))
    facts = dict(
        definition=function.__module__ + "." + function.__qualname__, sha256=hashlib.sha256(source.encode()).hexdigest()
    )
    return emitted, facts
