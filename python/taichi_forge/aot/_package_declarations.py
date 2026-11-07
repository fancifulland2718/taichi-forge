"""Expand proven declaration helpers, not numerical bodies or model execution."""

import ast
import copy
import hashlib
import inspect
import textwrap

from ._package_factories import _Unproven, _literal, _source_value, prove_factory


def expand_declarations(function, statement, scopes, *, _active=(), _prefix=None):
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
    if function in _active:
        raise _Unproven("recursive declaration helper dependencies are not supported")
    active = (*_active, function)
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
    prefix = _prefix or f"_forge_package_decl_{call.lineno}_{call.col_offset}"

    def input_node(node):
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name):
            if isinstance(node, ast.Name) and node.id in scopes:
                return copy.deepcopy(node)
            name = f"{prefix}_input_{len(inputs)}"
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
    dependencies = []
    extent_names = set()
    lexical_names = {
        node.id
        for node in ast.walk(tree.body[0])
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    } | set(signature.parameters)
    lexical_names.update(
        node.asname or node.name.split(".")[0] for node in ast.walk(tree.body[0]) if isinstance(node, ast.alias)
    )
    lexical_names.update(
        node.name
        for node in ast.walk(tree.body[0])
        if node is not tree.body[0] and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    operations = frozenset(
        ("record", "dim", "parent_dim", "derived_dim", "power_of_two_dim", "static", "constant", "value", "require_le")
    )

    def extent(node):
        if isinstance(node, ast.Name):
            return node.id in extent_names
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
            return extent(node.left) or extent(node.right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            return (
                isinstance(receiver, ast.Name)
                and receiver.id in known_scopes
                and node.func.attr in ("dim", "parent_dim", "derived_dim", "power_of_two_dim")
            ) or (node.func.attr in ("maximum", "minimum") and extent(receiver))
        return False

    def literal(node):
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            raise _Unproven("declaration helper control requires literal build inputs, not model values") from None
        if not _literal(value):
            raise _Unproven("declaration helper control requires immutable literals")
        return value

    def dictionary(pairs):
        keys, values, names = [], [], set()
        for key, value in pairs:
            key = expression(key)
            name = literal(key)
            if type(name) is not str or name in names:
                raise _Unproven("declaration dictionary requires distinct literal names")
            names.add(name)
            value = expression(value)
            temporary = local("dictionary_value")
            emit(ast.Assign([ast.Name(temporary, ast.Store())], value))
            if extent(value):
                extent_names.add(temporary)
            keys.append(key)
            values.append(ast.Name(temporary, ast.Load()))
        return ast.Dict(keys, values)

    def expression(node, *, allow_helper=False, void=False):
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
        if isinstance(node, ast.Dict) and all(key is not None for key in node.keys):
            if not allow_helper:
                raise _Unproven("declaration dictionaries require their own assignment or return")
            return dictionary(zip(node.keys, node.values))
        if isinstance(node, ast.DictComp) and len(node.generators) == 1:
            if not allow_helper:
                raise _Unproven("declaration dictionaries require their own assignment or return")
            loop = node.generators[0]
            values = expression(loop.iter)
            if (
                loop.is_async
                or loop.ifs
                or not isinstance(loop.target, ast.Name)
                or not isinstance(values, (ast.Tuple, ast.List))
            ):
                raise _Unproven("declaration dictionary requires one fixed literal sequence")
            literal(values)
            previous = environment.copy()
            keys, members = [], []
            for value in values.elts:
                environment[loop.target.id] = copy.deepcopy(value)
                item = dictionary(((node.key, node.value),))
                if any(literal(key) == literal(item.keys[0]) for key in keys):
                    raise _Unproven("declaration dictionary requires distinct literal names")
                keys.extend(item.keys)
                members.extend(item.values)
            environment.clear()
            environment.update(previous)
            return ast.Dict(keys, members)
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
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value
            target = _source_value(node.func, function.__globals__)
            if inspect.isfunction(target) and isinstance(root, ast.Name) and root.id not in lexical_names:
                if not allow_helper:
                    raise _Unproven("nested declaration helper calls require their own assignment or statement")
                forwarded = ast.copy_location(
                    ast.Call(
                        copy.deepcopy(node.func),
                        [expression(n) for n in node.args],
                        [ast.keyword(k.arg, expression(k.value)) for k in node.keywords],
                    ),
                    call,
                )
                normalized = None
                try:
                    normalized = prove_factory(target, forwarded, known_scopes, allow_void=void)
                except (TypeError, _Unproven):
                    pass
                if normalized is not None:
                    if not isinstance(normalized.func, ast.Attribute) or normalized.func.attr not in operations:
                        raise _Unproven("nested helper must forward supported ResourceScope declarations")
                    declaration_calls += 1
                    dependencies.append(normalized._package_factory)
                    return normalized
                name = local("result")
                nested = expand_declarations(
                    target,
                    ast.Assign([ast.Name(name, ast.Store())], forwarded),
                    known_scopes,
                    _active=active,
                    _prefix=name,
                )
                if nested is None:
                    raise _Unproven("nested declaration helper has no proved scope forwarding")
                body, facts = nested
                for child in body:
                    emit(child)
                    if isinstance(child, ast.Assign) and extent(child.value):
                        extent_names.update(t.id for t in child.targets if isinstance(t, ast.Name))
                declaration_calls += 1
                dependencies.append(facts)
                return ast.Name(name, ast.Load())
            function_node = expression(node.func)
            if (
                isinstance(function_node, ast.Attribute)
                and function_node.attr in ("maximum", "minimum")
                and extent(function_node.value)
                and len(node.args) == 1
                and not node.keywords
            ):
                return ast.Call(function_node, [expression(node.args[0])], [])
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
        return f"{prefix}_{serial}_{name}"

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
                if extent(value):
                    extent_names.add(name)
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
                assign(node.targets[0], expression(node.value, allow_helper=True))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                emit(ast.Expr(expression(node.value, allow_helper=True, void=True)))
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
                return expression(node.value, allow_helper=True) if node.value is not None else ast.Constant(None)
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
    if dependencies:
        facts["dependencies"] = dependencies
    return emitted, facts
