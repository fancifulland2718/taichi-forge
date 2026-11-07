"""Prove ordinary resource-factory forwarding without calling application code.

This accepts declaration forwarding and fixed frontend capability predicates,
not arbitrary Python execution, model preparation or numerical evaluation.
"""

import ast
from dataclasses import dataclass
import hashlib
import inspect
import textwrap
from types import ModuleType


class _Unproven(ValueError):
    pass


@dataclass(frozen=True)
class _Input:
    node: ast.AST


@dataclass(frozen=True)
class _ScopeInput(_Input):
    pass


@dataclass(frozen=True)
class _Method:
    receiver: _ScopeInput
    name: str


@dataclass(frozen=True)
class _Scope:
    arguments: tuple
    keywords: dict


@dataclass(frozen=True)
class _Invocation(_Scope):
    method: _Method


@dataclass(frozen=True)
class _FieldMetadata:
    """A source-contract obligation, never a concrete Field/model sample."""

    source: _Input


def _literal(value):
    return (
        value is None
        or type(value) in (bool, int, float, str)
        or (type(value) is tuple and all(_literal(item) for item in value))
    )


def _truth(value):
    if not _literal(value):
        raise _Unproven("factory condition depends on a model value")
    return bool(value)


def _frontend_module(value):
    return isinstance(value, ModuleType) and (
        value is inspect or value.__name__.split(".")[0] in ("taichi", "taichi_forge")
    )


def _source_value(node, namespace):
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.Attribute):
        parent = _source_value(node.value, namespace)
        if isinstance(parent, ModuleType):
            return vars(parent).get(node.attr)
    return None


def prove_factory(function, call, scopes=frozenset(), *, allow_void=False, absent_parameter=None):
    """Prove a scope constructor or dimension-method forwarding call."""
    from taichi_forge.lang._resource_declaration import ResourceScope

    if not inspect.isfunction(function) or "__wrapped__" in vars(function) or "__signature__" in vars(function):
        return None
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return None
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        return None
    definition = tree.body[0]
    # A candidate must mention the actual trusted declaration class, directly
    # or through a literal capability lookup, or receive an already proved
    # scope. Other Host calls remain untouched. Argument ASTs are forwarded;
    # the resource analyzer, not this proof, admits their dimension algebra.
    forwards_scope = any(
        isinstance(n, ast.Name) and n.id in scopes for n in (*call.args, *(k.value for k in call.keywords))
    )
    if (
        not forwards_scope
        and not any(_source_value(n, function.__globals__) is ResourceScope for n in ast.walk(definition))
        and not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "getattr"
            and len(n.args) >= 2
            and isinstance(n.args[1], ast.Constant)
            and n.args[1].value == "ResourceScope"
            for n in ast.walk(definition)
        )
    ):
        return None
    environment = {}
    signature = inspect.signature(function, follow_wrapped=False)
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values()):
        raise _Unproven("factory forwarding requires explicit parameters")

    def input_value(node):
        if isinstance(node, ast.Name) and node.id in scopes:
            return _ScopeInput(node)
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            if isinstance(node, ast.Name) or forwards_scope:
                return _Input(node)
            raise _Unproven("factory arguments require literal values or forwarded local names") from None
        if not _literal(value):
            raise _Unproven("factory arguments require immutable literal values")
        return value

    if any(k.arg is None for k in call.keywords):
        raise _Unproven("factory forwarding cannot infer expanded keyword arguments")
    bound = signature.bind(*(input_value(n) for n in call.args), **{k.arg: input_value(k.value) for k in call.keywords})
    bound.apply_defaults()
    if absent_parameter is not None:
        bound.arguments[absent_parameter] = None
    for name, value in bound.arguments.items():
        if not isinstance(value, _Input) and not _literal(value):
            raise _Unproven("factory defaults require immutable literal values")
        environment[name] = value
    trusted_builtins = dict(getattr=getattr, hasattr=hasattr, all=all, type=type, isinstance=isinstance)
    declaration_methods = {
        name: vars(ResourceScope)[name]
        for name in ("dim", "derived_dim", "power_of_two_dim", "borrow", "record", "owner_reference", "constant")
    }
    constructions = []
    field_obligations = {}
    nonstruct_obligations = {}
    builtin_namespace = function.__builtins__
    for name, value in trusted_builtins.items():
        if function.__globals__.get(name, value) is not value or builtin_namespace.get(name) is not value:
            raise _Unproven("factory capability checks cannot use shadowed builtins")

    def expression(node, local=None):
        local = environment if local is None else local
        if isinstance(node, ast.Constant) and _literal(node.value):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in local:
                return local[node.id]
            value = function.__globals__.get(node.id, trusted_builtins.get(node.id))
            if _frontend_module(value) or value is ResourceScope or any(value is v for v in trusted_builtins.values()):
                return value
            raise _Unproven("factory depends on an application global or unknown callable")
        if isinstance(node, (ast.Tuple, ast.List)):
            values = tuple(expression(n, local) for n in node.elts)
            if _literal(values):
                return values
        if isinstance(node, ast.Attribute):
            parent = expression(node.value, local)
            if _frontend_module(parent):
                if node.attr not in vars(parent):
                    raise _Unproven("factory references an unavailable frontend attribute")
                return vars(parent)[node.attr]
            if isinstance(parent, frozenset) and node.attr == "parameters":
                return parent
            if isinstance(parent, _ScopeInput) and node.attr in declaration_methods:
                return _Method(parent, node.attr)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not _truth(expression(node.operand, local))
        if isinstance(node, ast.BoolOp):
            for term in node.values:
                value = expression(term, local)
                if (isinstance(node.op, ast.And) and not _truth(value)) or (
                    isinstance(node.op, ast.Or) and _truth(value)
                ):
                    return value
            return value
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = expression(node.left, local), expression(node.comparators[0], local)
            op = node.ops[0]
            if isinstance(op, (ast.Is, ast.IsNot)) and right is None and type(left) is _Input:
                # This conditional proof is valid only if the resulting borrow
                # requires this exact input's declared Field provenance below.
                field_obligations[id(left)] = left
                return isinstance(op, ast.IsNot)
            if isinstance(op, (ast.Is, ast.IsNot)) and right is None and not isinstance(left, _Input):
                return (left is None) if isinstance(op, ast.Is) else (left is not None)
            if _literal(left) and (_literal(right) or isinstance(right, frozenset)):
                if isinstance(op, ast.Eq):
                    return left == right
                if isinstance(op, ast.NotEq):
                    return left != right
                if isinstance(op, (ast.In, ast.NotIn)):
                    result = left in right
                    return result if isinstance(op, ast.In) else not result
        if isinstance(node, ast.IfExp):
            return expression(node.body if _truth(expression(node.test, local)) else node.orelse, local)
        if isinstance(node, ast.Call) and not node.keywords:
            function_value = expression(node.func, local)
            if function_value is all and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
                generator = node.args[0]
                if len(generator.generators) != 1:
                    raise _Unproven("factory capability generator requires one literal sequence")
                loop = generator.generators[0]
                values = expression(loop.iter, local)
                if (
                    not isinstance(loop.target, ast.Name)
                    or loop.ifs
                    or loop.is_async
                    or not _literal(values)
                    or type(values) is not tuple
                ):
                    raise _Unproven("factory capability checks require literal names")
                return all(_truth(expression(generator.elt, {**local, loop.target.id: item})) for item in values)
            args = [expression(arg, local) for arg in node.args]
            if function_value is type and len(args) == 1 and isinstance(args[0], _ScopeInput):
                return ResourceScope
            if function_value is isinstance and len(args) == 2 and type(args[0]) is _Input:
                from taichi_forge.lang.struct import StructField

                if args[1] is StructField:
                    field_obligations[id(args[0])] = args[0]
                    nonstruct_obligations[id(args[0])] = args[0]
                    return False  # This helper branch explicitly excludes StructField inputs.
            if (function_value is getattr or function_value is hasattr) and len(args) in (2, 3):
                if function_value is hasattr and len(args) != 2:
                    raise _Unproven("hasattr capability checks require two arguments")
                target, name = args[:2]
                if type(target) is _Input and name == "_logical_resource_declaration":
                    field_obligations[id(target)] = target
                    return True if function_value is hasattr else _FieldMetadata(target)
                if type(name) is not str or not (_frontend_module(target) or target is ResourceScope):
                    raise _Unproven("factory attribute checks require frontend modules or ResourceScope")
                sentinel = object()
                value = vars(target).get(name, sentinel)
                if function_value is hasattr:
                    return value is not sentinel
                if value is sentinel and len(args) == 2:
                    raise _Unproven("factory capability is unavailable")
                return args[2] if value is sentinel else value
            if (
                function_value is inspect.signature
                and len(args) == 1
                and (args[0] is ResourceScope or any(args[0] is method for method in declaration_methods.values()))
            ):
                return frozenset(inspect.signature(args[0]).parameters)
        if isinstance(node, ast.Call):
            target = expression(node.func, local)
            args = tuple(expression(arg, local) for arg in node.args)
            method = next((name for name, value in declaration_methods.items() if target is value), None)
            if method is not None and args and isinstance(args[0], _ScopeInput):
                target, args = _Method(args[0], method), args[1:]
            if isinstance(target, _Method):
                if any(k.arg is None for k in node.keywords):
                    raise _Unproven("declaration forwarding requires explicit keywords")
                invocation = _Invocation(args, {k.arg: expression(k.value, local) for k in node.keywords}, target)
                constructions.append(invocation)
                return invocation
        if isinstance(node, ast.Call) and expression(node.func, local) is ResourceScope:
            if any(k.arg is None for k in node.keywords):
                raise _Unproven("resource constructor forwarding requires explicit keywords")
            scope = _Scope(
                tuple(expression(arg, local) for arg in node.args),
                {k.arg: expression(k.value, local) for k in node.keywords},
            )
            constructions.append(scope)
            return scope
        raise _Unproven(f"unproven factory expression at line {getattr(node, 'lineno', '?')}")

    fallthrough = object()

    def statements(body):
        for node in body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and type(node.value.value) is str:
                continue
            if isinstance(node, ast.Import) and all(n.name == "inspect" for n in node.names):
                environment.update((n.asname or n.name, inspect) for n in node.names)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                environment[node.targets[0].id] = expression(node.value)
            elif isinstance(node, ast.If):
                result = statements(node.body if _truth(expression(node.test)) else node.orelse)
                if result is not fallthrough:
                    return result
            elif isinstance(node, ast.Return):
                return expression(node.value) if node.value is not None else None
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if not isinstance(expression(node.value), _Invocation):
                    raise _Unproven("factory statements may only invoke declarations")
            else:
                raise _Unproven(f"factory has an unproven statement or side effect at line {node.lineno}")
        return fallthrough

    result = statements(definition.body)
    if allow_void and (result is fallthrough or result is None) and len(constructions) == 1:
        result = constructions[0]
    if not isinstance(result, _Scope) or len(constructions) != 1 or result is not constructions[0]:
        raise _Unproven("factory must return its single supported resource declaration on this capability path")
    forwarded = (*result.arguments, *result.keywords.values())
    if isinstance(result, _Invocation):
        forwarded = (*forwarded, result.method.receiver)
    if field_obligations:
        if (
            not isinstance(result, _Invocation)
            or result.method.name != "borrow"
            or len(result.arguments) != 2
            or result.keywords.get("source_scope") is not True
            or result.keywords.get("external", False) is not False
            or any(value is not result.arguments[1] for value in field_obligations.values())
        ):
            raise _Unproven("Field metadata branches require the same input's proven source-scope borrow")
    if absent_parameter is None and any(
        isinstance(value, _Input) and not any(item is value for item in forwarded) for value in bound.arguments.values()
    ):
        raise _Unproven("factory drops a nonliteral argument")

    def syntax(value):
        if isinstance(value, _Input):
            return value.node
        if _literal(value):
            return ast.parse(repr(value), mode="eval").body
        raise _Unproven("factory forwards an unproven resource argument")

    normalized = ast.Call(
        (
            ast.Attribute(result.method.receiver.node, result.method.name, ast.Load())
            if isinstance(result, _Invocation)
            else call.func
        ),
        [syntax(v) for v in result.arguments],
        [ast.keyword(k, syntax(v)) for k, v in result.keywords.items() if not (k == "builder" and v is None)],
    )
    if not isinstance(result, _Invocation):
        normalized._package_role = "experimental.ResourceScope"
    normalized._package_factory = dict(
        definition=function.__module__ + "." + function.__qualname__, sha256=hashlib.sha256(source.encode()).hexdigest()
    )
    if field_obligations and allow_void and absent_parameter is None:
        normalized._package_field_obligation = True
        normalized._package_nonstruct_borrow = bool(nonstruct_obligations)
        parameter = next(name for name, value in bound.arguments.items() if value is result.arguments[1])
        try:
            absence = prove_factory(function, call, scopes, allow_void=True, absent_parameter=parameter)
        except (TypeError, _Unproven):
            absence = None
        if (
            absence is not None
            and isinstance(absence.func, ast.Attribute)
            and absence.func.attr == "record"
            and ast.dump(absence.func.value) == ast.dump(normalized.func.value)
            and len(absence.args) == 2
            and ast.dump(absence.args[0]) == ast.dump(normalized.args[0])
            and isinstance(absence.args[1], ast.Constant)
            and absence.args[1].value is None
            and len(absence.keywords) == 1
            and absence.keywords[0].arg == "optional"
            and isinstance(absence.keywords[0].value, ast.Constant)
            and absence.keywords[0].value.value is True
        ):
            normalized._package_optional_borrow = True
    return ast.copy_location(normalized, call)


def normalize_factories(tree, namespace, module, fail, kind):
    """Normalize proved allocation declarations in constructors, not Host calls."""
    declarations = []
    from ._package_types import qualified_kind
    from ._package_declarations import expand_declarations

    replacements = {}

    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for function in cls.body:
            if not isinstance(function, ast.FunctionDef) or not (
                function.name == "__init__" or getattr(function, "_package_construction_method", False)
            ):
                continue
            scopes = set()

            def statements(body):
                for statement in body:
                    if getattr(statement, "_package_resource_context", None) is not None:
                        scopes.add(statement.items[0].optional_vars.id)
                        yield from statements(statement.body)
                    elif isinstance(statement, ast.If):
                        yield from statements(statement.body)
                        yield from statements(statement.orelse)
                    else:
                        yield statement

            for statement in statements(function.body):
                if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)) or not isinstance(
                    statement.value, ast.Call
                ):
                    continue
                call = statement.value
                target = _source_value(call.func, namespace)
                expanded = None
                try:
                    normalized = prove_factory(target, call, scopes, allow_void=isinstance(statement, ast.Expr))
                except (TypeError, _Unproven) as error:
                    try:
                        expanded = expand_declarations(target, statement, scopes)
                    except (TypeError, _Unproven) as expansion_error:
                        fail(module, call, f"Resource factory forwarding is unproven: {expansion_error}")
                    if expanded is None:
                        fail(module, call, f"Resource factory forwarding is unproven: {error}")
                    normalized = None
                if normalized is not None or expanded is not None:
                    root = call.func
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if not isinstance(root, ast.Name):
                        fail(module, call, "Resource factory requires an unshadowed module binding")
                    bindings = [
                        node
                        for node in ast.walk(function)
                        if (isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load) and node.id == root.id)
                        or (isinstance(node, ast.arg) and node.arg == root.id)
                        or (isinstance(node, (ast.Global, ast.Nonlocal)) and root.id in node.names)
                        or (isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == root.id)
                    ]
                    if bindings:
                        fail(module, call, "Resource factory binding is shadowed or reassigned in the constructor")
                    if expanded is not None:
                        body, facts = expanded
                        replacements[id(statement)] = body
                        declarations.append(
                            {**facts, "kind": "declaration_helper", "module": module, "line": call.lineno}
                        )
                    else:
                        statement.value = normalized
                        declarations.append({**normalized._package_factory, "module": module, "line": call.lineno})
                if (
                    qualified_kind(statement.value, kind) == "experimental.ResourceScope"
                    and isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                ):
                    scopes.add(statement.targets[0].id)

    class Replace(ast.NodeTransformer):
        def visit(self, node):
            replacement = replacements.get(id(node))
            return replacement if replacement is not None else super().visit(node)

    Replace().visit(tree)
    return declarations
