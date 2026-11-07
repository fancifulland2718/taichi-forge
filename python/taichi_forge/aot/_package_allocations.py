"""Project pure local allocation forwarding onto existing resource declarations."""

import ast
import copy
import hashlib
import inspect
import textwrap

from ._package_construction import construction_functions, ordinary_owner_lookup
from ._package_types import qualified_kind

_FIELDS = frozenset(("field", "vector_field", "matrix_field", "struct_field"))


def normalize_allocation_helpers(tree, namespace, kind, module, fail):
    facts = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for function in construction_functions(cls):
            scopes = {
                node.items[0].optional_vars.id
                for node in ast.walk(function)
                if getattr(node, "_package_resource_context", None) is not None
            }
            scopes.update(
                node.targets[0].id
                for node in ast.walk(function)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and qualified_kind(node.value, kind) == "experimental.ResourceScope"
            )
            helpers = {}
            for node in ast.walk(function):
                if not isinstance(node, ast.FunctionDef) or node is function:
                    continue
                body = [
                    n
                    for n in node.body
                    if not (
                        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and type(n.value.value) is str
                    )
                ]
                if len(body) != 1 or not isinstance(body[0], ast.Return):
                    continue
                value = body[0].value
                allocation = value.body if isinstance(value, ast.IfExp) else value
                if not (
                    isinstance(allocation, ast.Call)
                    and isinstance(allocation.func, ast.Attribute)
                    and isinstance(allocation.func.value, ast.Name)
                    and allocation.func.value.id in scopes
                    and allocation.func.attr in _FIELDS
                ):
                    continue
                args = node.args
                if (
                    node.decorator_list
                    or args.defaults
                    or any(n is not None for n in args.kw_defaults)
                    or args.vararg
                    or args.kwarg
                    or node.returns
                    or any(a.annotation for a in (*args.posonlyargs, *args.args, *args.kwonlyargs))
                ):
                    fail(module, node, "Local allocation helpers require plain explicit parameters")
                if isinstance(value, ast.IfExp) and not (
                    isinstance(value.orelse, ast.Constant) and value.orelse.value is None
                ):
                    fail(module, node, "Conditional allocation helpers require an absent None branch")
                if node.name in helpers:
                    fail(module, node, "Local allocation helper cannot be redefined")
                parameters = [inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_ONLY) for a in args.posonlyargs]
                parameters += [inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for a in args.args]
                parameters += [inspect.Parameter(a.arg, inspect.Parameter.KEYWORD_ONLY) for a in args.kwonlyargs]
                helpers[node.name] = (node, value, inspect.Signature(parameters))
            if not helpers:
                continue
            allowed = set()
            for node in ast.walk(function):
                if isinstance(node, ast.Name) and node.id in helpers and not isinstance(node.ctx, ast.Load):
                    fail(module, node, "Local allocation helper cannot be reassigned")
            serial = 0

            class Expand(ast.NodeTransformer):
                def visit_Expr(self, node):
                    nonlocal serial
                    call = node.value
                    if not (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "self"
                        and any(
                            isinstance(n, ast.Name) and n.id in helpers
                            for n in (*call.args, *(k.value for k in call.keywords))
                        )
                    ):
                        return self.generic_visit(node)
                    owner = namespace.get(cls.name)
                    if not ordinary_owner_lookup(owner) or any(
                        isinstance(n, ast.Attribute)
                        and isinstance(n.value, ast.Name)
                        and n.value.id == "self"
                        and n.attr == call.func.attr
                        and not isinstance(n.ctx, ast.Load)
                        for n in ast.walk(cls)
                    ):
                        fail(module, call, "Allocation forwarding requires unchanged ordinary method lookup")
                    method = inspect.getattr_static(owner, call.func.attr, None)
                    if not inspect.isfunction(method) or "__wrapped__" in vars(method):
                        fail(module, call, "Allocation forwarding requires an ordinary owner method")
                    try:
                        source = textwrap.dedent(inspect.getsource(method))
                        definition = ast.parse(source).body[0]
                        if not isinstance(definition, ast.FunctionDef) or definition.decorator_list:
                            raise ValueError("plain method required")
                        bound = inspect.signature(method).bind(
                            ast.Name("self", ast.Load()), *call.args, **{k.arg: k.value for k in call.keywords}
                        )
                    except (TypeError, ValueError, OSError) as error:
                        fail(module, call, f"Unproven allocation forwarding method: {error}")
                    environment = dict(bound.arguments)
                    if any(
                        not isinstance(value, ast.Name) or value.id not in {*scopes, *helpers, "self"}
                        for value in environment.values()
                    ):
                        fail(module, call, "Allocation forwarding method accepts only scopes and local allocators")
                    allowed.update(id(n) for n in environment.values() if n.id in helpers)
                    body = []

                    class BindMethod(ast.NodeTransformer):
                        def visit_Name(self, value):
                            if value.id in environment:
                                result = copy.deepcopy(environment[value.id])
                                result.ctx = value.ctx
                                return result
                            original = method.__globals__.get(value.id)
                            if original is None or namespace.get(value.id) is not original:
                                fail(module, call, "Allocation forwarding global requires the same visible definition")
                            return value

                    for statement in definition.body:
                        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                            continue
                        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                            fail(
                                module, call, "Allocation forwarding method must only declare local extents and fields"
                            )
                        target = statement.targets[0]
                        if isinstance(target, ast.Name):
                            if target.id in environment:
                                fail(
                                    module,
                                    call,
                                    "Allocation forwarding local cannot replace an input or previous value",
                                )
                            serial += 1
                            environment[target.id] = ast.Name(
                                f"_forge_package_alloc_{call.lineno}_{serial}_{target.id}", ast.Load()
                            )
                        copied = BindMethod().visit(copy.deepcopy(statement))
                        value, target = copied.value, copied.targets[0]
                        if not isinstance(value, ast.Call) or not (
                            isinstance(target, ast.Name)
                            and isinstance(value.func, ast.Attribute)
                            and isinstance(value.func.value, ast.Name)
                            and value.func.value.id in scopes
                            and value.func.attr in ("dim", "derived_dim", "parent_dim")
                            or isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id in scopes
                            and isinstance(value.func, ast.Name)
                            and value.func.id in helpers
                        ):
                            fail(
                                module,
                                call,
                                "Allocation forwarding method may only call its declared scope or allocator",
                            )
                        for part in ast.walk(copied):
                            if isinstance(part, (ast.expr, ast.stmt)):
                                part.lineno = part.end_lineno = call.lineno
                                part.col_offset = call.col_offset + len(body) + 1
                                part.end_col_offset = part.col_offset + 1
                        body.append(copied)
                    cls._package_allocation_methods = {
                        **getattr(cls, "_package_allocation_methods", {}),
                        call.func.attr: method,
                    }
                    facts.append(
                        dict(
                            kind="allocation_helper",
                            definition=method.__module__ + "." + method.__qualname__,
                            sha256=hashlib.sha256(source.encode()).hexdigest(),
                            line=call.lineno,
                        )
                    )
                    result = []
                    for statement in body:
                        expanded = self.visit(statement)
                        result.extend(expanded if isinstance(expanded, list) else [expanded])
                    return result

                def visit_FunctionDef(self, node):
                    return None if node.name in helpers else self.generic_visit(node)

                def visit_Assign(self, node):
                    nonlocal serial
                    call = node.value
                    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in helpers):
                        return self.generic_visit(node)
                    if not (
                        len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Attribute)
                        and isinstance(node.targets[0].value, ast.Name)
                        and node.targets[0].value.id in scopes
                    ):
                        fail(module, node, "Allocation helper result requires a declared scope member")
                    definition, value, signature = helpers[call.func.id]
                    allocation = value.body if isinstance(value, ast.IfExp) else value
                    if allocation.func.value.id != node.targets[0].value.id:
                        fail(module, call, "Allocation helper must assign into its original resource scope")
                    if call.lineno <= definition.lineno or any(k.arg is None for k in call.keywords):
                        fail(module, call, "Allocation helper requires a preceding definition and explicit arguments")
                    try:
                        bound = signature.bind(*call.args, **{k.arg: k.value for k in call.keywords})
                    except TypeError as error:
                        fail(module, call, str(error))
                    allowed.add(id(call.func))
                    references, prefix = {}, []
                    # Original Python evaluates all supplied arguments before
                    # entering the helper, including a disabled allocation.
                    for expression in (*call.args, *(k.value for k in call.keywords)):
                        name = next(name for name, actual in bound.arguments.items() if actual is expression)
                        serial += 1
                        local = f"_forge_package_alloc_{call.lineno}_{serial}_{name}"
                        prefix.append(ast.copy_location(ast.Assign([ast.Name(local, ast.Store())], expression), node))
                        reference = ast.Name(local, ast.Load())
                        reference._package_forwarded_argument = copy.deepcopy(expression)
                        references[name] = reference

                    class Bind(ast.NodeTransformer):
                        def visit_Name(self, node):
                            return copy.deepcopy(references.get(node.id, node))

                    result = Bind().visit(copy.deepcopy(value))
                    for part in ast.walk(result):
                        if isinstance(part, (ast.expr, ast.stmt)):
                            ast.copy_location(part, call)
                    node.value = result
                    return [*prefix, self.generic_visit(node)]

            original_uses = [n for n in ast.walk(function) if isinstance(n, ast.Name) and n.id in helpers]
            Expand().visit(function)
            if any(id(node) not in allowed for node in original_uses):
                fail(module, function, "Local allocation helper may only forward directly to declared members")
    return facts
