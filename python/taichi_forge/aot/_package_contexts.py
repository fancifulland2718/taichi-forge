"""Discover allocation boundaries inside ordinary generator context managers.

This reads declarations; it neither executes the generator nor substitutes its
Host lifetime behavior. The source recipe remains necessary for consumption.
"""

import ast
import contextlib
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap

from ._package_factories import _literal, _source_value, _Unproven, prove_factory


def resource_context_group(statement):
    """Keep independently defined preparation allocations in separate groups."""
    local = f"context_{statement.lineno}_{statement.col_offset}"
    definition = getattr(statement, "_package_group_definition", None)
    if definition is None:
        return local
    return "prepare_" + hashlib.sha256(definition.encode("utf8")).hexdigest() + "_" + local


def _generator_context(function):
    if not inspect.isfunction(function):
        return None
    wrapped = getattr(function, "__wrapped__", None)
    # Inspect the actual stdlib wrapper, not an application's decorator name.
    if (
        not inspect.isgeneratorfunction(wrapped)
        or function.__code__ is not contextlib.contextmanager(lambda: None).__code__
        or inspect.getclosurevars(function).nonlocals.get("func") is not wrapped
        or "__signature__" in vars(function)
    ):
        return None
    source = textwrap.dedent(inspect.getsource(wrapped))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    definition = tree.body[0]
    if (
        len(definition.decorator_list) != 1
        or _source_value(definition.decorator_list[0], wrapped.__globals__) is not contextlib.contextmanager
    ):
        return None
    return wrapped, source, definition


def prove_context(function, call):
    """Return a scoped allocation recipe, or None for unrelated Host contexts."""
    import taichi_forge as ti
    from taichi_forge.lang._resource_declaration import ResourceScope

    candidate = _generator_context(function)
    if candidate is None:
        return None
    wrapped, source, definition = candidate
    allocations = [
        node
        for node in ast.walk(definition)
        if isinstance(node, ast.Call) and _source_value(node.func, wrapped.__globals__) is ti.FieldsBuilder
    ]
    if not allocations:
        return None
    if any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in ("globals", "locals", "eval", "exec", "__import__")
        for n in ast.walk(definition)
    ):
        raise _Unproven("context allocation cannot depend on dynamic namespace inspection")
    if len(allocations) != 1:
        raise _Unproven("context requires one unconditional FieldsBuilder allocation")
    signature = inspect.signature(wrapped, follow_wrapped=False)
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values()):
        raise _Unproven("context forwarding requires explicit parameters")
    if any(k.arg is None for k in call.keywords):
        raise _Unproven("context arguments cannot use keyword expansion")
    bound = signature.bind(*call.args, **{k.arg: k.value for k in call.keywords})
    bound.apply_defaults()
    arguments = {}
    for name, value in bound.arguments.items():
        if not isinstance(value, ast.AST):
            if not _literal(value):
                raise _Unproven("context defaults require immutable literal values")
            value = ast.parse(repr(value), mode="eval").body
        arguments[name] = value
    bindings = {
        n.id for n in ast.walk(definition) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
    }
    if bindings.intersection(arguments):
        raise _Unproven("context parameters must not be reassigned")

    class Parameters(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in arguments:
                value = ast.copy_location(copy.deepcopy(arguments[node.id]), node)
                value._package_context_parameter = node.id
                return value
            return node

    builder_call = allocations[0]
    local_names = bindings | arguments.keys()

    def unshadowed(node):
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) and node.id not in local_names

    if not unshadowed(builder_call.func):
        raise _Unproven("context allocation requires an unshadowed frontend binding")
    builders = [
        n
        for n in definition.body
        if isinstance(n, ast.Assign)
        and n.value is builder_call
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
    ]
    if len(builders) != 1 or builder_call.args or builder_call.keywords:
        raise _Unproven("context requires one unconditional FieldsBuilder allocation")
    builder = builders[0].targets[0].id
    if (
        sum(isinstance(n, ast.Name) and n.id == builder and isinstance(n.ctx, ast.Store) for n in ast.walk(definition))
        != 1
    ):
        raise _Unproven("context builder must not be reassigned")
    scopes = []
    for node in definition.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        target = _source_value(node.value.func, wrapped.__globals__)
        forwarded = Parameters().visit(copy.deepcopy(node.value))
        if target is ResourceScope:
            normalized = forwarded
        else:
            normalized = prove_factory(target, forwarded)
        if normalized is None:
            continue
        if not unshadowed(node.value.func):
            raise _Unproven("context scope factory requires an unshadowed module binding")
        keywords = {k.arg: k.value for k in normalized.keywords}
        owner = normalized.args[0] if len(normalized.args) == 1 else None
        scope_builder = keywords.get("builder")
        if (
            len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or not isinstance(owner, ast.Name)
            or owner.id != "self"
            or not isinstance(scope_builder, ast.Name)
            or scope_builder.id != builder
            or set(keywords) - {"builder", "initialization"}
            or len(keywords) != len(normalized.keywords)
            or node.lineno <= builder_call.lineno
        ):
            raise _Unproven("context must forward its owner and unique builder to ResourceScope")
        policy = keywords.get("initialization", ast.Constant("consumer_explicit"))
        if not isinstance(policy, ast.Constant) or policy.value not in ("consumer_explicit", "zero"):
            raise _Unproven("context initialization requires a fixed existing policy")
        scopes.append((node.targets[0].id, node, normalized))
    if len(scopes) != 1:
        raise _Unproven("context must construct one ResourceScope before yielding")
    scope, allocation, normalized = scopes[0]
    yields = [n for n in ast.walk(definition) if isinstance(n, (ast.Yield, ast.YieldFrom))]
    if len(yields) != 1 or not isinstance(yields[0], ast.Yield):
        raise _Unproven("context must yield exactly its one declared scope")
    yielded = yields[0]
    if not isinstance(yielded.value, ast.Name) or yielded.value.id != scope:
        raise _Unproven("context must yield exactly its one declared scope")
    parents = {id(child): parent for parent in ast.walk(definition) for child in ast.iter_child_nodes(parent)}
    current = parents[id(yielded)]
    if not isinstance(current, ast.Expr):
        raise _Unproven("context yield cannot receive a value")
    while current is not definition:
        parent = parents[id(current)]
        if not isinstance(parent, (ast.FunctionDef, ast.Try)) or current not in parent.body:
            raise _Unproven("context allocation yield must be unconditional, optionally within try")
        current = parent
    if yielded.lineno <= allocation.lineno:
        raise _Unproven("context must allocate its scope before yielding")
    scope_uses = [n for n in ast.walk(definition) if isinstance(n, ast.Name) and n.id == scope]
    if any(n is not allocation.targets[0] and n is not yielded.value for n in scope_uses):
        raise _Unproven("context scope cannot escape or acquire hidden declarations")
    finalizers = [
        n
        for n in ast.walk(definition)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == builder
        and n.func.attr == "finalize"
        and not n.args
        and not n.keywords
    ]
    if not any(n.lineno > yielded.lineno for n in finalizers):
        raise _Unproven("context must retain an explicit finalization after its yield")
    permitted = {id(builders[0].targets[0]), *(id(n.func.value) for n in finalizers)}
    permitted.update(
        id(n.value)
        for n in allocation.value.keywords
        if n.arg == "builder" and isinstance(n.value, ast.Name) and n.value.id == builder
    )
    if any(isinstance(n, ast.Name) and n.id == builder and id(n) not in permitted for n in ast.walk(definition)):
        raise _Unproven("context builder may only construct its scope and finalize its tree")
    return dict(
        call=normalized,
        definition=wrapped.__module__ + "." + wrapped.__qualname__,
        sha256=hashlib.sha256(source.encode()).hexdigest(),
        allocation_line=builder_call.lineno,
        allocation_column=builder_call.col_offset,
        scope_line=allocation.lineno,
        scope_column=allocation.value.col_offset,
        owner_parameter=getattr(normalized.args[0], "_package_context_parameter", None),
        yield_line=yielded.lineno,
        finalization_lines=sorted(n.lineno for n in finalizers),
        lifecycle="original_host_generator_required",
    )


def discover_contexts(tree, namespace, module, fail):
    recipes = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for constructor in cls.body:
            if not isinstance(constructor, ast.FunctionDef) or not (
                constructor.name == "__init__"
                or getattr(constructor, "_package_construction_method", False)
                or getattr(constructor, "_package_preparation_method", False)
            ):
                continue
            parents = {id(child): parent for parent in ast.walk(constructor) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(constructor):
                if not isinstance(node, ast.With) or len(node.items) != 1:
                    continue
                item = node.items[0]
                if not isinstance(item.context_expr, ast.Call):
                    continue
                call = item.context_expr
                try:
                    recipe = prove_context(_source_value(call.func, namespace), call)
                except (TypeError, ValueError, OSError) as error:
                    fail(module, node, f"Resource context declaration is unproven: {error}")
                if recipe is None:
                    continue
                if not isinstance(item.optional_vars, ast.Name):
                    fail(module, node, "Resource contexts require one local scope binding")
                enclosing = parents[id(node)]
                while enclosing is not constructor:
                    if not isinstance(enclosing, ast.If):
                        fail(
                            module,
                            node,
                            "Resource contexts require constructor statements or declared conditional branches",
                        )
                    enclosing = parents[id(enclosing)]
                root = call.func
                while isinstance(root, ast.Attribute):
                    root = root.value
                if not isinstance(root, ast.Name) or any(
                    isinstance(n, ast.Name)
                    and n.id == root.id
                    and not isinstance(n.ctx, ast.Load)
                    or isinstance(n, ast.arg)
                    and n.arg == root.id
                    or isinstance(n, ast.alias)
                    and (n.asname or n.name.split(".")[0]) == root.id
                    for n in ast.walk(constructor)
                ):
                    fail(module, node, "Resource context requires an unshadowed module binding")
                node._package_resource_context = recipe
                node._package_deferred_context = constructor.name != "__init__"
                recipes.append({k: v for k, v in recipe.items() if k != "call"} | dict(module=module, line=node.lineno))
    return recipes


class HostContexts:
    """Keep required ordinary Host definitions in one namespace per source module.

    Only allocation calls are replaced. Python still executes the original
    generator's try/except/finally and bookkeeping; no lifecycle interpreter is
    introduced. Numerical definitions cannot enter this Host dependency closure.
    """

    def __init__(self, application_modules=()):
        self.modules = {}
        self.application_modules = frozenset(application_modules)

    def add(self, function, recipe):
        module = inspect.getmodule(function)
        if module.__name__ not in self.modules:
            path = inspect.getsourcefile(module)
            source = Path(path).read_text(encoding="utf8")
            tree = ast.parse(source)
            if any(
                not isinstance(
                    n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
                )
                and not (
                    isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
                )
                for n in tree.body
            ):
                raise _Unproven("context Host dependencies require explicit top-level definitions and initializers")
            self.modules[module.__name__] = dict(module=module, source=source, tree=tree, roots={}, selected=set())
        state = self.modules[module.__name__]
        name = function.__name__
        if function.__qualname__ != name:
            raise _Unproven("context Host definition must be a top-level function")
        previous = state["roots"].get(name)
        if previous is not None and previous["sha256"] != recipe["sha256"]:
            raise _Unproven("context Host source changed during export")
        state["roots"][name] = recipe
        return module.__name__, name

    def contains(self, value):
        return inspect.isfunction(value) and value.__module__ in self.modules

    def select(self, value):
        if value.__qualname__ != value.__name__:
            raise _Unproven("context Host import requires a top-level definition")
        self.modules[value.__module__]["selected"].add(value.__name__)
        return value.__module__, value.__name__

    def project_definition(self, module, node):
        state = self.modules.get(module)
        recipe = state["roots"].get(node.name) if state is not None else None
        if recipe is None:
            return node
        node = copy.deepcopy(node)
        first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        owner = recipe["owner_parameter"]
        if owner is None or any(
            isinstance(n, ast.Name)
            and n.id.startswith("_forge_package_")
            or isinstance(n, ast.arg)
            and n.arg.startswith("_forge_package_")
            for n in ast.walk(node)
        ):
            raise _Unproven("Context owner/group binding is ambiguous or uses reserved names")
        node.args.kwonlyargs.append(ast.arg("_forge_package_group"))
        node.args.kw_defaults.append(None)

        class Allocations(ast.NodeTransformer):
            def visit_Call(self, call):
                position = (call.lineno - first + 1, call.col_offset)
                if position == (recipe["allocation_line"], recipe["allocation_column"]):
                    return ast.copy_location(
                        ast.Call(
                            ast.Attribute(
                                ast.Name("_forge_package_runtime", ast.Load()), "context_builder", ast.Load()
                            ),
                            [ast.Name(owner, ast.Load()), ast.Name("_forge_package_group", ast.Load())],
                            [],
                        ),
                        call,
                    )
                if position == (recipe["scope_line"], recipe["scope_column"]):
                    return ast.copy_location(
                        ast.Call(
                            ast.Attribute(ast.Name("_forge_package_runtime", ast.Load()), "resource_scope", ast.Load()),
                            [ast.Name(owner, ast.Load())],
                            [],
                        ),
                        call,
                    )
                return self.generic_visit(call)

        return Allocations().visit(node)

    def render(self):
        import builtins
        import symtable

        payloads, manifest = {}, {}
        for module_name, state in self.modules.items():
            if module_name in self.application_modules:
                manifest[module_name] = dict(
                    module=module_name,
                    application_module=True,
                    definitions=sorted(set(state["roots"]) | state["selected"]),
                    source_sha256=hashlib.sha256(state["source"].encode()).hexdigest(),
                )
                continue
            definitions = {}
            for statement in state["tree"].body:
                if isinstance(statement, ast.Expr):
                    continue
                if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
                    names = [statement.name]
                elif isinstance(statement, ast.Import):
                    names = [n.asname or n.name.split(".")[0] for n in statement.names]
                elif isinstance(statement, ast.ImportFrom):
                    names = [n.asname or n.name for n in statement.names]
                else:
                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                    names = [
                        n.id
                        for target in targets
                        for n in ast.walk(target)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                    ]
                for name in names:
                    if name in definitions:
                        raise _Unproven(f"Context Host binding {name!r} is assigned more than once")
                    definitions[name] = statement
            chosen = {}

            def select(name):
                if name in chosen or name == "_forge_package_runtime":
                    return
                original = definitions.get(name)
                if original is None:
                    if name in vars(builtins):
                        return
                    raise _Unproven(f"Context Host global {name!r} has no source initializer")
                node = copy.deepcopy(original)
                chosen[name] = node
                if name in state["roots"]:
                    node = self.project_definition(module_name, node)
                    chosen[name] = node
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    value = vars(state["module"]).get(name)
                    if (
                        getattr(value, "_is_wrapped_kernel", False)
                        or getattr(value, "_is_taichi_function", False)
                        or getattr(value, "_data_oriented", False)
                    ):
                        raise _Unproven("Numerical context dependencies must join normal package compilation")
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.level:
                        raise _Unproven("Relative context Host dependencies require the application module closure")
                    origin = node.module if isinstance(node, ast.ImportFrom) else None
                    if (
                        (origin or "").split(".")[0] in ("taichi", "taichi_forge")
                        or isinstance(node, ast.Import)
                        and any(n.name.split(".")[0] in ("taichi", "taichi_forge") for n in node.names)
                    ):
                        raise _Unproven("Context Host closure retains an unprojected frontend dependency")
                    return
                fragment = ast.unparse(ast.fix_missing_locations(node))
                if any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id in ("globals", "locals", "eval", "exec", "__import__")
                    for n in ast.walk(node)
                ):
                    raise _Unproven("Context Host closure cannot infer dynamic namespace dependencies")
                table = symtable.symtable(fragment, "<context-host>", "exec")
                globals_used = set()

                def collect(table):
                    globals_used.update(
                        s.get_name() for s in table.get_symbols() if s.is_referenced() and s.is_global()
                    )
                    for child in table.get_children():
                        collect(child)

                collect(table)
                for dependency in globals_used:
                    select(dependency)

            for name in (*state["roots"], *sorted(state["selected"])):
                select(name)
            body, seen = [], set()
            for node in sorted(chosen.values(), key=lambda n: (n.lineno, n.col_offset)):
                position = (node.lineno, node.col_offset)
                if position not in seen:
                    seen.add(position)
                    body.append(node)
            prefix = ast.parse(
                "from taichi_forge_runtime.aot._package import bind_module as _forge_package_bind\n"
                "_forge_package_runtime = _forge_package_bind(__name__)\n"
            ).body
            future = [
                copy.deepcopy(n)
                for n in state["tree"].body
                if isinstance(n, ast.ImportFrom) and n.module == "__future__"
            ]
            tree = ast.fix_missing_locations(ast.Module(body=[*future, *prefix, *body], type_ignores=[]))
            source = ast.unparse(tree) + "\n"
            compile(source, "<projected-context-host>", "exec")
            key = "_forge_package_context_" + hashlib.sha256(module_name.encode()).hexdigest()
            payloads["host/" + key + ".py"] = source.encode("utf8")
            manifest[module_name] = dict(
                module=key,
                definitions=sorted(set(state["roots"]) | state["selected"]),
                source_sha256=hashlib.sha256(state["source"].encode()).hexdigest(),
            )
            if Path(inspect.getsourcefile(state["module"])).read_text(encoding="utf8") != state["source"]:
                raise _Unproven("Context Host source changed during export")
        return payloads, manifest
