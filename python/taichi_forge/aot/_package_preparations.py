"""Project ordinary free Host preparation through existing owner contracts."""

import ast
import copy
import hashlib
import inspect
import json
import textwrap

from ._package_construction import ordinary_owner_lookup
from ._package_contexts import discover_contexts, _generator_context
from ._package_factories import _source_value, normalize_factories, prove_factory
from ._package_function_inputs import discover
from ._package_input_families import source_class


def _declares_group(node, namespace):
    import taichi_forge as ti

    for part in ast.walk(node):
        if not isinstance(part, ast.With) or len(part.items) != 1:
            continue
        call = part.items[0].context_expr
        if not isinstance(call, ast.Call):
            continue
        candidate = _generator_context(_source_value(call.func, namespace))
        if candidate is not None and any(
            isinstance(item, ast.Call) and _source_value(item.func, candidate[0].__globals__) is ti.FieldsBuilder
            for item in ast.walk(candidate[2])
        ):
            return True
    return False


def _declares_values(node, namespace):
    """Recognize a normal scope use without interpreting a model's role list."""
    import taichi_forge as ti

    for statement in node.body:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
        ):
            continue
        call = statement.value
        target = _source_value(call.func, namespace)
        if target is not ti.experimental.ResourceScope:
            try:
                normalized = prove_factory(target, call)
            except (TypeError, ValueError, OSError):
                continue  # Full normalization still diagnoses unproved factories.
            if normalized is None or getattr(normalized, "_package_role", None) != "experimental.ResourceScope":
                continue
        scope = statement.targets[0].id
        if any(
            isinstance(part, ast.Call)
            and (
                isinstance(part.func, ast.Attribute)
                and isinstance(part.func.value, ast.Name)
                and part.func.value.id == scope
                or any(
                    isinstance(arg, ast.Name) and arg.id == scope
                    for arg in (*part.args, *(keyword.value for keyword in part.keywords))
                )
            )
            for part in ast.walk(node)
        ) or any(
            isinstance(part, ast.Attribute)
            and isinstance(part.ctx, ast.Store)
            and isinstance(part.value, ast.Name)
            and part.value.id == scope
            for part in ast.walk(node)
        ):
            return True
    return False


def is_preparation(function):
    """Owner preparation needs full analysis, not pure-factory folding."""
    if not inspect.isfunction(function):
        return False
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError):
        return False
    return (
        len(tree.body) == 1
        and isinstance(tree.body[0], ast.FunctionDef)
        and not tree.body[0].decorator_list
        and (
            _declares_group(tree.body[0], function.__globals__) or _declares_values(tree.body[0], function.__globals__)
        )
    )


def _candidates(sources, imported):
    candidates = {}
    for module, _, _, _, tree, kind, _, _ in sources:
        namespace = vars(imported[module])
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.decorator_list:
                continue
            # Explicit groups and value-only declarations use the same owner
            # projection. Raw allocations still fail full preparation admission.
            function = namespace.get(node.name)
            if inspect.isfunction(function) and (_declares_group(node, namespace) or _declares_values(node, namespace)):
                candidates[function] = (module, tree, kind, node)
    return candidates


def _guard_returns(body, module, fail):
    """Make fallthrough resource conditions explicit, preserving Host returns."""
    for index, statement in enumerate(body):
        if (
            isinstance(statement, ast.If)
            and not statement.orelse
            and len(statement.body) == 1
            and isinstance(statement.body[0], ast.Return)
            and (
                statement.body[0].value is None
                or isinstance(statement.body[0].value, ast.Constant)
                and statement.body[0].value.value is None
            )
        ):
            # The original predicate and return still execute normally. Only
            # the lexical placement of the remaining statements changes, so
            # the existing resource parser validates their actual guard.
            statement.orelse = _guard_returns(body[index + 1 :], module, fail)
            return [*body[:index], statement]
        if any(
            isinstance(node, ast.Return) and not (node is statement and index == len(body) - 1)
            for node in ast.walk(statement)
        ):
            fail(module, statement, "Preparation permits bare return guards or a terminal Host return")
    return body


def _normalize(function, original, module, kind, receiver, fail):
    arguments = (*original.args.posonlyargs, *original.args.args, *original.args.kwonlyargs)
    parameter_nodes = {id(part) for part in arguments if part.arg == receiver}
    if any(
        isinstance(part, (ast.Yield, ast.YieldFrom, ast.AsyncFunctionDef))
        or isinstance(part, ast.Name)
        and part.id == receiver
        and not isinstance(part.ctx, ast.Load)
        or isinstance(part, ast.arg)
        and part.arg == receiver
        and id(part) not in parameter_nodes
        or isinstance(part, ast.alias)
        and (part.asname or part.name.split(".")[0]) == receiver
        or isinstance(part, (ast.Global, ast.Nonlocal))
        and receiver in part.names
        or isinstance(part, ast.ExceptHandler)
        and part.name == receiver
        or receiver != "self"
        and (isinstance(part, ast.Name) and part.id == "self" or isinstance(part, ast.arg) and part.arg == "self")
        for part in ast.walk(original)
    ):
        fail(module, original, "Preparation requires an unchanged synchronous owner parameter")
    normalized = copy.deepcopy(original)
    normalized.body = _guard_returns(normalized.body, module, fail)

    class Receiver(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == receiver:
                node.id = "self"
            return node

        def visit_arg(self, node):
            if node.arg == receiver:
                node.arg = "self"
            return node

    normalized = Receiver().visit(normalized)
    normalized._package_preparation_method = True
    identity = "_forge_package_prepare_" + original.name
    owner = ast.copy_location(ast.ClassDef(identity, [], [], [normalized], []), original)
    wrapper = ast.Module(body=[owner], type_ignores=[])
    namespace = function.__globals__
    facts = discover_contexts(wrapper, namespace, module, fail)
    facts += normalize_factories(wrapper, namespace, module, fail, kind)
    from ._package_allocations import normalize_allocation_helpers
    from ._package_type_factories import normalize_type_member_factories

    facts += normalize_allocation_helpers(wrapper, namespace, kind, module, fail)
    facts += normalize_type_member_factories(wrapper, namespace, kind, module, fail)
    return owner, normalized, facts


def declaration_dependencies(sources, imported, fail):
    """Read declared domains before interpreting a caller's resource guards.

    Only source-proven owner receivers supply these facts. They are temporary
    inputs to schema resolution, never a replacement for the complete function
    projection performed by resolve(). No Host preparation is executed here.
    """
    from ._package_resources import scope_declarations

    candidates = _candidates(sources, imported)
    if not candidates:
        return {}
    receivers = {function: set() for function in candidates}
    discover(sources, imported, preparation_receivers=receivers)
    owners = {
        module + "." + name: vars(imported[module])[source_class(tree, name)]
        for module, _, _, _, tree, _, schemas, _ in sources
        for name in schemas
    }
    classes = {
        vars(imported[module]).get(node.name): (module, node, kind)
        for module, _, _, _, tree, kind, *_ in sources
        for node in tree.body
        if isinstance(node, ast.ClassDef) and inspect.isclass(vars(imported[module]).get(node.name))
    }
    constant_cache = {}

    def constants(cls):
        from ._package_construction import construction_functions

        if cls not in constant_cache:
            facts = {}
            for base in reversed(cls.__mro__):
                if base not in classes:
                    continue
                module, node, kind = classes[base]
                for function in construction_functions(node):
                    if getattr(function, "_package_preparation_method", False):
                        continue
                    declared = scope_declarations(
                        module,
                        copy.deepcopy(function),
                        kind,
                        fail,
                        vars(imported[module]),
                        owner_node=node,
                        owner_class=cls,
                        declaration_constants_only=True,
                    )
                    facts.update(declared)
            constant_cache[cls] = facts
        return constant_cache[cls]

    result = {}
    for function, (module, _, kind, original) in candidates.items():
        domains = receivers[function]
        parameters = {parameter for parameter, _ in domains}
        if not domains:
            continue
        if len(parameters) != 1:
            fail(module, original, "Preparation owner parameter changes between source calls")
        _, normalized, _ = _normalize(function, original, module, kind, next(iter(parameters)), fail)
        for _, target in domains:
            cls = owners[target]
            if not ordinary_owner_lookup(cls):
                fail(module, original, "Preparation requires ordinary owner lookup")
            declarations = scope_declarations(
                module,
                copy.deepcopy(normalized),
                kind,
                fail,
                function.__globals__,
                constants(cls),
                declaration_domains_only=True,
            )
            dependencies = result.setdefault(cls, {})
            for name, item in declarations.items():
                previous = dependencies.get(name)
                domain = {key: item[key] for key in ("kind", "choices", "kernel_only")}
                if previous is not None and previous != domain:
                    fail(module, original, "Preparation static domains conflict for " + name)
                dependencies[name] = domain
    return result


def resolve(sources, imported, owner_schema, fail, host_contexts):
    candidates = _candidates(sources, imported)
    if not candidates:
        return
    receivers = {function: set() for function in candidates}
    discover(sources, imported, preparation_receivers=receivers)
    owners = {
        module + "." + name: (schemas, name, vars(imported[module])[source_class(tree, name)])
        for module, _, _, _, tree, _, schemas, _ in sources
        for name in schemas
    }
    for function, (module, tree, kind, original) in candidates.items():
        domains = receivers[function]
        if not domains:
            continue  # Unused ordinary Host code keeps its normal projection.
        parameters = {parameter for parameter, _ in domains}
        if len(parameters) != 1:
            fail(module, original, "Preparation owner parameter changes between source calls")
        receiver = next(iter(parameters))
        owner, normalized, facts = _normalize(function, original, module, kind, receiver, fail)
        identity = owner.name
        namespace = vars(imported[module])
        local = None
        for _, target in sorted(domains):
            schemas, name, cls = owners[target]
            if not ordinary_owner_lookup(cls):
                fail(module, original, "Preparation requires ordinary owner lookup")
            schema = owner_schema(
                module, owner, kind, {**namespace, identity: cls}, functions=[normalized], earlier=schemas[name]
            )
            group_origins = {}
            for item in schema.values():
                if item["kind"] == "dense_field":
                    group = item.get("group", "fields")
                    origin = (item["declaration_module"], item["declaration_owner"])
                    if group_origins.setdefault(group, origin) != origin:
                        fail(module, original, "Preparation groups have ambiguous allocation identities")
            declarations = {
                key: item
                for key, item in schema.items()
                if item.get("declaration_module") == module and item.get("declaration_owner") == identity
            }
            # One Host definition has one allocation contract. Different
            # instance sizes are already symbolic; unrelated owner schemas do
            # not become alternative numerical implementations here.
            signature = json.dumps(declarations, sort_keys=True, default=ast.dump)
            if local is not None and local[0] != signature:
                fail(module, original, "Preparation declarations differ between owner contracts")
            local = signature, copy.deepcopy(declarations)
            schemas[name] = schema
        if not hasattr(tree, "_package_preparation_schemas"):
            tree._package_preparation_schemas = {}
            tree._package_preparation_owners = {}
        tree._package_preparation_schemas[identity] = local[1]
        tree._package_preparation_owners[identity] = {target for _, target in domains}
        original._package_free_preparation = (identity, receiver, normalized)
        tree._package_factory_declarations.extend(facts)
        tree._package_factory_declarations.append(
            dict(
                kind="preparation_function",
                module=module,
                definition=module + "." + original.name,
                sha256=hashlib.sha256(ast.dump(original).encode()).hexdigest(),
            )
        )
        for part in ast.walk(normalized):
            recipe = getattr(part, "_package_resource_context", None)
            if recipe is not None:
                host_contexts.add(_source_value(part.items[0].context_expr.func, namespace), recipe)
