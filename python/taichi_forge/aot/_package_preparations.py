"""Project ordinary free Host preparation through existing owner contracts."""

import ast
import copy
import hashlib
import inspect
import json

from ._package_construction import ordinary_owner_lookup
from ._package_contexts import discover_contexts, _generator_context
from ._package_factories import _source_value, normalize_factories
from ._package_function_inputs import discover
from ._package_input_families import source_class


def resolve(sources, imported, owner_schema, fail, host_contexts):
    import taichi_forge as ti

    candidates = {}
    for module, _, _, _, tree, kind, _, _ in sources:
        namespace = vars(imported[module])
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.decorator_list:
                continue
            # Explicit independent groups have the existing preparation
            # lifetime. Arbitrary Host functions/allocations are not lifted.
            contexts = []
            for part in ast.walk(node):
                if not isinstance(part, ast.With) or len(part.items) != 1:
                    continue
                call = part.items[0].context_expr
                if not isinstance(call, ast.Call):
                    continue
                candidate = _generator_context(_source_value(call.func, namespace))
                if candidate is not None and any(
                    isinstance(item, ast.Call)
                    and _source_value(item.func, candidate[0].__globals__) is ti.FieldsBuilder
                    for item in ast.walk(candidate[2])
                ):
                    contexts.append(part)
            function = namespace.get(node.name)
            if contexts and inspect.isfunction(function):
                candidates[function] = (module, tree, kind, node)
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
        namespace = vars(imported[module])
        facts = discover_contexts(wrapper, namespace, module, fail)
        facts += normalize_factories(wrapper, namespace, module, fail, kind)
        from ._package_allocations import normalize_allocation_helpers
        from ._package_type_factories import normalize_type_member_factories

        facts += normalize_allocation_helpers(wrapper, namespace, kind, module, fail)
        facts += normalize_type_member_factories(wrapper, namespace, kind, module, fail)
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
