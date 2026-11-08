"""Resolve ordinary owner inheritance without constructing sample models."""

import ast
import copy
import inspect

from ._package_construction import construction_functions, ordinary_owner_lookup


def resolve_owners(sources, imported, owner_schema, fail):
    classes = {}
    trees = {name: tree for name, _, _, _, tree, *_ in sources}
    definitions = {}
    for name, _, _, _, tree, kind, schemas, entries in sources:
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                cls = vars(imported[name]).get(node.name)
                if inspect.isclass(cls):
                    classes[cls] = (name, node, kind, schemas)
        for entry in entries:
            if entry["owner"] is not None:
                cls = vars(imported[name]).get(entry["owner"])
                definitions[cls, entry["node"].name] = entry

    def needs_following_constructor(cls):
        # object.__init__ cannot receive the forwarded constructor arguments.
        # Such a source class is a cooperative template, not a standalone owner.
        for base in cls.__mro__:
            if base is object or base not in classes:
                return False
            node = classes[base][1]
            constructor = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            if constructor is None:
                continue
            calls = [
                n
                for n in ast.walk(constructor)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "__init__"
            ]
            if len(calls) != 1:
                return False
            call = calls[0]
            receiver = call.func.value
            if not (
                isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Name)
                and receiver.func.id == "super"
                and not receiver.args
                and not receiver.keywords
                and any(isinstance(n, ast.Expr) and n.value is call for n in constructor.body)
            ):
                return False
            next_base = next(b for b in cls.__mro__[cls.__mro__.index(base) + 1 :] if "__init__" in vars(b))
            if next_base is object:
                return bool(call.args or call.keywords)
        return False

    templates = {
        cls
        for cls, (_, node, _, schemas) in classes.items()
        if (getattr(cls, "_data_oriented", False) or node.name in schemas) and needs_following_constructor(cls)
    }
    for cls in templates:
        name, node, _, schemas = classes[cls]
        schemas.pop(node.name, None)
        node._package_constructor_template = True
        for (owner, _), entry in definitions.items():
            if owner is cls:
                entry["requires_owner"] = True

    def constructor_schema(cls, start=0):
        mro = cls.__mro__
        index = next(i for i in range(start, len(mro)) if "__init__" in vars(mro[i]))
        base = mro[index]
        if start == 0 and cls in classes and getattr(classes[cls][1], "_package_inherited_constructor", False):
            base, index = cls, 0
        if base is object:
            return {}
        if base not in classes:
            name, node, _, _ = classes[cls]
            fail(name, node, "inherited constructor is outside the scanned source boundary")
        name, node, kind, schemas = classes[base]
        for method, original in getattr(node, "_package_allocation_methods", {}).items():
            if inspect.getattr_static(cls, method) is not original:
                fail(name, node, "Overridden allocation forwarding requires an explicit construction contract")
        for method, original in getattr(node, "_package_type_methods", {}).items():
            if inspect.getattr_static(cls, method) is not original:
                fail(name, node, "Overridden type construction requires a concrete constructor contract")
        for method in node.body:
            if getattr(method, "_package_construction_method", False) and (
                inspect.getattr_static(cls, method.name) is not inspect.getattr_static(base, method.name)
            ):
                fail(name, method, "Overridden construction resource methods require their own constructor contract")
        constructor = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if constructor is None:
            fail(name, node, "constructor replacement requires an explicit resource contract")
        calls = [
            n
            for n in ast.walk(constructor)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "__init__"
        ]
        inherited = {}
        if calls:
            call = calls[0]
            receiver = call.func.value
            if not (
                len(calls) == 1
                and isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Name)
                and receiver.func.id == "super"
                and not receiver.args
                and not receiver.keywords
                and any(isinstance(n, ast.Expr) and n.value is call for n in constructor.body)
            ):
                fail(name, constructor, "owner construction requires one unconditional super().__init__ call")
            inherited = constructor_schema(cls, index + 1)
        own = owner_schema(
            name,
            node,
            kind,
            vars(imported[name]),
            functions=[
                method
                for method in construction_functions(node)
                if not getattr(method, "_package_preparation_method", False)
            ],
            inherited=inherited,
        )
        if any(item.get("owner_dimension", [None])[0] in inherited for item in own.values()):
            node._package_inherited_resource_dependencies = True
        if calls:
            for n in ast.walk(constructor):
                if (
                    isinstance(n, ast.Attribute)
                    and isinstance(n.ctx, ast.Store)
                    and isinstance(n.value, ast.Name)
                    and n.value.id == "self"
                    and n.attr in own
                    and n.lineno < call.lineno
                ):
                    fail(name, n, "declared owner assignments must follow super().__init__")
        for key in inherited.keys() & own.keys():
            left, right = inherited[key], own[key]
            origins = {"scope_allocation", "scope_allocations", "declaration_module", "declaration_owner"}
            if left["kind"] == right["kind"] == "dimension" and (
                {k: v for k, v in left.items() if k not in origins}
                == {k: v for k, v in right.items() if k not in origins}
            ):
                continue
            if (
                left["kind"] != "value"
                or right["kind"] != "value"
                or "type_node" in left
                or "type_node" in right
                or left["dtype"] != right["dtype"]
            ):
                fail(name, constructor, f"inherited resource/type replacement for {key!r} requires further support")
        result = {**inherited, **own}
        from ._package_constants import class_constant, check_constant_slot

        for key, item in result.items():
            if item["kind"] == "constant":
                try:
                    check_constant_slot(cls, key)
                except ValueError as error:
                    fail(name, constructor, str(error))
            if item.get("constant_from_class"):
                if key not in own and any(
                    isinstance(n, ast.Attribute) and n.attr == key and not isinstance(n.ctx, ast.Load)
                    for n in ast.walk(constructor)
                ):
                    fail(name, constructor, "Inherited definition constant has an unproven assignment: " + key)
                try:
                    item["value"] = class_constant(cls, key)
                except (AttributeError, ValueError) as error:
                    fail(name, constructor, f"Definition constant {key!r}: {error}")
        field_owners = {
            (item["declaration_module"], item["declaration_owner"])
            for item in result.values()
            if item["kind"] == "dense_field"
        }
        if len(field_owners) > 1 and any(
            item["kind"] == "dense_field" and not item.get("group_builder") for item in result.values()
        ):
            fail(name, constructor, "Fields allocated by multiple constructors require separate native group contracts")
        # Projection needs the original allocation sites even for a plain
        # resource mixin. This is metadata, not an additional application decorator.
        if base in templates:
            tree = trees[name]
            if not hasattr(tree, "_package_construction_schemas"):
                tree._package_construction_schemas = {}
                tree._package_construction_owners = {}
            local = tree._package_construction_schemas.setdefault(node.name, {})

            def description(value):
                if isinstance(value, ast.AST):
                    return ast.dump(value, include_attributes=False)
                if isinstance(value, dict):
                    return {key: description(item) for key, item in value.items()}
                if isinstance(value, (tuple, list)):
                    return [description(item) for item in value]
                return value

            for key, item in own.items():
                if key in local and description(local[key]) != description(item):
                    fail(name, node, "Cooperative construction has incompatible resource projections: " + key)
                local[key] = copy.deepcopy(item)
            concrete_module, concrete_node, *_ = classes[cls]
            tree._package_construction_owners.setdefault(node.name, set()).add(
                concrete_module + "." + concrete_node.name
            )
        elif base is not cls and own:
            schemas.setdefault(node.name, own)
        return result

    pending = []
    for name, _, _, _, tree, _, schemas, entries in sources:
        for node in tree.body:
            cls = vars(imported[name]).get(getattr(node, "name", ""))
            if (
                not isinstance(node, ast.ClassDef)
                or not inspect.isclass(cls)
                or cls in templates
                or not (getattr(cls, "_data_oriented", False) or node.name in schemas)
            ):
                continue
            schema = constructor_schema(cls)
            # Preparation follows effective Python method lookup independently
            # of constructor inheritance. A plain mixin need not have __init__.
            methods = []
            seen_methods = set()
            for base in cls.__mro__:
                for method_name in vars(base):
                    if method_name in seen_methods:
                        continue
                    seen_methods.add(method_name)
                    if base not in classes:
                        continue
                    origin, base_node, base_kind, _ = classes[base]
                    method = next(
                        (m for m in base_node.body if isinstance(m, ast.FunctionDef) and m.name == method_name), None
                    )
                    if method is not None and getattr(method, "_package_preparation_method", False):
                        methods.append((base, origin, base_node, base_kind, method))
            if methods and not ordinary_owner_lookup(cls):
                fail(name, node, "Preparation inheritance requires ordinary owner lookup")
            for base, origin, base_node, base_kind, method in sorted(
                methods, key=lambda item: (-cls.__mro__.index(item[0]), item[4].lineno)
            ):
                schema = owner_schema(
                    origin, base_node, base_kind, vars(imported[origin]), functions=[method], earlier=schema
                )
                if base is not cls:
                    local = {
                        key: item
                        for key, item in schema.items()
                        if item.get("declaration_module") == origin and item.get("declaration_owner") == base_node.name
                    }
                    tree_origin = trees[origin]
                    if not hasattr(tree_origin, "_package_preparation_schemas"):
                        tree_origin._package_preparation_schemas = {}
                        tree_origin._package_preparation_owners = {}
                    tree_origin._package_preparation_schemas.setdefault(base_node.name, {}).update(copy.deepcopy(local))
                    receivers = tree_origin._package_preparation_owners.setdefault(base_node.name, set())
                    receivers.add(name + "." + node.name)
                    if base_node.name in classes[base][3]:
                        receivers.add(origin + "." + base_node.name)
            group_origins = {}
            for item in schema.values():
                if item["kind"] != "dense_field":
                    continue
                group = item.get("group", "fields")
                origin = (item["declaration_module"], item["declaration_owner"])
                if group_origins.setdefault(group, origin) != origin:
                    fail(name, node, "Inherited resource groups have ambiguous allocation identities")
            effective = []
            seen = set()
            for base in cls.__mro__:
                for method in vars(base):
                    if method in seen:
                        continue
                    seen.add(method)
                    entry = definitions.get((base, method))
                    if entry is None:
                        value = vars(base)[method]
                        if getattr(value, "_is_wrapped_kernel", False) or getattr(value, "_is_taichi_function", False):
                            fail(
                                name,
                                node,
                                f"inherited numerical definition {base.__name__}.{method} is outside the scanned source boundary",
                            )
                        continue
                    if base is not cls:
                        effective.append(
                            {
                                **entry,
                                "owner": node.name,
                                "name": node.name + "." + method,
                                "inherited": True,
                                "requires_owner": False,
                            }
                        )
            pending.append((schemas, node.name, schema, entries, effective))
    for schemas, name, schema, entries, effective in pending:
        schemas[name] = copy.deepcopy(schema)
        entries.extend(effective)
    for cls in templates:
        name, node, *_ = classes[cls]
        if node.name not in getattr(trees[name], "_package_construction_owners", {}):
            fail(name, node, "cooperative constructor has no exported concrete receiver")
    bound_nodes = {
        id(entry["node"])
        for _, _, _, _, _, _, _, entries in sources
        for entry in entries
        if not entry.get("requires_owner")
    }
    for name, _, _, _, _, _, _, entries in sources:
        for entry in entries:
            if (
                entry.get("requires_owner")
                and not entry.get("closure_definition")
                and entry["kind"] in ("kernel", "pyfunc")
                and not entry.get("error_interface")
                and id(entry["node"]) not in bound_nodes
            ):
                fail(name, entry["node"], "numerical mixin has no exported concrete data_oriented receiver")
