"""Resolve ordinary owner inheritance without constructing sample models."""

import ast
import copy
import inspect


def resolve_owners(sources, imported, owner_schema, fail):
    classes = {}
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

    def constructor_schema(cls, start=0):
        mro = cls.__mro__
        index = next(i for i in range(start, len(mro)) if "__init__" in vars(mro[i]))
        base = mro[index]
        if base is object:
            return {}
        if base not in classes:
            name, node, _, _ = classes[cls]
            fail(name, node, "inherited constructor is outside the scanned source boundary")
        name, node, kind, schemas = classes[base]
        constructor = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if constructor is None:
            fail(name, node, "constructor replacement requires an explicit resource contract")
        own = owner_schema(name, node, kind, vars(imported[name]))
        for item in own.values():
            item["declaration_module"] = name
            item["declaration_owner"] = node.name
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
        if len(field_owners) > 1:
            fail(name, constructor, "Fields allocated by multiple constructors require separate native group contracts")
        # Projection needs the original allocation sites even for a plain
        # resource mixin. This is metadata, not an additional application decorator.
        if base is not cls and own:
            schemas.setdefault(node.name, own)
        return result

    pending = []
    for name, _, _, _, tree, _, schemas, entries in sources:
        for node in tree.body:
            cls = vars(imported[name]).get(getattr(node, "name", ""))
            if (
                not isinstance(node, ast.ClassDef)
                or not inspect.isclass(cls)
                or not (getattr(cls, "_data_oriented", False) or node.name in schemas)
            ):
                continue
            schema = constructor_schema(cls)
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
