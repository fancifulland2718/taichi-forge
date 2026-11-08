"""Resolve declared input types from normal constructor calls, without executing them."""

import ast
import inspect


def default_owner_call(constructor, parameter, before):
    """Prove the ordinary ``if input is None: input = Owner(...)`` fallback.

    The caller checks that this is the parameter's only assignment. No Host
    branch or factory is executed to discover the resource type.
    """
    for statement in constructor.body:
        if statement.lineno >= before or not isinstance(statement, ast.If):
            continue
        test = statement.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == parameter
            and len(test.ops) == len(test.comparators) == 1
            and isinstance(test.ops[0], ast.Is)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
            and not statement.orelse
            and len(statement.body) == 1
        ):
            continue
        assignment = statement.body[0]
        if (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and assignment.targets[0].id == parameter
            and isinstance(assignment.value, ast.Call)
        ):
            return assignment.value
    return None


def resolve_inputs(sources, imported):
    from ._package_factories import _source_value
    from ._package_input_families import source_class, split_field_inputs

    owners = {
        name + "." + owner: (schema, getattr(imported[name], source_class(tree, owner)))
        for name, _, _, _, tree, _, schemas, _ in sources
        for owner, schema in schemas.items()
    }
    inputs = {
        (owner, key): item
        for owner, (schema, _) in owners.items()
        for key, item in schema.items()
        if item["kind"] in ("owner_input", "borrowed_field") and "input_parameter" in item
    }
    aliases = {
        (owner, key): item
        for owner, (schema, _) in owners.items()
        for key, item in schema.items()
        if item["kind"] == "owner_input" and "reference_path" in item
    }
    if not inputs and not aliases:
        return
    classes = {cls: name for name, (_, cls) in owners.items()}
    annotated = set()
    for key, item in inputs.items():
        constructor = owners[key[0]][1].__init__
        parameter = inspect.signature(constructor).parameters.get(item["input_parameter"])
        if parameter is None:
            raise ValueError("Inherited owner inputs require a resolved constructor parameter mapping: " + key[0])
        if item["kind"] != "owner_input":
            if item.get("field_obligation") and parameter.default is None and not item.get("optional_capable"):
                raise ValueError("Optional Field helper requires a proven absence declaration: " + key[0])
            if item.get("optional_capable") and parameter.default is None:
                item["optional_input"] = True
            continue
        annotation = parameter.annotation
        if isinstance(annotation, str):
            expression = ast.parse(annotation, mode="eval").body
            # Future annotations may retain another pair of forward-reference
            # quotes. Resolve only names/module attributes; never eval Python.
            while isinstance(expression, ast.Constant) and isinstance(expression.value, str):
                expression = ast.parse(expression.value, mode="eval").body
            annotation = _source_value(expression, constructor.__globals__)
        if inspect.isclass(annotation) and annotation in classes:
            item["owner_type"] = classes[annotation]
            annotated.add(key)
        fallback = item.get("default_constructor_call")
        if fallback is not None:
            default_type = _source_value(fallback.func, constructor.__globals__)
            if parameter.default is not None or not inspect.isclass(default_type) or default_type not in classes:
                raise ValueError("Default owner input requires None and a declared owner constructor: " + key[0])
            if key in annotated and annotation is not default_type:
                raise ValueError("Default owner input contradicts its declared source type: " + key[0])
            item["owner_type"] = classes[default_type]
            annotated.add(key)
    targets = {owner for owner, _ in inputs}
    calls = {}
    declared_sites = set()
    for caller, (schema, _) in owners.items():
        for item in schema.values():
            if item["kind"] == "owner":
                call = item["constructor_call"]
            elif "default_constructor_call" in item:
                call = item["default_constructor_call"]
            else:
                continue
            if item["owner_type"] not in targets:
                continue
            declared_sites.add(
                (item.get("declaration_module", caller.rpartition(".")[0]), call.lineno, call.col_offset)
            )
            if any(isinstance(arg, ast.Starred) for arg in call.args) or any(k.arg is None for k in call.keywords):
                raise ValueError("Declared constructor inputs require explicit arguments")
            target = item["owner_type"]
            try:
                bound = inspect.signature(owners[target][1].__init__).bind(
                    None, *call.args, **{k.arg: k.value for k in call.keywords}
                )
                bound.apply_defaults()
            except TypeError as error:
                raise ValueError("Declared constructor argument mismatch: " + target) from error
            calls.setdefault(target, []).append((caller, bound.arguments, item))
    untyped_targets = {owner for owner, name in inputs if (owner, name) not in annotated}
    for name, _, _, _, tree, *_ in sources:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or (name, node.lineno, node.col_offset) in declared_sites:
                continue
            constructor = _source_value(node.func, vars(imported[name]))
            if inspect.isclass(constructor) and classes.get(constructor) in untyped_targets:
                raise ValueError(
                    "Unscoped constructor input requires declared source provenance: " + classes[constructor]
                )

    def origin(item):
        if item["kind"] == "owner_input":
            return (item["owner_type"], ()) if "owner_type" in item else None
        return (item["source_type"], tuple(item["source_member"])) if "source_type" in item else None

    def member_type(owner, path, visited=()):
        if not path:
            return owner, ()
        key, *rest = path
        item = owners.get(owner, ({}, None))[0].get(key, {})
        kind = item.get("kind")
        if kind in ("owner", "owner_input"):
            return member_type(item["owner_type"], rest, visited) if "owner_type" in item else None
        if kind == "dense_field" and not item.get("conditions"):
            if not rest or (item.get("struct_field") and len(rest) == 1):
                return owner, tuple(path)
        if kind == "borrowed_field" and not rest and (owner, key) not in visited:
            if "input_parameter" in item:
                return origin(item)
            return member_type(owner, item["source_path"], (*visited, (owner, key)))
        return None

    def input_type(owner, expression):
        path = []
        while isinstance(expression, ast.Attribute):
            path.insert(0, expression.attr)
            expression = expression.value
        if not isinstance(expression, ast.Name):
            return None
        if expression.id != "self":
            references = [
                origin(item) for item in owners[owner][0].values() if item.get("input_parameter") == expression.id
            ]
            if not references or None in references or len(set(references)) != 1:
                return None
            owner, prefix = references[0]
            path = [*prefix, *path]
        return member_type(owner, path)

    absent = (None, ())

    def input_types(owner, expression):
        if expression is None or isinstance(expression, ast.Constant) and expression.value is None:
            return {absent}
        if isinstance(expression, ast.IfExp):
            left, right = input_types(owner, expression.body), input_types(owner, expression.orelse)
            return None if left is None or right is None else left | right
        value = input_type(owner, expression)
        if value is None:
            return None
        optional = any(
            item.get("optional_input")
            and (
                isinstance(expression, ast.Name)
                and item.get("input_parameter") == expression.id
                or isinstance(expression, ast.Attribute)
                and isinstance(expression.value, ast.Name)
                and expression.value.id == "self"
                and key == expression.attr
            )
            for key, item in owners[owner][0].items()
        )
        return {value, absent} if optional else {value}

    while True:
        changed = False
        for (owner, name), item in aliases.items():
            resolved = member_type(owner, item["reference_path"])
            if resolved is None:
                continue
            source_type, source_member = resolved
            if source_member:
                raise ValueError(f"Owner reference {owner}.{name} requires an object source")
            if item.get("owner_type") != source_type:
                item["owner_type"] = source_type
                changed = True
        for (owner, name), item in inputs.items():
            observed_all = {
                identity
                for caller, args, _ in calls.get(owner, ())
                for identity in (input_types(caller, args.get(item["input_parameter"])) or ())
            }
            observed = observed_all - {absent}
            if absent in observed_all and item.get("optional_capable") and not item.get("optional_input"):
                item["optional_input"] = True
                changed = True
            if len(observed) > 1:
                if item["kind"] == "borrowed_field":
                    split_field_inputs(sources, owner, inputs, calls.get(owner, ()), input_types)
                    return resolve_inputs(sources, imported)
                raise ValueError(f"Owner input {owner}.{name} has incompatible source definitions: {sorted(observed)}")
            if (owner, name) in annotated and observed - {origin(item)}:
                raise ValueError(f"Owner input {owner}.{name} contradicts its declared source type")
            if observed and origin(item) is None:
                source_type, source_member = observed.pop()
                if item["kind"] == "owner_input":
                    if source_member:
                        raise ValueError(f"Owner input {owner}.{name} requires an object source")
                    item["owner_type"] = source_type
                else:
                    if not source_member:
                        raise ValueError(f"Field input {owner}.{name} requires a declared allocation member")
                    item.update(source_type=source_type, source_member=list(source_member))
                changed = True
        if not changed:
            break
    for (owner, name), item in aliases.items():
        if "owner_type" not in item:
            raise ValueError(f"Owner reference {owner}.{name} requires a resolved declared owner path")
    for owner, (schema, _) in owners.items():
        for item in schema.values():
            if "owner_dimension" not in item:
                continue
            reference, dimension = item["owner_dimension"]
            source_type = schema[reference].get("owner_type")
            source = owners.get(source_type, ({}, None))[0].get(dimension, {})
            if source.get("kind") != "dimension" or source.get("conditions") or source.get("owner_dimension"):
                raise ValueError(
                    f"Referenced dimension {owner}.{reference}.{dimension} requires an unconditional declared source dimension"
                )
    for (owner, name), item in inputs.items():
        if (owner, name) in annotated:
            continue
        sites = calls.get(owner, ())
        if (
            not sites
            or origin(item) is None
            or any(
                (actual := input_types(caller, args.get(item["input_parameter"]))) is None
                or not actual <= ({origin(item), absent} if item.get("optional_input") else {origin(item)})
                for caller, args, _ in sites
            )
        ):
            raise ValueError(f"Owner input {owner}.{name} requires a proven source at every declared constructor call")
    for (owner, name), item in inputs.items():
        if item["kind"] != "borrowed_field":
            continue
        # A private provenance link exposes the original allocation to the
        # existing Component layout/alias machinery. It creates no runtime state.
        source = "_forge_package_source_" + name
        item["source_path"] = [source, *item["source_member"]]
        if item.get("optional_input") and not item.get("conditions"):
            presence = "_forge_package_input_present_" + item["input_parameter"]
            owners[owner][0][presence] = dict(
                kind="static",
                choices=[("bool", False), ("bool", True)],
                kernel_only=False,
                presence_parameter=item["input_parameter"],
                presence_owner=owner,
                presence_nonnull=True,
            )
            item["conditions"] = {presence: True}
        owners[owner][0][source] = dict(
            kind="owner_input",
            owner_type=item["source_type"],
            field_input=name,
            **({"conditions": item["conditions"]} if item.get("conditions") else {}),
        )
