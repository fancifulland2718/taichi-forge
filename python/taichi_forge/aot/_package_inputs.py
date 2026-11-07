"""Resolve declared input types from normal constructor calls, without executing them."""

import ast
import inspect


def resolve_inputs(sources, imported):
    from ._package_factories import _source_value

    owners = {
        name + "." + owner: (schema, getattr(imported[name], owner))
        for name, _, _, _, _, _, schemas, _ in sources
        for owner, schema in schemas.items()
    }
    inputs = {
        (owner, key): item
        for owner, (schema, _) in owners.items()
        for key, item in schema.items()
        if item["kind"] == "owner_input"
    }
    if not inputs:
        return
    classes = {cls: name for name, (_, cls) in owners.items()}
    annotated = set()
    for key, item in inputs.items():
        constructor = owners[key[0]][1].__init__
        parameter = inspect.signature(constructor).parameters.get(item["input_parameter"])
        if parameter is None:
            raise ValueError("Inherited owner inputs require a resolved constructor parameter mapping: " + key[0])
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
    targets = {owner for owner, _ in inputs}
    calls = {}
    declared_sites = set()
    for caller, (schema, _) in owners.items():
        for item in schema.values():
            if item["kind"] != "owner" or item["owner_type"] not in targets:
                continue
            call = item["constructor_call"]
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
            except TypeError as error:
                raise ValueError("Declared constructor argument mismatch: " + target) from error
            calls.setdefault(target, []).append((caller, bound.arguments))
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

    def input_type(owner, expression):
        path = []
        while isinstance(expression, ast.Attribute):
            path.insert(0, expression.attr)
            expression = expression.value
        if not isinstance(expression, ast.Name):
            return None
        if expression.id != "self":
            references = [
                item.get("owner_type")
                for item in owners[owner][0].values()
                if item["kind"] == "owner_input" and item["input_parameter"] == expression.id
            ]
            if not references or None in references or len(set(references)) != 1:
                return None
            owner = references[0]
        for part in path:
            item = owners.get(owner, ({}, None))[0].get(part, {})
            if item.get("kind") not in ("owner", "owner_input"):
                return None
            owner = item.get("owner_type")
            if owner is None:
                return None
        return owner

    while True:
        changed = False
        for (owner, name), item in inputs.items():
            observed = {
                identity
                for caller, args in calls.get(owner, ())
                if (identity := input_type(caller, args.get(item["input_parameter"]))) is not None
            }
            if len(observed) > 1:
                raise ValueError(f"Owner input {owner}.{name} has incompatible source definitions: {sorted(observed)}")
            if (owner, name) in annotated and observed - {item["owner_type"]}:
                raise ValueError(f"Owner input {owner}.{name} contradicts its declared source type")
            if observed and "owner_type" not in item:
                item["owner_type"] = observed.pop()
                changed = True
        if not changed:
            break
    for (owner, name), item in inputs.items():
        if (owner, name) in annotated:
            continue
        sites = calls.get(owner, ())
        if (
            not sites
            or "owner_type" not in item
            or any(
                input_type(caller, args.get(item["input_parameter"])) != item["owner_type"] for caller, args in sites
            )
        ):
            raise ValueError(f"Owner input {owner}.{name} requires a proven source at every declared constructor call")
