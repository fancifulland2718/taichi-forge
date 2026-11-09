"""Keep a borrowed allocation's declared layout closure, not its owner's algorithms."""

import ast


def allocation_schema(schema, source_paths):
    selected, pending, children = set(), [], {}

    def require_path(path):
        path = tuple(path)
        matches = [
            (key, tuple(item.get("path", (key,))))
            for key, item in schema.items()
            if path[: len(item.get("path", (key,)))] == tuple(item.get("path", (key,)))
        ]
        if not matches:
            raise ValueError("Borrowed allocation dependency has no declared resource path: " + ".".join(path))
        key, prefix = max(matches, key=lambda match: len(match[1]))
        require(key)
        item = schema[key]
        if item["kind"] in ("owner", "owner_input"):
            if len(path) == len(prefix):
                raise ValueError("Borrowed allocation requires a concrete member of its source owner")
            children.setdefault(key, set()).add(path[len(prefix) :])
        elif len(path) != len(prefix) and not item.get("struct_field"):
            raise ValueError("Borrowed allocation dependency is not a declared Field member")

    def require(key):
        if key in schema and key not in selected:
            selected.add(key)
            pending.append(key)

    def expression_dependencies(value):
        if isinstance(value, str):
            require(value)
        elif isinstance(value, (list, tuple)):
            for child in value:
                expression_dependencies(child)
        elif isinstance(value, dict):
            for child in value.values():
                expression_dependencies(child)

    for path in source_paths:
        require_path(path)
    while pending:
        key = pending.pop()
        item = schema[key]
        path = tuple(item.get("path", (key,)))
        for parent, declaration in schema.items():
            prefix = tuple(declaration.get("path", (parent,)))
            if declaration["kind"] == "record" and len(prefix) < len(path) and path[: len(prefix)] == prefix:
                require(parent)
        for condition in item.get("conditions", {}):
            require(condition)
        for part in ("shape", "expression", "dimension_expressions", "dimensions", "constraints"):
            expression_dependencies(item.get(part, ()))
        for part in ("source_path", "like_source_path", "reference_path", "owner_dimension"):
            if item.get(part):
                require_path(item[part])
        for part in ("shape_policy", "dtype_policy", "source_policy"):
            if part not in item:
                continue
            policy = item[part]
            if "condition_path" in policy:
                require_path(policy["condition_path"])
            else:
                require(policy["condition"])
            if part == "source_policy":
                require_path(policy["if_true"])
                require_path(policy["if_false"])
            elif part == "shape_policy":
                expression_dependencies((policy["if_true"], policy["if_false"]))
        if item["kind"] == "dense_field":
            # Native group placement includes siblings even when only one Field
            # is borrowed. Keep that ABI, including its original declaration order.
            for sibling, declaration in schema.items():
                if declaration["kind"] == "dense_field" and declaration.get("group", "fields") == item.get(
                    "group", "fields"
                ):
                    require(sibling)
        nodes = [item["type_node"]] if "type_node" in item else []
        nodes.extend(ast.parse(source, mode="eval").body for source in item.get("dimension_sources", {}).values())
        for node in nodes:
            intermediate = {id(part.value) for part in ast.walk(node) if isinstance(part, ast.Attribute)}
            for part in ast.walk(node):
                require(getattr(part, "_package_model_extent", None))
                if isinstance(part, ast.Attribute) and id(part) not in intermediate:
                    suffix, base = [], part
                    while isinstance(base, ast.Attribute):
                        suffix.insert(0, base.attr)
                        base = base.value
                    if isinstance(base, ast.Name) and base.id == "self":
                        if suffix[-1] == "dtype":
                            suffix.pop()
                        if suffix:
                            require_path(suffix)
    return {key: item for key, item in schema.items() if key in selected}, children
