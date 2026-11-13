"""Keep a borrowed allocation's declared layout closure, not its owner's algorithms."""

import ast


class AllocationSchema:
    """Index one build-owned declaration snapshot, never resource instances.

    OwnerGraph keeps the source immutable and copies the selected declarations
    before applying choices. Indices may therefore be reused by all borrowed
    members without merging their input slots or native allocation identities.
    """

    def __init__(self, schema):
        self.schema = schema
        self.paths, self.records, self.groups = {}, {}, {}
        for key, item in schema.items():
            path = tuple(item.get("path", (key,)))
            self.paths.setdefault(path, []).append(key)
            if item["kind"] == "record":
                self.records.setdefault(path, []).append(key)
            if item["kind"] == "dense_field":
                self.groups.setdefault(item.get("group", "fields"), []).append(key)

    def select(self, source_paths):
        return _allocation_schema(self, source_paths)


def allocation_schema(schema, source_paths):
    return AllocationSchema(schema).select(source_paths)


def _allocation_schema(index, source_paths):
    schema = index.schema
    selected, pending, children = set(), [], {}
    expanded_groups = set()

    def require_path(path):
        path = tuple(path)
        matches = [
            (key, path[:length])
            for length in range(len(path) + 1)
            for key in index.paths.get(path[:length], ())
        ]
        if not matches:
            raise ValueError("Borrowed allocation dependency has no declared resource path: " + ".".join(path))
        key, prefix = max(matches, key=lambda match: len(match[1]))
        require(key)
        # owner_dim has an exact declaration at e.g. parent.n. Keeping that
        # leaf also requires the non-owning parent and its dimension witness.
        for parent, owner_path in matches:
            if schema[parent]["kind"] in ("owner", "owner_input") and len(owner_path) < len(path):
                require(parent)
                children.setdefault(parent, set()).add(path[len(owner_path) :])
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
        # A recursive local function retains itself and, through require(),
        # the source schema until cyclic GC. Keep this temporary walk acyclic.
        values = [value]
        while values:
            value = values.pop()
            if isinstance(value, str):
                require(value)
            elif isinstance(value, (list, tuple)):
                values.extend(reversed(value))
            elif isinstance(value, dict):
                values.extend(reversed(tuple(value.values())))

    for path in source_paths:
        require_path(path)
    while pending:
        key = pending.pop()
        item = schema[key]
        path = tuple(item.get("path", (key,)))
        for length in range(len(path)):
            for parent in index.records.get(path[:length], ()):
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
            group = item.get("group", "fields")
            if group not in expanded_groups:
                expanded_groups.add(group)
                for sibling in index.groups[group]:
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
