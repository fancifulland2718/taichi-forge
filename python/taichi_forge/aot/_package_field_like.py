"""Resolve field_like from declared allocation types, without sampling instances."""

import ast
import copy


def declared_struct_members(item):
    """Read the already-normalized member types, without constructing a Field."""
    node = item.get("type_node")
    if not item.get("struct_field"):
        return {}
    if not isinstance(node, ast.Dict) or not all(
        isinstance(key, ast.Constant) and isinstance(key.value, str) for key in node.keys
    ):
        raise ValueError("Struct Field inventory requires resolved member declarations")
    return {key.value: value for key, value in zip(node.keys, node.values)}


def resolve_field_likes(sources):
    owners = {
        module + "." + name: (module, schema)
        for module, _, _, _, _, _, schemas, _ in sources
        for name, schema in schemas.items()
    }

    def allocation(owner, path, visiting=(), *, present=False):
        if not path:
            raise ValueError("field_like requires a declared complete Field allocation")
        token = (owner, tuple(path))
        if token in visiting:
            raise ValueError("Cyclic field_like source declaration")
        schema = owners[owner][1]
        name, *rest = path
        item = schema.get(name, {})
        if item.get("conditions") and not present:
            raise ValueError(
                "field_like currently requires an unconditional original allocation"
            )
        if item.get("kind") in ("owner", "owner_input"):
            if len({choice["owner_type"] for choice in item.get("owner_alternatives", ())}) > 1:
                raise ValueError("field_like across child implementation choices requires a resolved allocation type")
            identity, field, prefix = allocation(item["owner_type"], rest, (*visiting, token), present=present)
            return identity, field, [name, *prefix]
        if item.get("kind") == "borrowed_field" and not rest:
            return allocation(
                owner, item["source_path"], (*visiting, token), present=present or "input_parameter" in item
            )
        if item.get("kind") == "field_like":
            resolve(owner, name, (*visiting, token))
            item = schema[name]
        if item.get("kind") == "dense_field" and len(rest) == 1:
            members = declared_struct_members(item)
            if rest[0] in members:
                item = copy.deepcopy(item)
                item["type_node"] = copy.deepcopy(members[rest[0]])
                item.pop("struct_field", None)
                used = {axis for axis in item["shape"] if isinstance(axis, str)} | {
                    getattr(node, "_package_model_extent", None) for node in ast.walk(item["type_node"])
                }
                item["dimension_sources"] = {
                    key: value for key, value in item["dimension_sources"].items() if key in used
                }
                rest = []
        if item.get("kind") != "dense_field" or rest:
            raise ValueError("field_like requires a declared complete Field allocation")
        return owner, item, []

    def resolve(owner, name, visiting=()):
        _, schema = owners[owner]
        item = schema[name]
        source_owner, original, prefix = allocation(
            owner, item["source_path"], visiting
        )
        if source_owner == owner and original["allocation"] >= item["allocation"]:
            raise ValueError(
                "field_like source must precede the independent allocation"
            )
        source_expression = "self" + "".join("." + part for part in prefix)
        remap = {
            key: "_forge_package_like_" + name + "_" + key
            for key in original["dimension_sources"]
        }
        field = copy.deepcopy(original)
        # The clone owns its placement. Its sizes are read from the proved
        # original namespace in retained Host construction, never from a model
        # witness or another instance's allocation.
        field.update(
            allocation=item["allocation"],
            scope_allocation=item["scope_allocation"],
            group=item["group"] if "group_builder" in item else "field_like_" + name,
            layout=item["layout"],
            shape=[remap.get(axis, axis) for axis in original["shape"]],
            dimension_sources={remap[key]: source_expression + "." + key for key in original["dimension_sources"]},
            dimension_expressions={},
            like_source_path=item["source_path"],
            type_declaration_module=original.get(
                "type_declaration_module", original.get("declaration_module", owners[source_owner][0])
            ),
            declaration_module=item.get("declaration_module", owners[owner][0]),
        )
        field.pop("declaration_owner", None)
        if "declaration_owner" in item:
            field["declaration_owner"] = item["declaration_owner"]
        for key in (
            "local_type_source",
            "member_initialization_node",
            "member_initialization_source",
            "dimension_alias_checks",
            "placement_alias",
            "placement",
            "group_builder",
            "order",
            "order_policy",
            "order_source",
            "conditions",
        ):
            field.pop(key, None)
        if "group_builder" in item:
            # Clone the element/layout contract, not the source tree owner.
            # Explicit construction owns all copies in this original group.
            field["group_builder"] = copy.deepcopy(item["group_builder"])
        for node in ast.walk(field["type_node"]):
            axis = getattr(node, "_package_model_extent", None)
            if axis in remap:
                node._package_model_extent = remap[axis]
        schema[name] = field

    for owner, (_, schema) in owners.items():
        for name, item in tuple(schema.items()):
            if item["kind"] == "field_like":
                resolve(owner, name)


def project_field_likes(environment, schema):
    from dataclasses import replace
    from ._component import _capture_id, _ModelTensorField

    declarations = dict(environment.declarations)
    derived = set(environment._derived_attributes)
    for name, item in schema.items():
        if "like_source_path" not in item:
            continue
        target_path = tuple(item.get("path", (name,)))
        source_path = tuple(item["like_source_path"])
        members = item["element"]["members"] if item.get("struct_field") else (None,)
        for member in members:
            suffix = () if member is None else (member,)
            target = declarations[_capture_id((*target_path, *suffix))]
            source = declarations[_capture_id((*source_path, *suffix))]
            pairs = list(zip(target.shape, source.shape))
            if isinstance(target, _ModelTensorField):
                pairs.extend(zip(target.element_shape, source.element_shape))
            for left, right in pairs:
                if isinstance(left, str):
                    expression = (
                        ("dim", right) if isinstance(right, str) else ("const", right)
                    )
                    declarations[left] = replace(
                        declarations[left], expression=expression
                    )
                    derived.add(left)
                elif left != right:
                    raise ValueError("field_like source has incompatible constant axes")
    environment._set_declarations(declarations)
    environment._derived_attributes = frozenset(derived)
