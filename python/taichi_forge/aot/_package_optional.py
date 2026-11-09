"""Project optional input records and complete allocation groups.

Whole optional Dense groups retain their native layout and allocation lifetime;
partial or mixed-layout conditions keep the existing specialization contract.
"""

def project_input_records(owner, schema, declarations):
    from taichi_forge.types import primitive_types
    from ._component import _InputPresence, _InputRecord
    from ._package_records import write_member

    for name, record in list(schema.items()):
        if record["kind"] != "record" or not record.get("optional") or record.get("struct_field"):
            continue
        path = tuple(record.get("path", (name,)))
        members = {
            key: item
            for key, item in schema.items()
            if len(item.get("path", (key,))) > len(path) and tuple(item.get("path", (key,)))[: len(path)] == path
        }
        if not members or any(
            not (
                item["kind"] == "dimension"
                or item["kind"] == "value"
                and item.get("element", {}).get("kind", "scalar") == "scalar"
                or item["kind"] == "ndarray"
                and item.get("external")
                and not item.get("optional_input")
            )
            for item in members.values()
        ):
            continue
        # Whole present-only groups can share code without sharing storage.
        # Partial group layouts and dimensions escaping the record need their
        # existing layout contract, not an inferred nullable descriptor.
        axes = {key for key, item in members.items() if item["kind"] == "dimension"}

        def uses_axis(value):
            if isinstance(value, str):
                return value in axes
            if isinstance(value, dict):
                return any(uses_axis(child) for child in value.values())
            return isinstance(value, (tuple, list)) and any(uses_axis(child) for child in value)

        conditioned = {key: item for key, item in schema.items() if name in item.get("conditions", {})}
        storage = {key: item for key, item in conditioned.items() if item["kind"] == "dense_field"}
        groups = {item.get("group", "fields") for item in storage.values()}
        if (
            any(
                item.get("conditions") != {name: True}
                or item["kind"] not in ("dense_field", "dimension")
                or item["kind"] == "dense_field"
                and (item.get("struct_field") or item["element"].get("model_tensor") or not item.get("group_builder"))
                for item in conditioned.values()
            )
            or any(
                item["kind"] == "dense_field" and item.get("group", "fields") in groups and key not in storage
                for key, item in schema.items()
            )
            or any(
                key not in members
                and any(
                    uses_axis(item.get(part, ())) for part in ("expression", "shape", "shape_expressions", "element")
                )
                for key, item in schema.items()
            )
        ):
            # A first edge leaving the nullable namespace also covers derived
            # dimensions that subsequently feed an owned layout or another
            # input. Those consumers need a joint presence/binding contract.
            continue
        presence = "_forge_package_record_present_" + name
        record.update(dynamic_input_record=True, input_presence_attribute=presence)
        schema[presence] = dict(kind="value", dtype="i32", input_presence=True)
        declarations[presence] = _InputPresence(primitive_types.i32)
        declarations[name] = _InputRecord(presence)
        write_member(owner, presence, 1, schema)
        for item in members.values():
            item["input_record"] = name
        for item in storage.values():
            item["input_record"] = name
            item["dynamic_input_storage"] = True


def project_boolean_groups(schema, declarations):
    """Reuse nullable Dense arguments for a whole present-only allocation group.

    Keep the original Host boolean declaration and allocation. Only its device
    use becomes an immutable invocation value. Mixed groups, layout selection,
    child construction and other conditional declarations remain specialized.
    """
    from taichi_forge.types import primitive_types
    from ._component import _InputPresence

    candidates = {
        name
        for name, predicate in schema.items()
        if predicate["kind"] == "static"
        and not predicate.get("conditions")
        and not predicate.get("kernel_only")
        and not predicate.get("static_dtype")
        and {tuple(value) for value in predicate.get("choices", ())} == {("bool", False), ("bool", True)}
    }

    def eligible(name):
        conditioned = {key: item for key, item in schema.items() if name in item.get("conditions", {})}
        storage = {key: item for key, item in conditioned.items() if item["kind"] == "dense_field"}
        groups = {item.get("group", "fields") for item in storage.values()}
        return not (
            not storage
            or any(
                not item.get("conditions", {}).keys() <= candidates
                or any(value is not True for value in item.get("conditions", {}).values())
                # A conditionally declared dimension can be read without its
                # Field. It needs a presence contract of its own before it can
                # use a nullable argument, unlike an unconditional shape axis.
                or item["kind"] != "dense_field"
                or item.get("struct_field")
                or item["element"].get("model_tensor")
                or not item.get("group_builder")
                for item in conditioned.values()
            )
            or any(
                item["kind"] == "dense_field" and item.get("group", "fields") in groups and key not in storage
                for key, item in schema.items()
            )
            or any(
                name in item.get(guards, {})
                for item in schema.values()
                for guards in ("choice_guards", "shape_guards", "dtype_guards", "source_guards")
            )
        )

    # A nested group's predicates form one eligibility closure. If one guard
    # also selects a type/layout or a partial group, retain specialization for
    # every group relying on that guard. No application policy is inferred.
    while True:
        accepted = {name for name in candidates if eligible(name)}
        if accepted == candidates:
            break
        candidates = accepted
    for name in sorted(candidates):
        predicate = schema[name]
        predicate.update(dynamic_storage_condition=True, input_presence_attribute=name)
        declarations[name] = _InputPresence(primitive_types.i32)
    for item in schema.values():
        conditions = item.get("conditions", {})
        if item["kind"] == "dense_field" and conditions and conditions.keys() <= candidates:
            names = sorted(conditions)
            item.update(input_record=names[0], dynamic_input_storage=True)
            if len(names) > 1:
                item["input_conditions"] = names
