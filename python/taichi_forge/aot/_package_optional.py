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
                or item["element"].get("model_tensor")
                or any(member.get("model_tensor") for member in item["element"].get("members", {}).values())
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


def project_owned_records(environment, schema, *, borrowed=False, uses=()):
    """Project a complete optional child's numeric state through existing ports.

    A typed child with no implementation/layout alternatives is a nullable
    namespace, not another numerical algorithm. Keep its Host construction and
    native layouts; use the existing record-presence and Dense argument ABI.
    Borrowed queries run after entry-specific static classification, so unused
    producer configuration and prepared numeric predicates need no presence
    variants. Their complete Field views keep the source owner's native layout.
    """
    from taichi_forge.types import primitive_types
    from ._component import Array, DenseField, _ConditionalDenseField, _InputPresence, _InputRecord
    from taichi_forge_runtime._dimensions import extent_dependencies

    declarations = dict(environment.declarations)
    for name, record in schema.items():
        conditions = record.get("conditions", {})
        if (record["kind"] != ("owner_input" if borrowed else "owner") or record.get("owner_alternatives")
                or record.get("dynamic_input_record") or len(conditions) != 1):
            continue
        flag, expected = next(iter(conditions.items()))
        predicate = schema[flag]
        if (expected is not True or predicate["kind"] != "static" or predicate.get("conditions")
                or {tuple(v) for v in predicate["choices"]} != {("bool", False), ("bool", True)}):
            continue
        path = tuple(record.get("path", (name,)))
        members = {key: item for key, item in schema.items()
                   if len(item.get("path", (key,))) > len(path)
                   and tuple(item.get("path", (key,)))[:len(path)] == path}

        def numeric_member(key, item):
            if borrowed and any(item.get(part) for part in (
                    "owner_alternatives", "choice_guards", "shape_guards", "dtype_guards", "source_guards",
                    "shape_policy", "dtype_policy", "source_policy")):
                return False
            if item["kind"] in ("dimension", "value", "dense_field"):
                return True
            if not borrowed:
                return False
            return (item["kind"] == "record" and not item.get("optional")
                    or item["kind"] == "ndarray" and item.get("external") and not item.get("optional_input")
                    or item["kind"] == "static" and (
                        item.get("prepare_static") or environment._paths[key][-1] not in uses))

        if not members or any(not numeric_member(key, item)
                              or item.get("conditions") != conditions for key, item in members.items()):
            continue
        storage = {key: item for key, item in members.items() if item["kind"] == "dense_field"}
        groups = {declarations[key].group for key in storage if isinstance(declarations[key], DenseField)}
        axes = {key for key, item in members.items() if item["kind"] == "dimension"}
        if (not storage or any(item.get("struct_field") or item["element"].get("model_tensor")
                               or not borrowed and not item.get("group_builder") for item in storage.values())
                or any(isinstance(value, DenseField) and value.group in groups and key not in storage
                       for key, value in declarations.items())
                or any(flag in item.get("conditions", {}) and key not in members and key != name
                       for key, item in schema.items())
                or any(flag in item.get(guards, {}) for item in schema.values()
                       for guards in ("choice_guards", "shape_guards", "dtype_guards", "source_guards"))
                or any(key not in members and (
                    axes.intersection(item.get("shape", ()))
                    or axes.intersection(extent_dependencies(item["expression"]) if item.get("expression") else ())
                    or borrowed and any(axes.intersection(extent_dependencies(axis))
                                        for axis in item.get("shape_expressions", ()))
                    or tuple(item.get("source_path", ()))[:len(path)] == path)
                       for key, item in schema.items())):
            continue
        # Entry-specific borrowed views share the original schema dictionaries.
        # Publish new metadata only in this projection, never in another entry.
        if borrowed:
            record = schema[name] = dict(record)
            predicate = schema[flag] = dict(predicate)
            # Presence is fixed by the normal preparation binding. Reuse the
            # existing finite scalar contract so consumption folds the absent
            # query before device generation, and rebinding invalidates it.
            predicate["prepare_static"] = True
        record.update(dynamic_input_record=True, input_presence_attribute=flag)
        predicate.update(dynamic_storage_condition=True, input_presence_attribute=flag)
        declarations[name] = _InputRecord(flag)
        declarations[flag] = _InputPresence(primitive_types.i32)
        for key, item in members.items():
            if borrowed:
                item = schema[key] = dict(item)
            item["input_record"] = name
            if key in storage:
                value = declarations[key]
                declarations[key] = _ConditionalDenseField(value.dtype, value.shape, value.group,
                                                           ((flag, True),))
                item["dynamic_input_storage"] = True
                if borrowed:
                    item["shape"] = list(value.shape)
            elif borrowed and item["kind"] == "ndarray":
                # Presence is recorded by the enclosing input namespace. Use
                # the same nullable external descriptor and lease as records;
                # it is not an allocation owned by the query's caller.
                value = declarations[key]
                declarations[key] = Array(value.dtype, value.ndim, value.shape, value.resource_id)
    environment._set_declarations(declarations)


def project_struct_member_inputs(environment, schema):
    """Use existing Dense member descriptors for a whole optional StructField.

    This is compilation metadata only: the Host still owns one original group
    and record. No new allocation, aggregate pointer ABI or layout is invented.
    """
    from ._component import _capture_id

    for name, item in list(schema.items()):
        if not item.get("dynamic_input_storage"):
            continue
        if not item.get("struct_field"):
            item["shape"] = list(environment.declarations[name].shape)
            continue
        path = tuple(item.get("path", (name,)))
        for member, element in item["element"]["members"].items():
            key = _capture_id((*path, member))
            schema[key] = {
                **{k: v for k, v in item.items() if k != "struct_field"},
                "path": [*path, member],
                "element": element,
                "shape": list(environment.declarations[key].shape),
            }
