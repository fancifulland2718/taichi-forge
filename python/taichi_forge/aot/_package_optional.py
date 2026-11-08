"""Project typed input records without specializing their presence.

Allocation-controlling records retain their layout contract until conditional
owned storage can use the same invocation binding rules.
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
        # A record whose presence or dimensions allocate internal storage is
        # not merely a nullable input namespace. Preserve its existing layout.
        axes = {key for key, item in members.items() if item["kind"] == "dimension"}

        def uses_axis(value):
            if isinstance(value, str):
                return value in axes
            if isinstance(value, dict):
                return any(uses_axis(child) for child in value.values())
            return isinstance(value, (tuple, list)) and any(uses_axis(child) for child in value)

        if any(
            name in item.get("conditions", {})
            or key not in members
            and any(uses_axis(item.get(part, ())) for part in ("expression", "shape", "shape_expressions", "element"))
            for key, item in schema.items()
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
