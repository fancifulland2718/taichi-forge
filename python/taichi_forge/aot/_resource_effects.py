"""Compiler-owned write facts for cold preparation, never numerical execution."""


def package_effects(metadata, arguments, captures):
    descriptors = [*arguments, *captures]
    resources = {
        index
        for index, item in enumerate(descriptors)
        if item["kind"] in ("ndarray", "dense_field", "model_tensor_field", "local_tensor_workspace")
    }
    proven = bool(metadata.get("available")) and not metadata.get("opaque", True)
    accesses = {index: "none" for index in resources}
    for effect in metadata.get("effects", ()):
        path, access = effect.get("arg_id"), effect.get("access")
        if (
            effect.get("resource_kind") != "argument"
            or len(path or ()) != 1
            or path[0] not in resources
            or access not in ("read", "write", "read_write", "atomic", "opaque")
        ):
            proven = False
            break
        index = path[0]
        previous = accesses[index]
        accesses[index] = access if previous in ("none", access) else "read_write"
    return dict(
        precision="compiler_proven" if proven else "conservative",
        arguments={str(index): accesses[index] if proven else "opaque" for index in sorted(resources)},
    )


def written_resources(owner, record):
    """Return possible written captures of an owner's normal Graph recording.

    Recording may run ordinary Host preparation, but never a numerical step.
    Unknown compiler effects retain their resources. Explicit writable array
    arguments require caller binding and are rejected by this owner-only query.
    The result holds original objects, not copied data or independent storage.
    """
    from ._component import Component

    if getattr(record, "__self__", None) is not owner:
        raise ValueError("Write inspection requires the owner's bound recording method")
    artifact = Component.from_owner(owner).compile_graph(record)
    try:
        contract = artifact.contract
        for entry in contract["compilation"]["entries"]:
            for effect in entry.get("effects", ()):
                if effect.get("argument") in contract["explicit"] and effect["access"] not in ("read", "none"):
                    raise ValueError("Write inspection requires bindings for explicit writable resources")
        sources = contract["compilation"]["input_sources"]["resources"]
        result = {}
        for name, resource in contract["resources"].items():
            if resource.get("effects_precision") == "compiler_proven" and resource["access"] in ("read", "none"):
                continue
            for source in sources[name].values():
                if source["kind"] != "attribute":
                    raise ValueError("Write inspection requires an owned resource source")
                value = owner
                for part in source["path"]:
                    value = value[part] if isinstance(part, int) else getattr(value, part)
                result[id(value)] = value
        return tuple(result.values())
    finally:
        artifact._graph.close()
