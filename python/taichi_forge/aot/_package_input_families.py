"""Typed constructor contexts from declared calls, never from executed models."""

import copy
import hashlib
import json


def source_class(tree, name):
    return getattr(tree, "_package_owner_contexts", {}).get(name, name)


def split_constructor_inputs(sources, owner, inputs, sites, input_types):
    """Split resource contracts only for actual, fully proved argument tuples.

    Numerical definitions and the projected Host class remain single definitions.
    Model sizes and the ordering/count of constructed instances are not keys.
    """
    parameters = {name: item for (identity, name), item in inputs.items() if identity == owner}
    selections = []
    for caller, arguments, declaration in sites:
        selection = {}
        for name, item in parameters.items():
            values = input_types(caller, arguments.get(item["input_parameter"]))
            if values is None or len(values) != 1 or next(iter(values))[0] is None:
                raise ValueError("Typed resource inputs require one proven source at each constructor call: " + owner)
            identity, path = next(iter(values))
            if (item["kind"] == "owner_input") == bool(path):
                raise ValueError("Typed constructor input differs from its resource kind: " + owner)
            selection[name] = dict(source_type=identity, source_member=list(path))
        selections.append((selection, declaration))
    if not selections:
        raise ValueError("Typed resource inputs require declared constructor calls: " + owner)
    module, _, base = owner.rpartition(".")
    source = next(source for source in sources if source[0] == module)
    tree, schemas = source[4], source[6]
    contexts = getattr(tree, "_package_owner_contexts", {})
    families = getattr(tree, "_package_input_families", {})
    if base in contexts or base in families:
        raise ValueError("Recursive typed resource input contexts require resolved source definitions: " + owner)
    original = schemas.pop(base)
    variants = {}
    for selection, declaration in selections:
        digest = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
        name = base + "__input_" + digest
        if name not in variants:
            schema = copy.deepcopy(original)
            for attribute, origin in selection.items():
                if schema[attribute]["kind"] == "owner_input":
                    schema[attribute]["owner_type"] = origin["source_type"]
                else:
                    schema[attribute].update(origin)
            schemas[name] = schema
            contexts[name] = base
            variants[name] = {
                "owner": module + "." + name,
                "inputs": {
                    parameters[attribute]["input_parameter"]: {"kind": parameters[attribute]["kind"], **origin}
                    for attribute, origin in selection.items()
                },
            }
        declaration["owner_type"] = module + "." + name
    families[base] = list(variants.values())
    tree._package_owner_contexts = contexts
    tree._package_input_families = families
