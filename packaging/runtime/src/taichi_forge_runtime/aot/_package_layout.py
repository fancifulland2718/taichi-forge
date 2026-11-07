"""Select declared SNode members; the native schema compiler owns byte layout."""

import json


def layout_selection(contract, owner):
    names = {name for condition in contract.get("conditions", {}).values() for name in condition}
    selected = {name: getattr(owner, name) for name in names}
    if any(type(value) is not bool for value in selected.values()):
        raise ValueError("Conditional layout requires its declared boolean configuration")
    return selected


def select_layout(contract, selected):
    if not contract.get("conditions"):
        return contract
    active = {
        name: paths
        for name, paths in contract["members"].items()
        if all(selected[key] is value for key, value in contract["conditions"].get(name, {}).items())
    }
    # Remove inactive place nodes and their now-unused Dense ancestors. This
    # changes construction topology only: no strides, offsets, padding or sizes
    # are calculated here. instantiate() still uses the original StructCompiler.
    needed = {()}
    for paths in active.values():
        for path in paths:
            needed.update(tuple(path[:length]) for length in range(len(path) + 1))
    for container in contract.get("containers", ()):
        if all(selected[key] is value for key, value in container["conditions"].items()):
            path = container["path"]
            needed.update(tuple(path[:length]) for length in range(len(path) + 1))
    source = json.loads(contract["schema"])
    paths, children, retained, new_paths, new_children, nodes = {}, {}, {}, {}, {}, []
    for index, node in enumerate(source["nodes"]):
        parent = node["parent"]
        ordinal = children.get(parent, 0)
        children[parent] = ordinal + 1
        path = () if parent < 0 else (*paths[parent], ordinal)
        paths[index] = path
        if path not in needed:
            continue
        retained[index] = len(nodes)
        new_parent = -1 if parent < 0 else retained[parent]
        new_ordinal = new_children.get(new_parent, 0)
        new_children[new_parent] = new_ordinal + 1
        new_paths[path] = () if parent < 0 else (*new_paths[paths[parent]], new_ordinal)
        nodes.append({**node, "parent": new_parent})
    if not needed <= new_paths.keys():
        raise ValueError("Conditional layout references a missing native member path")
    dimensions = sorted({extent["parameter"] for node in nodes for extent in node["extents"] if extent["parameter"]})
    return dict(
        schema=json.dumps({**source, "nodes": nodes}, separators=(",", ":")),
        members={name: [list(new_paths[tuple(path)]) for path in paths] for name, paths in active.items()},
        dimensions=dimensions,
    )
