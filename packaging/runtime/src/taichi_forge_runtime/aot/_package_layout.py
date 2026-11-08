"""Select declared SNode members; the native schema compiler owns byte layout."""

import json


def layout_selection(contract, owner):
    names = {name for condition in contract.get("conditions", {}).values() for name in condition}
    names.update(policy["condition"] for policy in contract.get("orders", {}).values())
    selected = {name: getattr(owner, name) for name in names}
    if any(type(value) is not bool for value in selected.values()):
        raise ValueError("Conditional layout requires its declared boolean configuration")
    return selected


def _select_members(contract, selected):
    if not contract.get("conditions"):
        return contract
    active = {
        name: paths
        for name, paths in contract["members"].items()
        if all(selected[key] is value for key, value in contract["conditions"].get(name, {}).items())
    }
    source = json.loads(contract["schema"])
    source_paths, _ = _tree(source["nodes"])

    def member_paths(selector):
        if selector[0] == "schema_nodes":
            return [source_paths[index] for index in selector[1]]
        return selector

    # Remove inactive place nodes and their now-unused Dense ancestors. This
    # changes construction topology only: no strides, offsets, padding or sizes
    # are calculated here. instantiate() still uses the original StructCompiler.
    needed = {()}
    for selector in active.values():
        for path in member_paths(selector):
            needed.update(tuple(path[:length]) for length in range(len(path) + 1))
    for container in contract.get("containers", ()):
        if all(selected[key] is value for key, value in container["conditions"].items()):
            path = container["path"]
            needed.update(tuple(path[:length]) for length in range(len(path) + 1))
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
    dimensions = sorted({extent["parameter"] for node in nodes
                         for extent in node["extents"] + node.get("tensor_extents", []) if extent["parameter"]})
    return dict(
        **{
            key: value
            for key, value in contract.items()
            if key not in ("schema", "members", "dimensions", "conditions", "containers")
        },
        schema=json.dumps({**source, "nodes": nodes}, separators=(",", ":")),
        members={
            name: (
                ["schema_nodes", [retained[index] for index in selector[1]]]
                if selector[0] == "schema_nodes"
                else [list(new_paths[tuple(path)]) for path in selector]
            )
            for name, selector in active.items()
        },
        dimensions=dimensions,
    )


def _tree(nodes):
    paths, children = {}, {}
    for index, node in enumerate(nodes):
        parent = node["parent"]
        siblings = children.setdefault(parent, [])
        paths[index] = () if parent < 0 else (*paths[parent], len(siblings))
        siblings.append(index)
    return paths, children


def _select_orders(contract, selected):
    if not contract.get("orders"):
        return contract
    source = json.loads(contract["schema"])
    nodes = source["nodes"]
    paths, children = _tree(nodes)
    by_path = {path: index for index, path in paths.items()}

    def member_nodes(selector):
        return list(selector[1]) if selector[0] == "schema_nodes" else [by_path[tuple(path)] for path in selector]

    replacements = {}
    for name, policy in contract["orders"].items():
        if name not in contract["members"]:
            continue
        leaves = member_nodes(contract["members"][name])
        parent = nodes[leaves[0]]["parent"]
        if parent <= 0 or leaves != children.get(parent) or any(children.get(leaf) for leaf in leaves):
            raise ValueError("Field order selection requires one independent AOS allocation branch")
        chain = []
        index = parent
        while index:
            node = nodes[index]
            if node["type"] != 1 or node.get("tensor_extents") or not node["extents"]:
                raise ValueError("Field order selection requires ordinary independent Dense index nodes")
            if index != parent and children[index] != [chain[-1]]:
                raise ValueError("Field order selection cannot rearrange shared placement")
            chain.append(index)
            index = node["parent"]
        extents = {extent["axis"]: extent for index in chain for extent in nodes[index]["extents"]}
        if len(extents) != sum(len(nodes[index]["extents"]) for index in chain) or sorted(extents) != list(
            range(len(extents))
        ):
            raise ValueError("Field order selection requires each logical index axis exactly once")
        order = policy["if_true" if selected[policy["condition"]] else "if_false"]
        axes = list(range(len(extents))) if order is None else [ord(axis) - ord("i") for axis in order]
        if sorted(axes) != list(range(len(extents))):
            raise ValueError("Field order selection differs from the declared logical axes")
        root = chain[-1]
        if root in replacements:
            raise ValueError("Field order selection cannot duplicate an allocation branch")
        groups = [[extents[axis] for axis in axes]] if order is None else [[extents[axis]] for axis in axes]
        replacements[root] = (groups, leaves)

    updated, mapping = [], {}

    def emit(index, parent):
        node = nodes[index]
        if index in replacements:
            groups, leaves = replacements[index]
            for extents in groups:
                updated.append({**node, "parent": parent, "extents": extents})
                parent = len(updated) - 1
            for leaf in leaves:
                emit(leaf, parent)
            return
        mapping[index] = len(updated)
        updated.append({**node, "parent": parent})
        for child in children.get(index, ()):
            emit(child, mapping[index])

    emit(0, -1)
    new_paths, _ = _tree(updated)
    members = {}
    for name, selector in contract["members"].items():
        indices = [mapping[index] for index in member_nodes(selector)]
        members[name] = (
            ["schema_nodes", indices]
            if selector[0] == "schema_nodes"
            else [list(new_paths[index]) for index in indices]
        )
    # This changes only the declared Dense construction sequence. Dtypes,
    # tensor axes and logical extents are retained from native capture. The
    # native schema compiler still computes every byte offset and stride.
    return {**contract, "schema": json.dumps({**source, "nodes": updated}, separators=(",", ":")), "members": members}


def select_layout(contract, selected):
    return _select_orders(_select_members(contract, selected), selected)
