"""Compact cold storage recipes; warm replay is entirely native."""

import ctypes as C


def storage_recipe_version(recipe):
    return 22 if isinstance(recipe, dict) and recipe.get("version") == 2 else 21


def validate_storage_recipe(document, replay_arguments):
    recipe = document.get("storage_recipe")
    if recipe is None:
        return
    if (
        not isinstance(recipe, dict)
        or recipe.get("kind") != "ordered_temporary_ring"
        or type(recipe.get("version")) is not int
        or recipe["version"] not in (1, 2)
    ):
        raise ValueError("Invalid ordered storage ring recipe")
    from ._runtime import _name

    if recipe["version"] == 1:
        if set(recipe) != {"kind", "version", "capacity", "arrays"}:
            raise ValueError("Invalid ordered storage ring recipe")
        groups = [dict(capacity=recipe["capacity"], arrays=recipe["arrays"])]
    else:
        if set(recipe) != {"kind", "version", "groups"}:
            raise ValueError("Invalid grouped storage ring recipe")
        groups = recipe["groups"]
        if not isinstance(groups, list) or not groups:
            raise ValueError("Storage ring groups must be nonempty")
    names = []
    for group in groups:
        if (
            not isinstance(group, dict)
            or set(group) != {"capacity", "arrays"}
            or type(group["capacity"]) is not int
            or not 0
            < group["capacity"]
            <= min(0x7FFFFFFFFFFFFFFF, C.c_size_t(-1).value)
        ):
            raise ValueError("Invalid ordered storage ring group")
        if not isinstance(group["arrays"], list) or not group["arrays"]:
            raise ValueError("Storage ring requires private arrays")
        for name in group["arrays"]:
            _name(name)
        names.extend(group["arrays"])
    selected = set(names)
    if len(selected) != len(names):
        raise ValueError("Duplicate storage ring array")
    operations = document["operations"]
    if (
        not operations
        or any(
            key in document
            for key in ("binding_strategy", "submission_recipe", "initialization")
        )
        or any(
            op["kind"] not in ("kernel", "copy", "fill32") or "launch" in op
            for op in operations
        )
    ):
        raise ValueError("Storage rings require ordinary fixed kernel/copy/fill replay")
    resources = document.get("resources", [])
    by_name = {r["name"]: r for r in resources}
    for name in selected:
        if (
            by_name.get(name, {}).get("kind") != "ndarray"
            or name not in replay_arguments
        ):
            raise ValueError("Storage ring arrays must be used private ndarrays")
    for resource in resources:
        if resource["name"] in selected:
            continue
        references = []
        if resource["kind"] in ("extent", "ndarray_view", "ndarray_member"):
            references.append(resource["array"])
        if resource["kind"] == "native":
            references.extend(resource["bindings"].values())
        if resource["kind"] == "ndarray" and "allocator" in resource:
            references.append(resource["allocator"])
        if resource["kind"] in ("ndarray", "ndarray_view"):
            references.extend(
                axis["array"] for axis in resource["shape"] if isinstance(axis, dict)
            )
        if selected.intersection(references):
            raise ValueError("Shared resources cannot depend on storage ring arrays")
    for primal, gradient in document.get("gradients", {}).items():
        if primal in selected or gradient in selected:
            raise ValueError("Storage ring arrays cannot own gradient bindings")
