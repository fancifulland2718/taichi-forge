"""Cold, device-independent validation of persistent resource dependencies."""


def retained_arguments(resources, gradients, arguments):
    """Runtime closure of validated resources; shape metadata is cold-only."""
    by_name = {resource["name"]: resource for resource in resources}
    retained, pending = set(arguments), list(arguments)
    for name in pending:
        resource = by_name.get(name, {})
        refs = list(resource.get("bindings", {}).values()) if resource.get("kind") == "native" else []
        if name in gradients:
            refs.append(gradients[name])
        for ref in refs:
            if ref not in retained:
                retained.add(ref)
                pending.append(ref)
    return frozenset(retained)


def external_arguments(resources, gradients, arguments):
    from ._runtime import _name
    from ._native import NATIVE_FIELDS, native_references

    if not isinstance(resources, list) or not isinstance(gradients, dict):
        raise ValueError("Execution resources must be a list and gradients a name mapping")
    if not resources and not gradients:
        return tuple(sorted(arguments))
    by_name, references = {}, {}

    def word(value, *, positive=False):
        if type(value) is not int or not int(positive) <= value <= 0xFFFFFFFF:
            raise ValueError("Execution resource dimensions/axes must fit uint32")

    def extent(value):
        _name(value["array"])
        word(value["axis"])
        return value["array"]

    for resource in resources:
        if not isinstance(resource, dict):
            raise ValueError("Execution resource entries must be objects")
        kind = resource.get("kind")
        if kind == "ndarray":
            required, optional = {"kind", "name", "dtype", "shape"}, {"element_shape"}
        elif kind == "extent":
            required, optional = {"kind", "name", "dtype", "array", "axis"}, set()
        elif kind == "native":
            required, optional = NATIVE_FIELDS | {"name"}, set()
        else:
            raise ValueError(f"Unsupported execution resource kind: {kind}")
        if not required <= set(resource) or set(resource) - required - optional:
            raise ValueError(f"Invalid {kind} resource fields")
        name = resource["name"]
        _name(name)
        if kind != "native":
            _name(resource["dtype"])
        if name in by_name:
            raise ValueError(f"Duplicate execution resource: {name}")
        by_name[name] = resource
        refs = references[name] = []
        if kind == "native":
            refs.extend(native_references(resource))
        elif kind == "ndarray":
            shape, element_shape = resource["shape"], resource.get("element_shape", [])
            if not isinstance(shape, list) or not isinstance(element_shape, list):
                raise ValueError("Execution resource shapes must be arrays")
            for dim in shape:
                if isinstance(dim, dict):
                    if set(dim) != {"array", "axis"}:
                        raise ValueError("Execution shape reference requires array and axis")
                    refs.append(extent(dim))
                else:
                    word(dim)
            for dim in element_shape:
                word(dim, positive=True)
        else:
            if resource["dtype"] not in ("i32", "u32", "i64", "u64"):
                raise ValueError("Execution extent requires a 32/64-bit integer dtype")
            refs.append(extent(resource))

    indegree, dependents = dict.fromkeys(by_name, 0), {name: [] for name in by_name}
    for name, refs in references.items():
        for parent in refs:
            if parent in by_name:
                if by_name[name]["kind"] != "native" and by_name[parent]["kind"] != "ndarray":
                    raise ValueError(f"Execution shape reference requires an ndarray: {parent}")
                indegree[name] += 1
                dependents[parent].append(name)
    ready = [name for name, degree in indegree.items() if degree == 0]
    for parent in ready:
        for child in dependents[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(ready) != len(by_name):
        raise ValueError("Execution resource dependency cycle")

    for primal, gradient in gradients.items():
        _name(primal)
        _name(gradient)
        for name in (primal, gradient):
            if name in by_name and by_name[name]["kind"] != "ndarray":
                raise ValueError(f"Execution gradient reference requires an ndarray: {name}")
        arguments.update((primal, gradient))
    pending, seen, external = list(arguments), set(), set()
    for name in pending:
        if name not in by_name:
            external.add(name)
        elif name not in seen:
            seen.add(name)
            pending.extend(references[name])
    return tuple(sorted(external))
