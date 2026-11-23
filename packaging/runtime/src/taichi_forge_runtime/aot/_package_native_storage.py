"""Instance-owned SNode storage for projected Host objects."""

from contextlib import closing

from .._dimensions import evaluate_extent
from ._package_values import StructField, dimension_value, owner_type, path_value


class NativeStorage:
    def __init__(self, package, owner):
        self.package = package
        self.groups = {}
        self.module = None
        self.kernels = {}
        self.layouts = {}
        self.owner = owner

    def group(self, name):
        if name not in self.groups:
            key = f"{owner_type(self.owner)}:{name}"
            spec = self.package._manifest["native_storage_groups"][key]
            self.groups[name] = spec
        return self.groups[name]

    def layout(self, name):
        if self.module is not None:
            return self.layouts[name]
        variants = self.group(name)["slots"]
        matches = [item for item in variants if all(
            dimension_value(self.owner, key) is value for key, value in item["conditions"].items())]
        if len(matches) != 1:
            raise ValueError(f"Native storage group {name!r} has an ambiguous or missing layout")
        return matches[0]

    def bind(self):
        if self.module is None:
            dimensions = {}
            layouts = {}
            for group, spec in self.groups.items():
                layouts[group] = self.layout(group)
                slot = layouts[group]["slot"]
                dimensions[slot] = {
                    symbol: evaluate_extent(fact["expression"], lambda key: path_value(self.owner, fact["paths"][key]))
                    for symbol, fact in spec["dimensions"].items()
                }
            view = self.package._module._bind_native_storage(dimensions)
            self.package._native_views.add(view)
            self.layouts = layouts
            self.module = view
        return self.module

    def close(self):
        if self.module is not None:
            self.module.close()
            self.package._native_views.discard(self.module)
            self.module = None
        self.kernels.clear()
        self.layouts.clear()

    def kernel(self, name):
        if name not in self.kernels:
            self.kernels[name] = self.bind().kernel(name)
        return self.kernels[name]


def entry_storage(owner, contract):
    storage = None
    for binding in contract.get("native_storage", ()):
        allocated = path_value(owner, binding["owner_path"])
        state = allocated._forge_package_native_storage
        if state.layout(binding["group"])["slot"] != binding["slot"]:
            raise ValueError("Native storage layout differs from the selected numerical entry")
        if storage is not None and state is not storage:
            raise ValueError("Numerical entry uses independently owned native storage modules")
        storage = state
    return storage


class StorageNode:
    def __init__(self, state, group, attribute):
        self.state, self.group, self.attribute = state, group, attribute

    def deactivate_all(self, *, builder=None):
        state = self.state
        layout = state.layout(self.group)
        for name in layout["deactivate"][self.attribute]:
            if builder is None:
                state.kernel(name)()
            else:
                builder._dispatch_native_storage(state.owner, self.group, layout["slot"], name)


class StorageField:
    def __init__(self, state, group, path, element, shape):
        self.state, self.group, self.path = state, group, path
        self.dtype, self.element_shape = element["dtype"], tuple(element["element_shape"])
        self.shape = shape

    def _transfer(self, direction, values=None):
        state = self.state
        symbol = state.layout(self.group)["members"][".".join(self.path)][direction]
        with closing(state.package.runtime.ndarray(self.dtype, self.shape + self.element_shape)) as buffer:
            if direction == "read":
                # The transfer visits active cells only. NumPy returns zero
                # for every absent cell, including after deactivation.
                buffer.fill(0)
                state.kernel(symbol)(buffer)
                return buffer.to_numpy()
            buffer.from_numpy(values)
            state.kernel(symbol)(buffer)

    def to_numpy(self):
        return self._transfer("read")

    def from_numpy(self, values):
        self._transfer("write", values)


def field(state, group, name):
    spec = state.group(group)["members"][name]
    shape = tuple(evaluate_extent(expr, lambda key: dimension_value(state.owner, key))
                  for expr in spec["shape_expressions"])

    def member(path, element):
        if element["kind"] == "struct":
            return StructField({key: member((*path, key), child) for key, child in element["members"].items()})
        return StorageField(state, group, path, element, shape)

    return member((name,), spec["element"])
