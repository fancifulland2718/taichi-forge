"""Explicit consumer inputs for qualified opaque-resource reconstruction."""

from collections.abc import Mapping

from taichi_forge.lang.exception import TaichiRuntimeError


class ResourceInputBindings:
    def __init__(self, bindings, private_names):
        if bindings is not None and not isinstance(bindings, Mapping):
            raise TypeError("resource_bindings must map resource owners to named input ports")
        self.owners = {}
        self.names = set()
        self.used = set()
        self.used_names = set()
        for owner, ports in (bindings or {}).items():
            if owner is None or not isinstance(ports, Mapping) or not ports:
                raise TypeError("Resource bindings require an owner and a nonempty port mapping")
            ports = dict(ports)
            if any(not isinstance(value, str) or not value or "\0" in value for value in (*ports, *ports.values())):
                raise ValueError("Resource port and input names must be nonempty strings without NUL")
            if set(ports.values()) & set(private_names):
                raise ValueError("Resource reconstruction inputs cannot name private Graph storage")
            self.owners[id(owner)] = (owner, ports)
            self.names.update(ports.values())

    def require(self, owner, ports):
        entry = self.owners.get(id(owner))
        if entry is None or entry[0] is not owner:
            raise TaichiRuntimeError("AOT resource reconstruction requires explicit resource_bindings")
        if entry[1].keys() != set(ports):
            raise ValueError("Resource reconstruction ports must be exactly: " + ", ".join(sorted(ports)))
        self.used.add(id(owner))
        self.used_names.update(entry[1].values())
        return dict(entry[1])

    def validate_used(self):
        if self.owners.keys() - self.used:
            raise ValueError("resource_bindings contains owners unused by the exported Graph")
