"""Explicit construction inputs or public owners for qualified native resources."""

from collections.abc import Mapping

from taichi_forge.lang.exception import TaichiRuntimeError


def _asset_exporter(value):
    # Only concrete qualified adapters opt in; inherited hooks may lose state.
    return type(value).__dict__.get("_export_aot_resource")


class ResourceInputBindings:
    def __init__(self, bindings, private_names):
        if bindings is not None and not isinstance(bindings, Mapping):
            raise TypeError("resource_bindings must map owners to input ports or public resource names")
        self.owners = {}
        self.public_owners = {}
        self.public_names = set()
        self.names = set()
        self.used = set()
        self.used_names = set()
        for owner, ports in (bindings or {}).items():
            if isinstance(ports, str):
                if owner is None or not ports or "\0" in ports:
                    raise ValueError("Public resource bindings require an owner and a nonempty name without NUL")
                if ports in private_names:
                    raise ValueError("Public resource bindings cannot name private Graph storage")
                if ports in self.public_names:
                    raise ValueError("Distinct resource owners cannot share one public resource name")
                self.public_owners[id(owner)] = (owner, ports)
                self.public_names.add(ports)
                self.names.add(ports)
                continue
            if owner is None or not isinstance(ports, Mapping) or not ports:
                raise TypeError("Resource bindings require an owner and a nonempty port mapping")
            ports = dict(ports)
            names = [*ports, *(value for value in ports.values() if isinstance(value, str))]
            if any(not isinstance(value, str) or not value or "\0" in value for value in names):
                raise ValueError("Resource port and input names must be nonempty strings without NUL")
            if any(not isinstance(value, str) and not callable(_asset_exporter(value)) for value in ports.values()):
                raise ValueError("Resource inputs require symbolic names or qualified immutable assets")
            slots = {value for value in ports.values() if isinstance(value, str)}
            if slots & set(private_names):
                raise ValueError("Resource reconstruction inputs cannot name private Graph storage")
            self.owners[id(owner)] = (owner, ports)
            self.names.update(slots)
        if self.public_names & {
            name for _, ports in self.owners.values() for name in ports.values() if isinstance(name, str)
        }:
            raise ValueError("Public resource and construction input names must be distinct")

    def public(self, owner):
        """Opt-in by a qualified exporter; other resource families still reject."""
        entry = self.public_owners.get(id(owner))
        if entry is None:
            return None
        if entry[0] is not owner:
            raise ValueError("Public resource owner identity changed during export")
        self.used.add(id(owner))
        self.used_names.add(entry[1])
        return entry[1]

    def has_inputs(self, owner):
        """Whether this owner has explicit construction inputs (without consuming them)."""
        return id(owner) in self.owners

    def require(self, owner, ports, *, asset_ports=()):
        if id(owner) in self.public_owners:
            raise TaichiRuntimeError("This resource exporter does not support a public owner binding")
        entry = self.owners.get(id(owner))
        if entry is None or entry[0] is not owner:
            raise TaichiRuntimeError("AOT resource reconstruction requires explicit resource_bindings")
        if entry[1].keys() != set(ports):
            raise ValueError("Resource reconstruction ports must be exactly: " + ", ".join(sorted(ports)))
        if any(not isinstance(value, str) and port not in asset_ports for port, value in entry[1].items()):
            raise ValueError("This resource port requires a symbolic input binding")
        self.used.add(id(owner))
        self.used_names.update(value for value in entry[1].values() if isinstance(value, str))
        return dict(entry[1])

    def validate_used(self):
        if (self.owners.keys() | self.public_owners.keys()) - self.used:
            raise ValueError("resource_bindings contains owners unused by the exported Graph")
