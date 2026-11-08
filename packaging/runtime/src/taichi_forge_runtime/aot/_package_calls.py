"""Reuse ordinary definition-call binding while its observed inputs stay fixed.

This caches native prepared execution, not device results. Changes re-enter
the existing admission/binding path; replay retains the native resource leases.
No module-global resource pointer is changed to select an instance.
"""

import operator
import struct


_MISSING = object()
_PATH_KEYS = frozenset(("path", "source_path", "reference_path", "flag_path", "nullable_record", "layout_owner_path"))


def specialization_entries(path, target, entries):
    """Read native execution policy once, outside ordinary call dispatch.

    The native loader validates the same metadata. Unsupported formats/tasks
    keep ordinary AOT; compilation or binding errors are never swallowed.
    """
    from ._component_info import _read_document

    if target not in ("x64", "arm64", "cuda", "vulkan"):
        return frozenset()
    llvm = target != "vulkan"
    metadata = _read_document(path, "aot_metadata.json" if llvm else "metadata.json")
    if llvm and metadata.get("kernel_format") != "bc":
        return frozenset()
    # Old metadata without an explicit task policy cannot prove eligibility.
    if "device_range_dispatch" not in metadata:
        return frozenset()
    excluded = {item["name"] for item in metadata["device_range_dispatch"] if item["tasks"]}
    aliases = {item["key"]: item["value"] for item in metadata.get("kernel_aliases", ())}
    return frozenset(name for name in entries if name not in excluded and aliases.get(name, name) not in excluded)


def _token(value):
    # Exact types distinguish bool/int and +0/-0. Keep the observed objects
    # alive in the guard/call so an identity cannot be recycled underneath it.
    kind = type(value)
    if kind in (type(None), bool, int, str, bytes):
        return kind, value
    if kind is float:
        return kind, struct.pack("=d", value)
    if kind in (tuple, list):
        return kind, tuple(_token(x) for x in value)
    if kind is dict:
        return kind, tuple((_token(k), _token(v)) for k, v in value.items())
    return kind, id(value)


class OwnerGuard:
    """Read only contract dependencies; do not repeat schema interpretation."""

    def __init__(self, entry, selected, owner):
        paths = set()

        def path(value):
            if isinstance(value, (tuple, list)) and all(isinstance(x, str) for x in value):
                paths.add(tuple(value))

        def visit(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in _PATH_KEYS:
                        path(child)
                    else:
                        visit(child)
            elif isinstance(value, (tuple, list)):
                if len(value) == 2 and value[0] == "dim" and isinstance(value[1], str):
                    path(getattr(owner, "_forge_package_dimension_paths", {}).get(value[1], (value[1],)))
                for child in value:
                    visit(child)

        contract = selected.contract
        visit(contract.get("admission", {}))
        for binding in contract.get("admission", {}).get("owner_bindings", {}).values():
            for name in ("_forge_package_contract", "_forge_package_type", "_forge_package_owner"):
                path((*binding["path"], name))
        for spec in contract["captures"]:
            visit(spec)
            if spec["kind"] != "local_tensor_workspace" and not spec.get("dimension_relations"):
                path(spec.get("path", (spec["name"],)))
            if owner is not None:
                from ._package_values import owner_type, path_value

                prefix = tuple(spec.get("layout_owner_path", ()))
                allocated = path_value(owner, prefix)
                if allocated is None:
                    continue
                schema = entry.package.manifest["owners"].get(owner_type(allocated), {})
                selections = getattr(allocated, "_forge_package_member_layouts", {}).get(
                    spec.get("layout_member", spec["name"]), {})
                for name in selections:
                    path((*prefix, *schema.get(name, {}).get("path", (name,))))
        # Selection is a guard too: a previously selected variant cannot hide
        # a changed static option or concrete receiver contract.
        candidates = [entry, selected]
        if entry.family and "owner_inputs" in entry.family:
            from ._package_values import owner_type

            candidates.append(entry._selected[entry.family["owner_inputs"][owner_type(owner)]])
        for candidate in candidates:
            family = candidate.family
            if family:
                for name in family.get("domain", {}):
                    path(family.get("paths", {}).get(name, (name,)))
                for name in candidate._selection_order:
                    path(family.get("paths", {}).get(name, (name,)))
        if owner is not None:
            paths.add(("_forge_package_contract",))
            paths.add(("_forge_package_type",))
            paths.add(("_forge_package_owner",))
        self.readers = tuple(operator.attrgetter(".".join(p)) for p in sorted(paths) if p)
        self.values = self.read(owner)
        self.tokens = tuple(_token(x) for x in self.values)

    def read(self, owner):
        values = []
        for reader in self.readers:
            try:
                values.append(reader(owner))
            except AttributeError:
                # An absent optional record is part of the observed state.
                # If it becomes present its member reads invalidate the guard.
                values.append(_MISSING)
        return tuple(values)

    def matches(self, owner):
        return tuple(_token(x) for x in self.read(owner)) == self.tokens


class CachedCall:
    def __init__(self, guard, values, call, workspaces, *, specialize=False, constant_args=()):
        self.guard, self.values, self.call = guard, values, call
        self.tokens = tuple(_token(x) for x in values)
        self.workspaces = tuple(workspaces)
        self.prepared = None
        self.specialize, self.constant_args = specialize, constant_args

    def matches(self, owner, values):
        return tuple(_token(x) for x in values) == self.tokens and self.guard.matches(owner)

    def run(self, *, reuse=False):
        # Do native preparation when an unchanged binding is actually reused.
        # One-off calls and continually changing scalar inputs keep the ordinary
        # AOT launch path instead of building/retiring a native graph each time.
        if reuse and self.prepared is None:
            self.prepared = self.call.prepare(specialize=self.specialize, constant_args=self.constant_args)
        if self.prepared is None:
            return self.call.run()
        # Existing backend preparation consumes authoritative JIT/AOT views,
        # freezes launch geometry and retains its own argument frame. Shared
        # code never implies shared mutable instance state.
        self.prepared.run()
        if self.prepared._return_layout is not None:
            return self.prepared.result()

    def close(self):
        # Changed bindings are a cold boundary. Retire the previous immutable
        # invocation before dropping its owners or reusing its workspace.
        if self.prepared is not None:
            self.prepared.close()
