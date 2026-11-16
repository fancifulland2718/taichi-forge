"""Reuse ordinary definition-call binding while its observed inputs stay fixed.

This caches native prepared execution, not device results. Resource/owner changes
re-enter admission; live values keep the binding and snapshot their current bytes.
No module-global resource pointer is changed to select an instance.
"""

import ctypes as C
import operator
import struct

from ._package_values import HostTensor

_MISSING = object()
_PATH_KEYS = frozenset(
    ("path", "source_path", "reference_path", "flag_path", "nullable_record", "nullable_boolean", "layout_owner_path")
)


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
    if llvm and metadata.get("kernel_format") not in ("bc", "bc-library"):
        return frozenset()
    # Old metadata without an explicit task policy cannot prove eligibility.
    if "device_range_dispatch" not in metadata:
        return frozenset()
    excluded = {item["name"] for item in metadata["device_range_dispatch"] if item["tasks"]}
    aliases = {item["key"]: item["value"] for item in metadata.get("kernel_aliases", ())}
    # Sampled images retain the existing managed-resource kernel route. Package
    # automatic specialization currently admits ordinary value/array bindings.
    return frozenset(name for name in entries if name not in excluded and aliases.get(name, name) not in excluded
                     and not any(item["kind"] == "texture" for item in entries[name]["arguments"]))


def specialization_constant_args(contract, owner_schema, owners=None):
    """Select fixed scalar facts from the existing declaration and entry ABI.

    Plan-affecting integer captures are model policy, observed by OwnerGuard
    (or snapshotted by Graph.bind). Ordinary arguments, unmarked values and
    device contents remain live. A declaration with a different path or dtype
    cannot authorize specialization of a similarly named capture.
    """
    result = []
    for index, capture in enumerate(contract["captures"], len(contract["arguments"])):
        if capture["kind"] == "dimension" or "prepare_static" in capture:
            result.append(index)
            continue
        name = capture["name"]
        declaration = owner_schema.get(name, {})
        capture_path = tuple(capture.get("path", (name,)))
        if not declaration and owners is not None:
            schema, remaining = owner_schema, capture_path
            while remaining:
                match = next(((key, item) for key, item in schema.items()
                              if tuple(item.get("path", (key,))) == remaining), None)
                if match is not None:
                    key, declaration = match
                    break
                parent = next(((tuple(item.get("path", (key,))), item) for key, item in schema.items()
                               if item["kind"] in ("owner", "owner_input")
                               and remaining[:len(item.get("path", (key,)))] == tuple(item.get("path", (key,)))
                               and 0 < len(item.get("path", (key,))) < len(remaining)), None)
                if parent is None:
                    break
                prefix, item = parent
                identity = item["owner_type"]
                schema, remaining = owners.get(identity, {}), remaining[len(prefix):]
            if declaration:
                # The input path is proven by the package's owner contracts;
                # qualify only this local view, never mutate the shared schema.
                declaration = {**declaration, "path": capture_path}
        if (capture["kind"] == declaration.get("kind") == "value"
                and declaration.get("affects_plan") is True
                and capture.get("dtype") in ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64")
                and capture["dtype"] == declaration.get("dtype")
                and not capture.get("element_shape")
                and not declaration.get("element", {}).get("element_shape")
                and capture_path == tuple(declaration.get("path", (name,)))):
            result.append(index)
    return tuple(result)


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
    if isinstance(value, HostTensor):
        # CPU value storage only. Device arrays retain their identity token;
        # changing their contents must neither read them back nor reprepare.
        array = value.entries
        return kind, array.dtype.str, array.shape, array.tobytes()
    return kind, id(value)


def reusable_value(value):
    """Values whose complete mutable contents are represented by _token."""
    if type(value) in (bool, int, float) or isinstance(value, HostTensor):
        return True
    if type(value) in (tuple, list):
        return all(reusable_value(item) for item in value)
    if type(value) is dict:
        return all(reusable_value(item) for item in value.values())
    return False


class OwnerGuard:
    """Read only contract dependencies; do not repeat schema interpretation."""

    def __init__(self, entry, selected, owner):
        candidates = [entry, selected]
        if entry.family and "owner_inputs" in entry.family:
            from ._package_values import owner_type

            candidates.append(entry._selected[entry.family["owner_inputs"][owner_type(owner)]])
        self._initialize(entry.package, owner, (selected.contract,), candidates)

    @classmethod
    def for_graph(cls, package, owner, contracts):
        guard = cls.__new__(cls)
        guard._initialize(package, owner, contracts, ())
        return guard

    def _initialize(self, package, owner, contracts, candidates):
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

        captures = []
        for contract in contracts:
            visit(contract.get("admission", {}))
            for binding in contract.get("admission", {}).get("owner_bindings", {}).values():
                for name in ("_forge_package_contract", "_forge_package_type", "_forge_package_owner"):
                    path((*binding["path"], name))
            captures.extend(contract["captures"])
        for spec in captures:
            visit(spec)
            if spec["kind"] != "local_tensor_workspace" and not spec.get("dimension_relations"):
                path(spec.get("path", (spec["name"],)))
            if owner is not None:
                from ._package_values import capture_is_absent, owner_type, path_value

                if capture_is_absent(owner, spec):
                    continue
                prefix = tuple(spec.get("layout_owner_path", ()))
                allocated = path_value(owner, prefix)
                if allocated is None:
                    continue
                schema = package.manifest["owners"].get(owner_type(allocated), {})
                selections = getattr(allocated, "_forge_package_member_layouts", {}).get(
                    spec.get("layout_member", spec["name"]), {})
                for name in selections:
                    path((*prefix, *(name if isinstance(name, tuple) else schema.get(name, {}).get("path", (name,)))))
        # Selection is a guard too: a previously selected variant cannot hide
        # a changed static option or concrete receiver contract.
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
    def __init__(self, guard, values, call, workspaces, *, arguments, argument_specs,
                 specialize=False, constant_args=(), specialized_launch=False):
        self.guard, self.values, self.call = guard, values, call
        self.value_layout = call._entry._argument_layout
        # Compare by-value tensors/records through their compiler-owned ABI,
        # never Python identity. The tokens describe the actual bound snapshot.
        self.tokens = tuple(
            self._encoded_token(call._arguments[index])
            if index in self.value_layout.encoders else _token(value)
            for index, value in enumerate(values)
        )
        self.workspaces = tuple(workspaces)
        self.prepared = None
        self.specialize, self.constant_args = specialize, constant_args
        self.specialized_launch = specialized_launch
        self.arguments, self.argument_specs = tuple(arguments), tuple(argument_specs)

    @staticmethod
    def _encoded_token(argument):
        value = argument.value.value
        return C.string_at(value.data, value.size)

    def matches(self, owner, values):
        if not self.guard.matches(owner):
            return False
        buffers = []
        return tuple(
            self._encoded_token(self.value_layout.encode(index, value, buffers))
            if index in self.value_layout.encoders else _token(value)
            for index, value in enumerate(values)
        ) == self.tokens

    def bind_values(self, owner, values):
        """Refresh explicit values when every resource and owner fact is unchanged."""
        if not self.specialized_launch or not self.guard.matches(owner):
            return None
        from ._runtime import Scalar

        arguments = list(self.arguments)
        for index, (spec, value) in enumerate(zip(self.argument_specs, values)):
            if spec["kind"] == "value":
                if type(value) not in (bool, int, float):
                    return None
                arguments[index] = Scalar(spec["dtype"], value)
            elif spec["kind"] == "compound":
                arguments[index] = value
            elif _token(value) != self.tokens[index]:
                return None
        # The common ABI binder snapshots/validates all values before replacing
        # the cached invocation. Fixed dimensions, workspaces, optional presence,
        # resource layouts and selection are already covered by OwnerGuard.
        call = self.call._entry.bind(*arguments)
        return CachedCall(self.guard, values, call, self.workspaces,
                          specialize=self.specialize, constant_args=self.constant_args,
                          specialized_launch=True, arguments=arguments,
                          argument_specs=self.argument_specs)

    def run(self, *, reuse=False):
        # Eligible LLVM packages prepare on their first call, before any generic
        # backend compilation. Other formats prepare on binding reuse. Neither
        # path rebuilds a descriptor while its captured inputs remain unchanged.
        if reuse and self.prepared is None:
            self.prepared = self.call.prepare(specialize=self.specialize, constant_args=self.constant_args)
        if self.prepared is None:
            if self.specialized_launch:
                return self.call._run_specialized(self.constant_args)
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
