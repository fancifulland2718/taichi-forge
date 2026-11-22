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


def specialization_entries(path, target, entries, *, unified=False):
    """Read native execution policy once, outside ordinary call dispatch.

    The native loader validates the same metadata. Unsupported formats/tasks
    keep ordinary AOT; compilation or binding errors are never swallowed.
    """
    from ._component_info import _read_document

    if target not in ("x64", "arm64", "cuda", "vulkan"):
        return frozenset()
    if unified:
        # The shared library retains pre-offload loops. The selected device
        # creates its task dispatch during prepare.
        return frozenset(entries)
    llvm = target != "vulkan"
    metadata = _read_document(path, "aot_metadata.json" if llvm else "metadata.json")
    if llvm and metadata.get("kernel_format") not in ("bc", "bc-library"):
        return frozenset()
    # Old metadata without an explicit task policy cannot prove eligibility.
    if "device_range_dispatch" not in metadata:
        return frozenset()
    excluded = {item["name"] for item in metadata["device_range_dispatch"] if item["tasks"]}
    if llvm:
        aliases = {item["key"]: item["value"] for item in metadata.get("kernel_aliases", ())}
    else:
        aliases = {item["name"]: metadata["kernels"][item["kernel_index"]]["name"]
                   for item in metadata.get("kernel_aliases", ())}
    excluded_resources = {"rw_texture"}
    if target != "vulkan":
        excluded_resources.update(("texture", "acceleration_structure"))
    return frozenset(name for name in entries if name not in excluded and aliases.get(name, name) not in excluded
                     and not any(item["kind"] in excluded_resources
                                 for item in (*entries[name]["arguments"], *entries[name]["captures"])))


def specialization_constant_args(contract, owner_schema, owners=None):
    """Select fixed scalar facts from the existing declaration and entry ABI.

    OwnerGuard observes plan policy and typed optional input presence. An
    explicit Graph.bind snapshots both. Ordinary arguments, unmarked values
    and device contents remain live. The declaration must match the capture's
    path and dtype.
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
        declared_path = tuple(declaration.get("path", (name,)))
        if (capture["kind"] == "value" and capture.get("dtype") == "i32"
                and not capture.get("element_shape") and capture_path == declared_path
                and declaration.get("kind") == "static"
                and declaration.get("dynamic_storage_condition") is True):
            # Optional storage shares producer code across layouts. Its declared
            # presence is fixed for this binding just like its resource handles.
            result.append(index)
        elif (capture["kind"] == declaration.get("kind") == "value"
                and (declaration.get("affects_plan") is True
                     or declaration.get("input_presence") is True)
                and capture.get("dtype") in ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64")
                and capture["dtype"] == declaration.get("dtype")
                and not capture.get("element_shape")
                and not declaration.get("element", {}).get("element_shape")
                and capture_path == declared_path):
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
    from ._package_native import NativeObject
    if isinstance(value, NativeObject):
        return kind, id(value), id(value.resource)
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

        def visit(value, dimensions=None):
            if isinstance(value, dict):
                # Match capture/admission's declared short circuit. An inactive
                # input observes its enabling predicates, not unreachable
                # fields or layout dependencies. A predicate change rebuilds
                # the guard through ordinary admission before using the input.
                predicates = [(item["path"], False) for item in
                              (*value.get("presence_conditions", ()), *value.get("nullable_booleans", ()))]
                if value.get("nullable_record"):
                    predicates.append((value["nullable_record"], True))
                if value.get("nullable_boolean"):
                    predicates.append((value["nullable_boolean"], False))
                for parts, record in predicates:
                    path(parts)
                    actual = owner
                    for part in parts:
                        actual = getattr(actual, part, _MISSING)
                        if actual is _MISSING or actual is None:
                            break
                    if actual is _MISSING or (actual is None if record else not actual):
                        return False
                dimensions = {**(dimensions or {}),
                              **value.get("dimensions", {}),
                              **value.get("dimension_relations", {}),
                              **value.get("model_dimension_relations", {})}
                for key, child in value.items():
                    if key == "owner_bindings":
                        for binding in child.values():
                            if visit(binding, dimensions):
                                for name in ("_forge_package_contract", "_forge_package_type", "_forge_package_owner"):
                                    path((*binding["path"], name))
                    elif key in _PATH_KEYS:
                        path(child)
                    else:
                        visit(child, dimensions)
            elif isinstance(value, (tuple, list)):
                if len(value) == 2 and value[0] == "dim" and isinstance(value[1], str):
                    # Relation symbols are ABI names, not necessarily owner
                    # attributes. Observe the same path capture_value reads;
                    # recursive relation declarations are visited separately.
                    declaration = (dimensions or {}).get(value[1])
                    path(declaration["path"] if declaration else
                         getattr(owner, "_forge_package_dimension_paths", {}).get(value[1], (value[1],)))
                for child in value:
                    visit(child, dimensions)
            return True

        captures = []
        for contract in contracts:
            visit(contract.get("admission", {}))
            captures.extend(contract["captures"])
        for spec in captures:
            if package._manifest.get("frontend_target") in spec.get("absent_devices", ()):
                continue
            if not visit(spec):
                continue
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
                schema = package._manifest["owners"].get(owner_type(allocated), {})
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
        # An absent parent makes every descendant unreachable. Observe the
        # first absent namespace once; if it appears, ordinary admission and
        # binding rebuild the complete dependency snapshot. This is especially
        # relevant to optional nested inputs in a shared generic helper.
        reachable = set()
        for parts in paths:
            value = owner
            for index, part in enumerate(parts):
                try:
                    value = getattr(value, part)
                except AttributeError:
                    value = _MISSING
                if value is None or value is _MISSING:
                    reachable.add(parts[:index + 1])
                    break
            else:
                reachable.add(parts)
        versions = getattr(getattr(package, "runtime", None), "_owner_namespace_versions", None)
        self._bind_paths(owner, reachable, versions)

    def _bind_paths(self, owner, paths, versions):
        stamp = versions(owner) if versions and type(type(owner)) is type and len(paths) > 1 else None
        self._children = []
        if stamp is not None:
            # A write to the parent need not invalidate hundreds of independent
            # reads on its children. Watch a plain instance link on the parent
            # and the suffixes on the child whose namespace actually owns them.
            groups = {}
            for parts in paths:
                if len(parts) > 1:
                    groups.setdefault(parts[0], set()).add(parts[1:])
            for name, suffixes in groups.items():
                if len(suffixes) < 2 or any(name in cls.__dict__ for cls in type(owner).__mro__):
                    continue
                child = getattr(owner, name, _MISSING)
                if child is _MISSING or type(type(child)) is not type or versions(child) is None:
                    continue
                paths.difference_update((name, *suffix) for suffix in suffixes)
                guard = type(self).__new__(type(self))
                guard._bind_paths(child, suffixes, versions)
                self._children.append((name, guard))
        names = [".".join(p) for p in sorted(paths) if p]
        self.readers = tuple(operator.attrgetter(name) for name in names)
        self.values = self.read(owner)
        # Batch present attributes in the C accessor. Missing top-level names
        # use getattr's default instead of raising/catching in Python each step.
        # Keep the individual readers for changed/deleted optional paths.
        present = [name for name, value in zip(names, self.values) if value is not _MISSING]
        missing = [name for name, value in zip(names, self.values) if value is _MISSING]
        ordered = present + missing
        if ordered != names:
            by_name = dict(zip(names, self.values))
            self.readers = tuple(operator.attrgetter(name) for name in ordered)
            self.values = tuple(by_name[name] for name in ordered)
        self._present_reader = operator.attrgetter(*present) if present else None
        self._present_count = len(present)
        self._missing_readers = tuple((name, operator.attrgetter(name)) for name in missing)
        self.tokens = tuple(_token(x) for x in self.values)
        from ._package_native import NativeObject
        self.identity_safe = tuple(type(x) not in (tuple, list, dict) and not isinstance(x, (HostTensor, NativeObject))
                                   for x in self.values)
        self._all_identity_safe = all(self.identity_safe)
        self._namespace_versions = None
        if stamp is not None and (stamp == versions(owner) or self._children):
            # Versioned dictionaries cover plain top-level attributes, including
            # absence. Properties, nested paths and mutable contents have their
            # own semantics and must still be read on every implicit replay.
            classes = type(owner).__mro__
            residual = []
            for index, name in enumerate(ordered):
                value = next((cls.__dict__[name] for cls in classes if name in cls.__dict__), _MISSING)
                if ("." in name or not self.identity_safe[index]
                        or (value is not _MISSING and type(value) not in (bool, int, float, str, bytes, type(None)))):
                    residual.append(index)
            if len(residual) < len(ordered) or self._children:
                self._namespace_versions = versions
                self._namespace_stamp = stamp
                self._namespace_type = type(owner)
                self._residual = [(self.readers[i], self.values[i], self.tokens[i], self.identity_safe[i])
                                  for i in residual]

    def read(self, owner):
        if hasattr(self, "_present_count"):
            try:
                current = self._present_reader(owner) if self._present_count else ()
                if self._present_count == 1:
                    current = (current,)
                if not self._missing_readers:
                    return current
                values = []
                for name, reader in self._missing_readers:
                    if "." not in name:
                        values.append(getattr(owner, name, _MISSING))
                    else:
                        try:
                            values.append(reader(owner))
                        except AttributeError:
                            values.append(_MISSING)
                return current + tuple(values)
            except AttributeError:
                # A formerly present path disappeared. Preserve normal guard
                # invalidation, including optional parents and descriptors.
                pass
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
        if not self._matches_attributes(owner):
            return False
        for name, guard in self._children:
            child = getattr(owner, name, _MISSING)
            if child is _MISSING or child is None or not guard.matches(child):
                return False
        return True

    def _matches_attributes(self, owner):
        # Identity is sufficient for immutable values and resource handles.
        # CPU tensors/containers still need content tokens even when their
        # Python object is unchanged. No device data is inspected here.
        versions = self._namespace_versions
        stamp = versions(owner) if versions is not None and type(owner) is self._namespace_type else None
        if self._children and (stamp is None or stamp[1:] != self._namespace_stamp[1:]):
            # A new descriptor can change lookup of a formerly plain child.
            # Let the next prepare choose the appropriate observation boundary.
            return False
        if stamp is not None and stamp == self._namespace_stamp:
            for reader, previous, token, safe in self._residual:
                try:
                    value = reader(owner)
                except AttributeError:
                    value = _MISSING
                if not (safe and value is previous) and _token(value) != token:
                    return False
            return not self._residual or versions(owner) == stamp
        current = self.read(owner)
        matched = self._all_identity_safe and all(map(operator.is_, current, self.values))
        if not matched:
            matched = all((safe and value is previous) or _token(value) == token
                          for value, previous, token, safe in
                          zip(current, self.values, self.tokens, self.identity_safe))
        if matched and stamp is not None and stamp[1:] == self._namespace_stamp[1:]:
            # Only the instance changed. Class mutation may install a descriptor
            # and requires the full read until the next binding.
            self._namespace_stamp = stamp
        return matched


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

    def matches_values(self, values):
        if not values:
            return not self.tokens
        buffers = []
        return tuple(
            self._encoded_token(self.value_layout.encode(index, value, buffers))
            if index in self.value_layout.encoders else _token(value)
            for index, value in enumerate(values)
        ) == self.tokens

    def bind_values(self, values):
        """Refresh explicit values after the caller checked the owner binding."""
        if not self.specialized_launch:
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
        # The caller has already selected the entry's backend and task contract.
        # Prepare an eligible stable binding before the first device launch.
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
