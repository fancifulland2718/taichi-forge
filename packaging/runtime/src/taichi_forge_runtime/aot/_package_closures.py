"""Bind precompiled factory entries to existing allocation provenance."""

from types import SimpleNamespace
import inspect

from .._static_values import read_static_key, static_key
from ._package_values import path_value, owner_type


def _declared_field_path(package, owner, path):
    schema = package.manifest["owners"].get(owner, {})
    for name, item in schema.items():
        prefix = item.get("path", [name])
        if prefix == path:
            return item["kind"] in ("dense_field", "model_tensor_field", "borrowed_field", "fixed_snode_field")
        if item["kind"] in ("owner", "owner_input") and path[: len(prefix)] == prefix and len(prefix) < len(path):
            return _declared_field_path(package, item["owner_type"], path[len(prefix) :])
    return False


def bind_closure(package, identity, values, *, options=None):
    package._check()
    matches, templates = [], []
    options = {} if options is None else options
    option_keys = {name: static_key(value) for name, value in options.items()}
    for candidate in package.manifest.get("closure_entries", {}).get(identity, ()):
        expected_options = {
            name: static_key(read_static_key(value)) for name, value in candidate.get("options", {}).items()
        }
        if option_keys != expected_options:
            continue
        owners, paths = [], []

        def matches_value(value, binding):
            kind, payload = binding
            if kind == "static":
                try:
                    return static_key(value) == static_key(read_static_key(payload))
                except ValueError:
                    return False
            if kind == "sequence":
                return (
                    type(value) in (tuple, list)
                    and len(value) == len(payload)
                    and all(matches_value(item, spec) for item, spec in zip(value, payload))
                )
            if kind == "path":
                source = getattr(value, "_forge_package_field_source", None)
                if source is None:
                    return False
                owner, member = source
                if (
                    getattr(owner, "_forge_package_owner", None) is not package
                    or owner_type(owner) != candidate["owner_type"]
                    or list(member) != payload
                    or path_value(owner, payload) is not value
                ):
                    return False
                owners.append(owner)
                paths.append(payload)
                return True
            raise ValueError("Unsupported cold factory binding")

        bindings = candidate["bindings"]
        if set(values) != set(bindings) or not all(
            matches_value(values[name], spec) for name, spec in bindings.items()
        ):
            continue
        if candidate.get("template_owner"):
            if owners:
                raise ValueError("Template factory cannot implicitly bind lexical Field owners")
            templates.append(candidate)
            continue
        if not owners or any(owner is not owners[0] for owner in owners):
            continue
        # A lexical closure retains the objects present at construction. Later
        # owner attribute replacement must not redirect an existing invocation.
        snapshot = SimpleNamespace(**vars(owners[0]))
        copied = set()
        for path in paths:
            current = snapshot
            for depth, part in enumerate(path[:-1]):
                prefix = tuple(path[: depth + 1])
                if prefix not in copied:
                    setattr(current, part, SimpleNamespace(**vars(getattr(current, part))))
                    copied.add(prefix)
                current = getattr(current, part)
        matches.append((candidate["symbol"], snapshot))
    if templates:
        if matches:
            raise ValueError("Cold factory has ambiguous receiver binding modes")
        return _TemplateEntry(package, templates)
    if not matches or len({symbol for symbol, _ in matches}) != 1:
        raise ValueError("Cold factory inputs have no unambiguous declared source/type binding")
    from ._package import entry

    symbol, owner = matches[0]
    return entry(package, symbol).__get__(owner)


class _TemplateEntry:
    """Share immutable code selection; bind each receiver at the invocation."""

    def __init__(self, package, candidates):
        self.package = package
        by_type = {}
        for candidate in candidates:
            previous = by_type.setdefault(candidate["owner_type"], candidate["symbol"])
            if previous != candidate["symbol"]:
                raise ValueError("Cold template factory has ambiguous code for its receiver")
        from ._package import entry

        self._entries = {identity: entry(package, symbol) for identity, symbol in by_type.items()}

    def __get__(self, owner, cls=None):
        self.package._check()
        if owner is None:
            return self
        return self._bind(owner)

    def _bind(self, owner):
        self.package._check()
        identity = owner_type(owner)
        if getattr(owner, "_forge_package_owner", None) is not self.package or identity not in self._entries:
            raise ValueError("Cold template factory requires its declared receiver in the same package")
        return self._entries[identity].__get__(owner)

    def __call__(descriptor, *args, **kwargs):
        if args:
            if "self" in kwargs:
                raise TypeError("Cold template factory received its receiver twice")
            owner, args = args[0], args[1:]
        elif "self" in kwargs:
            owner = kwargs.pop("self")
        else:
            raise TypeError("Cold template factory requires its explicit self receiver")
        return descriptor._bind(owner)(*args, **kwargs)


class _FunctionEntry:
    """An original free Kernel with an instance-bound template receiver."""

    def __init__(self, package, identity):
        from ._package import entry, _entry_contract

        self.package = package
        spec = package.manifest.get("function_entries", {}).get(identity)
        if not isinstance(spec, dict) or set(spec) != {"receiver", "parameters", "templates", "owners"}:
            raise ValueError("Free Kernel requires a complete receiver contract")
        parameters, receiver, owners = spec["parameters"], spec["receiver"], spec["owners"]
        templates = spec["templates"]
        if (
            not isinstance(parameters, list)
            or not all(isinstance(name, str) and name.isidentifier() for name in parameters)
            or len(set(parameters)) != len(parameters)
            or receiver not in parameters
            or not isinstance(templates, list)
            or not all(isinstance(name, str) and name in parameters for name in templates)
            or len(set(templates)) != len(templates)
            or receiver not in templates
            or not isinstance(owners, dict)
            or not owners
        ):
            raise ValueError("Free Kernel has an invalid receiver signature")
        self.receiver = receiver
        self.templates = tuple(templates)
        self.static_parameters = tuple(name for name in templates if name != receiver)
        self.signature = inspect.Signature(
            [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in parameters]
        )
        self._entries = {}
        self._resource_entries = {}
        for owner, cases in owners.items():
            if owner not in package.manifest["owners"] or not isinstance(cases, list) or not cases:
                raise ValueError("Free Kernel receiver is outside the declared package")
            choices, resource_choices, seen = {}, [], set()
            for case in cases:
                if (
                    not isinstance(case, dict)
                    or set(case) not in ({"symbol", "values"}, {"symbol", "values", "resources"})
                    or not isinstance(case["symbol"], str)
                    or not isinstance(case["values"], dict)
                    or not isinstance(case.get("resources", {}), dict)
                ):
                    raise ValueError("Free Kernel has an invalid template input case")
                resources = case.get("resources", {})
                if set(resources) & case["values"].keys() or set(resources) | case["values"].keys() != set(
                    self.static_parameters
                ):
                    raise ValueError("Free Kernel has an invalid template input case")
                if (
                    resources
                    and package.manifest["version"] < 52
                    or any(
                        not isinstance(path, list)
                        or not path
                        or any(not isinstance(part, str) or not part.isidentifier() for part in path)
                        for path in resources.values()
                    )
                ):
                    raise ValueError("Free Kernel has invalid template resource paths")
                if any(not _declared_field_path(package, owner, path) for path in resources.values()):
                    raise ValueError("Template resource path is outside its declared owner")
                values = {name: static_key(read_static_key(value)) for name, value in case["values"].items()}
                key = tuple(
                    ("path", tuple(resources[name])) if name in resources else ("static", values[name])
                    for name in self.static_parameters
                )
                if key in seen:
                    raise ValueError("Free Kernel has duplicate template input cases")
                seen.add(key)
                bound = entry(package, case["symbol"])
                if [item["name"] for item in bound.contract["arguments"]] != [
                    n for n in parameters if n not in templates
                ]:
                    raise ValueError("Free Kernel receiver binding differs from its numerical ABI")
                if resources:
                    contracts = (
                        [bound.contract]
                        if bound.family is None
                        else [
                            _entry_contract(package.manifest, variant["symbol"]) for variant in bound.family["variants"]
                        ]
                    )
                    for contract in contracts:
                        if any(
                            resources.get(name) != path for name, path in contract.get("template_resources", {}).items()
                        ):
                            raise ValueError("Template resource paths differ from the compiled resource ABI")
                    resource_choices.append((values, resources, bound))
                else:
                    choices[tuple(values[name] for name in self.static_parameters)] = bound
            self._entries[owner] = choices
            self._resource_entries[owner] = resource_choices

    def _bind(self, inputs):
        self.package._check()
        owner = inputs[self.receiver]
        identity = owner_type(owner)
        if getattr(owner, "_forge_package_owner", None) is not self.package or identity not in self._entries:
            raise ValueError("Free Kernel requires its declared receiver in the same package")
        try:
            key = tuple(static_key(inputs[name]) for name in self.static_parameters)
        except ValueError:
            key = None
        selected = self._entries[identity].get(key)
        matches = [] if selected is None else [selected]
        for values, paths, candidate in self._resource_entries[identity]:
            try:
                valid = all(static_key(inputs[name]) == value for name, value in values.items()) and all(
                    path_value(owner, path) is inputs[name] for name, path in paths.items()
                )
            except (AttributeError, ValueError):
                valid = False
            if valid:
                matches.append(candidate)
        if len({candidate.symbol for candidate in matches}) != 1:
            raise ValueError("Free Kernel template inputs are outside their definition-proven domain")
        return matches[0].__get__(owner)

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        inputs = {name: bound.arguments.pop(name) for name in self.templates}
        return self._bind(inputs)(**bound.arguments)

    def _graph_bind(self, args, templates):
        if not isinstance(templates, dict) or set(templates) != set(self.templates):
            raise ValueError("Free Kernel Graph requires its declared template receiver")
        return self._bind(templates), args


class _MethodEntry(_FunctionEntry):
    """Preserve Python method binding without sharing mutable instance pointers."""

    def __init__(self, package, identity):
        super().__init__(package, identity)
        if next(iter(self.signature.parameters)) != self.receiver or package.manifest["version"] < 51:
            raise ValueError("Method template inputs require a leading receiver and package format 51")

    def __get__(self, owner, cls=None):
        self.package._check()
        return self if owner is None else _BoundMethodEntry(self, owner)


class _BoundMethodEntry:
    def __init__(self, entry, owner):
        self.entry, self.owner = entry, owner

    def __call__(self, *args, **kwargs):
        return self.entry(self.owner, *args, **kwargs)

    def _graph_bind(self, args, templates):
        templates = {} if templates is None else dict(templates)
        receiver = self.entry.receiver
        if receiver in templates and templates[receiver] is not self.owner:
            raise ValueError("Method Graph receiver differs from its bound instance")
        templates[receiver] = self.owner
        return self.entry._graph_bind(args, templates)
