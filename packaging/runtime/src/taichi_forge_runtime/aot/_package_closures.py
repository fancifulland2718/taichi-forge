"""Bind precompiled factory entries to existing allocation provenance."""

from types import SimpleNamespace

from .._static_values import read_static_key, static_key
from ._package_values import path_value


def bind_closure(package, identity, values):
    package._check()
    matches = []
    for candidate in package.manifest.get("closure_entries", {}).get(identity, ()):
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
                    or getattr(type(owner), "_forge_package_type", None) != candidate["owner_type"]
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
    if not matches or len({symbol for symbol, _ in matches}) != 1:
        raise ValueError("Cold factory inputs have no unambiguous declared source/type binding")
    from ._package import entry

    symbol, owner = matches[0]
    return entry(package, symbol).__get__(owner)
