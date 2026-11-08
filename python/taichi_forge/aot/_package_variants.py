"""Explore only static dependencies observed by the existing kernel compiler."""

from collections import deque

from taichi_forge_runtime._static_values import read_static_key, static_key
from ._package_records import read_member, write_member
from ._component import _DefinitionMissingRecord


def compile_static_variants(environment, entry, arguments, owner, schema, *, template_args=None):
    """Cover a kernel's finite declaration domain without multiplying owners.

    A compiled contract covers every assignment of unread configuration values.
    Neighboring assignments vary only values read by that compilation. When a
    different branch reads another value, that compiler result extends coverage.
    No application constructor, Host recording or numerical execution is used.
    """
    domains = {
        name: tuple(read_static_key(value) for value in item["choices"])
        for name, item in schema.items()
        if item["kind"] == "static"
    }
    # Constants are fixed definition dependencies, not consumer configuration
    # axes. Include their one value in the existing coverage/guard machinery.
    domains.update(
        (name, (read_static_key(item["value"]),)) for name, item in schema.items() if item["kind"] == "constant"
    )
    records = {
        name: read_member(owner, name, schema)
        for name, item in schema.items()
        if item["kind"] == "record" and item.get("optional") and not item.get("dynamic_input_record")
    }
    if not domains and not records:
        return [environment.compile(entry, *arguments, owner=owner, template_args=template_args)]
    initial = {name: values[0] for name, values in domains.items()}
    initial.update((name, True) for name in records)
    previous = {name: read_member(owner, name, schema) for name in domains}
    pending, visited, codes = deque([initial]), set(), []
    try:
        while pending:
            selection = pending.popleft()
            keys = {name: static_key(value) for name, value in selection.items()}
            identity = tuple(sorted(keys.items()))
            if identity in visited:
                continue
            visited.add(identity)
            if any(
                all(keys[name] == value for name, value in code.static.items())
                and all(selection[name] is value for name, value in code.objects.items() if name in records)
                for code in codes
            ):
                continue
            for name, value in selection.items():
                if name not in records:
                    write_member(owner, name, value, schema)
            for name, value in records.items():
                write_member(owner, name, value if selection[name] else None, schema)
            try:
                code = environment.compile(entry, *arguments, owner=owner, template_args=template_args)
            except _DefinitionMissingRecord as missing:
                # Only an explicitly identified absent required record makes a
                # selection inadmissible. Keep exploring dependencies reached
                # on that branch; all other compilation failures still abort.
                code = missing
            else:
                codes.append(code)
            if not code.static.keys() <= domains.keys():
                raise ValueError("Kernel static dependencies are outside the declared package domain")
            # Forget unread values. They are unconstrained by this code, not a
            # second identity for it. A changed read value may expose a new path.
            reads = {*code.static, *(name for name in code.objects if name in records)}
            selected = {**initial, **{name: selection[name] for name in reads}}
            for name in reads:
                for value in domains.get(name, (True, False)):
                    if static_key(value) != keys[name]:
                        pending.append({**selected, name: value})
            # Restore namespaces before selecting any of their scalar leaves.
            for name, value in records.items():
                write_member(owner, name, value, schema)
    finally:
        for name, value in records.items():
            write_member(owner, name, value, schema)
        for name, value in previous.items():
            write_member(owner, name, value, schema)
    return codes
