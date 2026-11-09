"""Explore only static dependencies observed by the existing kernel compiler."""

from collections import deque

from taichi_forge_runtime._static_values import read_static_key, static_key
from ._package_records import read_member, write_member
from ._component import _DefinitionMissingResource
from ._package_types import producer_static


def compile_static_variants(environment, entry, arguments, owner, schema, *, template_args=None, fixed=None):
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
    for name, value in (fixed or {}).items():
        if name not in domains or static_key(value) not in tuple(static_key(v) for v in domains[name]):
            raise ValueError("Child implementation guard is outside the declared static domain")
        domains[name] = (value,)
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
    initial_identity = tuple(sorted((name, static_key(value)) for name, value in initial.items()))
    pending, scheduled, codes = deque([initial]), {initial_identity}, []
    try:
        while pending:
            selection = pending.popleft()
            keys = {name: static_key(value) for name, value in selection.items()}
            if any(
                all(keys[name] == value for name, value in code.static.items())
                and all(selection[name] is value for name, value in code.objects.items() if name in records)
                for code in codes
            ):
                continue
            for name, value in selection.items():
                if name not in records:
                    write_member(owner, name, producer_static(value), schema)
            for name, value in records.items():
                write_member(owner, name, value if selection[name] else None, schema)
            try:
                code = environment.compile(entry, *arguments, owner=owner, template_args=template_args)
            except _DefinitionMissingResource as missing:
                # Only an explicitly identified absent required resource makes a
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
                        candidate = {**selected, name: value}
                        identity = tuple(sorted((key, static_key(item)) for key, item in candidate.items()))
                        # Different read branches can discover the same state.
                        # Keep one queued visit rather than retaining duplicate
                        # owner selections until they are eventually popped.
                        if identity not in scheduled:
                            scheduled.add(identity)
                            pending.append(candidate)
            # Restore namespaces before selecting any of their scalar leaves.
            for name, value in records.items():
                write_member(owner, name, value, schema)
    finally:
        for name, value in records.items():
            write_member(owner, name, value, schema)
        for name, value in previous.items():
            write_member(owner, name, value, schema)
    if not codes:
        raise ValueError("Numerical entry has no admissible resource/configuration choice")
    return codes


def compile_owner_variants(graph, identity, entry, arguments, *, template_args=None):
    """Explore child implementations only after an entry actually uses them.

    The parent class/instance and Host constructor remain single definitions.
    These temporary compiler witnesses describe local method/layout choices;
    they never become separately exported parent profiles or mutable state.
    """
    from ._package_owner_graph import owner_bindings
    from types import SimpleNamespace

    pending, visited, emitted, results = deque([{}]), set(), set(), []
    while pending:
        selection = pending.popleft()
        key = tuple(sorted(selection.items()))
        if key in visited:
            continue
        visited.add(key)
        owner, _, environment, schema = graph.context(identity, selection)
        choices = {name: item for name, item in schema.items() if item.get("owner_alternatives")}
        shape_choices = {name: item for name, item in schema.items() if item.get("shape_policy")}
        dtype_choices = {name: item for name, item in schema.items() if item.get("dtype_policy")}
        source_choices = {name: item for name, item in schema.items() if item.get("source_policy")}
        fixed = {}
        for item in (*choices.values(), *shape_choices.values(), *dtype_choices.values(), *source_choices.values()):
            for name, value in {
                **item.get("choice_guards", {}),
                **item.get("shape_guards", {}),
                **item.get("dtype_guards", {}),
                **item.get("source_guards", {}),
            }.items():
                if name in fixed and fixed[name] is not value:
                    raise ValueError("Child implementation guards contradict one another")
                fixed[name] = value
        selected_entry = (
            SimpleNamespace(_primal=entry._primal, _kernel_owner=owner)
            if getattr(entry, "_kernel_owner", None) is not None
            else entry
        )
        codes = compile_static_variants(
            environment,
            selected_entry,
            arguments,
            owner,
            schema,
            template_args=template_args(owner) if template_args else None,
            fixed=fixed,
        )
        for code in codes:
            bindings = owner_bindings(environment, schema, code.used)
            # The existing compiler still decides physical code sharing. This
            # merely avoids emitting the same logical selection twice when a
            # branch did not visit the changed child.
            signature = (
                code.kernel,
                code.key,
                tuple(sorted(code.static.items())),
                tuple(sorted(code.objects.items())),
                tuple(sorted((name, item["owner_type"]) for name, item in bindings.items())),
            )
            if signature not in emitted:
                emitted.add(signature)
                results.append((code, owner, environment, schema))
            for name, policy in (
                *((name, item["shape_policy"]) for name, item in shape_choices.items()),
                *((name, item["dtype_policy"]) for name, item in dtype_choices.items()),
                *((name, item["source_policy"]) for name, item in source_choices.items()),
            ):
                guard = policy["condition"]
                if name in code.used or guard in code.static:
                    for value in (False, True):
                        pending.append({**selection, tuple(environment._paths[guard]): value})
            for name, item in choices.items():
                # A parent method can also read the same configuration without
                # touching the child. Its own static branches remain complete.
                if name not in bindings and not code.static.keys() & item.get("choice_guards", {}).keys():
                    continue
                for alternative in item["owner_alternatives"]:
                    guards = {
                        tuple(environment._paths[flag]): value
                        for flag, value in alternative.get("conditions", {}).items()
                        if flag not in item.get("conditions", {})
                    }
                    pending.append({**selection, **guards})
    return results
