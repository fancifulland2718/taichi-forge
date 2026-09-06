"""Cold-boundary metric declarations; never infer timing semantics from names."""


def _metric_definitions(evaluation_contract, cost_profiles):
    facts = {} if evaluation_contract is None else evaluation_contract.facts
    definitions = facts.get("metric_definitions", {})
    if not isinstance(definitions, dict):
        raise TypeError("Graph evaluation metric_definitions must be a mapping")
    for name, definition in definitions.items():
        if not name or not isinstance(definition, dict):
            raise TypeError("A metric definition needs a name and a mapping")
        for field in ("unit", "scope", "source", "interval"):
            if not isinstance(definition.get(field), str) or not definition[field].strip():
                raise ValueError(f"Metric {name!r} needs an explicit {field}")
    for profile in cost_profiles.values():
        for phase in ("setup", "first", "steady"):
            name = profile.get(phase)
            if name in definitions and definitions[name]["unit"] != profile["unit"]:
                raise ValueError(f"Metric {name!r} has conflicting units in its definition and cost profile")
    return definitions


def _measurement_definitions(metric_names, definitions):
    return {
        name: {
            "status": "declared" if name in definitions else "undeclared",
            "declaration": definitions.get(name),
            "authority": "caller_declared_not_independently_verified",
        }
        for name in sorted(metric_names)
    }


def _measurement_markdown(annotations):
    declarations = {}
    for annotation in annotations:
        declarations.update(annotation.get("measurement", {}).get("metric_definitions", {}))
    if not declarations:
        return []

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    lines = [
        "",
        "## Metric definitions",
        "",
        "Scopes, units, sources and intervals are caller declarations, not inferred from metric names. "
        "A device event interval may contain idle gaps; it is not the sum or union of active kernels. "
        "Host submission, synchronized elapsed time, active device work and memory observations remain separate. "
        "Undeclared semantics do not invalidate a trial, but do not justify a device-speed claim.",
        "",
        "| Metric | Unit | Scope | Source | Interval |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, entry in sorted(declarations.items()):
        declared = entry["declaration"] or {}
        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    name,
                    *(declared.get(field, "undeclared") for field in ("unit", "scope", "source", "interval")),
                )
            )
            + " |"
        )
    return lines
