"""Phase projections of a validated component Plan, without new execution semantics."""

import json

from ._plan import ExecutionPlan
from ._constraints import validate_constraints


def split_component_lifecycle(plan):
    """Keep model initialization separate from preparation-owned replay scratch.

    Initializers may only reference caller/instance bindings. Replay-only compact
    ndarrays retain the ordinary Plan allocation and retirement contract. Shared
    scratch, native resources, recipes and cross-phase constraints need their own
    lifetime qualification; do not silently reconstruct them for each phase.
    """
    document = json.loads(plan.to_json())
    if not document.get("initialization") or any(document.get(key) for key in (
            "gradients", "binding_strategy", "submission_recipe", "storage_recipe")):
        raise ValueError("Component initializer requires qualified instance-owned resources")
    resources = document.get("resources", [])
    allowed = {"kind", "name", "dtype", "shape", "element_shape", "zero_initialize"}
    if any(resource.get("kind") != "ndarray" or set(resource) - allowed
           or resource.get("zero_initialize") is not True for resource in resources):
        raise ValueError("Component initializer private resources require shared-lifetime qualification")
    # The existing Plan validator collects references through nested controls,
    # launch descriptors and native operations. Do not scan kernel source here.
    initializer = ExecutionPlan(document["initialization"])
    private = {resource["name"] for resource in resources}
    shared = private.intersection(initializer.argument_names)
    if shared:
        raise ValueError("Component initializer references Plan-private resources requiring "
                         "shared-lifetime qualification: " + ", ".join(sorted(shared)))
    replay = dict(document)
    replay.pop("initialization")
    replay.pop("version")
    constraints = replay.pop("constraints", [])
    replay_plan = ExecutionPlan(**replay)
    initial_constraints, replay_constraints = [], []
    for constraint in constraints:
        # Collect references through the authoritative validator, including
        # access groups and aliases. A requirement is copied whole, never
        # weakened into per-phase fragments or silently dropped.
        references = validate_constraints([constraint], plan._retained_argument_names)
        in_initial = references <= initializer._retained_argument_names
        in_replay = references <= replay_plan._retained_argument_names
        if not in_initial and not in_replay:
            raise ValueError("Component cross-phase binding constraint requires shared-lifetime qualification")
        if in_initial:
            initial_constraints.append(constraint)
        if in_replay:
            replay_constraints.append(constraint)
    initializer = ExecutionPlan(document["initialization"], constraints=initial_constraints or None)
    replay["constraints"] = replay_constraints or None
    return initializer, ExecutionPlan(**replay)


def split_component_setup(initializer, boundaries):
    """Project cold entry points; Host preparation keeps its normal ordering."""
    document = json.loads(initializer.to_json())
    operations = document["operations"]
    start, names, phases = 0, set(), []
    for item in boundaries:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Invalid component setup boundary")
        name, end = item
        if (not isinstance(name, str) or not name.isidentifier() or name in names
                or type(end) is not int or not start < end < len(operations)):
            raise ValueError("Invalid component setup boundary")
        names.add(name)
        phases.append((name, operations[start:end]))
        start = end
    phases.append((None, operations[start:]))
    plans = [(name, ExecutionPlan(ops)) for name, ops in phases]
    constraints = [[] for _ in plans]
    for requirement in document.get("constraints", []):
        refs = validate_constraints([requirement], initializer._retained_argument_names)
        selected = [i for i, (_, plan) in enumerate(plans) if refs <= plan._retained_argument_names]
        if not selected:
            raise ValueError("Component setup cross-phase constraint requires shared-lifetime qualification")
        for index in selected:
            constraints[index].append(requirement)
    result = {name: ExecutionPlan(ops, constraints=rules or None)
              for (name, ops), rules in zip(phases, constraints)}
    return {name: plan for name, plan in result.items() if name is not None}, result[None]


def split_component_operations(replay, boundaries):
    """Select normal operation regions using existing Plan reference validation.

    State shared across entries must be instance-owned. Private zero scratch
    belongs to exactly one entry; sharing it across separately prepared calls
    would otherwise silently change its lifetime and initialization semantics.
    """
    document = json.loads(replay.to_json())
    if set(document) - {"version", "operations", "constraints", "resources"}:
        raise ValueError("Component operation recipes require shared-lifetime qualification")
    operations = document["operations"]
    start, names, phases = 0, set(), []
    if not isinstance(boundaries, (list, tuple)) or not boundaries:
        raise ValueError("Invalid component operation boundary")
    for item in boundaries:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Invalid component operation boundary")
        name, end = item
        if (not isinstance(name, str) or not name.isidentifier() or name in names
                or type(end) is not int or not start < end < len(operations)):
            raise ValueError("Invalid component operation boundary")
        names.add(name)
        phases.append((name, operations[start:end]))
        start = end
    phases.append((None, operations[start:]))
    raw = [ExecutionPlan(ops) for _, ops in phases]
    resources = [[] for _ in phases]
    for resource in document.get("resources", []):
        selected = [i for i, plan in enumerate(raw) if resource["name"] in plan.argument_names]
        if len(selected) != 1:
            raise ValueError("Component operations cannot share Plan-private resources; use instance-owned state")
        resources[selected[0]].append(resource)
    plans = [ExecutionPlan(ops, resources=private or None)
             for (_, ops), private in zip(phases, resources)]
    constraints = [[] for _ in plans]
    for requirement in document.get("constraints", []):
        refs = validate_constraints([requirement], replay._retained_argument_names)
        selected = [i for i, plan in enumerate(plans) if refs <= plan._retained_argument_names]
        if not selected:
            raise ValueError("Component operation cross-phase constraint requires shared-lifetime qualification")
        for index in selected:
            constraints[index].append(requirement)
    result = {name: ExecutionPlan(ops, resources=private or None, constraints=rules or None)
              for (name, ops), private, rules in zip(phases, resources, constraints)}
    return {name: plan for name, plan in result.items() if name is not None}, result[None]
