"""Phase projections of a validated component Plan, without new execution semantics."""

import json

from ._plan import ExecutionPlan


def split_component_lifecycle(plan):
    """Keep model initialization separate from preparation-owned replay scratch.

    Initializers may only reference caller/instance bindings. Replay-only compact
    ndarrays retain the ordinary Plan allocation and retirement contract. Shared
    scratch, native resources, recipes and cross-phase constraints need their own
    lifetime qualification; do not silently reconstruct them for each phase.
    """
    document = json.loads(plan.to_json())
    if not document.get("initialization") or any(document.get(key) for key in (
            "gradients", "constraints", "binding_strategy", "submission_recipe", "storage_recipe")):
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
    return initializer, ExecutionPlan(**replay)
