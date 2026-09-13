"""Cold ray plan identity; live aliases are not serialized resource addresses."""

import hashlib
import json
from dataclasses import dataclass
from functools import partial


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, eq=False)
class RayResourceIdentity:
    """Identity-equal alias token using Graph's existing static-resource slots.

    The JSON describes allocation-independent plans and immutable imported assets,
    not geometry contents or mutable transforms. Those remain caller workload
    data. Separate owners always have separate alias tokens, even for equal plans.
    """

    _plan_json: str

    def __init__(self, kind, *, children=(), **facts):
        slots = {}
        topology = []
        plans = []
        for child in children:
            if child not in slots:
                slots[child] = len(plans)
                plans.append(json.loads(child._plan_json))
            topology.append(slots[child])
        object.__setattr__(
            self,
            "_plan_json",
            _canonical(dict(kind=kind, facts=facts, children=plans, topology=topology)),
        )


def identify_ray_recording(recording, operation, resource, **parameters):
    """Defer hashing to Graph compilation, not direct trace/refit invocation."""
    object.__setattr__(
        recording,
        "_graph_identity_factory",
        partial(
            _recording_identity,
            operation,
            resource,
            recording.binding_names,
            parameters,
        ),
    )


def _recording_identity(operation, resource, bindings, parameters):
    semantics = dict(operation=operation, bindings=bindings, **parameters)
    semantic_json = _canonical(semantics)
    # Asset applicability is explicitly separated from live allocation identity.
    # It participates in the frozen plan digest so replacing baked input cannot
    # silently reuse measurements of another asset. No runtime snapshot is read.
    physical_json = _canonical(
        dict(semantics=semantics, resource_plan=json.loads(resource._plan_json))
    )
    return (
        hashlib.sha256(semantic_json.encode()).hexdigest(),
        "ray-plan:" + hashlib.sha256(physical_json.encode()).hexdigest(),
    )
