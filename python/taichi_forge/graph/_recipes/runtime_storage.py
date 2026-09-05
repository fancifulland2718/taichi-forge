"""Cold, provider-owned storage contributions to a runtime Graph instance."""

from contextlib import ExitStack
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphRuntimeStoragePlan:
    """An allocation owner, not a replay hook or an external-pointer escape hatch.

    The factory returns an owner with ``allocate(dtype, shape)`` and ``close()``.
    Allocations must be ordinary Program-registered ScalarNdarrays. Closing the
    factory must preserve storage retained by bindings or in-flight execution.
    """

    plan_id: str
    binding_names: tuple[str, ...]
    temporary_arena: bool
    factory: object = field(repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.plan_id, str) or not self.plan_id:
            raise ValueError("Graph storage plan requires a stable identity")
        names = tuple(self.binding_names)
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Graph storage bindings must be unique nonempty names")
        if not names and not self.temporary_arena:
            raise ValueError("Graph storage plan must own private bindings or a temporary arena")
        if not callable(self.factory):
            raise TypeError("Graph storage owner factory must be callable")
        object.__setattr__(self, "binding_names", names)


def validate_storage_plans(spec, plans):
    """Resolve ownership once, before creating any allocation or executable."""
    from taichi_forge.graph._graph import _GraphInternalNdarraySpec

    by_name = {}
    identities = set()
    arena_owner = None
    aliases = {}
    for name, value in spec.fixed_runtime_args.items():
        if isinstance(value, _GraphInternalNdarraySpec):
            aliases.setdefault(id(value), set()).add(name)
    private_names = set().union(*aliases.values()) if aliases else set()
    for plan in plans:
        if not isinstance(plan, GraphRuntimeStoragePlan):
            raise TypeError("runtime Graph storage contribution must be a GraphRuntimeStoragePlan")
        if plan.plan_id in identities:
            raise ValueError("runtime Graph storage plan is selected twice")
        identities.add(plan.plan_id)
        selected = set(plan.binding_names)
        if not selected <= private_names:
            raise ValueError("Graph storage plans can allocate only declared private bindings")
        if selected.intersection(by_name):
            raise ValueError("Graph private binding has multiple allocation owners")
        if any(selected.intersection(group) and not group <= selected for group in aliases.values()):
            raise ValueError("Graph private binding aliases must share one allocation owner")
        by_name.update((name, plan) for name in selected)
        if plan.temporary_arena:
            if arena_owner is not None:
                raise ValueError("Graph temporary arena has multiple allocation owners")
            if not spec.temporary_memory_plan.allocations:
                raise ValueError("Graph storage plan selected an absent temporary arena")
            arena_owner = plan


def create_storage_owners(instance, plans):
    """Publish each owner immediately so partial construction can retire it."""
    allocators = {}
    arena_allocator = None
    for plan in plans:
        owner = plan.factory()
        instance._storage_owners += (owner,)
        allocate = owner.allocate
        allocators.update((name, allocate) for name in plan.binding_names)
        if plan.temporary_arena:
            arena_allocator = allocate
    return allocators, arena_allocator


def retire_storage_owners(instance):
    owners, instance._storage_owners = instance._storage_owners, ()
    # Retire every factory even if a provider reports a cleanup failure. The
    # existing allocation leases, not this callback stack, protect GPU uses.
    with ExitStack() as cleanup:
        for owner in owners:
            cleanup.callback(owner.close)


__all__ = ["GraphRuntimeStoragePlan"]
