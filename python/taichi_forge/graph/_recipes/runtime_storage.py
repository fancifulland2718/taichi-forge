"""Cold, provider-owned storage contributions to a runtime Graph instance."""

from contextlib import ExitStack
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class GraphStoragePoolReport:
    """Explicit cold-boundary measurements, separate from requested storage."""

    allocator: str
    allocation_members: tuple[str, ...]
    allocation_count: int
    requested_bytes: int
    used_current_bytes: int | None
    reserved_current_bytes: int | None
    used_high_bytes: int | None
    reserved_high_bytes: int | None
    release_threshold_bytes: int | None
    closed: bool
    instance_index: int = 0


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
    temporary_capacity: int | None = None

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
        if self.temporary_capacity is not None and (
            not self.temporary_arena
            or isinstance(self.temporary_capacity, bool)
            or not isinstance(self.temporary_capacity, int)
            or not 1 <= self.temporary_capacity <= 64
        ):
            raise ValueError("Graph storage temporary capacity requires an arena and a bounded slot count")
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
    arena_capacity = None
    for plan in plans:
        owner = plan.factory()
        instance._storage_owners += (owner,)
        allocate = owner.allocate
        allocators.update((name, allocate) for name in plan.binding_names)
        if plan.temporary_arena:
            arena_allocator = allocate
            arena_capacity = plan.temporary_capacity
    return allocators, arena_allocator, arena_capacity


def storage_pool_reports(instances):
    """Only called by explicit execution_stats, never by replay or acquisition."""
    return tuple(
        replace(observe(), instance_index=index)
        for index, instance in enumerate(instances)
        for owner in instance._storage_owners
        if (observe := getattr(owner, "storage_pool_report", None)) is not None
    )


def retire_storage_owners(instance):
    owners, instance._storage_owners = instance._storage_owners, ()
    # Retire every factory even if a provider reports a cleanup failure. The
    # existing allocation leases, not this callback stack, protect GPU uses.
    with ExitStack() as cleanup:
        for owner in owners:
            cleanup.callback(owner.close)


__all__ = ["GraphRuntimeStoragePlan"]
