"""Cold reconstruction of already-qualified queue-ordered Graph scratch."""

from taichi_forge.lang.exception import TaichiRuntimeError


def declare_ordered_temporaries(context, graph, allocator):
    from taichi_forge.graph._graph import (
        _GraphOrderedTemporaryArena,
        _GraphTemporaryArena,
    )
    from taichi_forge.graph._recipes.runtime_storage import (
        ordered_temporary_reuse_eligible,
    )

    spec = graph._spec
    arena = graph._instance._temporary_arena
    if not ordered_temporary_reuse_eligible(spec) or type(arena) not in (
        _GraphOrderedTemporaryArena,
        _GraphTemporaryArena,
    ):
        raise TaichiRuntimeError(
            "AOT temporary storage requires a qualified ordered CUDA schedule"
        )
    if len(arena._slots) != arena.capacity:
        raise TaichiRuntimeError(
            "AOT temporary storage requires an eagerly materialized arena"
        )
    # A completion arena can reuse any completed slot. Scratch is not persistent
    # Graph state. Preserve capacity and allocations; the same proven queue
    # order protects native cyclic reuse. Validate every slot's callback mapping
    # at this cold boundary, then serialize only one template. Existing cached
    # bindings are reused, and scratch contents are never read back.
    for slot in arena._slots:
        bindings = slot["bindings"]
        resolved = spec.bind_temporary_args(bindings)
        for action in spec.temporary_actions:
            for symbol, name in action.temporary_bindings.items():
                if (
                    bindings[name].offset != 0
                    or resolved[symbol] is not bindings[name].storage
                ):
                    raise TaichiRuntimeError(
                        "AOT temporary binding must preserve whole ordered storage"
                    )
    bindings = arena._slots[0]["bindings"]
    arrays = []
    for action in spec.temporary_actions:
        for symbol, name in action.temporary_bindings.items():
            owner = bindings[name].storage
            if symbol in spec.fixed_runtime_args or (
                symbol in context.private_bindings
                and context.private_bindings[symbol] is not owner
            ):
                raise TaichiRuntimeError(
                    "AOT temporary symbol conflicts with another private binding"
                )
            context.private_bindings[symbol] = owner
            context._private_names[symbol] = context._resource(
                owner,
                dict(
                    kind="ndarray",
                    dtype=str(owner.dtype),
                    shape=owner.shape,
                    allocator=allocator,
                ),
            )
            arrays.append(context._private_names[symbol])
        context.reconstructed_temporary_actions.add(id(action))
    if type(arena) is _GraphTemporaryArena:
        context.storage_recipe = dict(
            kind="ordered_temporary_ring",
            version=1,
            capacity=arena.capacity,
            arrays=list(dict.fromkeys(arrays)),
        )
