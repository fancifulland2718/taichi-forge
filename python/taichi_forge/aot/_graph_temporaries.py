"""Cold reconstruction of already-qualified queue-ordered Graph scratch."""

from taichi_forge.lang.exception import TaichiRuntimeError


def declare_ordered_temporaries(context, graph, allocator):
    from taichi_forge.graph._graph import _GraphOrderedTemporaryArena
    from taichi_forge.graph._recipes.runtime_storage import ordered_temporary_reuse_eligible

    spec = graph._spec
    arena = graph._instance._temporary_arena
    if not ordered_temporary_reuse_eligible(spec) or type(arena) is not _GraphOrderedTemporaryArena:
        raise TaichiRuntimeError("AOT temporary storage requires the selected ordered CUDA arena")
    # Materialization already resolves and checks each callback against these
    # whole buffers. This cold cache lookup reuses its result, never serializes
    # callbacks, and never reads scratch contents back to the host.
    bindings = arena._lease.bindings
    resolved = spec.bind_temporary_args(bindings)
    for action in spec.temporary_actions:
        for symbol, name in action.temporary_bindings.items():
            binding = bindings[name]
            owner = binding.storage
            if binding.offset != 0 or resolved[symbol] is not owner:
                raise TaichiRuntimeError("AOT temporary binding must preserve whole ordered storage")
            if symbol in spec.fixed_runtime_args or (
                symbol in context.private_bindings and context.private_bindings[symbol] is not owner
            ):
                raise TaichiRuntimeError("AOT temporary symbol conflicts with another private binding")
            context.private_bindings[symbol] = owner
            context._private_names[symbol] = context._resource(
                owner,
                dict(kind="ndarray", dtype=str(owner.dtype), shape=owner.shape, allocator=allocator),
            )
        context.reconstructed_temporary_actions.add(id(action))
