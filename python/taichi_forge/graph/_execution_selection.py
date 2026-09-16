"""Cold selection of an established whole-Graph execution contract.

This is an explicit construction policy, not performance search or a new
materializer. Provider-owned choices remain private and CompileIQ is unchanged.
"""


def select_execution_recipe(definition, *, queue, binding_reuse):
    from taichi_forge.graph._optimization_api import GraphRecipeHandle
    from taichi_forge.graph._recipes.binding_frames import GraphBindingFrameRecipeProvider
    from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
    from taichi_forge.graph._recipes.providers import GraphRecipeProviderError

    if queue not in ("preserve", "graphics"):
        raise ValueError("execution queue must be 'preserve' or 'graphics'")
    if binding_reuse not in ("require", "prefer"):
        raise ValueError("binding_reuse must be 'require' or 'prefer'")
    provider = GraphBindingFrameRecipeProvider()
    catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(), provider))
    candidates = tuple(
        entry.recipe for entry in catalog.entries() if provider.matches_execution_contract(entry.recipe, queue=queue)
    )
    if len(candidates) > 1:
        raise GraphRecipeProviderError(
            "More than one recipe implements the requested execution contract; use recipe search",
            error_key="execution_contract_ambiguous",
            provider_namespace=provider.descriptor.namespace,
        )
    if candidates:
        recipe = candidates[0]
    elif queue == "preserve" and binding_reuse == "prefer":
        recipe = catalog.baseline.recipe
    else:
        reason = provider.explain_discovery(definition)
        raise GraphRecipeProviderError(
            f"No complete Graph recipe satisfies queue={queue!r}, binding_reuse={binding_reuse!r}. "
            "The graphics queue requirement never falls back to baseline. "
            f"Discovery: {reason or catalog.discovery_report()}",
            error_key="execution_contract_unavailable",
            provider_namespace=provider.descriptor.namespace,
        )
    return GraphRecipeHandle._from_recipe(definition, recipe, catalog.provider_set)
