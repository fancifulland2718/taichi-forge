"""Frozen source-addon identity and caller workspace, without producer pointers."""

from taichi_forge.hardware._native_adapter import validate_runtime_generation
from taichi_forge.lang.exception import TaichiRuntimeError


def frozen_parameters(plan, dtype):
    from taichi_forge.hardware._cub_source_provider import CubSourcePlan

    if type(plan) is not CubSourcePlan:
        raise TaichiRuntimeError("CUB plan variants require their own AOT contract")
    validate_runtime_generation(plan.provider, "AOT CUB export requires a live producer generation")
    return plan._resource_definition(dtype).parameter_dict()


def export_segmented(recording, context):
    recipe = recording._cuda_capture_recipe
    plan, bindings = recipe.plan, recipe.bindings
    names, arrays = tuple(bindings), tuple(bindings.values())
    data = recording._aot_heads_data
    if type(data) is not bytes or len(data) != arrays[1].shape[0] * 4:
        raise TaichiRuntimeError("AOT CUB segmented scan requires its frozen topology payload")
    from taichi_forge.hardware._cub_source_provider import CubSourcePlan

    if type(plan) is not CubSourcePlan:
        raise TaichiRuntimeError("CUB plan variants require their own AOT contract")
    validate_runtime_generation(plan.provider, "AOT CUB export requires a live producer generation")
    heads = context.data_resource(arrays[1], data, dtype="u32", shape=arrays[1].shape, readonly=True)
    plan._native_binding(arrays[0].dtype, (names[0], heads, names[2])).lower(context)
    # Heads are immutable topology, scratch belongs to the reconstructed plan.
    return names[1], names[3]
