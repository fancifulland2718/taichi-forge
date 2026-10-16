"""Frozen source-addon identity and caller workspace, without producer pointers."""

from taichi_forge.hardware._native_adapter import validate_runtime_generation
from taichi_forge.lang.exception import TaichiRuntimeError


def frozen_parameters(plan, dtype):
    from taichi_forge.hardware._cub_source_provider import CubSourcePlan

    if type(plan) is not CubSourcePlan:
        raise TaichiRuntimeError("CUB plan variants require their own AOT contract")
    validate_runtime_generation(plan.provider, "AOT CUB export requires a live producer generation")
    manifest = plan.provider.manifest
    info = plan.provider._library.info
    contract = {} if manifest.build_profile is None else manifest.build_profile["driver_contract"]
    return dict(
        operation=plan.operation,
        num_items=plan.num_items,
        dtype=str(dtype),
        workspace_bytes=plan.workspace_bytes,
        cub_version=int(info.cub_version),
        cuda_runtime_version=int(info.cuda_runtime_version),
        binary_sha256=manifest.binary_sha256,
        target_code=list(manifest.toolchain["target_code"]),
        minimum_driver_api_version=contract.get("minimum_api_version", 0),
        ptx_driver_api_version=contract.get("ptx_api_version", 0),
    )


def export_segmented(recording, context):
    recipe = recording._cuda_capture_recipe
    plan, bindings = recipe.plan, recipe.bindings
    names, arrays = tuple(bindings), tuple(bindings.values())
    data = recording._aot_heads_data
    if type(data) is not bytes or len(data) != arrays[1].shape[0] * 4:
        raise TaichiRuntimeError("AOT CUB segmented scan requires its frozen topology payload")
    parameters = frozen_parameters(plan, arrays[0].dtype)
    resource = context.resource_once(
        plan,
        lambda: context.resource(plan, "cub_source", "plan", parameters=parameters),
        policy=("cub_source", "plan", parameters["dtype"]),
    )
    heads = context.data_resource(arrays[1], data, dtype="u32", shape=arrays[1].shape, readonly=True)
    context.native(
        "cub_source", "execute", bindings=dict(plan=resource, input0=names[0], input1=heads, output0=names[2])
    )
    # Heads are immutable topology, scratch belongs to the reconstructed plan.
    return names[1], names[3]
