"""Freeze a selected complete matmul recipe without producer pointers or code."""

from taichi_forge.hardware._native_adapter import validate_runtime_generation
from taichi_forge.lang.exception import TaichiRuntimeError


def frozen_parameters(recording):
    from taichi_forge.hardware.source_providers._cutlass_matmul import _Recording

    if type(recording) is not _Recording:
        raise TaichiRuntimeError("CUTLASS recording variants require their own AOT contract")
    validate_runtime_generation(recording.library, "AOT CUTLASS export requires a live producer generation")
    manifest = recording.library.manifest
    source = next(x for x in manifest.toolchain["source_dependencies"] if x.name == "cutlass")
    major, minor, patch = (int(x) for x in source.version.split("."))
    contract = manifest.build_profile["driver_contract"]
    semantics = recording.semantics
    if semantics["batch_count"] != 1:
        raise TaichiRuntimeError("AOT CUTLASS requires the selected single-matrix contract")
    return dict(
        **{
            key: semantics[key]
            for key in ("m", "n", "k", "transpose_a", "transpose_b", "alpha", "beta", "activation")
        },
        strategy=recording.strategy,
        workspace_bytes=recording.workspace_bytes,
        cutlass_version=major * 10000 + minor * 100 + patch,
        binary_sha256=manifest.binary_sha256,
        target_code=list(manifest.toolchain["target_code"]),
        minimum_driver_api_version=contract["minimum_api_version"],
        ptx_driver_api_version=contract["ptx_api_version"],
    )


def export_recording(recording, context):
    parameters = frozen_parameters(recording)
    resource = context.resource_once(
        recording,
        lambda: context.resource(recording, "cutlass_source", "plan", parameters=parameters),
        policy=("cutlass_source", "plan"),
    )
    context.native(
        "cutlass_source",
        "execute",
        bindings=dict(plan=resource, **dict(zip(("a", "b", "output"), recording.binding_names[:3]))),
    )
    # The exact typed operation restores this recording's scalar f32 shape and
    # disjoint-output checks, including aliases hidden by imported allocations.
    context.reconstructed_binding_validators.add(id(recording))
    return recording.binding_names[3:]  # Scratch is recreated by the native plan.
