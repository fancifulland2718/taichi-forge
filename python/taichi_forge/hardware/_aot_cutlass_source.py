"""Freeze a selected complete matmul recipe without producer pointers or code."""

from taichi_forge.hardware._native_adapter import validate_runtime_generation
from taichi_forge.lang.exception import TaichiRuntimeError


def frozen_parameters(recording):
    from taichi_forge.hardware.source_providers._cutlass_matmul import _Recording

    if type(recording) is not _Recording:
        raise TaichiRuntimeError("CUTLASS recording variants require their own AOT contract")
    validate_runtime_generation(recording.library, "AOT CUTLASS export requires a live producer generation")
    return recording._definition.parameter_dict()


def export_recording(recording, context):
    from taichi_forge.hardware.source_providers._cutlass_matmul import _Recording

    if type(recording) is not _Recording:
        raise TaichiRuntimeError("CUTLASS recording variants require their own AOT contract")
    validate_runtime_generation(recording.library, "AOT CUTLASS export requires a live producer generation")
    recording._native_binding().lower(context)
    # The exact typed operation restores this recording's scalar f32 shape and
    # disjoint-output checks, including aliases hidden by imported allocations.
    context.reconstructed_binding_validators.add(id(recording))
    return recording.binding_names[3:]  # Scratch is recreated by the native plan.
