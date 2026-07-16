from taichi_forge._lib import core as _ti_core

Layout = _ti_core.Layout
AutodiffMode = _ti_core.AutodiffMode
SNodeGradType = _ti_core.SNodeGradType
Format = _ti_core.Format
BoundaryMode = _ti_core.BoundaryMode


def to_boundary_enum(boundary):
    if boundary == "clamp":
        return BoundaryMode.CLAMP
    if boundary == "unsafe":
        return BoundaryMode.UNSAFE
    raise ValueError(f"Invalid boundary argument: {boundary}")


class DeviceCapability:
    spirv_version_1_3 = "spirv_version=66304"
    spirv_version_1_4 = "spirv_version=66560"
    spirv_version_1_5 = "spirv_version=66816"
    spirv_has_int8 = "spirv_has_int8"
    spirv_has_int16 = "spirv_has_int16"
    spirv_has_int64 = "spirv_has_int64"
    spirv_has_float16 = "spirv_has_float16"
    spirv_has_float64 = "spirv_has_float64"
    spirv_has_atomic_int64 = "spirv_has_atomic_int64"
    spirv_has_atomic_float16 = "spirv_has_atomic_float16"
    spirv_has_atomic_float16_add = "spirv_has_atomic_float16_add"
    spirv_has_atomic_float16_minmax = "spirv_has_atomic_float16_minmax"
    spirv_has_atomic_float = "spirv_has_atomic_float"
    spirv_has_atomic_float_add = "spirv_has_atomic_float_add"
    spirv_has_atomic_float_minmax = "spirv_has_atomic_float_minmax"
    spirv_has_atomic_float64 = "spirv_has_atomic_float64"
    spirv_has_atomic_float64_add = "spirv_has_atomic_float64_add"
    spirv_has_atomic_float64_minmax = "spirv_has_atomic_float64_minmax"
    spirv_has_variable_ptr = "spirv_has_variable_ptr"
    spirv_has_physical_storage_buffer = "spirv_has_physical_storage_buffer"
    spirv_has_subgroup_basic = "spirv_has_subgroup_basic"
    spirv_has_subgroup_vote = "spirv_has_subgroup_vote"
    spirv_has_subgroup_arithmetic = "spirv_has_subgroup_arithmetic"
    spirv_has_subgroup_ballot = "spirv_has_subgroup_ballot"
    spirv_has_non_semantic_info = "spirv_has_non_semantic_info"
    spirv_has_no_integer_wrap_decoration = "spirv_has_no_integer_wrap_decoration"

    @staticmethod
    def cuda_compute_capability(value):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("CUDA compute capability must be a positive integer such as 60, 86, or 90")
        return f"cuda_compute_capability={value}"


__all__ = ["Layout", "AutodiffMode", "SNodeGradType", "Format", "DeviceCapability"]
