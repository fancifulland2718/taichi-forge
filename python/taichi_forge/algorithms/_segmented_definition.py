"""Selected segmented math shared by direct calls and Graph reconstruction.

This is a recipe's semantic definition, not another execution IR. The existing
kernel dispatches/native children still own preparation and submission.
"""

from dataclasses import dataclass, field

from ._primitive_definitions import TransformDefinition


@dataclass(frozen=True)
class SegmentedDefinition:
    kind: str
    dtype: object
    method: str
    ndarray_mode: bool
    capacity: int
    num_segments: int
    inclusive: bool = True
    in_place: bool = False
    copy: TransformDefinition | None = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "copy",
            (
                TransformDefinition(self.capacity, self.dtype, 1, 0)
                if self.kind == "scan" and self.method == "global_scan" and not self.in_place
                else None
            ),
        )

    @property
    def serial_kernel(self):
        from taichi_forge import _kernels

        if self.kind == "reduce":
            return _kernels.segmented_reduce_sum_ndarray if self.ndarray_mode else _kernels.segmented_reduce_sum_field
        return (
            _kernels.segmented_scan_sum_serial_ndarray
            if self.ndarray_mode
            else _kernels.segmented_scan_sum_serial_field
        )

    @property
    def correction_kernels(self):
        from taichi_forge import _kernels

        if self.ndarray_mode:
            return _kernels.segmented_scan_gather_bases_ndarray, _kernels.segmented_scan_apply_bases_ndarray
        return _kernels.segmented_scan_gather_bases_field, _kernels.segmented_scan_apply_bases_field
