"""Generated CUDA kernels for complete segmented-scan Graph recipes."""

from taichi_forge.lang import ops, simt
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.lang.misc import loop_config
from taichi_forge.types import ndarray_type
from taichi_forge.types.primitive_types import i32


_SEGMENT_SCAN_KERNELS = {}
_GLOBAL_CORRECTION_KERNELS = {}


def generated_segment_chunk_kernel(dtype, block_dim, segment_count, *, indexed):
    """Build one fixed-topology segmented scan with device-only chunk carry."""

    key = (dtype, int(block_dim), int(segment_count), bool(indexed))
    cached = _SEGMENT_SCAN_KERNELS.get(key)
    if cached is not None:
        return cached
    worker_count = int(block_dim) * int(segment_count)

    if indexed:

        @kernel
        def scan_kernel(
            values: ndarray_type.ndarray(dtype=dtype, ndim=1),
            offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
            segment_indices: ndarray_type.ndarray(dtype=i32, ndim=1),
            output: ndarray_type.ndarray(dtype=dtype, ndim=1),
            inclusive: i32,
        ):
            loop_config(block_dim=block_dim)
            for worker in range(worker_count):
                lane = worker % block_dim
                segment = segment_indices[worker // block_dim]
                begin = offsets[segment]
                end = offsets[segment + 1]
                carry = ops.cast(0, dtype)
                chunk = begin
                pad = simt.block.SharedArray((block_dim,), dtype)
                while chunk < end:
                    index = chunk + lane
                    value = ops.cast(0, dtype)
                    if index < end:
                        value = values[index]
                    pad[lane] = value
                    simt.block.sync()
                    stride = 1
                    while stride < block_dim:
                        addend = ops.cast(0, dtype)
                        if lane >= stride:
                            addend = pad[lane - stride]
                        simt.block.sync()
                        pad[lane] += addend
                        simt.block.sync()
                        stride *= 2
                    if index < end:
                        if inclusive != 0:
                            output[index] = carry + pad[lane]
                        else:
                            output[index] = carry + pad[lane] - value
                    carry += pad[block_dim - 1]
                    simt.block.sync()
                    chunk += block_dim

    else:

        @kernel
        def scan_kernel(
            values: ndarray_type.ndarray(dtype=dtype, ndim=1),
            offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
            output: ndarray_type.ndarray(dtype=dtype, ndim=1),
            inclusive: i32,
        ):
            loop_config(block_dim=block_dim)
            for worker in range(worker_count):
                lane = worker % block_dim
                segment = worker // block_dim
                begin = offsets[segment]
                end = offsets[segment + 1]
                carry = ops.cast(0, dtype)
                chunk = begin
                pad = simt.block.SharedArray((block_dim,), dtype)
                while chunk < end:
                    index = chunk + lane
                    value = ops.cast(0, dtype)
                    if index < end:
                        value = values[index]
                    pad[lane] = value
                    simt.block.sync()
                    stride = 1
                    while stride < block_dim:
                        addend = ops.cast(0, dtype)
                        if lane >= stride:
                            addend = pad[lane - stride]
                        simt.block.sync()
                        pad[lane] += addend
                        simt.block.sync()
                        stride *= 2
                    if index < end:
                        if inclusive != 0:
                            output[index] = carry + pad[lane]
                        else:
                            output[index] = carry + pad[lane] - value
                    carry += pad[block_dim - 1]
                    simt.block.sync()
                    chunk += block_dim

    _SEGMENT_SCAN_KERNELS[key] = scan_kernel
    return scan_kernel


def generated_global_correction_kernels(dtype, block_dim, segment_count, *, inclusive):
    """Build a fixed base snapshot and block-parallel segment correction.

    The snapshot is a separate kernel because the last value of one segment is
    the base of the next.  Keeping a grid-wide boundary between snapshot and
    correction avoids a cross-block read/write race without requiring a
    replay-time host synchronization.
    """

    key = (dtype, int(block_dim), int(segment_count), bool(inclusive))
    cached = _GLOBAL_CORRECTION_KERNELS.get(key)
    if cached is not None:
        return cached
    worker_count = int(block_dim) * int(segment_count)

    @kernel
    def gather_bases(
        scanned: ndarray_type.ndarray(dtype=dtype, ndim=1),
        offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
        bases: ndarray_type.ndarray(dtype=dtype, ndim=1),
    ):
        for segment in range(segment_count):
            begin = offsets[segment]
            base = ops.cast(0, dtype)
            if begin > 0:
                base = scanned[begin - 1]
            bases[segment] = base

    @kernel
    def apply_correction(
        values: ndarray_type.ndarray(dtype=dtype, ndim=1),
        scanned: ndarray_type.ndarray(dtype=dtype, ndim=1),
        offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
        bases: ndarray_type.ndarray(dtype=dtype, ndim=1),
    ):
        loop_config(block_dim=block_dim)
        for worker in range(worker_count):
            lane = worker % block_dim
            segment = worker // block_dim
            begin = offsets[segment]
            end = offsets[segment + 1]
            base = bases[segment]
            index = begin + lane
            while index < end:
                if inclusive:
                    scanned[index] -= base
                else:
                    scanned[index] = scanned[index] - base - values[index]
                index += block_dim

    result = (gather_bases, apply_correction)
    _GLOBAL_CORRECTION_KERNELS[key] = result
    return result


__all__ = []
