"""Generated CUDA kernels for complete segmented-scan Graph recipes."""

from taichi_forge.lang import ops, simt
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.lang.misc import loop_config
from taichi_forge.types import ndarray_type
from taichi_forge.types.primitive_types import i32


_SEGMENT_SCAN_KERNELS = {}


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


__all__ = []
