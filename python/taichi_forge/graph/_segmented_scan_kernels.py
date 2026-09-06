"""Generated CUDA kernels for complete segmented-scan Graph recipes."""

from taichi_forge.lang import impl, ops, simt
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.lang.misc import loop_config
from taichi_forge.types import ndarray_type
from taichi_forge.types.primitive_types import i32, u32


_SEGMENT_SCAN_KERNELS = {}
_GLOBAL_CORRECTION_KERNELS = {}
_SHUFFLE_SCAN_KERNELS = {}


def generated_segment_shuffle_kernel(dtype, block_dim, segment_count):
    """Register prefix scan, with shared storage only between complete warps.

    A whole block owns one segment. Even the final partial chunk executes all
    shuffle lanes with zero padding; no inactive lane is read. Two barriers
    protect cross-warp totals when present. All arithmetic is modulo 2**32,
    including the signed storage route, rather than relying on signed overflow.
    """
    if dtype not in (i32, u32) or block_dim not in (32, 128):
        raise ValueError("segmented shuffle requires i32/u32 and a complete warp block")
    key = (dtype, int(block_dim), int(segment_count))
    cached = _SHUFFLE_SCAN_KERNELS.get(key)
    if cached is not None:
        return cached
    worker_count = int(block_dim) * int(segment_count)
    warp_count = int(block_dim) // 32

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
            warp_lane = lane % 32
            warp_index = lane // 32
            segment = worker // block_dim
            begin = offsets[segment]
            end = offsets[segment + 1]
            carry = ops.cast(0, u32)
            chunk = begin
            # The unused single-warp declaration is eliminated during lowering.
            totals = simt.block.SharedArray((warp_count,), u32)
            while chunk < end:
                index = chunk + lane
                value = ops.cast(0, u32)
                if index < end:
                    value = ops.cast(values[index], u32)
                prefix = value
                for shift in impl.static((1, 2, 4, 8, 16)):
                    addend = ops.bit_cast(
                        simt.warp.shfl_up_i32(ops.cast(-1, u32), ops.bit_cast(prefix, i32), shift), u32
                    )
                    if warp_lane >= shift:
                        prefix += addend
                prior_warps = ops.cast(0, u32)
                total = ops.cast(0, u32)
                if impl.static(warp_count == 1):
                    total = ops.bit_cast(simt.warp.shfl_sync_i32(ops.cast(-1, u32), ops.bit_cast(prefix, i32), 31), u32)
                else:
                    if warp_lane == 31:
                        totals[warp_index] = prefix
                    simt.block.sync()
                    for previous in impl.static(range(warp_count)):
                        warp_total = totals[previous]
                        total += warp_total
                        if previous < warp_index:
                            prior_warps += warp_total
                if index < end:
                    result = carry + prior_warps + prefix
                    if inclusive == 0:
                        result -= value
                    output[index] = ops.cast(result, dtype)
                carry += total
                if impl.static(warp_count > 1):
                    # No warp may overwrite totals for the next chunk before
                    # the last reader of this chunk has consumed them.
                    simt.block.sync()
                chunk += block_dim

    _SHUFFLE_SCAN_KERNELS[key] = scan_kernel
    return scan_kernel


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
