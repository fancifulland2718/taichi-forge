import numpy as np

from taichi_forge._kernels import (
    blit_from_field_to_field,
    bucket_copy_offsets_to_cursor_field,
    bucket_copy_offsets_to_cursor_ndarray,
    bucket_count_i32_field,
    bucket_count_i32_ndarray,
    bucket_prefix_offsets_i32_field_serial,
    bucket_scatter_i32_field,
    bucket_scatter_i32_ndarray,
    compact_flags_to_prefix_field,
    compact_flags_to_prefix_ndarray_from_field,
    compact_scatter_field_from_prefix_ndarray,
    compact_scatter_field,
    compact_single_item_field,
    gather_f32_field,
    gather_f32_ndarray,
    gather_i32_field,
    gather_i32_ndarray,
    grouped_reduce_sum_i32_field,
    grouped_reduce_sum_i32_ndarray,
    histogram_i32_field_direct,
    histogram_i32_field_private_count,
    histogram_i32_field_private_reduce,
    reduce_f32_field,
    reduce_f32_field_private_count,
    reduce_f32_field_private_reduce,
    reduce_i32_field,
    reduce_i32_field_private_count,
    reduce_i32_field_private_reduce,
    scan_add_inclusive,
    scan_add_inclusive_cuda,
    scan_add_inclusive_ndarray,
    sort_stage,
    sort_copy_key_buffer_to_field_u32,
    sort_copy_key_buffer_to_ndarray_u32,
    sort_copy_value_buffer_to_field,
    sort_copy_value_buffer_to_i32_ndarray,
    sort_init_key_buffer_u32,
    sort_init_key_buffer_u32_ndarray,
    sort_init_value_buffer,
    sort_init_value_buffer_i32_ndarray,
    sort_radix_count_zero_bits_u32,
    sort_radix_count_zero_bits_u32_ndarray,
    sort_radix_scatter_keys_u32_ndarray,
    sort_radix_scatter_u32,
    sort_radix_scatter_u32_i32_ndarray,
    sort_radix_store_zero_count,
    sort_radix_store_zero_count_ndarray,
    scatter_f32_field,
    scatter_f32_ndarray,
    scatter_add_f32_field,
    scatter_add_f32_ndarray,
    scatter_add_i32_field,
    scatter_add_i32_ndarray,
    scatter_i32_field,
    scatter_i32_ndarray,
    transform_affine_f32_field,
    transform_affine_f32_ndarray,
    transform_affine_i32_field,
    transform_affine_i32_ndarray,
    uniform_add,
    uniform_add_cuda,
    uniform_add_ndarray,
    warp_shfl_up_i32,
)
from taichi_forge.lang.impl import current_cfg, field, ndarray as ti_ndarray
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.kernel_impl import data_oriented
from taichi_forge.lang.misc import arm64, cuda, vulkan, x64
from taichi_forge.lang.runtime_ops import sync
from taichi_forge.lang.simt import subgroup
from taichi_forge.types.primitive_types import f32, f64, i32, i64, u32, u64

_CUDA_CUB_SORT_METHODS = {"cuda_cub_native", "cuda_cub_split32", "cuda_cub_u32"}
_SUPPORTED_SORT_METHODS = {
    "auto",
    "host_stable",
    "legacy",
    "radix_u32",
    "vulkan_graph_radix_u32",
    "vulkan_native_radix_u32",
    "vulkan_radix_u32",
    *_CUDA_CUB_SORT_METHODS,
}
_SUPPORTED_SORT_PRECISIONS = {"exact"}
_SUPPORTED_NAN_POLICIES = {"last", "bitwise"}
_SUPPORTED_REDUCE_METHODS = {
    "auto",
    "cuda_cub",
    "vulkan_native",
    "cpu_native",
    "field_atomic",
}
_SUPPORTED_REDUCE_OPS = {"sum": 0, "min": 1, "max": 2}
_SUPPORTED_TRANSFORM_METHODS = {
    "auto",
    "cuda_device",
    "vulkan_native",
    "cpu_native",
    "kernel",
    "field_kernel",
}
_SUPPORTED_INDEXED_COPY_METHODS = {
    "auto",
    "cuda_device",
    "vulkan_native",
    "cpu_native",
    "kernel",
    "field_kernel",
}
_SUPPORTED_SCATTER_ADD_METHODS = {
    "auto",
    "cuda_device",
    "vulkan_native",
    "cpu_native",
    "kernel",
    "field_kernel",
}
_SUPPORTED_BUCKET_BUILDER_METHODS = {
    "auto",
    "cuda_device",
    "vulkan_native",
    "cpu_native",
    "kernel",
    "field_kernel",
}
_SUPPORTED_GROUPED_REDUCE_METHODS = {
    "auto",
    "cuda_device",
    "cuda_segmented",
    "vulkan_native",
    "vulkan_segmented",
    "segmented",
    "cpu_native",
    "kernel",
    "field_kernel",
}
_SUPPORTED_GROUPED_REDUCE_OPS = {"sum": 0}
_REDUCE_FIELD_PRIVATE_MIN_N = 65536
_REDUCE_FIELD_PRIVATE_CHUNK_SIZE = 2048
_HISTOGRAM_FIELD_PRIVATE_MIN_N = 65536
_HISTOGRAM_FIELD_PRIVATE_MAX_BINS = 512
_HISTOGRAM_FIELD_PRIVATE_CHUNK_SIZE = 2048


class SortWorkspace:
    """Workspace handle for future backend sort implementations.

    The current implementation is intentionally metadata-only. It gives the new
    sort API a stable place to attach allocation accounting without changing the
    legacy odd-even fallback behavior.
    """

    def __init__(self, max_items=None, device=None):
        self.max_items = max_items
        self.device = device
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._reserved_specs = []
        self._radix_u32_buffers = {}
        self._vulkan_graph_u32_buffers = {}
        self._vulkan_graph_u32_execs = {}
        self._cuda_cub_active = False
        self._vulkan_native_active = False

    def reserve(self, dtype=None, value_dtype=None, n=None):
        if n is not None and n < 0:
            raise ValueError("SortWorkspace.reserve() expects n >= 0.")
        if self.max_items is not None and n is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} sort items, exceeding max_items={self.max_items}."
            )
        self._reserved_specs.append(
            {"dtype": dtype, "value_dtype": value_dtype, "n": n}
        )
        return self

    def clear(self):
        if self._cuda_cub_active:
            self._clear_cuda_cub_backend_workspace()
        if self._vulkan_native_active:
            self._clear_vulkan_native_backend_workspace()
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._reserved_specs.clear()
        self._radix_u32_buffers.clear()
        self._vulkan_graph_u32_buffers.clear()
        self._vulkan_graph_u32_execs.clear()
        self._cuda_cub_active = False
        self._vulkan_native_active = False

    def _clear_cuda_cub_backend_workspace(self):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if hasattr(prog, "cuda_cub_radix_sort_clear_workspace"):
            prog.cuda_cub_radix_sort_clear_workspace()

    def _clear_vulkan_native_backend_workspace(self):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if hasattr(prog, "vulkan_radix_sort_clear_workspace"):
            prog.vulkan_radix_sort_clear_workspace()

    def _get_radix_u32_buffers(self, n, value_dtype, use_values):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} sort items, exceeding max_items={self.max_items}."
            )
        key = (n, str(value_dtype), use_values)
        if key not in self._radix_u32_buffers:
            buffers = {
                "key_in": field(u32, shape=n),
                "key_out": field(u32, shape=n),
                "zero_prefix": field(i32, shape=n),
                "zero_count": field(i32, shape=()),
                "scanner": PrefixSumExecutor(n),
            }
            if use_values:
                buffers["value_in"] = field(value_dtype, shape=n)
                buffers["value_out"] = field(value_dtype, shape=n)
            self._radix_u32_buffers[key] = buffers
            bytes_used = 2 * n * 4 + n * 4 + 4
            if use_values:
                bytes_used += 2 * n * _dtype_nbytes(value_dtype)
            self.workspace_bytes_current += bytes_used
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._radix_u32_buffers[key]

    def _get_vulkan_graph_u32_buffers(self, n, use_values):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} sort items, exceeding max_items={self.max_items}."
            )
        key = (n, use_values)
        if key not in self._vulkan_graph_u32_buffers:
            scan_size = _scan_workspace_size(n)
            buffers = {
                "key_in": ti_ndarray(u32, shape=n),
                "key_out": ti_ndarray(u32, shape=n),
                "scan_arr": ti_ndarray(i32, shape=scan_size),
                "zero_count": ti_ndarray(i32, shape=1),
            }
            if use_values:
                buffers["value_in"] = ti_ndarray(i32, shape=n)
                buffers["value_out"] = ti_ndarray(i32, shape=n)
            self._vulkan_graph_u32_buffers[key] = buffers
            bytes_used = 2 * n * 4 + scan_size * 4 + 4
            if use_values:
                bytes_used += 2 * n * 4
            self.workspace_bytes_current += bytes_used
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._vulkan_graph_u32_buffers[key]


class CompactWorkspace:
    """Workspace for experimental stable flag compaction.

    The field path keeps a Forge-kernel prefix buffer plus a PrefixSumExecutor.
    CUDA ndarray fast path uses CUB DeviceSelect and reports the cached CUB temp
    storage through Program.
    """

    def __init__(self, max_items=None):
        self.max_items = max_items
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers = {}
        self._cuda_field_buffers = {}
        self._cuda_cub_active = False
        self._cuda_cub_scan_active = False
        self._vulkan_native_active = False

    def clear(self):
        if self._cuda_cub_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "cuda_cub_select_clear_workspace"):
                prog.cuda_cub_select_clear_workspace()
        if self._cuda_cub_scan_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "cuda_cub_scan_clear_workspace"):
                prog.cuda_cub_scan_clear_workspace()
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "vulkan_compact_clear_workspace"):
                prog.vulkan_compact_clear_workspace()
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers.clear()
        self._cuda_field_buffers.clear()
        self._cuda_cub_active = False
        self._cuda_cub_scan_active = False
        self._vulkan_native_active = False

    def _get_field_buffers(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} compact items, exceeding max_items={self.max_items}."
            )
        key = n
        if key not in self._field_buffers:
            scanner = PrefixSumExecutor(n)
            buffers = {
                "scanner": scanner,
            }
            self._field_buffers[key] = buffers
            bytes_used = scanner.workspace_length * 4
            self.workspace_bytes_current += bytes_used
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._field_buffers[key]

    def _get_cuda_field_buffers(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} compact items, exceeding max_items={self.max_items}."
            )
        key = n
        if key not in self._cuda_field_buffers:
            buffers = {
                "prefix": ti_ndarray(i32, shape=n),
                "scanner": PrefixSumExecutor(n),
                "prefix_bytes": n * 4,
            }
            self._cuda_field_buffers[key] = buffers
            self.workspace_bytes_current += buffers["prefix_bytes"]
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._cuda_field_buffers[key]


class ReduceWorkspace:
    """Workspace for experimental reductions.

    CUDA ndarray fast path uses CUB DeviceReduce. Field/SNode fallback stays in
    Forge kernels to preserve layout and offset semantics.
    """

    def __init__(self, max_items=None):
        self.max_items = max_items
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers = {}
        self._cuda_cub_active = False
        self._vulkan_native_active = False

    def clear(self):
        if self._cuda_cub_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "cuda_cub_reduce_clear_workspace"):
                prog.cuda_cub_reduce_clear_workspace()
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "vulkan_reduce_clear_workspace"):
                prog.vulkan_reduce_clear_workspace()
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers.clear()
        self._cuda_cub_active = False
        self._vulkan_native_active = False

    def check_shape(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} reduce items, exceeding max_items={self.max_items}."
            )

    def _get_field_private_buffers(self, n, dtype):
        self.check_shape(n)
        chunk_size = _REDUCE_FIELD_PRIVATE_CHUNK_SIZE
        num_chunks = (n + chunk_size - 1) // chunk_size
        key = (num_chunks, str(dtype))
        if key not in self._field_buffers:
            partial = field(dtype, shape=num_chunks)
            self._field_buffers[key] = {
                "partial": partial,
                "chunk_size": chunk_size,
                "num_chunks": num_chunks,
            }
            bytes_used = num_chunks * _dtype_nbytes(dtype)
            self.workspace_bytes_current += bytes_used
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._field_buffers[key]


class HistogramWorkspace:
    """Workspace for experimental fixed-bin histogram.

    CUDA ndarray fast path uses CUB DeviceHistogram. Vulkan ndarray fast path
    uses native compute shaders. Field fallback uses Forge kernels, selecting a
    zero-workspace direct path for small inputs and a chunk-private path for
    larger fixed-bin histograms.
    """

    def __init__(self, max_items=None, max_bins=None):
        self.max_items = max_items
        self.max_bins = max_bins
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers = {}
        self._cuda_cub_active = False
        self._vulkan_native_active = False

    def clear(self):
        if self._cuda_cub_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "cuda_cub_histogram_clear_workspace"):
                prog.cuda_cub_histogram_clear_workspace()
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "vulkan_histogram_clear_workspace"):
                prog.vulkan_histogram_clear_workspace()
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._field_buffers.clear()
        self._cuda_cub_active = False
        self._vulkan_native_active = False

    def check_shape(self, n, num_bins):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} histogram items, exceeding max_items={self.max_items}."
            )
        if self.max_bins is not None and num_bins > self.max_bins:
            raise ValueError(
                f"Requested {num_bins} histogram bins, exceeding max_bins={self.max_bins}."
            )

    def _get_field_private_buffers(self, n, num_bins):
        self.check_shape(n, num_bins)
        chunk_size = _HISTOGRAM_FIELD_PRIVATE_CHUNK_SIZE
        num_chunks = (n + chunk_size - 1) // chunk_size
        key = (num_chunks, num_bins)
        if key not in self._field_buffers:
            partial = field(i32, shape=num_chunks * num_bins)
            self._field_buffers[key] = {
                "partial": partial,
                "chunk_size": chunk_size,
                "num_chunks": num_chunks,
            }
            bytes_used = num_chunks * num_bins * 4
            self.workspace_bytes_current += bytes_used
            self.workspace_bytes_peak = max(
                self.workspace_bytes_peak, self.workspace_bytes_current
            )
        return self._field_buffers[key]


class TransformWorkspace:
    """Workspace metadata for experimental affine transforms.

    CUDA driver and CPU native paths are zero-workspace. Vulkan native uses one
    cached 8-byte params buffer; field/SNode fallback stays in Forge kernels.
    """

    def __init__(self, max_items=None):
        self.max_items = max_items
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False

    def check_shape(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} transform items, exceeding max_items={self.max_items}."
            )

    def clear(self):
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "vulkan_transform_clear_workspace"):
                prog.vulkan_transform_clear_workspace()
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False


class IndexedCopyWorkspace:
    """Workspace metadata for experimental indexed gather/scatter.

    Current native paths are zero-workspace. The class exists to keep the
    public experimental primitive contract aligned with sort/scan/transform and
    to leave room for future cached staging or validation buffers.
    """

    def __init__(self, max_items=None):
        self.max_items = max_items
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False

    def check_shape(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} indexed-copy items, exceeding max_items={self.max_items}."
            )

    def clear(self):
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "vulkan_indexed_copy_clear_workspace"):
                prog.vulkan_indexed_copy_clear_workspace()
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False


class ScatterAddWorkspace:
    """Workspace metadata for experimental indexed scatter-add.

    Native paths currently use no extra device workspace. The object mirrors the
    other experimental primitive workspaces so future segmented or bucketed
    implementations can report temporary storage without changing the API.
    """

    def __init__(self, max_items=None):
        self.max_items = max_items
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False

    def check_shape(self, n):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} scatter-add items, exceeding max_items={self.max_items}."
            )

    def clear(self):
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "vulkan_scatter_add_clear_workspace"):
                prog.vulkan_scatter_add_clear_workspace()
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._vulkan_native_active = False


class BucketBuilderWorkspace:
    """Workspace for experimental fixed-bin bucket range construction.

    Native CUDA/Vulkan paths use one cached i32 cursor buffer of length
    ``num_bins``. Field/SNode fallback uses a field cursor plus the existing
    prefix-sum executor over ``offsets``.
    """

    def __init__(self, max_items=None, max_bins=None):
        self.max_items = max_items
        self.max_bins = max_bins
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._cursor_ndarray = None
        self._cursor_field = None
        self._scanner_cache = {}
        self._vulkan_native_active = False

    def check_shape(self, n, num_bins):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} bucket items, exceeding max_items={self.max_items}."
            )
        if self.max_bins is not None and num_bins > self.max_bins:
            raise ValueError(
                f"Requested {num_bins} buckets, exceeding max_bins={self.max_bins}."
            )

    def clear(self):
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "vulkan_bucket_builder_clear_workspace"):
                prog.vulkan_bucket_builder_clear_workspace()
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._cursor_ndarray = None
        self._cursor_field = None
        self._scanner_cache.clear()
        self._vulkan_native_active = False

    def _reserve_bytes(self, bytes_used):
        self.workspace_bytes_current += bytes_used
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak, self.workspace_bytes_current
        )

    def _get_cursor_ndarray(self, num_bins):
        if self._cursor_ndarray is None or self._cursor_ndarray.shape[0] < num_bins:
            self._cursor_ndarray = ti_ndarray(i32, shape=num_bins)
            self._reserve_bytes(num_bins * 4)
        return self._cursor_ndarray

    def _get_cursor_field(self, num_bins):
        if self._cursor_field is None or self._cursor_field.shape[0] < num_bins:
            self._cursor_field = field(i32, shape=num_bins)
            self._reserve_bytes(num_bins * 4)
        return self._cursor_field

    def _get_scanner(self, length):
        if length not in self._scanner_cache:
            scanner = PrefixSumExecutor(length)
            self._scanner_cache[length] = scanner
            self._reserve_bytes(scanner.workspace_length * 4)
        return self._scanner_cache[length]


class GroupedReduceWorkspace:
    """Workspace for experimental grouped reductions.

    Native paths build fixed-bin bucket ranges and then reduce each bucket.
    The Python-owned ndarrays keep the external API call free of repeated
    allocation once the workspace is reused.
    """

    def __init__(self, max_items=None, max_groups=None):
        self.max_items = max_items
        self.max_groups = max_groups
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._offsets_ndarray = None
        self._scratch_ndarray = None
        self._cursor_ndarray = None
        self._vulkan_native_active = False

    def check_shape(self, n, num_groups):
        if self.max_items is not None and n > self.max_items:
            raise ValueError(
                f"Requested {n} grouped-reduce items, exceeding max_items={self.max_items}."
            )
        if self.max_groups is not None and num_groups > self.max_groups:
            raise ValueError(
                f"Requested {num_groups} grouped-reduce groups, exceeding max_groups={self.max_groups}."
            )

    def clear(self):
        if self._vulkan_native_active:
            from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

            prog = impl.get_runtime().prog
            if hasattr(prog, "vulkan_grouped_reduce_clear_workspace"):
                prog.vulkan_grouped_reduce_clear_workspace()
        self.workspace_bytes_current = 0
        self.workspace_bytes_peak = 0
        self._offsets_ndarray = None
        self._scratch_ndarray = None
        self._cursor_ndarray = None
        self._vulkan_native_active = False

    def _reserve_bytes(self, bytes_used):
        self.workspace_bytes_current += bytes_used
        self.workspace_bytes_peak = max(
            self.workspace_bytes_peak, self.workspace_bytes_current
        )

    def _get_native_buffers(self, n, num_groups):
        if (
            self._offsets_ndarray is None
            or self._offsets_ndarray.shape[0] < num_groups + 1
        ):
            self._offsets_ndarray = ti_ndarray(i32, shape=num_groups + 1)
            self._reserve_bytes((num_groups + 1) * 4)
        if self._scratch_ndarray is None or self._scratch_ndarray.shape[0] < n:
            self._scratch_ndarray = ti_ndarray(i32, shape=n)
            self._reserve_bytes(n * 4)
        if self._cursor_ndarray is None or self._cursor_ndarray.shape[0] < num_groups:
            self._cursor_ndarray = ti_ndarray(i32, shape=num_groups)
            self._reserve_bytes(num_groups * 4)
        return self._offsets_ndarray, self._scratch_ndarray, self._cursor_ndarray


def _dtype_nbytes(dtype):
    text = str(dtype)
    if "64" in text:
        return 8
    if "16" in text:
        return 2
    if "8" in text:
        return 1
    return 4


def _scan_layout(length):
    block_sz = 64
    ele_nums = [length]
    ele_nums_pos = [0]
    ele_num = length
    start_pos = 0
    while ele_num > 1:
        ele_num = int((ele_num + block_sz - 1) / block_sz)
        ele_nums.append(ele_num)
        start_pos += block_sz * ele_num
        ele_nums_pos.append(start_pos)
    return ele_nums, ele_nums_pos, max(start_pos, length)


def _scan_workspace_size(length):
    return _scan_layout(length)[2]


def _parallel_sort_legacy(keys, values=None):
    """Odd-even merge sort implementation used as compatibility fallback.

    References:
        https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
        https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
    """
    N = keys.shape[0]

    num_stages = 0
    p = 1
    while p < N:
        k = p
        while k >= 1:
            invocations = int((N - k - k % p) / (2 * k)) + 1
            if values is None:
                sort_stage(keys, 0, keys, N, p, k, invocations)
            else:
                sort_stage(keys, 1, values, N, p, k, invocations)
            num_stages += 1
            sync()
            k = int(k / 2)
        p = int(p * 2)


def _check_sort_request(
    keys,
    values,
    stable,
    descending,
    method,
    precision,
    workspace,
    nan_policy,
):
    if not hasattr(keys, "shape") or len(keys.shape) != 1:
        raise ValueError("sort() currently expects a 1D Taichi field or ndarray.")
    if values is not None:
        if not hasattr(values, "shape") or len(values.shape) != 1:
            raise ValueError("sort() values must be a 1D Taichi field or ndarray.")
        if values.shape[0] != keys.shape[0]:
            raise ValueError("sort() keys and values must have the same length.")
    if stable is not True:
        raise NotImplementedError("Only stable sort is currently implemented.")
    if descending:
        raise NotImplementedError("descending=True is not implemented yet.")
    if method not in _SUPPORTED_SORT_METHODS:
        raise NotImplementedError(f"sort method '{method}' is not implemented yet.")
    if precision not in _SUPPORTED_SORT_PRECISIONS:
        raise NotImplementedError(f"sort precision '{precision}' is not implemented yet.")
    if nan_policy not in _SUPPORTED_NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {sorted(_SUPPORTED_NAN_POLICIES)}, got {nan_policy!r}."
        )
    if workspace is not None and not isinstance(workspace, SortWorkspace):
        raise TypeError("workspace must be a SortWorkspace instance or None.")


def _host_stable_sort(keys, values=None, descending=False, nan_policy="last"):
    if nan_policy == "bitwise":
        raise NotImplementedError("nan_policy='bitwise' needs a device sortable-key path.")
    keys_np = keys.to_numpy()
    if values is None:
        sorted_keys = np.sort(keys_np, kind="stable")
        if descending:
            sorted_keys = sorted_keys[::-1]
        keys.from_numpy(np.ascontiguousarray(sorted_keys))
        sync()
        return

    order = np.argsort(keys_np, kind="stable")
    if descending:
        order = order[::-1]
    values_np = values.to_numpy()
    keys.from_numpy(np.ascontiguousarray(keys_np[order]))
    values.from_numpy(np.ascontiguousarray(values_np[order]))
    sync()


def _host_stable_sort_by_key_parts(key_parts, values=None):
    key_arrays = [part.to_numpy() for part in key_parts]
    if len(key_arrays) == 1:
        order = np.argsort(key_arrays[0], kind="stable")
    else:
        stable_tie_breaker = np.arange(key_arrays[0].shape[0])
        order = np.lexsort((stable_tie_breaker, *reversed(key_arrays)))
    for part, part_np in zip(key_parts, key_arrays):
        part.from_numpy(part_np[order])
    if values is not None:
        values_np = values.to_numpy()
        values.from_numpy(values_np[order])
    sync()


def _radix_sort_u32(keys, values=None, workspace=None):
    arch = current_cfg().arch
    if arch not in (cuda, vulkan):
        raise RuntimeError(
            "method='radix_u32' is currently supported only on CUDA/Vulkan."
        )
    if keys.dtype not in (i32, u32):
        raise TypeError("method='radix_u32' currently supports only ti.i32 and ti.u32 keys.")
    N = keys.shape[0]
    if N <= 1:
        return

    signed = 1 if keys.dtype == i32 else 0
    use_values = values is not None
    if workspace is None:
        workspace = SortWorkspace(max_items=N)
    buffers = workspace._get_radix_u32_buffers(
        N, values.dtype if use_values else u32, use_values
    )
    key_in = buffers["key_in"]
    key_out = buffers["key_out"]
    zero_prefix = buffers["zero_prefix"]
    zero_count = buffers["zero_count"]
    scanner = buffers["scanner"]
    if use_values:
        value_in = buffers["value_in"]
        value_out = buffers["value_out"]
        sort_init_value_buffer(values, value_in, N)
    else:
        value_in = key_in
        value_out = key_out

    sort_init_key_buffer_u32(keys, key_in, N, signed)
    for bit in range(32):
        sort_radix_count_zero_bits_u32(key_in, zero_prefix, N, bit)
        scanner.run(zero_prefix)
        sort_radix_store_zero_count(zero_prefix, zero_count, N)
        sort_radix_scatter_u32(
            key_in,
            key_out,
            zero_prefix,
            value_in,
            value_out,
            1 if use_values else 0,
            N,
            bit,
            zero_count,
        )
        key_in, key_out = key_out, key_in
        if use_values:
            value_in, value_out = value_out, value_in

    sort_copy_key_buffer_to_field_u32(key_in, keys, N, signed)
    if use_values:
        sort_copy_value_buffer_to_field(value_in, values, N)
    sync()


class _VulkanGraphRadixU32Executor:
    def __init__(self, n, key_dtype, use_values, buffers):
        from taichi_forge import graph  # pylint: disable=import-outside-toplevel

        self.n = n
        self.key_dtype = key_dtype
        self.use_values = use_values
        self.buffers = buffers
        self.arg_values = {
            "N": n,
            "signed": 1 if key_dtype == i32 else 0,
            "use_values": 1 if use_values else 0,
        }

        key_arg = graph.Arg(graph.ArgKind.NDARRAY, "keys", key_dtype, ndim=1)
        key_in_arg = graph.Arg(graph.ArgKind.NDARRAY, "key_in", u32, ndim=1)
        key_out_arg = graph.Arg(graph.ArgKind.NDARRAY, "key_out", u32, ndim=1)
        scan_arg = graph.Arg(graph.ArgKind.NDARRAY, "scan_arr", i32, ndim=1)
        zero_count_arg = graph.Arg(
            graph.ArgKind.NDARRAY, "zero_count", i32, ndim=1
        )
        n_arg = graph.Arg(graph.ArgKind.SCALAR, "N", i32)
        signed_arg = graph.Arg(graph.ArgKind.SCALAR, "signed", i32)

        builder = graph.GraphBuilder()
        builder.dispatch(sort_init_key_buffer_u32_ndarray, key_arg, key_in_arg, n_arg, signed_arg)
        if use_values:
            values_arg = graph.Arg(graph.ArgKind.NDARRAY, "values", i32, ndim=1)
            value_in_arg = graph.Arg(graph.ArgKind.NDARRAY, "value_in", i32, ndim=1)
            value_out_arg = graph.Arg(graph.ArgKind.NDARRAY, "value_out", i32, ndim=1)
            use_values_arg = graph.Arg(graph.ArgKind.SCALAR, "use_values", i32)
            builder.dispatch(sort_init_value_buffer_i32_ndarray, values_arg, value_in_arg, n_arg)
        else:
            value_in_arg = None
            value_out_arg = None
            use_values_arg = None

        ele_nums, ele_nums_pos, _ = _scan_layout(n)
        scan_stage_args = []
        for level in range(len(ele_nums) - 1):
            beg_name = f"scan_beg_{level}"
            end_name = f"scan_end_{level}"
            single_name = f"scan_single_{level}"
            self.arg_values[beg_name] = ele_nums_pos[level]
            self.arg_values[end_name] = ele_nums_pos[level + 1]
            self.arg_values[single_name] = 1 if level == len(ele_nums) - 2 else 0
            scan_stage_args.append(
                (
                    graph.Arg(graph.ArgKind.SCALAR, beg_name, i32),
                    graph.Arg(graph.ArgKind.SCALAR, end_name, i32),
                    graph.Arg(graph.ArgKind.SCALAR, single_name, i32),
                )
            )

        key_read_arg = key_in_arg
        key_write_arg = key_out_arg
        value_read_arg = value_in_arg
        value_write_arg = value_out_arg
        for bit in range(32):
            bit_name = f"bit_{bit}"
            self.arg_values[bit_name] = bit
            bit_arg = graph.Arg(graph.ArgKind.SCALAR, bit_name, i32)
            builder.dispatch(
                sort_radix_count_zero_bits_u32_ndarray,
                key_read_arg,
                scan_arg,
                n_arg,
                bit_arg,
            )
            for beg_arg, end_arg, single_arg in scan_stage_args:
                builder.dispatch(
                    scan_add_inclusive_ndarray,
                    scan_arg,
                    beg_arg,
                    end_arg,
                    single_arg,
                )
            for level in range(len(ele_nums) - 3, -1, -1):
                beg_arg, end_arg, _ = scan_stage_args[level]
                builder.dispatch(uniform_add_ndarray, scan_arg, beg_arg, end_arg)
            builder.dispatch(
                sort_radix_store_zero_count_ndarray,
                scan_arg,
                zero_count_arg,
                n_arg,
            )
            if use_values:
                builder.dispatch(
                    sort_radix_scatter_u32_i32_ndarray,
                    key_read_arg,
                    key_write_arg,
                    scan_arg,
                    value_read_arg,
                    value_write_arg,
                    use_values_arg,
                    n_arg,
                    bit_arg,
                    zero_count_arg,
                )
            else:
                builder.dispatch(
                    sort_radix_scatter_keys_u32_ndarray,
                    key_read_arg,
                    key_write_arg,
                    scan_arg,
                    n_arg,
                    bit_arg,
                    zero_count_arg,
                )
            key_read_arg, key_write_arg = key_write_arg, key_read_arg
            if use_values:
                value_read_arg, value_write_arg = value_write_arg, value_read_arg

        builder.dispatch(
            sort_copy_key_buffer_to_ndarray_u32,
            key_read_arg,
            key_arg,
            n_arg,
            signed_arg,
        )
        if use_values:
            builder.dispatch(
                sort_copy_value_buffer_to_i32_ndarray,
                value_read_arg,
                values_arg,
                n_arg,
            )
        self.graph = builder.compile()

    def run(self, keys, values=None):
        args = {
            "keys": keys,
            "key_in": self.buffers["key_in"],
            "key_out": self.buffers["key_out"],
            "scan_arr": self.buffers["scan_arr"],
            "zero_count": self.buffers["zero_count"],
            **self.arg_values,
        }
        if self.use_values:
            args["values"] = values
            args["value_in"] = self.buffers["value_in"]
            args["value_out"] = self.buffers["value_out"]
        self.graph.run(args)
        sync()


def _vulkan_graph_radix_sort_u32(keys, values=None, workspace=None):
    arch = current_cfg().arch
    if arch != vulkan:
        raise RuntimeError("method='vulkan_graph_radix_u32' is supported only on Vulkan.")
    if not isinstance(keys, Ndarray):
        raise NotImplementedError(
            "method='vulkan_graph_radix_u32' currently supports only ti.ndarray keys."
        )
    if keys.dtype not in (i32, u32):
        raise TypeError(
            "method='vulkan_graph_radix_u32' currently supports only ti.i32 and ti.u32 keys."
        )
    use_values = values is not None
    if use_values:
        if not isinstance(values, Ndarray):
            raise NotImplementedError(
                "method='vulkan_graph_radix_u32' currently supports only ti.ndarray values."
            )
        if values.dtype != i32:
            raise TypeError(
                "method='vulkan_graph_radix_u32' currently supports only ti.i32 values."
            )
    n = keys.shape[0]
    if n <= 1:
        return
    if workspace is None:
        workspace = SortWorkspace(max_items=n)
    buffers = workspace._get_vulkan_graph_u32_buffers(n, use_values)
    cache_key = (n, str(keys.dtype), use_values)
    executor = workspace._vulkan_graph_u32_execs.get(cache_key)
    if executor is None:
        executor = _VulkanGraphRadixU32Executor(n, keys.dtype, use_values, buffers)
        workspace._vulkan_graph_u32_execs[cache_key] = executor
    executor.run(keys, values)


def _vulkan_native_radix_sort_u32(keys, values=None, workspace=None):
    arch = current_cfg().arch
    if arch != vulkan:
        raise RuntimeError("method='vulkan_native_radix_u32' is supported only on Vulkan.")
    if not isinstance(keys, Ndarray):
        raise NotImplementedError(
            "method='vulkan_native_radix_u32' currently supports only ti.ndarray keys."
        )
    if keys.dtype not in (i32, u32):
        raise TypeError(
            "method='vulkan_native_radix_u32' currently supports only ti.i32 and ti.u32 keys."
        )
    use_values = values is not None
    if use_values:
        if not isinstance(values, Ndarray):
            raise NotImplementedError(
                "method='vulkan_native_radix_u32' currently supports only ti.ndarray values."
            )
        if values.dtype != i32:
            raise TypeError(
                "method='vulkan_native_radix_u32' currently supports only ti.i32 values."
            )
    if keys.shape[0] <= 1:
        return

    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not prog.vulkan_radix_sort_available():
        raise RuntimeError("method='vulkan_native_radix_u32' requires Vulkan sort support.")
    key_type = {u32: 0, i32: 1}[keys.dtype]
    temp_bytes = (
        prog.vulkan_radix_sort_u32_ndarray(keys.arr, values.arr, key_type)
        if values is not None
        else prog.vulkan_radix_sort_u32_keys_ndarray(keys.arr, key_type)
    )
    if workspace is not None:
        workspace._vulkan_native_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )


def _cuda_cub_sort_native(
    keys,
    values=None,
    workspace=None,
    method="cuda_cub_native",
    nan_policy="last",
):
    arch = current_cfg().arch
    if arch != cuda:
        raise RuntimeError(f"method='{method}' is supported only on CUDA.")
    if not isinstance(keys, Ndarray):
        raise NotImplementedError(
            f"method='{method}' currently supports only 1D ti.ndarray keys."
        )
    if values is not None and not isinstance(values, Ndarray):
        raise NotImplementedError(
            f"method='{method}' currently supports only ti.ndarray payloads."
        )
    if keys.dtype not in (u32, i32, f32, u64, i64, f64):
        raise TypeError(
            f"method='{method}' currently supports ti.u32, ti.i32, ti.f32, "
            "ti.u64, ti.i64, and ti.f64 keys."
        )
    if values is not None and values.dtype != i32:
        raise TypeError(f"method='{method}' currently supports only ti.i32 values.")
    if method == "cuda_cub_split32" and keys.dtype not in (u64, i64, f64):
        raise TypeError(
            "method='cuda_cub_split32' supports only ti.u64, ti.i64, and ti.f64 keys."
        )
    if keys.shape[0] <= 1:
        return
    # Keep the implementation tied to Program instead of adding Python-side
    # per-pass behavior. `impl.get_runtime()` is imported lazily to avoid
    # expanding the algorithms module's public surface.
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not prog.cuda_cub_radix_sort_available():
        raise RuntimeError(
            f"method='{method}' requires CUDA CUB sort support and a "
            "discoverable CUDA runtime library."
        )
    key_type = {u32: 0, i32: 1, f32: 2, u64: 3, i64: 4, f64: 5}[keys.dtype]
    mode = 1 if method == "cuda_cub_split32" else 0
    nan_policy_id = {"last": 0, "bitwise": 1}[nan_policy]
    temp_bytes = (
        prog.cuda_cub_radix_sort_ndarray(
            keys.arr, values.arr, key_type, mode, nan_policy_id
        )
        if values is not None
        else prog.cuda_cub_radix_sort_keys_ndarray(
            keys.arr, key_type, mode, nan_policy_id
        )
    )
    if workspace is not None:
        workspace._cuda_cub_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )


def _auto_sort(keys, values=None, workspace=None, nan_policy="last"):
    arch = current_cfg().arch
    if (
        arch == cuda
        and isinstance(keys, Ndarray)
        and (values is None or isinstance(values, Ndarray))
        and keys.dtype in (u32, i32, f32, u64, i64, f64)
        and (values is None or values.dtype == i32)
    ):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        if impl.get_runtime().prog.cuda_cub_radix_sort_available():
            _cuda_cub_sort_native(
                keys,
                values,
                workspace=workspace,
                method="cuda_cub_native",
                nan_policy=nan_policy,
            )
            return

    if (
        arch == vulkan
        and isinstance(keys, Ndarray)
        and (values is None or isinstance(values, Ndarray))
        and keys.dtype in (u32, i32)
        and (values is None or values.dtype == i32)
        and nan_policy == "last"
    ):
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        if impl.get_runtime().prog.vulkan_radix_sort_available():
            _vulkan_native_radix_sort_u32(keys, values, workspace=workspace)
            return

    _host_stable_sort(keys, values, nan_policy=nan_policy)


def sort(
    keys,
    values=None,
    *,
    stable=True,
    descending=False,
    method="auto",
    precision="exact",
    workspace=None,
    nan_policy="last",
):
    """Sort keys, optionally carrying a value payload.

    This is a Taichi Forge extension. `auto` selects the native CUDA CUB
    DeviceRadixSort path on CUDA when available, the native Vulkan radix8 path
    for supported 32-bit ndarray keys on Vulkan, and otherwise falls back to a
    host stable sort. Use ``method="legacy"`` for the original odd-even merge
    implementation. ``cuda_cub_split32`` is an explicit opt-in method only.
    """

    _check_sort_request(
        keys, values, stable, descending, method, precision, workspace, nan_policy
    )
    if method == "auto":
        _auto_sort(keys, values, workspace=workspace, nan_policy=nan_policy)
    elif method == "host_stable":
        _host_stable_sort(keys, values, descending=descending, nan_policy=nan_policy)
    elif method == "legacy":
        _parallel_sort_legacy(keys, values)
    elif method in ("radix_u32", "vulkan_radix_u32"):
        _radix_sort_u32(keys, values, workspace=workspace)
    elif method == "vulkan_graph_radix_u32":
        _vulkan_graph_radix_sort_u32(keys, values, workspace=workspace)
    elif method == "vulkan_native_radix_u32":
        _vulkan_native_radix_sort_u32(keys, values, workspace=workspace)
    elif method in _CUDA_CUB_SORT_METHODS:
        _cuda_cub_sort_native(
            keys, values, workspace=workspace, method=method, nan_policy=nan_policy
        )
    else:
        _host_stable_sort(keys, values, descending=descending, nan_policy=nan_policy)


def sort_by_key(
    key_parts,
    values=None,
    *,
    stable=True,
    order="lexicographic",
    method="auto",
    workspace=None,
):
    """Sort values by key parts.

    Single-part keys route to :func:`sort`. Multi-part exact lexicographic sort
    needs the future stable radix primitive so all key parts and payloads can be
    permuted together.
    """

    if order != "lexicographic":
        raise ValueError("sort_by_key() currently only supports order='lexicographic'.")
    if isinstance(key_parts, (list, tuple)):
        parts = list(key_parts)
    else:
        parts = [key_parts]
    if len(parts) == 0:
        raise ValueError("sort_by_key() expects at least one key part.")
    _check_sort_request(parts[0], values, stable, False, method, "exact", workspace, "last")
    for part in parts[1:]:
        if not hasattr(part, "shape") or len(part.shape) != 1:
            raise ValueError("sort_by_key() key parts must be 1D Taichi fields.")
        if part.shape[0] != parts[0].shape[0]:
            raise ValueError("sort_by_key() key parts must have the same length.")
    if len(parts) > 1 and method == "legacy":
        raise NotImplementedError("Multi-part sort_by_key() needs a stable backend.")
    if len(parts) > 1:
        _host_stable_sort_by_key_parts(parts, values)
        return
    sort(
        parts[0],
        values,
        stable=stable,
        method=method,
        workspace=workspace,
    )


_SUPPORTED_COMPACT_METHODS = {
    "auto",
    "cpu_native",
    "cuda_cub",
    "field_scan",
    "vulkan_native",
}


def _is_1d(obj):
    return hasattr(obj, "shape") and len(obj.shape) == 1


def _check_compact_request(values, flags, output, count, method, workspace):
    if method not in _SUPPORTED_COMPACT_METHODS:
        raise NotImplementedError(f"compact method '{method}' is not implemented.")
    if not (_is_1d(values) and _is_1d(flags) and _is_1d(output)):
        raise ValueError("experimental_compact() expects 1D values, flags, and output.")
    if values.shape[0] != flags.shape[0]:
        raise ValueError("experimental_compact() values and flags must have the same length.")
    if output.shape[0] < values.shape[0]:
        raise ValueError("experimental_compact() output must have at least input length.")
    for name, arr in (("values", values), ("flags", flags), ("output", output)):
        if arr.dtype != i32:
            raise TypeError(
                f"experimental_compact() currently supports only ti.i32 {name}."
            )
    if count.dtype != i32:
        raise TypeError("experimental_compact() currently expects ti.i32 count.")
    if isinstance(values, Ndarray) or isinstance(flags, Ndarray) or isinstance(output, Ndarray):
        if not (
            isinstance(values, Ndarray)
            and isinstance(flags, Ndarray)
            and isinstance(output, Ndarray)
            and isinstance(count, Ndarray)
        ):
            raise TypeError(
                "experimental_compact() ndarray mode requires values, flags, "
                "output, and count all to be ti.ndarray."
            )
        if not _is_1d(count) or count.shape[0] < 1:
            raise ValueError("experimental_compact() ndarray count must be shape >= 1.")
    else:
        if isinstance(count, Ndarray) or not hasattr(count, "shape") or count.shape != ():
            raise TypeError(
                "experimental_compact() field mode requires a scalar ti.field count."
            )
    if workspace is not None and not isinstance(workspace, CompactWorkspace):
        raise TypeError("workspace must be a CompactWorkspace instance or None.")


def _try_cuda_cub_compact(values, flags, output, count, workspace):
    if current_cfg().arch != cuda:
        return False
    if not (
        isinstance(values, Ndarray)
        and isinstance(flags, Ndarray)
        and isinstance(output, Ndarray)
        and isinstance(count, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cuda_cub_select_available"):
        return False
    if not prog.cuda_cub_select_available():
        return False
    temp_bytes = prog.cuda_cub_select_i32_ndarray(
        values.arr, flags.arr, output.arr, count.arr
    )
    if workspace is not None:
        workspace._cuda_cub_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _try_vulkan_native_compact(values, flags, output, count, workspace):
    if current_cfg().arch != vulkan:
        return False
    if not (
        isinstance(values, Ndarray)
        and isinstance(flags, Ndarray)
        and isinstance(output, Ndarray)
        and isinstance(count, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_compact_available"):
        return False
    if not prog.vulkan_compact_available():
        return False
    temp_bytes = prog.vulkan_compact_i32_ndarray(
        values.arr, flags.arr, output.arr, count.arr
    )
    if workspace is not None:
        workspace._vulkan_native_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _try_cpu_native_compact(values, flags, output, count, workspace):
    if current_cfg().arch not in [x64, arm64]:
        return False
    if not (
        isinstance(values, Ndarray)
        and isinstance(flags, Ndarray)
        and isinstance(output, Ndarray)
        and isinstance(count, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_compact_available"):
        return False
    if not prog.cpu_compact_available():
        return False
    temp_bytes = prog.cpu_compact_i32_ndarray(
        values.arr, flags.arr, output.arr, count.arr
    )
    if workspace is not None:
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _cuda_cub_scan_available():
    if current_cfg().arch != cuda:
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    return (
        hasattr(prog, "cuda_cub_scan_available")
        and prog.cuda_cub_scan_available()
    )


def _compact_field_cuda_scan(values, flags, output, count, workspace, n):
    buffers = workspace._get_cuda_field_buffers(n)
    prefix = buffers["prefix"]
    scanner = buffers["scanner"]
    compact_flags_to_prefix_ndarray_from_field(flags, prefix, n)
    scanner.run(prefix)
    compact_scatter_field_from_prefix_ndarray(values, flags, prefix, output, count, n)
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if hasattr(prog, "cuda_cub_scan_workspace_bytes"):
        workspace._cuda_cub_scan_active = True
        scan_bytes = int(prog.cuda_cub_scan_workspace_bytes())
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak,
            workspace.workspace_bytes_current + scan_bytes,
        )
    sync()
    return workspace


def _compact_field_scan(values, flags, output, count, workspace):
    if isinstance(values, Ndarray) or isinstance(flags, Ndarray) or isinstance(output, Ndarray):
        raise NotImplementedError(
            "method='field_scan' supports only ti.field values/flags/output."
        )
    n = values.shape[0]
    if workspace is None:
        workspace = CompactWorkspace(max_items=n)
    if n <= 1:
        compact_single_item_field(values, flags, output, count, n)
        sync()
        return workspace
    if _cuda_cub_scan_available():
        return _compact_field_cuda_scan(values, flags, output, count, workspace, n)
    buffers = workspace._get_field_buffers(n)
    scanner = buffers["scanner"]
    prefix = scanner._ensure_large_arr()
    compact_flags_to_prefix_field(flags, prefix, n)
    scanner._run_field_workspace(prefix)
    compact_scatter_field(values, flags, prefix, output, count, n)
    sync()
    return workspace


def experimental_compact(
    values,
    flags,
    output,
    count,
    *,
    method="auto",
    workspace=None,
):
    """Stable compact values where ``flags[i] != 0``.

    This is an experimental Forge primitive. ``count`` remains on device:
    ndarray mode expects a one-element ndarray, while field mode expects a
    scalar field. The initial implementation supports i32 payloads and i32
    flags/count.
    """

    _check_compact_request(values, flags, output, count, method, workspace)
    if values.shape[0] == 0:
        return
    if workspace is None:
        workspace = CompactWorkspace(max_items=values.shape[0])
    if method in ("auto", "cuda_cub") and _try_cuda_cub_compact(
        values, flags, output, count, workspace
    ):
        return
    if method == "cuda_cub":
        raise RuntimeError(
            "method='cuda_cub' requires CUDA ndarray inputs and available CUB DeviceSelect."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_native_compact(
        values, flags, output, count, workspace
    ):
        return
    if method == "vulkan_native":
        raise RuntimeError(
            "method='vulkan_native' requires Vulkan ndarray inputs and available "
            "native compact."
        )
    if method in ("auto", "cpu_native") and _try_cpu_native_compact(
        values, flags, output, count, workspace
    ):
        return
    if method == "cpu_native":
        raise RuntimeError(
            "method='cpu_native' requires CPU ndarray inputs and available native "
            "compact."
        )
    _compact_field_scan(values, flags, output, count, workspace)


def _check_reduce_request(values, output, op, method, workspace):
    if method not in _SUPPORTED_REDUCE_METHODS:
        raise NotImplementedError(f"reduce method '{method}' is not implemented.")
    if op not in _SUPPORTED_REDUCE_OPS:
        raise ValueError(f"reduce op must be one of {sorted(_SUPPORTED_REDUCE_OPS)}.")
    if not _is_1d(values):
        raise ValueError("experimental_reduce() expects 1D values.")
    if values.shape[0] <= 0:
        raise ValueError("experimental_reduce() expects at least one input item.")
    if values.dtype not in (i32, f32):
        raise TypeError("experimental_reduce() currently supports ti.i32 and ti.f32.")
    if output.dtype != values.dtype:
        raise TypeError("experimental_reduce() values and output dtype must match.")
    if isinstance(values, Ndarray) or isinstance(output, Ndarray):
        if not (isinstance(values, Ndarray) and isinstance(output, Ndarray)):
            raise TypeError(
                "experimental_reduce() ndarray mode requires both values and output "
                "to be ti.ndarray."
            )
        if not _is_1d(output) or output.shape[0] < 1:
            raise ValueError("experimental_reduce() ndarray output must be shape >= 1.")
    else:
        if not hasattr(output, "shape") or output.shape != ():
            raise TypeError(
                "experimental_reduce() field mode requires a scalar ti.field output."
            )
    if workspace is not None and not isinstance(workspace, ReduceWorkspace):
        raise TypeError("workspace must be a ReduceWorkspace instance or None.")


def _reduce_value_type(dtype):
    if dtype == i32:
        return 0
    if dtype == f32:
        return 1
    raise TypeError("unsupported reduce dtype")


def _try_cuda_cub_reduce(values, output, op, workspace):
    if current_cfg().arch != cuda:
        return False
    if not (isinstance(values, Ndarray) and isinstance(output, Ndarray)):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cuda_cub_reduce_available"):
        return False
    if not prog.cuda_cub_reduce_available():
        return False
    temp_bytes = prog.cuda_cub_reduce_ndarray(
        values.arr, output.arr, _reduce_value_type(values.dtype), _SUPPORTED_REDUCE_OPS[op]
    )
    if workspace is not None:
        workspace._cuda_cub_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _try_vulkan_reduce(values, output, op, workspace):
    if current_cfg().arch != vulkan:
        return False
    if not (isinstance(values, Ndarray) and isinstance(output, Ndarray)):
        return False
    if values.dtype != i32:
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_reduce_available"):
        return False
    if not prog.vulkan_reduce_available():
        return False
    temp_bytes = prog.vulkan_reduce_i32_ndarray(
        values.arr, output.arr, _SUPPORTED_REDUCE_OPS[op]
    )
    if workspace is not None:
        workspace._vulkan_native_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _try_cpu_reduce(values, output, op, workspace):
    if current_cfg().arch not in [x64, arm64]:
        return False
    if not (isinstance(values, Ndarray) and isinstance(output, Ndarray)):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_reduce_available"):
        return False
    if not prog.cpu_reduce_available():
        return False
    temp_bytes = prog.cpu_reduce_ndarray(
        values.arr, output.arr, _reduce_value_type(values.dtype), _SUPPORTED_REDUCE_OPS[op]
    )
    if workspace is not None:
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _reduce_field_atomic(values, output, op, workspace):
    op_id = _SUPPORTED_REDUCE_OPS[op]
    if values.shape[0] >= _REDUCE_FIELD_PRIVATE_MIN_N:
        buffers = workspace._get_field_private_buffers(values.shape[0], values.dtype)
        partial = buffers["partial"]
        if values.dtype == i32:
            reduce_i32_field_private_count(
                values,
                partial,
                values.shape[0],
                buffers["chunk_size"],
                buffers["num_chunks"],
                op_id,
            )
            reduce_i32_field_private_reduce(
                partial, output, buffers["num_chunks"], op_id
            )
        else:
            reduce_f32_field_private_count(
                values,
                partial,
                values.shape[0],
                buffers["chunk_size"],
                buffers["num_chunks"],
                op_id,
            )
            reduce_f32_field_private_reduce(
                partial, output, buffers["num_chunks"], op_id
            )
        sync()
        return
    if values.dtype == i32:
        reduce_i32_field(values, output, values.shape[0], op_id)
    else:
        reduce_f32_field(values, output, values.shape[0], op_id)
    sync()


def experimental_reduce(values, output, *, op="sum", method="auto", workspace=None):
    """Reduce a 1D array into a scalar output.

    This experimental primitive currently supports ``sum``, ``min``, and
    ``max`` for i32/f32 values. CUDA ndarray input uses CUB DeviceReduce when
    available. Vulkan i32 ndarray input uses native compute shaders. CPU
    ndarray input uses a host native path. Field/SNode fallback stays in Forge
    kernels.
    """

    _check_reduce_request(values, output, op, method, workspace)
    if workspace is None:
        workspace = ReduceWorkspace(max_items=values.shape[0])
    workspace.check_shape(values.shape[0])
    if method in ("auto", "cuda_cub") and _try_cuda_cub_reduce(
        values, output, op, workspace
    ):
        return
    if method == "cuda_cub":
        raise RuntimeError(
            "method='cuda_cub' requires CUDA ndarray inputs and available CUB DeviceReduce."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_reduce(
        values, output, op, workspace
    ):
        return
    if method == "vulkan_native":
        raise RuntimeError(
            "method='vulkan_native' requires Vulkan i32 ndarray inputs and "
            "available native reduce shaders."
        )
    if method in ("auto", "cpu_native") and _try_cpu_reduce(
        values, output, op, workspace
    ):
        return
    if method == "cpu_native":
        raise RuntimeError(
            "method='cpu_native' requires CPU ndarray inputs and available native reduce."
        )
    if isinstance(values, Ndarray):
        raise RuntimeError(
            "experimental_reduce() ndarray input is currently supported only "
            "by native CPU/CUDA/Vulkan reduce fast paths. Use a field input "
            "or an available native backend."
        )
    _reduce_field_atomic(values, output, op, workspace)


_SUPPORTED_HISTOGRAM_METHODS = {
    "auto",
    "cuda_cub",
    "vulkan_native",
    "cpu_native",
    "field_atomic",
    "field_direct",
    "field_private",
}


def _check_histogram_request(values, bins, method, workspace):
    if method not in _SUPPORTED_HISTOGRAM_METHODS:
        raise NotImplementedError(f"histogram method '{method}' is not implemented.")
    if not (_is_1d(values) and _is_1d(bins)):
        raise ValueError("experimental_histogram() expects 1D values and bins.")
    if values.dtype != i32 or bins.dtype != i32:
        raise TypeError("experimental_histogram() currently expects ti.i32 values and bins.")
    if bins.shape[0] <= 0:
        raise ValueError("experimental_histogram() expects at least one bin.")
    if isinstance(values, Ndarray) or isinstance(bins, Ndarray):
        if not (isinstance(values, Ndarray) and isinstance(bins, Ndarray)):
            raise TypeError(
                "experimental_histogram() ndarray mode requires both values and bins "
                "to be ti.ndarray."
            )
    if workspace is not None and not isinstance(workspace, HistogramWorkspace):
        raise TypeError("workspace must be a HistogramWorkspace instance or None.")


def _try_cuda_cub_histogram(values, bins, workspace):
    if current_cfg().arch != cuda:
        return False
    if not (isinstance(values, Ndarray) and isinstance(bins, Ndarray)):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cuda_cub_histogram_available"):
        return False
    if not prog.cuda_cub_histogram_available():
        return False
    temp_bytes = prog.cuda_cub_histogram_i32_ndarray(values.arr, bins.arr)
    if workspace is not None:
        workspace._cuda_cub_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _try_vulkan_histogram(values, bins, workspace):
    if current_cfg().arch != vulkan:
        return False
    if not (isinstance(values, Ndarray) and isinstance(bins, Ndarray)):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_histogram_available"):
        return False
    if not prog.vulkan_histogram_available():
        return False
    temp_bytes = prog.vulkan_histogram_i32_ndarray(values.arr, bins.arr)
    if workspace is not None:
        workspace._vulkan_native_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _try_cpu_native_histogram(values, bins, workspace):
    if current_cfg().arch not in [x64, arm64]:
        return False
    if not (isinstance(values, Ndarray) and isinstance(bins, Ndarray)):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_histogram_available"):
        return False
    if not prog.cpu_histogram_available():
        return False
    temp_bytes = prog.cpu_histogram_i32_ndarray(values.arr, bins.arr)
    if workspace is not None:
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _histogram_should_use_private(n, num_bins):
    return (
        n >= _HISTOGRAM_FIELD_PRIVATE_MIN_N
        and num_bins <= _HISTOGRAM_FIELD_PRIVATE_MAX_BINS
    )


def _histogram_field_direct(values, bins, n, num_bins):
    histogram_i32_field_direct(values, bins, n, num_bins)
    sync()


def _histogram_field_private(values, bins, workspace, n, num_bins):
    buffers = workspace._get_field_private_buffers(n, num_bins)
    histogram_i32_field_private_count(
        values,
        buffers["partial"],
        n,
        num_bins,
        buffers["chunk_size"],
        buffers["num_chunks"],
    )
    histogram_i32_field_private_reduce(
        buffers["partial"], bins, num_bins, buffers["num_chunks"]
    )
    sync()


def _histogram_field_atomic(values, bins, workspace, method):
    if isinstance(values, Ndarray) or isinstance(bins, Ndarray):
        raise NotImplementedError(
            f"method='{method}' supports only ti.field values and bins."
        )
    n = values.shape[0]
    num_bins = bins.shape[0]
    if workspace is not None:
        workspace.check_shape(n, num_bins)
    if method == "field_private" or (
        method in ("auto", "field_atomic")
        and _histogram_should_use_private(n, num_bins)
    ):
        _histogram_field_private(values, bins, workspace, n, num_bins)
    else:
        _histogram_field_direct(values, bins, n, num_bins)
    return workspace


def experimental_histogram(values, bins, *, method="auto", workspace=None):
    """Count i32 bin ids in ``values`` into i32 ``bins``.

    ``values[i]`` is interpreted as a bin id. Values outside
    ``[0, bins.shape[0])`` are ignored by the field fallback. The CUDA CUB path
    is intended for inputs in that same range.
    """

    _check_histogram_request(values, bins, method, workspace)
    if workspace is None:
        workspace = HistogramWorkspace(max_items=values.shape[0], max_bins=bins.shape[0])
    workspace.check_shape(values.shape[0], bins.shape[0])
    if method in ("auto", "cuda_cub") and _try_cuda_cub_histogram(
        values, bins, workspace
    ):
        return
    if method == "cuda_cub":
        raise RuntimeError(
            "method='cuda_cub' requires CUDA ndarray inputs and available CUB DeviceHistogram."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_histogram(
        values, bins, workspace
    ):
        return
    if method == "vulkan_native":
        raise RuntimeError(
            "method='vulkan_native' requires Vulkan ndarray inputs and available native histogram."
        )
    if method in ("auto", "cpu_native") and _try_cpu_native_histogram(
        values, bins, workspace
    ):
        return
    if method == "cpu_native":
        raise RuntimeError(
            "method='cpu_native' requires CPU ndarray inputs and available native histogram."
        )
    _histogram_field_atomic(values, bins, workspace, method)


def _transform_value_type(dtype):
    if dtype == i32:
        return 0
    if dtype == f32:
        return 1
    raise TypeError("experimental_transform() currently supports ti.i32 and ti.f32.")


def _as_i32_transform_arg(name, value):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"experimental_transform() i32 {name} must be integral.") from exc
    if result != value:
        raise TypeError(f"experimental_transform() i32 {name} must be integral.")
    if result < -(1 << 31) or result > (1 << 31) - 1:
        raise ValueError(f"experimental_transform() i32 {name} is out of range.")
    return result


def _normalize_transform_args(dtype, scale, bias):
    if dtype == i32:
        return (
            _as_i32_transform_arg("scale", scale),
            _as_i32_transform_arg("bias", bias),
        )
    if dtype == f32:
        return float(scale), float(bias)
    raise TypeError("experimental_transform() currently supports ti.i32 and ti.f32.")


def _check_transform_request(src, dst, method, workspace):
    if method not in _SUPPORTED_TRANSFORM_METHODS:
        raise NotImplementedError(f"transform method '{method}' is not implemented.")
    if not (_is_1d(src) and _is_1d(dst)):
        raise ValueError("experimental_transform() expects 1D source and destination.")
    if src.shape[0] != dst.shape[0]:
        raise ValueError("experimental_transform() source and destination sizes differ.")
    if src.dtype != dst.dtype:
        raise TypeError("experimental_transform() source and destination dtype must match.")
    if src.dtype not in (i32, f32):
        raise TypeError("experimental_transform() currently supports ti.i32 and ti.f32.")
    if isinstance(src, Ndarray) or isinstance(dst, Ndarray):
        if not (isinstance(src, Ndarray) and isinstance(dst, Ndarray)):
            raise TypeError(
                "experimental_transform() ndarray mode requires source and "
                "destination both to be ti.ndarray."
            )
    if workspace is not None and not isinstance(workspace, TransformWorkspace):
        raise TypeError("workspace must be a TransformWorkspace instance or None.")


def _try_cuda_device_transform(src, dst, value_type, scale, bias):
    if current_cfg().arch != cuda:
        return False
    if not (isinstance(src, Ndarray) and isinstance(dst, Ndarray)):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cuda_device_transform_available"):
        return False
    if not prog.cuda_device_transform_available():
        return False
    prog.cuda_device_transform_affine_ndarray(src.arr, dst.arr, value_type, scale, bias)
    return True


def _try_vulkan_transform(src, dst, value_type, scale, bias, workspace):
    if current_cfg().arch != vulkan:
        return False
    if not (isinstance(src, Ndarray) and isinstance(dst, Ndarray)):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_transform_available"):
        return False
    if not prog.vulkan_transform_available():
        return False
    temp_bytes = prog.vulkan_transform_affine_ndarray(
        src.arr, dst.arr, value_type, scale, bias
    )
    if workspace is not None:
        workspace._vulkan_native_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _try_cpu_transform(src, dst, value_type, scale, bias):
    if current_cfg().arch not in [x64, arm64]:
        return False
    if not (isinstance(src, Ndarray) and isinstance(dst, Ndarray)):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_transform_available"):
        return False
    if not prog.cpu_transform_available():
        return False
    prog.cpu_transform_affine_ndarray(src.arr, dst.arr, value_type, scale, bias)
    return True


def _transform_kernel(src, dst, scale, bias):
    n = src.shape[0]
    if isinstance(src, Ndarray):
        if src.dtype == i32:
            transform_affine_i32_ndarray(src, dst, scale, bias, n)
        else:
            transform_affine_f32_ndarray(src, dst, scale, bias, n)
    else:
        if src.dtype == i32:
            transform_affine_i32_field(src, dst, scale, bias, n)
        else:
            transform_affine_f32_field(src, dst, scale, bias, n)
    sync()


def experimental_transform(
    src,
    dst,
    *,
    scale=1,
    bias=0,
    method="auto",
    workspace=None,
):
    """Apply ``dst[i] = src[i] * scale + bias`` to a 1D array.

    This is an experimental primitive. Contiguous ndarray inputs route to
    backend native implementations when available: CUDA uses driver-level
    device API/PTX, Vulkan uses compute shaders, and CPU uses a host native
    loop. Field/SNode fallback stays in Forge kernels to preserve layout and
    offset semantics.
    """

    _check_transform_request(src, dst, method, workspace)
    if workspace is None:
        workspace = TransformWorkspace(max_items=src.shape[0])
    workspace.check_shape(src.shape[0])
    scale, bias = _normalize_transform_args(src.dtype, scale, bias)
    value_type = _transform_value_type(src.dtype)
    if src.shape[0] == 0:
        return workspace
    if method in ("auto", "cuda_device") and _try_cuda_device_transform(
        src, dst, value_type, scale, bias
    ):
        return workspace
    if method == "cuda_device":
        raise RuntimeError(
            "method='cuda_device' requires CUDA ndarray inputs and available "
            "CUDA driver transform support."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_transform(
        src, dst, value_type, scale, bias, workspace
    ):
        return workspace
    if method == "vulkan_native":
        raise RuntimeError(
            "method='vulkan_native' requires Vulkan ndarray inputs and available "
            "native transform shaders."
        )
    if method in ("auto", "cpu_native") and _try_cpu_transform(
        src, dst, value_type, scale, bias
    ):
        return workspace
    if method == "cpu_native":
        raise RuntimeError(
            "method='cpu_native' requires CPU ndarray inputs and available native "
            "transform."
        )
    if method in ("kernel", "field_kernel", "auto"):
        _transform_kernel(src, dst, scale, bias)
        return workspace
    raise RuntimeError("experimental_transform() could not find an available backend.")


def _check_indexed_copy_request(src, indices, dst, method, workspace, op_name):
    if method not in _SUPPORTED_INDEXED_COPY_METHODS:
        raise NotImplementedError(f"{op_name} method '{method}' is not implemented.")
    if not (_is_1d(src) and _is_1d(indices) and _is_1d(dst)):
        raise ValueError(f"{op_name} expects 1D source, indices, and destination.")
    if indices.dtype != i32:
        raise TypeError(f"{op_name} currently expects ti.i32 indices.")
    if src.dtype != dst.dtype:
        raise TypeError(f"{op_name} source and destination dtype must match.")
    if src.dtype not in (i32, f32):
        raise TypeError(f"{op_name} currently supports ti.i32 and ti.f32 values.")
    if isinstance(src, Ndarray) or isinstance(indices, Ndarray) or isinstance(dst, Ndarray):
        if not (
            isinstance(src, Ndarray)
            and isinstance(indices, Ndarray)
            and isinstance(dst, Ndarray)
        ):
            raise TypeError(
                f"{op_name} ndarray mode requires source, indices, and "
                "destination all to be ti.ndarray."
            )
    if workspace is not None and not isinstance(workspace, IndexedCopyWorkspace):
        raise TypeError("workspace must be an IndexedCopyWorkspace instance or None.")


def _indexed_copy_item_count(src, indices, dst, scatter):
    if scatter:
        if src.shape[0] != indices.shape[0]:
            raise ValueError(
                "experimental_scatter() expects source and indices sizes to match."
            )
    else:
        if indices.shape[0] != dst.shape[0]:
            raise ValueError(
                "experimental_gather() expects indices and destination sizes to match."
            )
    return indices.shape[0]


def _try_cuda_device_indexed_copy(src, indices, dst, scatter):
    if current_cfg().arch != cuda:
        return False
    if not (
        isinstance(src, Ndarray)
        and isinstance(indices, Ndarray)
        and isinstance(dst, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cuda_device_indexed_copy_available"):
        return False
    if not prog.cuda_device_indexed_copy_available():
        return False
    if scatter:
        prog.cuda_device_scatter_ndarray(src.arr, indices.arr, dst.arr)
    else:
        prog.cuda_device_gather_ndarray(src.arr, indices.arr, dst.arr)
    return True


def _try_vulkan_indexed_copy(src, indices, dst, scatter, workspace):
    if current_cfg().arch != vulkan:
        return False
    if not (
        isinstance(src, Ndarray)
        and isinstance(indices, Ndarray)
        and isinstance(dst, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_indexed_copy_available"):
        return False
    if not prog.vulkan_indexed_copy_available():
        return False
    temp_bytes = (
        prog.vulkan_scatter_ndarray(src.arr, indices.arr, dst.arr)
        if scatter
        else prog.vulkan_gather_ndarray(src.arr, indices.arr, dst.arr)
    )
    if workspace is not None:
        workspace._vulkan_native_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _try_cpu_indexed_copy(src, indices, dst, scatter):
    if current_cfg().arch not in [x64, arm64]:
        return False
    if not (
        isinstance(src, Ndarray)
        and isinstance(indices, Ndarray)
        and isinstance(dst, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_indexed_copy_available"):
        return False
    if not prog.cpu_indexed_copy_available():
        return False
    if scatter:
        prog.cpu_scatter_ndarray(src.arr, indices.arr, dst.arr)
    else:
        prog.cpu_gather_ndarray(src.arr, indices.arr, dst.arr)
    return True


def _indexed_copy_kernel(src, indices, dst, scatter):
    n = indices.shape[0]
    if isinstance(src, Ndarray):
        if scatter:
            if src.dtype == i32:
                scatter_i32_ndarray(src, indices, dst, n)
            else:
                scatter_f32_ndarray(src, indices, dst, n)
        else:
            if src.dtype == i32:
                gather_i32_ndarray(src, indices, dst, n)
            else:
                gather_f32_ndarray(src, indices, dst, n)
    else:
        if scatter:
            if src.dtype == i32:
                scatter_i32_field(src, indices, dst, n)
            else:
                scatter_f32_field(src, indices, dst, n)
        else:
            if src.dtype == i32:
                gather_i32_field(src, indices, dst, n)
            else:
                gather_f32_field(src, indices, dst, n)
    sync()


def _experimental_indexed_copy(src, indices, dst, *, method, workspace, scatter):
    op_name = "experimental_scatter()" if scatter else "experimental_gather()"
    _check_indexed_copy_request(src, indices, dst, method, workspace, op_name)
    n = _indexed_copy_item_count(src, indices, dst, scatter)
    if workspace is None:
        workspace = IndexedCopyWorkspace(max_items=n)
    workspace.check_shape(n)
    if n == 0:
        return workspace
    if method in ("auto", "cuda_device") and _try_cuda_device_indexed_copy(
        src, indices, dst, scatter
    ):
        return workspace
    if method == "cuda_device":
        raise RuntimeError(
            f"{op_name} method='cuda_device' requires CUDA ndarray inputs and "
            "available CUDA driver indexed-copy support."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_indexed_copy(
        src, indices, dst, scatter, workspace
    ):
        return workspace
    if method == "vulkan_native":
        raise RuntimeError(
            f"{op_name} method='vulkan_native' requires Vulkan ndarray inputs "
            "and available native indexed-copy shaders."
        )
    if method in ("auto", "cpu_native") and _try_cpu_indexed_copy(
        src, indices, dst, scatter
    ):
        return workspace
    if method == "cpu_native":
        raise RuntimeError(
            f"{op_name} method='cpu_native' requires CPU ndarray inputs and "
            "available native indexed-copy support."
        )
    if method in ("kernel", "field_kernel", "auto"):
        _indexed_copy_kernel(src, indices, dst, scatter)
        return workspace
    raise RuntimeError(f"{op_name} could not find an available backend.")


def experimental_gather(src, indices, dst, *, method="auto", workspace=None):
    """Apply ``dst[i] = src[indices[i]]`` for 1D arrays.

    Indices must be valid. Native ndarray paths are provided for CUDA, Vulkan,
    and CPU. Field/SNode inputs use Forge kernels.
    """

    return _experimental_indexed_copy(
        src, indices, dst, method=method, workspace=workspace, scatter=False
    )


def experimental_scatter(src, indices, dst, *, method="auto", workspace=None):
    """Apply ``dst[indices[i]] = src[i]`` for 1D arrays.

    Indices must be valid and unique for deterministic native scatter. Duplicate
    write conflict handling belongs to future scatter-add / segmented-reduction
    primitives.
    """

    return _experimental_indexed_copy(
        src, indices, dst, method=method, workspace=workspace, scatter=True
    )


def _check_scatter_add_request(src, indices, dst, method, workspace):
    op_name = "experimental_scatter_add()"
    if method not in _SUPPORTED_SCATTER_ADD_METHODS:
        raise NotImplementedError(f"{op_name} method '{method}' is not implemented.")
    if not (_is_1d(src) and _is_1d(indices) and _is_1d(dst)):
        raise ValueError(f"{op_name} expects 1D source, indices, and destination.")
    if indices.dtype != i32:
        raise TypeError(f"{op_name} currently expects ti.i32 indices.")
    if src.dtype != dst.dtype:
        raise TypeError(f"{op_name} source and destination dtype must match.")
    if src.dtype not in (i32, f32):
        raise TypeError(f"{op_name} currently supports ti.i32 and ti.f32 values.")
    if src.shape[0] != indices.shape[0]:
        raise ValueError(f"{op_name} expects source and indices sizes to match.")
    if isinstance(src, Ndarray) or isinstance(indices, Ndarray) or isinstance(dst, Ndarray):
        if not (
            isinstance(src, Ndarray)
            and isinstance(indices, Ndarray)
            and isinstance(dst, Ndarray)
        ):
            raise TypeError(
                f"{op_name} ndarray mode requires source, indices, and "
                "destination all to be ti.ndarray."
            )
    if workspace is not None and not isinstance(workspace, ScatterAddWorkspace):
        raise TypeError("workspace must be a ScatterAddWorkspace instance or None.")


def _scatter_add_value_type(dtype):
    return 0 if dtype == i32 else 1


def _try_cuda_device_scatter_add(src, indices, dst):
    if current_cfg().arch != cuda:
        return False
    if not (
        isinstance(src, Ndarray)
        and isinstance(indices, Ndarray)
        and isinstance(dst, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cuda_device_scatter_add_available"):
        return False
    if not prog.cuda_device_scatter_add_available():
        return False
    prog.cuda_device_scatter_add_ndarray(
        src.arr, indices.arr, dst.arr, _scatter_add_value_type(src.dtype)
    )
    return True


def _try_vulkan_scatter_add(src, indices, dst, workspace):
    if current_cfg().arch != vulkan:
        return False
    if src.dtype != i32:
        return False
    if not (
        isinstance(src, Ndarray)
        and isinstance(indices, Ndarray)
        and isinstance(dst, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_scatter_add_available"):
        return False
    if not prog.vulkan_scatter_add_available():
        return False
    temp_bytes = prog.vulkan_scatter_add_ndarray(
        src.arr, indices.arr, dst.arr, _scatter_add_value_type(src.dtype)
    )
    if workspace is not None:
        workspace._vulkan_native_active = True
        workspace.workspace_bytes_current = max(
            workspace.workspace_bytes_current, temp_bytes
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak, workspace.workspace_bytes_current
        )
    return True


def _try_cpu_scatter_add(src, indices, dst):
    if current_cfg().arch not in [x64, arm64]:
        return False
    if not (
        isinstance(src, Ndarray)
        and isinstance(indices, Ndarray)
        and isinstance(dst, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_scatter_add_available"):
        return False
    if not prog.cpu_scatter_add_available():
        return False
    prog.cpu_scatter_add_ndarray(
        src.arr, indices.arr, dst.arr, _scatter_add_value_type(src.dtype)
    )
    return True


def _scatter_add_kernel(src, indices, dst):
    n = indices.shape[0]
    if isinstance(src, Ndarray):
        if src.dtype == i32:
            scatter_add_i32_ndarray(src, indices, dst, n)
        else:
            scatter_add_f32_ndarray(src, indices, dst, n)
    else:
        if src.dtype == i32:
            scatter_add_i32_field(src, indices, dst, n)
        else:
            scatter_add_f32_field(src, indices, dst, n)
    sync()


def experimental_scatter_add(src, indices, dst, *, method="auto", workspace=None):
    """Apply ``dst[indices[i]] += src[i]`` for 1D arrays.

    Invalid indices are ignored. Duplicate target indices are accumulated using
    backend atomics; floating-point accumulation order is backend-dependent.
    """

    _check_scatter_add_request(src, indices, dst, method, workspace)
    n = indices.shape[0]
    if workspace is None:
        workspace = ScatterAddWorkspace(max_items=n)
    workspace.check_shape(n)
    if n == 0:
        return workspace
    if method in ("auto", "cuda_device") and _try_cuda_device_scatter_add(
        src, indices, dst
    ):
        return workspace
    if method == "cuda_device":
        raise RuntimeError(
            "experimental_scatter_add() method='cuda_device' requires CUDA "
            "ndarray inputs and CUDA toolkit scatter-add support."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_scatter_add(
        src, indices, dst, workspace
    ):
        return workspace
    if method == "vulkan_native":
        raise RuntimeError(
            "experimental_scatter_add() method='vulkan_native' currently "
            "requires Vulkan ndarray inputs, i32 values, and available native "
            "scatter-add shaders. f32 uses the Forge kernel path on Vulkan to "
            "avoid unsafe high-contention CAS."
        )
    if method in ("auto", "cpu_native") and _try_cpu_scatter_add(src, indices, dst):
        return workspace
    if method == "cpu_native":
        raise RuntimeError(
            "experimental_scatter_add() method='cpu_native' requires CPU ndarray "
            "inputs and available native scatter-add support."
        )
    if method in ("kernel", "field_kernel", "auto"):
        _scatter_add_kernel(src, indices, dst)
        return workspace
    raise RuntimeError("experimental_scatter_add() could not find an available backend.")


def _check_bucket_builder_request(keys, values, offsets, output, method, workspace):
    if method not in _SUPPORTED_BUCKET_BUILDER_METHODS:
        raise NotImplementedError(
            f"bucket builder method '{method}' is not implemented."
        )
    if not (_is_1d(keys) and _is_1d(values) and _is_1d(offsets) and _is_1d(output)):
        raise ValueError(
            "experimental_bucket_builder() expects 1D keys, values, offsets, and output."
        )
    if keys.dtype != i32 or values.dtype != i32 or offsets.dtype != i32 or output.dtype != i32:
        raise TypeError(
            "experimental_bucket_builder() currently expects ti.i32 keys, values, offsets, and output."
        )
    if keys.shape[0] != values.shape[0]:
        raise ValueError("experimental_bucket_builder() keys and values sizes must match.")
    if offsets.shape[0] < 2:
        raise ValueError("experimental_bucket_builder() offsets must have at least 2 items.")
    if output.shape[0] < values.shape[0]:
        raise ValueError(
            "experimental_bucket_builder() output must have at least values length."
        )
    if isinstance(keys, Ndarray) or isinstance(values, Ndarray) or isinstance(offsets, Ndarray) or isinstance(output, Ndarray):
        if not (
            isinstance(keys, Ndarray)
            and isinstance(values, Ndarray)
            and isinstance(offsets, Ndarray)
            and isinstance(output, Ndarray)
        ):
            raise TypeError(
                "experimental_bucket_builder() ndarray mode requires all inputs "
                "and outputs to be ti.ndarray."
            )
    if workspace is not None and not isinstance(workspace, BucketBuilderWorkspace):
        raise TypeError("workspace must be a BucketBuilderWorkspace instance or None.")


def _try_cuda_device_bucket_builder(keys, values, offsets, output, workspace, num_bins):
    if current_cfg().arch != cuda:
        return False
    if not (
        isinstance(keys, Ndarray)
        and isinstance(values, Ndarray)
        and isinstance(offsets, Ndarray)
        and isinstance(output, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cuda_device_bucket_builder_available"):
        return False
    if not prog.cuda_device_bucket_builder_available():
        return False
    cursor = workspace._get_cursor_ndarray(num_bins)
    temp_bytes = prog.cuda_device_bucket_builder_i32_ndarray(
        keys.arr, values.arr, offsets.arr, output.arr, cursor.arr
    )
    workspace.workspace_bytes_peak = max(
        workspace.workspace_bytes_peak,
        workspace.workspace_bytes_current + temp_bytes,
    )
    return True


def _try_vulkan_bucket_builder(keys, values, offsets, output, workspace, num_bins):
    if current_cfg().arch != vulkan:
        return False
    if not (
        isinstance(keys, Ndarray)
        and isinstance(values, Ndarray)
        and isinstance(offsets, Ndarray)
        and isinstance(output, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_bucket_builder_available"):
        return False
    if not prog.vulkan_bucket_builder_available():
        return False
    cursor = workspace._get_cursor_ndarray(num_bins)
    temp_bytes = prog.vulkan_bucket_builder_i32_ndarray(
        keys.arr, values.arr, offsets.arr, output.arr, cursor.arr
    )
    workspace._vulkan_native_active = True
    workspace.workspace_bytes_peak = max(
        workspace.workspace_bytes_peak,
        workspace.workspace_bytes_current + temp_bytes,
    )
    return True


def _try_cpu_bucket_builder(keys, values, offsets, output, workspace):
    if current_cfg().arch not in [x64, arm64]:
        return False
    if not (
        isinstance(keys, Ndarray)
        and isinstance(values, Ndarray)
        and isinstance(offsets, Ndarray)
        and isinstance(output, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_bucket_builder_available"):
        return False
    if not prog.cpu_bucket_builder_available():
        return False
    temp_bytes = prog.cpu_bucket_builder_i32_ndarray(
        keys.arr, values.arr, offsets.arr, output.arr
    )
    workspace.workspace_bytes_peak = max(
        workspace.workspace_bytes_peak,
        workspace.workspace_bytes_current + temp_bytes,
    )
    return True


def _bucket_builder_kernel(keys, values, offsets, output, workspace, num_bins):
    n = keys.shape[0]
    if isinstance(keys, Ndarray):
        cursor = workspace._get_cursor_ndarray(num_bins)
        bucket_count_i32_ndarray(keys, offsets, n, num_bins)
        PrefixSumExecutor(num_bins + 1).run(offsets)
        bucket_copy_offsets_to_cursor_ndarray(offsets, cursor, num_bins)
        bucket_scatter_i32_ndarray(keys, values, cursor, output, n, num_bins)
    else:
        cursor = workspace._get_cursor_field(num_bins)
        bucket_count_i32_field(keys, offsets, n, num_bins)
        if current_cfg().arch in [x64, arm64]:
            bucket_prefix_offsets_i32_field_serial(offsets, num_bins)
        else:
            scanner = workspace._get_scanner(num_bins + 1)
            scanner.run(offsets)
        bucket_copy_offsets_to_cursor_field(offsets, cursor, num_bins)
        bucket_scatter_i32_field(keys, values, cursor, output, n, num_bins)
    sync()


def experimental_bucket_builder(
    keys, values, offsets, output, *, method="auto", workspace=None
):
    """Build fixed-bin bucket ranges and compacted values.

    ``keys[i]`` is interpreted as a bucket id. Valid ids are in
    ``[0, offsets.shape[0] - 1)``. Invalid keys are ignored. On return,
    ``offsets`` has length ``num_bins + 1`` and stores exclusive bucket
    ranges; ``output[offsets[b]:offsets[b + 1]]`` contains values for bucket
    ``b`` in an unspecified order.
    """

    _check_bucket_builder_request(keys, values, offsets, output, method, workspace)
    n = keys.shape[0]
    num_bins = offsets.shape[0] - 1
    if workspace is None:
        workspace = BucketBuilderWorkspace(max_items=n, max_bins=num_bins)
    workspace.check_shape(n, num_bins)
    if method in ("auto", "cuda_device") and _try_cuda_device_bucket_builder(
        keys, values, offsets, output, workspace, num_bins
    ):
        return workspace
    if method == "cuda_device":
        raise RuntimeError(
            "experimental_bucket_builder() method='cuda_device' requires CUDA "
            "ndarray inputs and CUDA toolkit bucket-builder support."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_bucket_builder(
        keys, values, offsets, output, workspace, num_bins
    ):
        return workspace
    if method == "vulkan_native":
        raise RuntimeError(
            "experimental_bucket_builder() method='vulkan_native' requires Vulkan "
            "ndarray inputs and available native bucket-builder shaders."
        )
    if method in ("auto", "cpu_native") and _try_cpu_bucket_builder(
        keys, values, offsets, output, workspace
    ):
        return workspace
    if method == "cpu_native":
        raise RuntimeError(
            "experimental_bucket_builder() method='cpu_native' requires CPU ndarray "
            "inputs and available native bucket-builder support."
        )
    if method in ("kernel", "field_kernel", "auto"):
        _bucket_builder_kernel(keys, values, offsets, output, workspace, num_bins)
        return workspace
    raise RuntimeError("experimental_bucket_builder() could not find an available backend.")


def _check_grouped_reduce_request(keys, values, output, op, method, workspace):
    if method not in _SUPPORTED_GROUPED_REDUCE_METHODS:
        raise NotImplementedError(f"grouped reduce method '{method}' is not implemented.")
    if op not in _SUPPORTED_GROUPED_REDUCE_OPS:
        raise ValueError(
            f"grouped reduce op must be one of {sorted(_SUPPORTED_GROUPED_REDUCE_OPS)}."
        )
    if not (_is_1d(keys) and _is_1d(values) and _is_1d(output)):
        raise ValueError("experimental_grouped_reduce() expects 1D keys, values, and output.")
    if keys.dtype != i32:
        raise TypeError("experimental_grouped_reduce() currently expects ti.i32 keys.")
    if values.dtype != i32 or output.dtype != i32:
        raise TypeError(
            "experimental_grouped_reduce() currently expects ti.i32 values and output."
        )
    if keys.shape[0] != values.shape[0]:
        raise ValueError("experimental_grouped_reduce() keys and values sizes must match.")
    if output.shape[0] <= 0:
        raise ValueError("experimental_grouped_reduce() output must contain at least one group.")
    if isinstance(keys, Ndarray) or isinstance(values, Ndarray) or isinstance(output, Ndarray):
        if not (
            isinstance(keys, Ndarray)
            and isinstance(values, Ndarray)
            and isinstance(output, Ndarray)
        ):
            raise TypeError(
                "experimental_grouped_reduce() ndarray mode requires keys, values, "
                "and output all to be ti.ndarray."
            )
    if workspace is not None and not isinstance(workspace, GroupedReduceWorkspace):
        raise TypeError("workspace must be a GroupedReduceWorkspace instance or None.")


def _try_cuda_device_grouped_reduce(
    keys, values, output, workspace, num_groups, op, *, segmented=False
):
    if current_cfg().arch != cuda:
        return False
    if not (
        isinstance(keys, Ndarray)
        and isinstance(values, Ndarray)
        and isinstance(output, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cuda_device_grouped_reduce_available"):
        return False
    if not prog.cuda_device_grouped_reduce_available():
        return False
    if not segmented and hasattr(prog, "cuda_device_grouped_reduce_i32_atomic_ndarray"):
        temp_bytes = prog.cuda_device_grouped_reduce_i32_atomic_ndarray(
            keys.arr,
            values.arr,
            output.arr,
            _SUPPORTED_GROUPED_REDUCE_OPS[op],
        )
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak,
            workspace.workspace_bytes_current + temp_bytes,
        )
        return True
    if not hasattr(prog, "cuda_device_grouped_reduce_i32_ndarray"):
        return False
    offsets, scratch, cursor = workspace._get_native_buffers(keys.shape[0], num_groups)
    temp_bytes = prog.cuda_device_grouped_reduce_i32_ndarray(
        keys.arr,
        values.arr,
        output.arr,
        offsets.arr,
        scratch.arr,
        cursor.arr,
        _SUPPORTED_GROUPED_REDUCE_OPS[op],
    )
    workspace.workspace_bytes_peak = max(
        workspace.workspace_bytes_peak,
        workspace.workspace_bytes_current + temp_bytes,
    )
    return True


def _try_vulkan_grouped_reduce(
    keys, values, output, workspace, num_groups, op, *, segmented=False
):
    if current_cfg().arch != vulkan:
        return False
    if not (
        isinstance(keys, Ndarray)
        and isinstance(values, Ndarray)
        and isinstance(output, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "vulkan_grouped_reduce_available"):
        return False
    if not prog.vulkan_grouped_reduce_available():
        return False
    if not segmented and hasattr(prog, "vulkan_grouped_reduce_i32_atomic_ndarray"):
        temp_bytes = prog.vulkan_grouped_reduce_i32_atomic_ndarray(
            keys.arr,
            values.arr,
            output.arr,
            _SUPPORTED_GROUPED_REDUCE_OPS[op],
        )
        workspace._vulkan_native_active = True
        workspace.workspace_bytes_peak = max(
            workspace.workspace_bytes_peak,
            workspace.workspace_bytes_current + temp_bytes,
        )
        return True
    if not hasattr(prog, "vulkan_grouped_reduce_i32_ndarray"):
        return False
    offsets, scratch, cursor = workspace._get_native_buffers(keys.shape[0], num_groups)
    temp_bytes = prog.vulkan_grouped_reduce_i32_ndarray(
        keys.arr,
        values.arr,
        output.arr,
        offsets.arr,
        scratch.arr,
        cursor.arr,
        _SUPPORTED_GROUPED_REDUCE_OPS[op],
    )
    workspace._vulkan_native_active = True
    workspace.workspace_bytes_peak = max(
        workspace.workspace_bytes_peak,
        workspace.workspace_bytes_current + temp_bytes,
    )
    return True


def _try_cpu_grouped_reduce(keys, values, output, workspace, op):
    if current_cfg().arch not in [x64, arm64]:
        return False
    if not (
        isinstance(keys, Ndarray)
        and isinstance(values, Ndarray)
        and isinstance(output, Ndarray)
    ):
        return False
    from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

    prog = impl.get_runtime().prog
    if not hasattr(prog, "cpu_grouped_reduce_available"):
        return False
    if not prog.cpu_grouped_reduce_available():
        return False
    temp_bytes = prog.cpu_grouped_reduce_i32_ndarray(
        keys.arr, values.arr, output.arr, _SUPPORTED_GROUPED_REDUCE_OPS[op]
    )
    workspace.workspace_bytes_peak = max(
        workspace.workspace_bytes_peak,
        workspace.workspace_bytes_current + temp_bytes,
    )
    return True


def _grouped_reduce_kernel(keys, values, output):
    n = keys.shape[0]
    num_groups = output.shape[0]
    if isinstance(keys, Ndarray):
        grouped_reduce_sum_i32_ndarray(keys, values, output, n, num_groups)
    else:
        grouped_reduce_sum_i32_field(keys, values, output, n, num_groups)
    sync()


def experimental_grouped_reduce(
    keys, values, output, *, op="sum", method="auto", workspace=None
):
    """Reduce values into fixed groups selected by ``keys``.

    Current scope is i32 sum. Invalid negative or out-of-range keys are ignored;
    empty groups produce zero. The default native ndarray paths use direct
    atomic accumulation to avoid distribution-dependent bucket overhead. The
    explicit ``method="segmented"`` routes through bucket ranges plus a
    per-group reduction, while field/SNode fallback stays in Forge kernels.
    """

    _check_grouped_reduce_request(keys, values, output, op, method, workspace)
    n = keys.shape[0]
    num_groups = output.shape[0]
    if workspace is None:
        workspace = GroupedReduceWorkspace(max_items=n, max_groups=num_groups)
    workspace.check_shape(n, num_groups)
    if method in ("auto", "cuda_device") and _try_cuda_device_grouped_reduce(
        keys, values, output, workspace, num_groups, op, segmented=False
    ):
        return workspace
    if method in ("segmented", "cuda_segmented") and _try_cuda_device_grouped_reduce(
        keys, values, output, workspace, num_groups, op, segmented=True
    ):
        return workspace
    if method == "cuda_segmented":
        raise RuntimeError(
            "experimental_grouped_reduce() method='cuda_segmented' requires CUDA "
            "ndarray inputs and CUDA toolkit segmented grouped-reduce support."
        )
    if method == "cuda_device":
        raise RuntimeError(
            "experimental_grouped_reduce() method='cuda_device' requires CUDA "
            "ndarray inputs and CUDA toolkit grouped-reduce support."
        )
    if method in ("auto", "vulkan_native") and _try_vulkan_grouped_reduce(
        keys, values, output, workspace, num_groups, op, segmented=False
    ):
        return workspace
    if method in ("segmented", "vulkan_segmented") and _try_vulkan_grouped_reduce(
        keys, values, output, workspace, num_groups, op, segmented=True
    ):
        return workspace
    if method == "vulkan_segmented":
        raise RuntimeError(
            "experimental_grouped_reduce() method='vulkan_segmented' requires Vulkan "
            "ndarray inputs and available native segmented grouped-reduce shaders."
        )
    if method == "vulkan_native":
        raise RuntimeError(
            "experimental_grouped_reduce() method='vulkan_native' requires Vulkan "
            "ndarray inputs and available native grouped-reduce shaders."
        )
    if method == "segmented":
        raise RuntimeError(
            "experimental_grouped_reduce() method='segmented' requires CUDA or "
            "Vulkan ndarray native segmented grouped-reduce support."
        )
    if method in ("auto", "cpu_native") and _try_cpu_grouped_reduce(
        keys, values, output, workspace, op
    ):
        return workspace
    if method == "cpu_native":
        raise RuntimeError(
            "experimental_grouped_reduce() method='cpu_native' requires CPU ndarray "
            "inputs and available native grouped-reduce support."
        )
    if method in ("kernel", "field_kernel", "auto"):
        _grouped_reduce_kernel(keys, values, output)
        return workspace
    raise RuntimeError("experimental_grouped_reduce() could not find an available backend.")


def parallel_sort(keys, values=None):
    """Compatibility wrapper for the legacy public sorting API."""

    sort(keys, values, stable=True, method="legacy", precision="exact")


@data_oriented
class PrefixSumExecutor:
    """Parallel Prefix Sum (Scan) Helper

    Use this helper to perform an inclusive in-place's parallel prefix sum.

    References:
        https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
        https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu
    """

    def __init__(self, length):
        self.sorting_length = length

        BLOCK_SZ = 256 if current_cfg().arch == cuda and length >= 65536 else 64
        self.block_sz = BLOCK_SZ
        GRID_SZ = int((length + BLOCK_SZ - 1) / BLOCK_SZ)

        # Buffer position and length
        # This is a single buffer implementation for ease of aot usage
        ele_num = length
        self.ele_nums = [ele_num]
        start_pos = 0
        self.ele_nums_pos = [start_pos]

        while ele_num > 1:
            ele_num = int((ele_num + BLOCK_SZ - 1) / BLOCK_SZ)
            self.ele_nums.append(ele_num)
            start_pos += BLOCK_SZ * ele_num
            self.ele_nums_pos.append(start_pos)

        self.workspace_length = start_pos
        self.large_arr = None

    def _ensure_large_arr(self):
        if self.large_arr is None:
            self.large_arr = field(i32, shape=self.workspace_length)
        return self.large_arr

    def _try_cuda_cub_scan(self, input_arr):
        if current_cfg().arch != cuda:
            return False
        if not isinstance(input_arr, Ndarray):
            return False
        if input_arr.dtype != i32:
            return False
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if not hasattr(prog, "cuda_cub_scan_available"):
            return False
        if not prog.cuda_cub_scan_available():
            return False
        prog.cuda_cub_inclusive_scan_ndarray(input_arr.arr, 0)
        return True

    def _try_vulkan_native_scan(self, input_arr):
        if current_cfg().arch != vulkan:
            return False
        if not isinstance(input_arr, Ndarray):
            return False
        if input_arr.dtype != i32:
            return False
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if not hasattr(prog, "vulkan_scan_available"):
            return False
        if not prog.vulkan_scan_available():
            return False
        prog.vulkan_inclusive_scan_ndarray(input_arr.arr, 0)
        return True

    def _try_cpu_native_scan(self, input_arr):
        if current_cfg().arch not in [x64, arm64]:
            return False
        if not isinstance(input_arr, Ndarray):
            return False
        if input_arr.dtype != i32:
            return False
        from taichi_forge.lang import impl  # pylint: disable=import-outside-toplevel

        prog = impl.get_runtime().prog
        if not hasattr(prog, "cpu_scan_available"):
            return False
        if not prog.cpu_scan_available():
            return False
        prog.cpu_inclusive_scan_ndarray(input_arr.arr, 0)
        return True

    def _run_field_workspace(self, large_arr):
        ele_nums = self.ele_nums
        ele_nums_pos = self.ele_nums_pos
        if current_cfg().arch == cuda:
            inclusive_add = warp_shfl_up_i32
            use_cuda_large_scan = self.block_sz == 256
            scan_kernel = scan_add_inclusive_cuda if use_cuda_large_scan else scan_add_inclusive
            uniform_kernel = uniform_add_cuda if use_cuda_large_scan else uniform_add
        elif current_cfg().arch == vulkan:
            inclusive_add = subgroup.inclusive_add
            use_cuda_large_scan = False
            scan_kernel = scan_add_inclusive
            uniform_kernel = uniform_add
        else:
            raise RuntimeError(f"{str(current_cfg().arch)} is not supported for prefix sum.")

        for i in range(len(ele_nums) - 1):
            single_block = i == len(ele_nums) - 2
            if use_cuda_large_scan:
                scan_kernel(large_arr, ele_nums_pos[i], ele_nums_pos[i + 1], single_block)
            else:
                scan_kernel(
                    large_arr,
                    ele_nums_pos[i],
                    ele_nums_pos[i + 1],
                    single_block,
                    inclusive_add,
                )

        for i in range(len(ele_nums) - 3, -1, -1):
            uniform_kernel(large_arr, ele_nums_pos[i], ele_nums_pos[i + 1])

    def run(self, input_arr):
        length = self.sorting_length

        if input_arr.dtype != i32:
            raise RuntimeError("Only ti.i32 type is supported for prefix sum.")
        if self._try_cuda_cub_scan(input_arr):
            return
        if self._try_vulkan_native_scan(input_arr):
            return
        if self._try_cpu_native_scan(input_arr):
            return
        if isinstance(input_arr, Ndarray):
            raise RuntimeError(
                "PrefixSumExecutor ndarray input is currently supported only "
                "by native CPU/CUDA/Vulkan scan fast paths. Ensure the backend "
                "runtime primitive is available, or use a field input."
            )

        large_arr = self._ensure_large_arr()
        blit_from_field_to_field(large_arr, input_arr, 0, length)
        self._run_field_workspace(large_arr)
        blit_from_field_to_field(input_arr, large_arr, 0, length)


__all__ = [
    "parallel_sort",
    "sort",
    "sort_by_key",
    "SortWorkspace",
    "PrefixSumExecutor",
    "CompactWorkspace",
    "ReduceWorkspace",
    "HistogramWorkspace",
    "TransformWorkspace",
    "IndexedCopyWorkspace",
    "ScatterAddWorkspace",
    "BucketBuilderWorkspace",
    "GroupedReduceWorkspace",
    "experimental_compact",
    "experimental_reduce",
    "experimental_histogram",
    "experimental_transform",
    "experimental_gather",
    "experimental_scatter",
    "experimental_scatter_add",
    "experimental_bucket_builder",
    "experimental_grouped_reduce",
]
