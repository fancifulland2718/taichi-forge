from taichi_forge._funcs import field_fill_taichi_scope
from taichi_forge._lib.utils import get_os_name
from taichi_forge.lang import ops
from taichi_forge.lang._ndrange import ndrange
from taichi_forge.lang.enums import Format
from taichi_forge.lang.expr import Expr
from taichi_forge.lang.field import ScalarField
from taichi_forge.lang.impl import grouped, static, static_assert
from taichi_forge.lang.kernel_impl import func, kernel
from taichi_forge.lang.misc import loop_config
from taichi_forge.lang.simt import block, subgroup, warp
from taichi_forge.lang.snode import deactivate
from taichi_forge.types import ndarray_type, texture_type, vector
from taichi_forge.types.annotations import template
from taichi_forge.types.primitive_types import f16, f32, f64, i32, u8, u32

from taichi_forge.math import vec3


_field_compact_static_kernel_cache = {}


# A set of helper (meta)functions
@kernel
def fill_field(field: template(), val: template()):
    value = ops.cast(val, field.dtype)
    for I in grouped(field):
        field[I] = value


@kernel
def fill_ndarray(ndarray: ndarray_type.ndarray(), val: template()):
    for I in grouped(ndarray):
        ndarray[I] = val


@kernel
def fill_ndarray_matrix(ndarray: ndarray_type.ndarray(), val: template()):
    for I in grouped(ndarray):
        ndarray[I] = val


@kernel
def tensor_to_ext_arr(tensor: template(), arr: ndarray_type.ndarray()):
    # default value of offset is [], replace it with [0] * len
    offset = static(tensor.snode.ptr.offset if len(tensor.snode.ptr.offset) != 0 else [0] * len(tensor.shape))

    for I in grouped(tensor):
        arr[I - offset] = tensor[I]


@kernel
def ndarray_to_ext_arr(ndarray: ndarray_type.ndarray(), arr: ndarray_type.ndarray()):
    for I in grouped(ndarray):
        arr[I] = ndarray[I]


@kernel
def ndarray_matrix_to_ext_arr(
    ndarray: ndarray_type.ndarray(),
    arr: ndarray_type.ndarray(),
    layout_is_aos: template(),
    as_vector: template(),
):
    for I in grouped(ndarray):
        for p in static(range(ndarray[I].n)):
            if static(as_vector):
                if static(layout_is_aos):
                    arr[I, p] = ndarray[I][p]
                else:
                    arr[p, I] = ndarray[I][p]
            else:
                for q in static(range(ndarray[I].m)):
                    if static(layout_is_aos):
                        arr[I, p, q] = ndarray[I][p, q]
                    else:
                        arr[p, q, I] = ndarray[I][p, q]


@kernel
def vector_to_fast_image(img: template(), out: ndarray_type.ndarray()):
    static_assert(len(img.shape) == 2)
    offset = static(img.snode.ptr.offset if len(img.snode.ptr.offset) != 0 else [0, 0])
    i_offset = static(offset[0])
    j_offset = static(offset[1])
    # FIXME: Why is ``for i, j in img:`` slower than:
    for i, j in ndrange(*img.shape):
        r, g, b = 0, 0, 0
        color = img[i + i_offset, (img.shape[1] + j_offset) - 1 - j]
        if static(img.dtype in [f16, f32, f64]):
            r, g, b = ops.min(255, ops.max(0, int(color * 255)))[:3]
        else:
            static_assert(img.dtype == u8)
            r, g, b = color[:3]

        idx = j * img.shape[0] + i
        # We use i32 for |out| since OpenGL and Metal doesn't support u8 types
        if static(get_os_name() != "osx"):
            out[idx] = (r << 16) + (g << 8) + b
        else:
            # What's -16777216?
            #
            # On Mac, we need to set the alpha channel to 0xff. Since Mac's GUI
            # is big-endian, the color is stored in ABGR order, and we need to
            # add 0xff000000, which is -16777216 in I32's legit range. (Albeit
            # the clarity, adding 0xff000000 doesn't work.)
            alpha = -16777216
            out[idx] = (b << 16) + (g << 8) + r + alpha


@kernel
def tensor_to_image(tensor: template(), arr: ndarray_type.ndarray()):
    # default value of offset is [], replace it with [0] * len
    offset = static(tensor.snode.ptr.offset if len(tensor.snode.ptr.offset) != 0 else [0] * len(tensor.shape))
    for I in grouped(tensor):
        t = ops.cast(tensor[I], f32)
        arr[I - offset, 0] = t
        arr[I - offset, 1] = t
        arr[I - offset, 2] = t


@kernel
def vector_to_image(mat: template(), arr: ndarray_type.ndarray()):
    # default value of offset is [], replace it with [0] * len
    offset = static(mat.snode.ptr.offset if len(mat.snode.ptr.offset) != 0 else [0] * len(mat.shape))
    for I in grouped(mat):
        for p in static(range(mat.n)):
            arr[I - offset, p] = ops.cast(mat[I][p], f32)
            if static(mat.n <= 2):
                arr[I - offset, 2] = 0


@kernel
def tensor_to_tensor(tensor: template(), other: template()):
    static_assert(tensor.shape == other.shape)
    shape = static(tensor.shape)
    tensor_offset = static(tensor.snode.ptr.offset if len(tensor.snode.ptr.offset) != 0 else [0] * len(shape))
    other_offset = static(other.snode.ptr.offset if len(other.snode.ptr.offset) != 0 else [0] * len(shape))

    for I in grouped(ndrange(*shape)):
        tensor[I + tensor_offset] = other[I + other_offset]


@kernel
def ext_arr_to_tensor(arr: ndarray_type.ndarray(), tensor: template()):
    # default value of offset is [], replace it with [0] * len
    offset = static(tensor.snode.ptr.offset if len(tensor.snode.ptr.offset) != 0 else [0] * len(tensor.shape))
    for I in grouped(tensor):
        tensor[I] = arr[I - offset]


@kernel
def ndarray_to_ndarray(ndarray: ndarray_type.ndarray(), other: ndarray_type.ndarray()):
    for I in grouped(ndarray):
        ndarray[I] = other[I]


@kernel
def transform_affine_i32_ndarray(
    src: ndarray_type.ndarray(dtype=i32, ndim=1),
    dst: ndarray_type.ndarray(dtype=i32, ndim=1),
    scale: i32,
    bias: i32,
    n: i32,
):
    for i in range(n):
        dst[i] = src[i] * scale + bias


@kernel
def transform_affine_f32_ndarray(
    src: ndarray_type.ndarray(dtype=f32, ndim=1),
    dst: ndarray_type.ndarray(dtype=f32, ndim=1),
    scale: f32,
    bias: f32,
    n: i32,
):
    for i in range(n):
        dst[i] = src[i] * scale + bias


@kernel
def transform_affine_i32_field(
    src: template(),
    dst: template(),
    scale: i32,
    bias: i32,
    n: i32,
):
    src_offset = static(src.snode.ptr.offset if len(src.snode.ptr.offset) != 0 else [0])
    dst_offset = static(dst.snode.ptr.offset if len(dst.snode.ptr.offset) != 0 else [0])
    for i in range(n):
        dst[i + dst_offset[0]] = src[i + src_offset[0]] * scale + bias


@kernel
def transform_affine_f32_field(
    src: template(),
    dst: template(),
    scale: f32,
    bias: f32,
    n: i32,
):
    src_offset = static(src.snode.ptr.offset if len(src.snode.ptr.offset) != 0 else [0])
    dst_offset = static(dst.snode.ptr.offset if len(dst.snode.ptr.offset) != 0 else [0])
    for i in range(n):
        dst[i + dst_offset[0]] = src[i + src_offset[0]] * scale + bias


@kernel
def gather_i32_ndarray(
    src: ndarray_type.ndarray(dtype=i32, ndim=1),
    indices: ndarray_type.ndarray(dtype=i32, ndim=1),
    dst: ndarray_type.ndarray(dtype=i32, ndim=1),
    n: i32,
):
    for i in range(n):
        index = indices[i]
        if index >= 0 and index < src.shape[0]:
            dst[i] = src[index]
        else:
            dst[i] = 0


@kernel
def gather_f32_ndarray(
    src: ndarray_type.ndarray(dtype=f32, ndim=1),
    indices: ndarray_type.ndarray(dtype=i32, ndim=1),
    dst: ndarray_type.ndarray(dtype=f32, ndim=1),
    n: i32,
):
    for i in range(n):
        index = indices[i]
        if index >= 0 and index < src.shape[0]:
            dst[i] = src[index]
        else:
            dst[i] = 0.0


@kernel
def scatter_i32_ndarray(
    src: ndarray_type.ndarray(dtype=i32, ndim=1),
    indices: ndarray_type.ndarray(dtype=i32, ndim=1),
    dst: ndarray_type.ndarray(dtype=i32, ndim=1),
    n: i32,
):
    for i in range(n):
        index = indices[i]
        if index >= 0 and index < dst.shape[0]:
            dst[index] = src[i]


@kernel
def scatter_f32_ndarray(
    src: ndarray_type.ndarray(dtype=f32, ndim=1),
    indices: ndarray_type.ndarray(dtype=i32, ndim=1),
    dst: ndarray_type.ndarray(dtype=f32, ndim=1),
    n: i32,
):
    for i in range(n):
        index = indices[i]
        if index >= 0 and index < dst.shape[0]:
            dst[index] = src[i]


@kernel
def scatter_add_i32_ndarray(
    src: ndarray_type.ndarray(dtype=i32, ndim=1),
    indices: ndarray_type.ndarray(dtype=i32, ndim=1),
    dst: ndarray_type.ndarray(dtype=i32, ndim=1),
    n: i32,
):
    for i in range(n):
        index = indices[i]
        if index >= 0 and index < dst.shape[0]:
            ops.atomic_add(dst[index], src[i])


@kernel
def scatter_add_f32_ndarray(
    src: ndarray_type.ndarray(dtype=f32, ndim=1),
    indices: ndarray_type.ndarray(dtype=i32, ndim=1),
    dst: ndarray_type.ndarray(dtype=f32, ndim=1),
    n: i32,
):
    for i in range(n):
        index = indices[i]
        if index >= 0 and index < dst.shape[0]:
            ops.atomic_add(dst[index], src[i])


@kernel
def bucket_count_i32_ndarray(
    keys: ndarray_type.ndarray(dtype=i32, ndim=1),
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    n: i32,
    num_bins: i32,
):
    for i in range(num_bins + 1):
        offsets[i] = 0
    for i in range(n):
        key = keys[i]
        if key >= 0 and key < num_bins:
            ops.atomic_add(offsets[key + 1], 1)


@kernel
def bucket_copy_offsets_to_cursor_ndarray(
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    cursor: ndarray_type.ndarray(dtype=i32, ndim=1),
    num_bins: i32,
):
    for i in range(num_bins):
        cursor[i] = offsets[i]


@kernel
def bucket_scatter_i32_ndarray(
    keys: ndarray_type.ndarray(dtype=i32, ndim=1),
    values: ndarray_type.ndarray(dtype=i32, ndim=1),
    cursor: ndarray_type.ndarray(dtype=i32, ndim=1),
    output: ndarray_type.ndarray(dtype=i32, ndim=1),
    n: i32,
    num_bins: i32,
):
    for i in range(n):
        key = keys[i]
        if key >= 0 and key < num_bins:
            out_idx = ops.atomic_add(cursor[key], 1)
            if out_idx >= 0 and out_idx < output.shape[0]:
                output[out_idx] = values[i]


@kernel
def grouped_reduce_sum_i32_ndarray(
    keys: ndarray_type.ndarray(dtype=i32, ndim=1),
    values: ndarray_type.ndarray(dtype=i32, ndim=1),
    output: ndarray_type.ndarray(dtype=i32, ndim=1),
    n: i32,
    num_groups: i32,
):
    for i in range(num_groups):
        output[i] = 0
    for i in range(n):
        key = keys[i]
        if key >= 0 and key < num_groups:
            ops.atomic_add(output[key], values[i])


@kernel
def gather_i32_field(src: template(), indices: template(), dst: template(), n: i32):
    src_offset = static(src.snode.ptr.offset if len(src.snode.ptr.offset) != 0 else [0])
    indices_offset = static(
        indices.snode.ptr.offset if len(indices.snode.ptr.offset) != 0 else [0]
    )
    dst_offset = static(dst.snode.ptr.offset if len(dst.snode.ptr.offset) != 0 else [0])
    for i in range(n):
        index = indices[i + indices_offset[0]]
        if index >= 0 and index < src.shape[0]:
            dst[i + dst_offset[0]] = src[index + src_offset[0]]
        else:
            dst[i + dst_offset[0]] = 0


@kernel
def gather_f32_field(src: template(), indices: template(), dst: template(), n: i32):
    src_offset = static(src.snode.ptr.offset if len(src.snode.ptr.offset) != 0 else [0])
    indices_offset = static(
        indices.snode.ptr.offset if len(indices.snode.ptr.offset) != 0 else [0]
    )
    dst_offset = static(dst.snode.ptr.offset if len(dst.snode.ptr.offset) != 0 else [0])
    for i in range(n):
        index = indices[i + indices_offset[0]]
        if index >= 0 and index < src.shape[0]:
            dst[i + dst_offset[0]] = src[index + src_offset[0]]
        else:
            dst[i + dst_offset[0]] = 0.0


@kernel
def scatter_i32_field(src: template(), indices: template(), dst: template(), n: i32):
    src_offset = static(src.snode.ptr.offset if len(src.snode.ptr.offset) != 0 else [0])
    indices_offset = static(
        indices.snode.ptr.offset if len(indices.snode.ptr.offset) != 0 else [0]
    )
    dst_offset = static(dst.snode.ptr.offset if len(dst.snode.ptr.offset) != 0 else [0])
    for i in range(n):
        index = indices[i + indices_offset[0]]
        if index >= 0 and index < dst.shape[0]:
            dst[index + dst_offset[0]] = src[i + src_offset[0]]


@kernel
def scatter_f32_field(src: template(), indices: template(), dst: template(), n: i32):
    src_offset = static(src.snode.ptr.offset if len(src.snode.ptr.offset) != 0 else [0])
    indices_offset = static(
        indices.snode.ptr.offset if len(indices.snode.ptr.offset) != 0 else [0]
    )
    dst_offset = static(dst.snode.ptr.offset if len(dst.snode.ptr.offset) != 0 else [0])
    for i in range(n):
        index = indices[i + indices_offset[0]]
        if index >= 0 and index < dst.shape[0]:
            dst[index + dst_offset[0]] = src[i + src_offset[0]]


@kernel
def scatter_add_i32_field(src: template(), indices: template(), dst: template(), n: i32):
    src_offset = static(src.snode.ptr.offset if len(src.snode.ptr.offset) != 0 else [0])
    indices_offset = static(
        indices.snode.ptr.offset if len(indices.snode.ptr.offset) != 0 else [0]
    )
    dst_offset = static(dst.snode.ptr.offset if len(dst.snode.ptr.offset) != 0 else [0])
    for i in range(n):
        index = indices[i + indices_offset[0]]
        if index >= 0 and index < dst.shape[0]:
            ops.atomic_add(dst[index + dst_offset[0]], src[i + src_offset[0]])


@kernel
def scatter_add_f32_field(src: template(), indices: template(), dst: template(), n: i32):
    src_offset = static(src.snode.ptr.offset if len(src.snode.ptr.offset) != 0 else [0])
    indices_offset = static(
        indices.snode.ptr.offset if len(indices.snode.ptr.offset) != 0 else [0]
    )
    dst_offset = static(dst.snode.ptr.offset if len(dst.snode.ptr.offset) != 0 else [0])
    for i in range(n):
        index = indices[i + indices_offset[0]]
        if index >= 0 and index < dst.shape[0]:
            ops.atomic_add(dst[index + dst_offset[0]], src[i + src_offset[0]])


@kernel
def bucket_count_i32_field(keys: template(), offsets: template(), n: i32, num_bins: i32):
    keys_offset = static(keys.snode.ptr.offset if len(keys.snode.ptr.offset) != 0 else [0])
    offsets_offset = static(
        offsets.snode.ptr.offset if len(offsets.snode.ptr.offset) != 0 else [0]
    )
    for i in range(num_bins + 1):
        offsets[i + offsets_offset[0]] = 0
    for i in range(n):
        key = keys[i + keys_offset[0]]
        if key >= 0 and key < num_bins:
            ops.atomic_add(offsets[key + 1 + offsets_offset[0]], 1)


@kernel
def bucket_prefix_offsets_i32_field_serial(offsets: template(), num_bins: i32):
    offsets_offset = static(
        offsets.snode.ptr.offset if len(offsets.snode.ptr.offset) != 0 else [0]
    )
    running = 0
    i = 0
    while i <= num_bins:
        running += offsets[i + offsets_offset[0]]
        offsets[i + offsets_offset[0]] = running
        i += 1


@kernel
def bucket_copy_offsets_to_cursor_field(
    offsets: template(), cursor: template(), num_bins: i32
):
    offsets_offset = static(
        offsets.snode.ptr.offset if len(offsets.snode.ptr.offset) != 0 else [0]
    )
    for i in range(num_bins):
        cursor[i] = offsets[i + offsets_offset[0]]


@kernel
def bucket_scatter_i32_field(
    keys: template(),
    values: template(),
    cursor: template(),
    output: template(),
    n: i32,
    num_bins: i32,
):
    keys_offset = static(keys.snode.ptr.offset if len(keys.snode.ptr.offset) != 0 else [0])
    values_offset = static(
        values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else [0]
    )
    output_offset = static(
        output.snode.ptr.offset if len(output.snode.ptr.offset) != 0 else [0]
    )
    for i in range(n):
        key = keys[i + keys_offset[0]]
        if key >= 0 and key < num_bins:
            out_idx = ops.atomic_add(cursor[key], 1)
            if out_idx >= 0 and out_idx < output.shape[0]:
                output[out_idx + output_offset[0]] = values[i + values_offset[0]]


@kernel
def grouped_reduce_sum_i32_field(
    keys: template(),
    values: template(),
    output: template(),
    n: i32,
    num_groups: i32,
):
    keys_offset = static(keys.snode.ptr.offset if len(keys.snode.ptr.offset) != 0 else [0])
    values_offset = static(
        values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else [0]
    )
    output_offset = static(
        output.snode.ptr.offset if len(output.snode.ptr.offset) != 0 else [0]
    )
    for i in range(num_groups):
        output[i + output_offset[0]] = 0
    for i in range(n):
        key = keys[i + keys_offset[0]]
        if key >= 0 and key < num_groups:
            ops.atomic_add(
                output[key + output_offset[0]], values[i + values_offset[0]]
            )


@kernel
def ext_arr_to_ndarray(arr: ndarray_type.ndarray(), ndarray: ndarray_type.ndarray()):
    for I in grouped(ndarray):
        ndarray[I] = arr[I]


@kernel
def ext_arr_to_ndarray_matrix(
    arr: ndarray_type.ndarray(),
    ndarray: ndarray_type.ndarray(),
    layout_is_aos: template(),
    as_vector: template(),
):
    for I in grouped(ndarray):
        for p in static(range(ndarray[I].n)):
            if static(as_vector):
                if static(layout_is_aos):
                    ndarray[I][p] = arr[I, p]
                else:
                    ndarray[I][p] = arr[p, I]
            else:
                for q in static(range(ndarray[I].m)):
                    if static(layout_is_aos):
                        ndarray[I][p, q] = arr[I, p, q]
                    else:
                        ndarray[I][p, q] = arr[p, q, I]


@kernel
def matrix_to_ext_arr(mat: template(), arr: ndarray_type.ndarray(), as_vector: template()):
    # default value of offset is [], replace it with [0] * len
    offset = static(mat.snode.ptr.offset if len(mat.snode.ptr.offset) != 0 else [0] * len(mat.shape))

    for I in grouped(mat):
        for p in static(range(mat.n)):
            for q in static(range(mat.m)):
                if static(as_vector):
                    if static(getattr(mat, "ndim", 2) == 1):
                        arr[I - offset, p] = mat[I][p]
                    else:
                        arr[I - offset, p] = mat[I][p, q]
                else:
                    if static(getattr(mat, "ndim", 2) == 1):
                        arr[I - offset, p, q] = mat[I][p]
                    else:
                        arr[I - offset, p, q] = mat[I][p, q]


@kernel
def ext_arr_to_matrix(arr: ndarray_type.ndarray(), mat: template(), as_vector: template()):
    # default value of offset is [], replace it with [0] * len
    offset = static(mat.snode.ptr.offset if len(mat.snode.ptr.offset) != 0 else [0] * len(mat.shape))

    for I in grouped(mat):
        for p in static(range(mat.n)):
            for q in static(range(mat.m)):
                if static(getattr(mat, "ndim", 2) == 1):
                    if static(as_vector):
                        mat[I][p] = arr[I - offset, p]
                    else:
                        mat[I][p] = arr[I - offset, p, q]
                else:
                    if static(as_vector):
                        mat[I][p, q] = arr[I - offset, p]
                    else:
                        mat[I][p, q] = arr[I - offset, p, q]


# extract ndarray of raw vulkan memory layout to normal memory layout.
# the vulkan layout stored in ndarray : width-by-width stored along n-
# darray's shape[1] which is the height-axis(So use [size // h, size %
#  h]). And the height-order of vulkan layout is flip up-down.(So take
# [size = (h - 1 - j) * w + i] to get the index)
@kernel
def arr_vulkan_layout_to_arr_normal_layout(vk_arr: ndarray_type.ndarray(), normal_arr: ndarray_type.ndarray()):
    static_assert(len(normal_arr.shape) == 2)
    w = normal_arr.shape[0]
    h = normal_arr.shape[1]
    for i, j in ndrange(w, h):
        normal_arr[i, j] = vk_arr[(h - 1 - j) * w + i]


# extract ndarray of raw vulkan memory layout into a taichi-field data
# structure with normal memory layout.
@kernel
def arr_vulkan_layout_to_field_normal_layout(vk_arr: ndarray_type.ndarray(), normal_field: template()):
    static_assert(len(normal_field.shape) == 2)
    w = static(normal_field.shape[0])
    h = static(normal_field.shape[1])
    offset = static(normal_field.snode.ptr.offset if len(normal_field.snode.ptr.offset) != 0 else [0, 0])
    i_offset = static(offset[0])
    j_offset = static(offset[1])

    for i, j in ndrange(w, h):
        normal_field[i + i_offset, j + j_offset] = vk_arr[(h - 1 - j) * w + i]


@kernel
def clear_gradients(_vars: template()):
    for I in grouped(ScalarField(Expr(_vars[0]))):
        for s in static(_vars):
            ScalarField(Expr(s))[I] = ops.cast(0, dtype=s.get_dt())


@kernel
def field_fill_python_scope(F: template(), val: template()):
    field_fill_taichi_scope(F, val)


@kernel
def snode_deactivate(b: template()):
    for I in grouped(b):
        deactivate(b, I)


# G11-D2 (P-Vulkan-Sparse-Deact-Merge): fused two-level deactivate.
#
# Background: `SNode.deactivate_all()` on a pointer with a bitmasked child
# (the canonical sparse layout) post-order recurses, issuing two separate
# Python @kernel calls — one per sparse SNode. On Vulkan each Python kernel
# call is a separate vk command-buffer submit, so per-frame `deactivate_all`
# pays double submit overhead (~0.13 ms each on RTX-class HW). Fusing both
# struct-fors into a single @kernel keeps them as two offloads but only one
# command-buffer submit; the inter-offload pipeline barrier already preserves
# child→parent ordering (child mask cleared before parent slot freed).
#
# Used by `SNode.deactivate_all` fast-path when self and its single child are
# both sparse (pointer/bitmasked) and the grandchildren are only `place`.
@kernel
def snode_deactivate_pair(parent: template(), child: template()):
    for I in grouped(child):
        deactivate(child, I)
    for J in grouped(parent):
        deactivate(parent, J)


@kernel
def snode_deactivate_dynamic(b: template()):
    for I in grouped(b.parent()):
        deactivate(b, I)


@kernel
def load_texture_from_numpy(
    tex: texture_type.rw_texture(num_dimensions=2, fmt=Format.rgba8, lod=0),
    img: ndarray_type.ndarray(dtype=vec3, ndim=2),
):
    for i, j in img:
        tex.store(
            vector(2, i32)([i, j]),
            vector(4, f32)([img[i, j][0], img[i, j][1], img[i, j][2], 0]) / 255.0,
        )


@kernel
def save_texture_to_numpy(
    tex: texture_type.rw_texture(num_dimensions=2, fmt=Format.rgba8, lod=0),
    img: ndarray_type.ndarray(dtype=vec3, ndim=2),
):
    for i, j in img:
        img[i, j] = ops.round(tex.load(vector(2, i32)([i, j])).rgb * 255)


# Odd-even merge sort
@kernel
def sort_stage(
    keys: template(),
    use_values: int,
    values: template(),
    N: int,
    p: int,
    k: int,
    invocations: int,
):
    keys_offset = static(keys.snode.ptr.offset if len(keys.snode.ptr.offset) != 0 else 0)
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    for inv in range(invocations):
        j = k % p + inv * 2 * k
        for i in range(0, ops.min(k, N - j - k)):
            a = i + j
            b = i + j + k
            if int(a / (p * 2)) == int(b / (p * 2)):
                key_a = keys[a + keys_offset]
                key_b = keys[b + keys_offset]
                if key_a > key_b:
                    keys[a + keys_offset] = key_b
                    keys[b + keys_offset] = key_a
                    if use_values != 0:
                        temp = values[a + values_offset]
                        values[a + values_offset] = values[b + values_offset]
                        values[b + values_offset] = temp


@kernel
def sort_init_key_buffer_u32(keys: template(), key_buffer: template(), N: int, signed: int):
    keys_offset = static(keys.snode.ptr.offset if len(keys.snode.ptr.offset) != 0 else 0)
    for i in range(N):
        key = ops.cast(keys[i + keys_offset], u32)
        if signed != 0:
            key = ops.bit_cast(keys[i + keys_offset], u32) ^ (ops.cast(1, u32) << 31)
        key_buffer[i] = key


@kernel
def sort_init_value_buffer(values: template(), value_buffer: template(), N: int):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    for i in range(N):
        value_buffer[i] = values[i + values_offset]


@kernel
def sort_radix_count_zero_bits_u32(keys: template(), zero_prefix: template(), N: int, bit: int):
    for i in range(N):
        zero_prefix[i] = 1 - ops.cast((keys[i] >> bit) & 1, i32)


@kernel
def sort_radix_store_zero_count(zero_prefix: template(), zero_count: template(), N: int):
    zero_count[None] = zero_prefix[N - 1]


@kernel
def sort_radix_scatter_u32(
    keys_in: template(),
    keys_out: template(),
    zero_prefix: template(),
    values_in: template(),
    values_out: template(),
    use_values: int,
    N: int,
    bit: int,
    zero_count: template(),
):
    for i in range(N):
        key = keys_in[i]
        is_one = ops.cast((key >> bit) & 1, i32)
        zero_before_or_at = zero_prefix[i]
        pos = zero_before_or_at - 1
        if is_one != 0:
            pos = zero_count[None] + i - zero_before_or_at
        keys_out[pos] = key
        if use_values != 0:
            values_out[pos] = values_in[i]


@kernel
def sort_copy_key_buffer_to_field_u32(
    key_buffer: template(), keys: template(), N: int, signed: int
):
    keys_offset = static(keys.snode.ptr.offset if len(keys.snode.ptr.offset) != 0 else 0)
    for i in range(N):
        key = key_buffer[i]
        if signed != 0:
            keys[i + keys_offset] = ops.bit_cast(key ^ (ops.cast(1, u32) << 31), i32)
        else:
            keys[i + keys_offset] = key


@kernel
def sort_copy_value_buffer_to_field(value_buffer: template(), values: template(), N: int):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    for i in range(N):
        values[i + values_offset] = value_buffer[i]


@kernel
def sort_init_key_buffer_u32_ndarray(
    keys: ndarray_type.ndarray(ndim=1),
    key_buffer: ndarray_type.ndarray(dtype=u32, ndim=1),
    N: i32,
    signed: i32,
):
    for i in range(N):
        key = ops.cast(keys[i], u32)
        if signed != 0:
            key = ops.bit_cast(keys[i], u32) ^ (ops.cast(1, u32) << 31)
        key_buffer[i] = key


@kernel
def sort_init_value_buffer_i32_ndarray(
    values: ndarray_type.ndarray(dtype=i32, ndim=1),
    value_buffer: ndarray_type.ndarray(dtype=i32, ndim=1),
    N: i32,
):
    for i in range(N):
        value_buffer[i] = values[i]


@kernel
def sort_radix_count_zero_bits_u32_ndarray(
    keys: ndarray_type.ndarray(dtype=u32, ndim=1),
    scan_arr: ndarray_type.ndarray(dtype=i32, ndim=1),
    N: i32,
    bit: i32,
):
    for i in range(N):
        scan_arr[i] = 1 - ops.cast((keys[i] >> bit) & 1, i32)


@kernel
def sort_radix_store_zero_count_ndarray(
    scan_arr: ndarray_type.ndarray(dtype=i32, ndim=1),
    zero_count: ndarray_type.ndarray(dtype=i32, ndim=1),
    N: i32,
):
    zero_count[0] = scan_arr[N - 1]


@kernel
def sort_radix_scatter_u32_i32_ndarray(
    keys_in: ndarray_type.ndarray(dtype=u32, ndim=1),
    keys_out: ndarray_type.ndarray(dtype=u32, ndim=1),
    scan_arr: ndarray_type.ndarray(dtype=i32, ndim=1),
    values_in: ndarray_type.ndarray(dtype=i32, ndim=1),
    values_out: ndarray_type.ndarray(dtype=i32, ndim=1),
    use_values: i32,
    N: i32,
    bit: i32,
    zero_count: ndarray_type.ndarray(dtype=i32, ndim=1),
):
    for i in range(N):
        key = keys_in[i]
        is_one = ops.cast((key >> bit) & 1, i32)
        zero_before_or_at = scan_arr[i]
        pos = zero_before_or_at - 1
        if is_one != 0:
            pos = zero_count[0] + i - zero_before_or_at
        keys_out[pos] = key
        if use_values != 0:
            values_out[pos] = values_in[i]


@kernel
def sort_radix_scatter_keys_u32_ndarray(
    keys_in: ndarray_type.ndarray(dtype=u32, ndim=1),
    keys_out: ndarray_type.ndarray(dtype=u32, ndim=1),
    scan_arr: ndarray_type.ndarray(dtype=i32, ndim=1),
    N: i32,
    bit: i32,
    zero_count: ndarray_type.ndarray(dtype=i32, ndim=1),
):
    for i in range(N):
        key = keys_in[i]
        is_one = ops.cast((key >> bit) & 1, i32)
        zero_before_or_at = scan_arr[i]
        pos = zero_before_or_at - 1
        if is_one != 0:
            pos = zero_count[0] + i - zero_before_or_at
        keys_out[pos] = key


@kernel
def sort_copy_key_buffer_to_ndarray_u32(
    key_buffer: ndarray_type.ndarray(dtype=u32, ndim=1),
    keys: ndarray_type.ndarray(ndim=1),
    N: i32,
    signed: i32,
):
    for i in range(N):
        key = key_buffer[i]
        if signed != 0:
            keys[i] = ops.bit_cast(key ^ (ops.cast(1, u32) << 31), i32)
        else:
            keys[i] = key


@kernel
def sort_copy_value_buffer_to_i32_ndarray(
    value_buffer: ndarray_type.ndarray(dtype=i32, ndim=1),
    values: ndarray_type.ndarray(dtype=i32, ndim=1),
    N: i32,
):
    for i in range(N):
        values[i] = value_buffer[i]


@kernel
def scan_add_inclusive_ndarray(
    arr_in: ndarray_type.ndarray(dtype=i32, ndim=1),
    in_beg: i32,
    in_end: i32,
    single_block: i32,
):
    WARP_SZ = 32
    BLOCK_SZ = 64
    loop_config(block_dim=64)
    for i in range(in_beg, in_end):
        val = arr_in[i]

        thread_id = i % BLOCK_SZ
        block_id = int((i - in_beg) // BLOCK_SZ)
        lane_id = thread_id % WARP_SZ
        warp_id = thread_id // WARP_SZ

        pad_shared = block.SharedArray((65,), i32)

        val = subgroup.inclusive_add(val)
        block.sync()

        if thread_id % WARP_SZ == WARP_SZ - 1:
            pad_shared[warp_id] = val
        block.sync()

        if warp_id == 0 and lane_id == 0:
            for k in range(1, BLOCK_SZ // WARP_SZ):
                pad_shared[k] += pad_shared[k - 1]
        block.sync()

        warp_sum = 0
        if warp_id > 0:
            warp_sum = pad_shared[warp_id - 1]
        val += warp_sum
        arr_in[i] = val

        if single_block == 0 and (thread_id == BLOCK_SZ - 1):
            arr_in[in_end + block_id] = val


@kernel
def uniform_add_ndarray(
    arr_in: ndarray_type.ndarray(dtype=i32, ndim=1),
    in_beg: i32,
    in_end: i32,
):
    BLOCK_SZ = 64
    loop_config(block_dim=64)
    for i in range(in_beg + BLOCK_SZ, in_end):
        block_id = int((i - in_beg) // BLOCK_SZ)
        arr_in[i] += arr_in[in_end + block_id - 1]


# Parallel Prefix Sum (Scan)
@func
def warp_shfl_up_i32(val: template()):
    global_tid = block.global_thread_idx()
    WARP_SZ = 32
    lane_id = global_tid % WARP_SZ
    # Intra-warp scan, manually unrolled
    offset_j = 1
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 2
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 4
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 8
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    offset_j = 16
    n = warp.shfl_up_i32(warp.active_mask(), val, offset_j)
    if lane_id >= offset_j:
        val += n
    return val


@kernel
def scan_add_inclusive(
    arr_in: template(),
    in_beg: i32,
    in_end: i32,
    single_block: template(),
    inclusive_add: template(),
):
    WARP_SZ = 32
    BLOCK_SZ = 64
    loop_config(block_dim=64)
    for i in range(in_beg, in_end):
        val = arr_in[i]

        thread_id = i % BLOCK_SZ
        block_id = int((i - in_beg) // BLOCK_SZ)
        lane_id = thread_id % WARP_SZ
        warp_id = thread_id // WARP_SZ

        pad_shared = block.SharedArray((65,), i32)

        val = inclusive_add(val)
        block.sync()

        # Put warp scan results to smem
        # TODO replace smem with real smem when available
        if thread_id % WARP_SZ == WARP_SZ - 1:
            pad_shared[warp_id] = val
        block.sync()

        # Inter-warp scan, use the first thread in the first warp
        if warp_id == 0 and lane_id == 0:
            for k in range(1, BLOCK_SZ // WARP_SZ):
                pad_shared[k] += pad_shared[k - 1]
        block.sync()

        # Update data with warp sums
        warp_sum = 0
        if warp_id > 0:
            warp_sum = pad_shared[warp_id - 1]
        val += warp_sum
        arr_in[i] = val

        # Update partial sums except the final block
        if not single_block and (thread_id == BLOCK_SZ - 1):
            arr_in[in_end + block_id] = val


@kernel
def scan_add_inclusive_cuda(arr_in: template(), in_beg: i32, in_end: i32, single_block: template()):
    WARP_SZ = 32
    BLOCK_SZ = 256
    loop_config(block_dim=256)
    for i in range(in_beg, in_end):
        val = arr_in[i]

        thread_id = i % BLOCK_SZ
        block_id = int((i - in_beg) // BLOCK_SZ)
        lane_id = thread_id % WARP_SZ
        warp_id = thread_id // WARP_SZ

        pad_shared = block.SharedArray((65,), i32)

        val = warp_shfl_up_i32(val)
        block.sync()

        if thread_id % WARP_SZ == WARP_SZ - 1:
            pad_shared[warp_id] = val
        block.sync()

        if warp_id == 0 and lane_id == 0:
            for k in range(1, BLOCK_SZ // WARP_SZ):
                pad_shared[k] += pad_shared[k - 1]
        block.sync()

        warp_sum = 0
        if warp_id > 0:
            warp_sum = pad_shared[warp_id - 1]
        val += warp_sum
        arr_in[i] = val

        if not single_block and (thread_id == BLOCK_SZ - 1):
            arr_in[in_end + block_id] = val


@kernel
def uniform_add(arr_in: template(), in_beg: i32, in_end: i32):
    BLOCK_SZ = 64
    loop_config(block_dim=64)
    for i in range(in_beg + BLOCK_SZ, in_end):
        block_id = int((i - in_beg) // BLOCK_SZ)
        arr_in[i] += arr_in[in_end + block_id - 1]


@kernel
def uniform_add_cuda(arr_in: template(), in_beg: i32, in_end: i32):
    BLOCK_SZ = 256
    loop_config(block_dim=256)
    for i in range(in_beg + BLOCK_SZ, in_end):
        block_id = int((i - in_beg) // BLOCK_SZ)
        arr_in[i] += arr_in[in_end + block_id - 1]


@kernel
def blit_from_field_to_field(dst: template(), src: template(), offset: i32, size: i32):
    dst_offset = static(dst.snode.ptr.offset if len(dst.snode.ptr.offset) != 0 else 0)
    src_offset = static(src.snode.ptr.offset if len(src.snode.ptr.offset) != 0 else 0)
    for i in range(size):
        dst[i + dst_offset + offset] = src[i + src_offset]


@kernel
def fill_i32_arange_ndarray(out: ndarray_type.ndarray(dtype=i32, ndim=1), N: i32):
    for i in range(N):
        out[i] = i


@kernel
def rle_mark_boundaries_ndarray(
    keys: ndarray_type.ndarray(ndim=1),
    flags: ndarray_type.ndarray(dtype=i32, ndim=1),
    N: i32,
    capacity: i32,
):
    for i in range(capacity):
        flags[i] = 0
        if i < N:
            is_start = i == 0
            if i > 0:
                is_start = keys[i] != keys[i - 1]
            flags[i] = 1 if is_start else 0


@kernel
def rle_mark_boundaries_and_starts_ndarray(
    keys: ndarray_type.ndarray(ndim=1),
    flags: ndarray_type.ndarray(dtype=i32, ndim=1),
    starts: ndarray_type.ndarray(dtype=i32, ndim=1),
    N: i32,
    capacity: i32,
):
    for i in range(capacity):
        flags[i] = 0
        starts[i] = i
        if i < N:
            is_start = i == 0
            if i > 0:
                is_start = keys[i] != keys[i - 1]
            flags[i] = 1 if is_start else 0


@kernel
def rle_finalize_lengths_ndarray(
    starts: ndarray_type.ndarray(dtype=i32, ndim=1),
    lengths: ndarray_type.ndarray(dtype=i32, ndim=1),
    count: ndarray_type.ndarray(dtype=i32, ndim=1),
    N: i32,
    capacity: i32,
):
    run_count = count[0]
    for i in range(capacity):
        if i < run_count:
            end = N
            if i + 1 < run_count:
                end = starts[i + 1]
            lengths[i] = end - starts[i]


@kernel
def rle_reset_count_ndarray(
    count: ndarray_type.ndarray(dtype=i32, ndim=1),
):
    count[0] = 0


@kernel
def rle_mark_boundaries_field(
    keys: template(),
    flags: template(),
    N: i32,
    capacity: i32,
):
    keys_offset = static(keys.snode.ptr.offset if len(keys.snode.ptr.offset) != 0 else 0)
    flags_offset = static(flags.snode.ptr.offset if len(flags.snode.ptr.offset) != 0 else 0)
    for i in range(capacity):
        flags[i + flags_offset] = 0
        if i < N:
            is_start = i == 0
            if i > 0:
                is_start = keys[i + keys_offset] != keys[i - 1 + keys_offset]
            flags[i + flags_offset] = 1 if is_start else 0


@kernel
def rle_mark_boundaries_and_starts_field(
    keys: template(),
    flags: template(),
    starts: template(),
    N: i32,
    capacity: i32,
):
    keys_offset = static(keys.snode.ptr.offset if len(keys.snode.ptr.offset) != 0 else 0)
    flags_offset = static(flags.snode.ptr.offset if len(flags.snode.ptr.offset) != 0 else 0)
    starts_offset = static(starts.snode.ptr.offset if len(starts.snode.ptr.offset) != 0 else 0)
    for i in range(capacity):
        flags[i + flags_offset] = 0
        starts[i + starts_offset] = i
        if i < N:
            is_start = i == 0
            if i > 0:
                is_start = keys[i + keys_offset] != keys[i - 1 + keys_offset]
            flags[i + flags_offset] = 1 if is_start else 0


@kernel
def rle_finalize_lengths_field(
    starts: template(),
    lengths: template(),
    count: template(),
    N: i32,
    capacity: i32,
):
    starts_offset = static(starts.snode.ptr.offset if len(starts.snode.ptr.offset) != 0 else 0)
    lengths_offset = static(
        lengths.snode.ptr.offset if len(lengths.snode.ptr.offset) != 0 else 0
    )
    run_count = count[None]
    for i in range(capacity):
        if i < run_count:
            end = N
            if i + 1 < run_count:
                end = starts[i + 1 + starts_offset]
            lengths[i + lengths_offset] = end - starts[i + starts_offset]


@kernel
def rle_reset_count_field(count: template()):
    count[None] = 0


@kernel
def segmented_reduce_sum_ndarray(
    values: ndarray_type.ndarray(ndim=1),
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    output: ndarray_type.ndarray(ndim=1),
    num_segments: i32,
):
    for segment in range(num_segments):
        begin = offsets[segment]
        end = offsets[segment + 1]
        if begin < end:
            acc = values[begin]
            loop_config(serialize=True)
            for local_index in range(end - begin - 1):
                acc += values[begin + local_index + 1]
            output[segment] = acc
        else:
            output[segment] = 0


@kernel
def segmented_reduce_sum_field(
    values: template(),
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    output: template(),
    num_segments: i32,
):
    values_offset = static(
        values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0
    )
    output_offset = static(
        output.snode.ptr.offset if len(output.snode.ptr.offset) != 0 else 0
    )
    for segment in range(num_segments):
        acc = ops.cast(0, values.dtype)
        begin = offsets[segment]
        end = offsets[segment + 1]
        loop_config(serialize=True)
        for local_index in range(end - begin):
            acc += values[begin + local_index + values_offset]
        output[segment + output_offset] = acc


@kernel
def segmented_scan_sum_serial_ndarray(
    values: ndarray_type.ndarray(ndim=1),
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    output: ndarray_type.ndarray(ndim=1),
    num_segments: i32,
    inclusive: i32,
):
    for segment in range(num_segments):
        begin = offsets[segment]
        end = offsets[segment + 1]
        if begin < end:
            acc = values[begin]
            if inclusive != 0:
                output[begin] = acc
            else:
                output[begin] = 0
            loop_config(serialize=True)
            for local_index in range(end - begin - 1):
                index = begin + local_index + 1
                value = values[index]
                if inclusive != 0:
                    acc += value
                    output[index] = acc
                else:
                    output[index] = acc
                    acc += value


@kernel
def segmented_scan_sum_serial_field(
    values: template(),
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    output: template(),
    num_segments: i32,
    inclusive: i32,
):
    values_offset = static(
        values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0
    )
    output_offset = static(
        output.snode.ptr.offset if len(output.snode.ptr.offset) != 0 else 0
    )
    for segment in range(num_segments):
        acc = ops.cast(0, values.dtype)
        begin = offsets[segment]
        end = offsets[segment + 1]
        loop_config(serialize=True)
        for local_index in range(end - begin):
            index = begin + local_index
            value = values[index + values_offset]
            if inclusive != 0:
                acc += value
                output[index + output_offset] = acc
            else:
                output[index + output_offset] = acc
                acc += value


@kernel
def segmented_scan_gather_bases_ndarray(
    scanned: ndarray_type.ndarray(ndim=1),
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    bases: ndarray_type.ndarray(ndim=1),
    num_segments: i32,
):
    for segment in range(num_segments):
        begin = offsets[segment]
        if begin > 0:
            bases[segment] = scanned[begin - 1]
        else:
            bases[segment] = 0


@kernel
def segmented_scan_gather_bases_field(
    scanned: template(),
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    bases: ndarray_type.ndarray(ndim=1),
    num_segments: i32,
):
    scanned_offset = static(
        scanned.snode.ptr.offset if len(scanned.snode.ptr.offset) != 0 else 0
    )
    for segment in range(num_segments):
        begin = offsets[segment]
        base = ops.cast(0, scanned.dtype)
        if begin > 0:
            base = scanned[begin - 1 + scanned_offset]
        bases[segment] = base


@kernel
def segmented_scan_apply_bases_ndarray(
    scanned: ndarray_type.ndarray(ndim=1),
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    bases: ndarray_type.ndarray(ndim=1),
    num_segments: i32,
    inclusive: i32,
):
    for segment in range(num_segments):
        begin = offsets[segment]
        end = offsets[segment + 1]
        base = bases[segment]
        if inclusive != 0:
            for local_index in range(end - begin):
                index = begin + local_index
                scanned[index] -= base
        else:
            loop_config(serialize=True)
            for reverse_index in range(end - begin):
                index = end - 1 - reverse_index
                if index == begin:
                    scanned[index] = scanned[index] * 0
                else:
                    scanned[index] = scanned[index - 1] - base


@kernel
def segmented_scan_apply_bases_field(
    scanned: template(),
    offsets: ndarray_type.ndarray(dtype=i32, ndim=1),
    bases: ndarray_type.ndarray(ndim=1),
    num_segments: i32,
    inclusive: i32,
):
    scanned_offset = static(
        scanned.snode.ptr.offset if len(scanned.snode.ptr.offset) != 0 else 0
    )
    for segment in range(num_segments):
        begin = offsets[segment]
        end = offsets[segment + 1]
        base = bases[segment]
        if inclusive != 0:
            for local_index in range(end - begin):
                index = begin + local_index + scanned_offset
                scanned[index] -= base
        else:
            loop_config(serialize=True)
            for reverse_index in range(end - begin):
                logical_index = end - 1 - reverse_index
                index = logical_index + scanned_offset
                if logical_index == begin:
                    scanned[index] = ops.cast(0, scanned.dtype)
                else:
                    scanned[index] = scanned[index - 1] - base


@kernel
def compact_flags_to_prefix_field(flags: template(), prefix: template(), N: i32):
    flags_offset = static(flags.snode.ptr.offset if len(flags.snode.ptr.offset) != 0 else 0)
    for i in range(N):
        prefix[i] = 1 if flags[i + flags_offset] != 0 else 0


@kernel
def compact_flags_to_prefix_ndarray_from_field(
    flags: template(),
    prefix: ndarray_type.ndarray(dtype=i32, ndim=1),
    N: i32,
):
    flags_offset = static(flags.snode.ptr.offset if len(flags.snode.ptr.offset) != 0 else 0)
    for i in range(N):
        prefix[i] = 1 if flags[i + flags_offset] != 0 else 0


@kernel
def compact_single_item_field(
    values: template(),
    flags: template(),
    output: template(),
    count: template(),
    N: i32,
):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    flags_offset = static(flags.snode.ptr.offset if len(flags.snode.ptr.offset) != 0 else 0)
    output_offset = static(output.snode.ptr.offset if len(output.snode.ptr.offset) != 0 else 0)
    count[None] = 0
    if N == 1 and flags[flags_offset] != 0:
        output[output_offset] = values[values_offset]
        count[None] = 1


@kernel
def compact_scatter_field(
    values: template(),
    flags: template(),
    prefix: template(),
    output: template(),
    count: template(),
    N: i32,
):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    flags_offset = static(flags.snode.ptr.offset if len(flags.snode.ptr.offset) != 0 else 0)
    output_offset = static(output.snode.ptr.offset if len(output.snode.ptr.offset) != 0 else 0)
    if N > 0:
        count[None] = prefix[N - 1]
    else:
        count[None] = 0
    for i in range(N):
        if flags[i + flags_offset] != 0:
            output[prefix[i] - 1 + output_offset] = values[i + values_offset]


@kernel
def compact_scatter_field_from_prefix_ndarray(
    values: template(),
    flags: template(),
    prefix: ndarray_type.ndarray(dtype=i32, ndim=1),
    output: template(),
    count: template(),
    N: i32,
):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    flags_offset = static(flags.snode.ptr.offset if len(flags.snode.ptr.offset) != 0 else 0)
    output_offset = static(output.snode.ptr.offset if len(output.snode.ptr.offset) != 0 else 0)
    if N > 0:
        count[None] = prefix[N - 1]
    else:
        count[None] = 0
    for i in range(N):
        if flags[i + flags_offset] != 0:
            output[prefix[i] - 1 + output_offset] = values[i + values_offset]


@kernel
def compact_stable_serial_field(
    values: template(),
    flags: template(),
    output: template(),
    count: template(),
    N: i32,
):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    flags_offset = static(flags.snode.ptr.offset if len(flags.snode.ptr.offset) != 0 else 0)
    output_offset = static(output.snode.ptr.offset if len(output.snode.ptr.offset) != 0 else 0)
    count[None] = 0
    loop_config(serialize=True)
    for i in range(N):
        if flags[i + flags_offset] != 0:
            output[count[None] + output_offset] = values[i + values_offset]
            count[None] += 1


def compact_stable_serial_field_static_n(values, flags, output, count, N):
    N = int(N)
    compact_kernel = _field_compact_static_kernel_cache.get(N)
    if compact_kernel is None:

        @kernel
        def compact_kernel(
            values: template(),
            flags: template(),
            output: template(),
            count: template(),
        ):
            values_offset = static(
                values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0
            )
            flags_offset = static(
                flags.snode.ptr.offset if len(flags.snode.ptr.offset) != 0 else 0
            )
            output_offset = static(
                output.snode.ptr.offset if len(output.snode.ptr.offset) != 0 else 0
            )
            count[None] = 0
            loop_config(serialize=True)
            for i in range(N):
                if flags[i + flags_offset] != 0:
                    output[count[None] + output_offset] = values[i + values_offset]
                    count[None] += 1

        _field_compact_static_kernel_cache[N] = compact_kernel
    compact_kernel(values, flags, output, count)


@kernel
def reduce_i32_field(values: template(), output: template(), N: i32, op: i32):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    if op == 0:
        output[None] = 0
        for i in range(N):
            ops.atomic_add(output[None], values[i + values_offset])
    elif op == 1:
        output[None] = 2147483647
        for i in range(N):
            ops.atomic_min(output[None], values[i + values_offset])
    else:
        output[None] = -2147483648
        for i in range(N):
            ops.atomic_max(output[None], values[i + values_offset])


@kernel
def reduce_f32_field(values: template(), output: template(), N: i32, op: i32):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    if op == 0:
        output[None] = 0.0
        for i in range(N):
            ops.atomic_add(output[None], values[i + values_offset])
    elif op == 1:
        output[None] = 3.4028234663852886e38
        for i in range(N):
            ops.atomic_min(output[None], values[i + values_offset])
    else:
        output[None] = -3.4028234663852886e38
        for i in range(N):
            ops.atomic_max(output[None], values[i + values_offset])


@kernel
def reduce_i32_field_private_count(
    values: template(),
    partial: template(),
    N: i32,
    chunk_size: i32,
    num_chunks: i32,
    op: i32,
):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    for chunk in range(num_chunks):
        if op == 0:
            partial[chunk] = 0
        elif op == 1:
            partial[chunk] = 2147483647
        else:
            partial[chunk] = -2147483648
    for i in range(N):
        chunk = i // chunk_size
        if op == 0:
            ops.atomic_add(partial[chunk], values[i + values_offset])
        elif op == 1:
            ops.atomic_min(partial[chunk], values[i + values_offset])
        else:
            ops.atomic_max(partial[chunk], values[i + values_offset])


@kernel
def reduce_i32_field_private_reduce(partial: template(), output: template(), num_chunks: i32, op: i32):
    if op == 0:
        output[None] = 0
        for chunk in range(num_chunks):
            ops.atomic_add(output[None], partial[chunk])
    elif op == 1:
        output[None] = 2147483647
        for chunk in range(num_chunks):
            ops.atomic_min(output[None], partial[chunk])
    else:
        output[None] = -2147483648
        for chunk in range(num_chunks):
            ops.atomic_max(output[None], partial[chunk])


@kernel
def reduce_f32_field_private_count(
    values: template(),
    partial: template(),
    N: i32,
    chunk_size: i32,
    num_chunks: i32,
    op: i32,
):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    for chunk in range(num_chunks):
        if op == 0:
            partial[chunk] = 0.0
        elif op == 1:
            partial[chunk] = 3.4028234663852886e38
        else:
            partial[chunk] = -3.4028234663852886e38
    for i in range(N):
        chunk = i // chunk_size
        if op == 0:
            ops.atomic_add(partial[chunk], values[i + values_offset])
        elif op == 1:
            ops.atomic_min(partial[chunk], values[i + values_offset])
        else:
            ops.atomic_max(partial[chunk], values[i + values_offset])


@kernel
def reduce_f32_field_private_reduce(partial: template(), output: template(), num_chunks: i32, op: i32):
    if op == 0:
        output[None] = 0.0
        for chunk in range(num_chunks):
            ops.atomic_add(output[None], partial[chunk])
    elif op == 1:
        output[None] = 3.4028234663852886e38
        for chunk in range(num_chunks):
            ops.atomic_min(output[None], partial[chunk])
    else:
        output[None] = -3.4028234663852886e38
        for chunk in range(num_chunks):
            ops.atomic_max(output[None], partial[chunk])


@kernel
def histogram_i32_field_direct(values: template(), bins: template(), N: i32, num_bins: i32):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    bins_offset = static(bins.snode.ptr.offset if len(bins.snode.ptr.offset) != 0 else 0)
    for i in range(num_bins):
        bins[i + bins_offset] = 0
    for i in range(N):
        bin_id = values[i + values_offset]
        if 0 <= bin_id < num_bins:
            ops.atomic_add(bins[bin_id + bins_offset], 1)


@kernel
def histogram_i32_field_private_count(
    values: template(),
    partial: template(),
    N: i32,
    num_bins: i32,
    chunk_size: i32,
    num_chunks: i32,
):
    values_offset = static(values.snode.ptr.offset if len(values.snode.ptr.offset) != 0 else 0)
    for i in range(num_chunks * num_bins):
        partial[i] = 0
    for i in range(N):
        bin_id = values[i + values_offset]
        if 0 <= bin_id < num_bins:
            chunk_id = i // chunk_size
            ops.atomic_add(partial[chunk_id * num_bins + bin_id], 1)


@kernel
def histogram_i32_field_private_reduce(
    partial: template(),
    bins: template(),
    num_bins: i32,
    num_chunks: i32,
):
    bins_offset = static(bins.snode.ptr.offset if len(bins.snode.ptr.offset) != 0 else 0)
    for bin_id in range(num_bins):
        total = 0
        for chunk_id in range(num_chunks):
            total += partial[chunk_id * num_bins + bin_id]
        bins[bin_id + bins_offset] = total
