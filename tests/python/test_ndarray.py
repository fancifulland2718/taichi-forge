import copy
import gc
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang import kernel_impl
from taichi_forge.lang.exception import TaichiIndexError, TaichiRuntimeError, TaichiSyntaxError, TaichiTypeError
from taichi_forge.lang.misc import get_host_arch_list
from taichi_forge.lang.util import has_pytorch
from taichi_forge.math import vec3, ivec3

import taichi_forge as ti
from tests import test_utils

if has_pytorch():
    import torch

# properties

data_types = [ti.i32, ti.f32, ti.i64, ti.f64]
ndarray_shapes = [(), 8, (6, 12)]
vector_dims = [3]
matrix_dims = [(1, 2), (2, 3)]
supported_archs_taichi_ndarray = [
    ti.cpu,
    ti.cuda,
    ti.opengl,
    ti.vulkan,
    ti.metal,
    ti.amdgpu,
]


def _test_scalar_ndarray(dtype, shape):
    x = ti.ndarray(dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)
    assert x.element_shape == ()

    assert x.dtype == dtype


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_scalar_ndarray(dtype, shape):
    _test_scalar_ndarray(dtype, shape)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_scalar_ndarray_f16_host_access():
    x = ti.ndarray(ti.f16, shape=(1,))
    x[0] = 1.5
    assert x[0] == test_utils.approx(1.5)


def _test_vector_ndarray(n, dtype, shape):
    x = ti.Vector.ndarray(n, dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)
    assert x.element_shape == (n,)

    assert x.dtype == dtype
    assert x.n == n


@pytest.mark.parametrize("n", vector_dims)
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_vector_ndarray(n, dtype, shape):
    _test_vector_ndarray(n, dtype, shape)


def _test_matrix_ndarray(n, m, dtype, shape):
    x = ti.Matrix.ndarray(n, m, dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)
    assert x.element_shape == (n, m)

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == m


@pytest.mark.parametrize("n,m", matrix_dims)
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_matrix_ndarray(n, m, dtype, shape):
    _test_matrix_ndarray(n, m, dtype, shape)


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_default_fp_ndarray(dtype):
    arch = ti.lang.impl.current_cfg().arch
    ti.reset()
    ti.init(arch=arch, default_fp=dtype)

    x = ti.Vector.ndarray(2, float, ())

    assert x.dtype == impl.get_runtime().default_fp


@pytest.mark.parametrize("dtype", [ti.i32, ti.i64])
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_default_ip_ndarray(dtype):
    arch = ti.lang.impl.current_cfg().arch
    ti.reset()
    ti.init(arch=arch, default_ip=dtype)

    x = ti.Vector.ndarray(2, int, ())

    assert x.dtype == impl.get_runtime().default_ip


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_1d():
    n = 4

    @ti.kernel
    def run(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in range(n):
            x[i] += i + y[i]

    a = ti.ndarray(ti.i32, shape=(n,))
    for i in range(n):
        a[i] = i * i
    b = np.ones((n,), dtype=np.int32)
    run(a, b)
    for i in range(n):
        assert a[i] == i * i + i + 1
    run(b, a)
    for i in range(n):
        assert b[i] == i * i + (i + 1) * 2


def _test_ndarray_2d():
    n = 4
    m = 7

    @ti.kernel
    def run(x: ti.types.ndarray(), y: ti.types.ndarray()):
        for i in range(n):
            for j in range(m):
                x[i, j] += i + j + y[i, j]

    a = ti.ndarray(ti.i32, shape=(n, m))
    for i in range(n):
        for j in range(m):
            a[i, j] = i * j
    b = np.ones((n, m), dtype=np.int32)
    run(a, b)
    for i in range(n):
        for j in range(m):
            assert a[i, j] == i * j + i + j + 1
    run(b, a)
    for i in range(n):
        for j in range(m):
            assert b[i, j] == i * j + (i + j + 1) * 2


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_2d():
    _test_ndarray_2d()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_compound_element():
    n = 10
    a = ti.ndarray(ti.i32, shape=(n,))

    vec3 = ti.types.vector(3, ti.i32)
    b = ti.ndarray(vec3, shape=(n, n))
    assert isinstance(b, ti.VectorNdarray)
    assert b.shape == (n, n)
    assert b.element_type.element_type() == ti.i32
    assert b.element_type.shape() == [3]

    matrix34 = ti.types.matrix(3, 4, float)
    c = ti.ndarray(matrix34, shape=(n, n + 1))
    assert isinstance(c, ti.MatrixNdarray)
    assert c.shape == (n, n + 1)
    assert c.element_type.element_type() == ti.f32
    assert c.element_type.shape() == [3, 4]


def test_struct_dtype_size_and_alignment_metadata():
    pixel = ti.types.struct(depth=ti.f32, color=ti.types.vector(3, ti.f32), idx=ti.i32)
    assert _ti_core.data_type_alignment(pixel.dtype) == 4
    assert _ti_core.data_type_size(pixel.dtype) == 20

    inner = ti.types.struct(a=ti.i8, b=ti.i32)
    outer = ti.types.struct(tag=ti.i8, payload=inner, weight=ti.f64)
    assert _ti_core.data_type_alignment(inner.dtype) == 4
    assert _ti_core.data_type_size(inner.dtype) == 8
    assert _ti_core.data_type_alignment(outer.dtype) == 8
    assert _ti_core.data_type_size(outer.dtype) == 24


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_struct_ndarray_numpy_roundtrip_and_copy():
    pixel = ti.types.struct(depth=ti.f32, color=ti.types.vector(3, ti.f32), idx=ti.i32)
    arr = ti.ndarray(pixel, shape=17)
    dst = ti.ndarray(pixel, shape=17)

    np_dtype = np.dtype(
        {
            "names": ["depth", "color", "idx"],
            "formats": [np.float32, (np.float32, (3,)), np.int32],
            "offsets": [0, 4, 16],
            "itemsize": 20,
        }
    )
    src = np.zeros((17,), dtype=np_dtype)
    src["depth"] = np.arange(17, dtype=np.float32) * 1.25
    src["color"] = np.arange(51, dtype=np.float32).reshape(17, 3)
    src["idx"] = np.arange(17, dtype=np.int32) * 3 - 5

    assert arr.numpy_dtype == np_dtype
    assert arr.element_shape == ()
    initial = arr.to_numpy()
    assert (initial["depth"] == 0).all()
    assert (initial["color"] == 0).all()
    assert (initial["idx"] == 0).all()
    arr.from_numpy(src)
    np.testing.assert_array_equal(arr.to_numpy(), src)

    dst.copy_from(arr)
    np.testing.assert_array_equal(dst.to_numpy(), src)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_struct_ndarray_native_bulk_no_compile():
    pixel = ti.types.struct(depth=ti.f32, color=ti.types.vector(3, ti.f32), idx=ti.i32)
    arr = ti.ndarray(pixel, shape=257)
    dst = ti.ndarray(pixel, shape=257)

    src = np.zeros((257,), dtype=arr.numpy_dtype)
    src["depth"] = np.arange(257, dtype=np.float32) * 0.5
    src["color"] = np.arange(257 * 3, dtype=np.float32).reshape(257, 3) - 11.0
    src["idx"] = np.arange(257, dtype=np.int32) * 7 - 13

    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    arr.from_numpy(src)
    np.testing.assert_array_equal(arr.to_numpy(), src)

    dst.fill(0)
    zeroed = dst.to_numpy()
    assert (zeroed["depth"] == 0).all()
    assert (zeroed["color"] == 0).all()
    assert (zeroed["idx"] == 0).all()

    dst.copy_from(arr)
    np.testing.assert_array_equal(dst.to_numpy(), src)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions


@test_utils.test(arch=get_host_arch_list())
def test_struct_ndarray_rejects_unsupported_python_access_and_dtype():
    pixel = ti.types.struct(depth=ti.f32, idx=ti.i32)
    arr = ti.ndarray(pixel, shape=4)
    bad_dtype = np.dtype([("depth", np.float32), ("idx", np.int64)])

    with pytest.raises(TypeError, match="Mismatch dtype"):
        arr.from_numpy(np.zeros((4,), dtype=bad_dtype))
    with pytest.raises(TaichiRuntimeError, match="Python item access is not supported yet"):
        _ = arr[0]
    with pytest.raises(TaichiRuntimeError, match="Python item assignment is not supported yet"):
        arr[0] = {"depth": 1.0, "idx": 1}
    ty = arr.get_type()
    assert ty.element_type is pixel
    assert ty.shape == (4,)
    assert not ty.needs_grad


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_struct_ndarray_kernel_argument_binding():
    pixel = ti.types.struct(depth=ti.f32, idx=ti.i32)
    arr = ti.ndarray(pixel, shape=8)
    out = ti.ndarray(ti.i32, shape=1)

    @ti.kernel
    def accept_typed(a: ti.types.ndarray(dtype=pixel, ndim=1), result: ti.types.ndarray(ti.i32, ndim=1)):
        result[0] = 7

    @ti.kernel
    def accept_untyped(a: ti.types.ndarray(), result: ti.types.ndarray(ti.i32, ndim=1)):
        result[0] += 5

    accept_typed(arr, out)
    accept_untyped(arr, out)
    assert out.to_numpy()[0] == 12


@test_utils.test(arch=get_host_arch_list())
def test_struct_ndarray_kernel_argument_dtype_mismatch():
    pixel = ti.types.struct(depth=ti.f32, idx=ti.i32)
    other = ti.types.struct(depth=ti.f64, idx=ti.i32)
    arr = ti.ndarray(pixel, shape=4)
    out = ti.ndarray(ti.i32, shape=1)

    @ti.kernel
    def accept_other(a: ti.types.ndarray(dtype=other, ndim=1), result: ti.types.ndarray(ti.i32, ndim=1)):
        result[0] = 1

    with pytest.raises(ValueError, match="required element type"):
        accept_other(arr, out)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_struct_ndarray_scalar_member_view_roundtrip():
    pixel = ti.types.struct(depth=ti.f32, color=ti.types.vector(3, ti.f32), idx=ti.i32)
    arr = ti.ndarray(pixel, shape=9)
    src = np.zeros((9,), dtype=arr.numpy_dtype)
    src["depth"] = np.arange(9, dtype=np.float32) + 0.5
    src["color"] = np.arange(27, dtype=np.float32).reshape(9, 3)
    src["idx"] = np.arange(9, dtype=np.int32) * 2 - 3
    arr.from_numpy(src)

    depth = arr.field("depth")
    idx = arr.field("idx")
    color = arr.field("color")
    color_y = arr.field("color", component=1)
    assert depth.base is arr
    assert depth.name == "depth"
    assert depth.dtype == ti.f32
    assert depth.shape == (9,)
    assert depth.offset == 0
    assert depth.stride == 20
    assert depth.element_size == 4
    assert depth.element_shape == ()
    assert idx.offset == 16
    assert color.name == "color"
    assert color.element_shape == (3,)
    assert color.offset == 4
    assert color.stride == 20
    assert color_y.name == "color[1]"
    assert color_y.offset == 8
    np.testing.assert_array_equal(depth.to_numpy(), src["depth"])
    np.testing.assert_array_equal(idx.to_numpy(), src["idx"])
    np.testing.assert_array_equal(color.to_numpy(), src["color"])
    np.testing.assert_array_equal(color_y.to_numpy(), src["color"][:, 1])

    updated_depth = np.arange(9, dtype=np.float32) * -1.25
    depth.from_numpy(updated_depth)
    expected = src.copy()
    expected["depth"] = updated_depth
    np.testing.assert_array_equal(arr.to_numpy(), expected)

    updated_color_y = np.arange(9, dtype=np.float32) * 2.5
    color_y.from_numpy(updated_color_y)
    expected["color"][:, 1] = updated_color_y
    np.testing.assert_array_equal(arr.to_numpy(), expected)

    updated_color = (np.arange(27, dtype=np.float32).reshape(9, 3) - 6.0) * 0.25
    color.from_numpy(updated_color)
    expected["color"] = updated_color
    np.testing.assert_array_equal(arr.to_numpy(), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_struct_ndarray_large_scalar_member_view_native_host_copy_no_compile():
    n = 65536
    pixel = ti.types.struct(depth=ti.f32, color=ti.types.vector(3, ti.f32), idx=ti.i32)
    arr = ti.ndarray(pixel, shape=n)
    src = np.zeros((n,), dtype=arr.numpy_dtype)
    src["depth"] = np.arange(n, dtype=np.float32) * 0.25
    src["color"] = np.arange(n * 3, dtype=np.float32).reshape(n, 3) - 9.0
    src["idx"] = np.arange(n, dtype=np.int32) * 2 - 5
    arr.from_numpy(src)

    depth = arr.field("depth")
    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    np.testing.assert_array_equal(depth.to_numpy(), src["depth"])
    assert depth._native_host_copy_tmp is not None

    updated = np.arange(n, dtype=np.float32) * -0.5
    depth.from_numpy(updated)
    expected = src.copy()
    expected["depth"] = updated
    np.testing.assert_array_equal(arr.to_numpy(), expected)

    fields = arr.to_numpy_fields("depth")
    np.testing.assert_array_equal(fields["depth"], updated)

    restored = src["depth"].copy()
    arr.from_numpy_fields({"depth": restored})
    expected["depth"] = restored
    np.testing.assert_array_equal(arr.to_numpy(), expected)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_struct_ndarray_large_tensor_member_view_native_host_copy_no_compile():
    n = 65536
    pixel = ti.types.struct(depth=ti.f32, color=ti.types.vector(3, ti.f32), idx=ti.i32)
    arr = ti.ndarray(pixel, shape=n)
    src = np.zeros((n,), dtype=arr.numpy_dtype)
    src["depth"] = np.arange(n, dtype=np.float32) * 0.25
    src["color"] = np.arange(n * 3, dtype=np.float32).reshape(n, 3) - 9.0
    src["idx"] = np.arange(n, dtype=np.int32) * 2 - 5
    arr.from_numpy(src)

    color = arr.field("color")
    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    np.testing.assert_array_equal(color.to_numpy(), src["color"])
    assert color._native_host_copy_tmp is not None

    updated = np.arange(n * 3, dtype=np.float32).reshape(n, 3) * -0.125
    color.from_numpy(updated)
    expected = src.copy()
    expected["color"] = updated
    np.testing.assert_array_equal(arr.to_numpy(), expected)

    fields = arr.to_numpy_fields("color")
    np.testing.assert_array_equal(fields["color"], updated)

    restored = src["color"].copy()
    arr.from_numpy_fields({"color": restored})
    expected["color"] = restored
    np.testing.assert_array_equal(arr.to_numpy(), expected)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions


@test_utils.test(arch=get_host_arch_list())
def test_struct_ndarray_scalar_member_view_rejections():
    inner = ti.types.struct(a=ti.i32)
    pixel = ti.types.struct(depth=ti.f32, color=ti.types.vector(3, ti.f32), payload=inner)
    arr = ti.ndarray(pixel, shape=4)
    depth = arr.field("depth")

    with pytest.raises(TypeError, match="expects a string"):
        arr.field(1)
    with pytest.raises(KeyError, match="no member"):
        arr.field("missing")
    with pytest.raises(TypeError, match="primitive scalar leaves"):
        arr.field("payload")
    with pytest.raises(TypeError, match="component=.*vector/matrix"):
        arr.field("depth", component=0)
    with pytest.raises(IndexError, match="out of bounds"):
        arr.field("color", component=3)
    with pytest.raises(ValueError, match="Mismatch shape"):
        depth.from_numpy(np.zeros((5,), dtype=np.float32))
    with pytest.raises(TypeError, match="Mismatch dtype"):
        depth.from_numpy(np.zeros((4,), dtype=np.float64))
    ty = depth.get_type()
    assert ty.element_type == ti.f32
    assert ty.shape == (4,)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_struct_ndarray_nested_member_views_and_host_helpers():
    payload = ti.types.struct(a=ti.i32, b=ti.types.vector(2, ti.f32))
    pixel = ti.types.struct(depth=ti.f32, payload=payload)
    arr = ti.ndarray(pixel, shape=6)
    src = np.zeros((6,), dtype=arr.numpy_dtype)
    src["depth"] = np.arange(6, dtype=np.float32) + 0.25
    src["payload"]["a"] = np.arange(6, dtype=np.int32) * 3 - 2
    src["payload"]["b"] = np.arange(12, dtype=np.float32).reshape(6, 2)
    arr.from_numpy(src)

    nested_a = arr.field("payload.a")
    nested_b1 = arr.field(("payload", "b"), component=1)
    np.testing.assert_array_equal(nested_a.to_numpy(), src["payload"]["a"])
    np.testing.assert_array_equal(nested_b1.to_numpy(), src["payload"]["b"][:, 1])

    updated_a = np.arange(6, dtype=np.int32) * -5
    updated_b1 = np.arange(6, dtype=np.float32) + 11.0
    nested_a.from_numpy(updated_a)
    nested_b1.from_numpy(updated_b1)
    expected = src.copy()
    expected["payload"]["a"] = updated_a
    expected["payload"]["b"][:, 1] = updated_b1
    np.testing.assert_array_equal(arr.to_numpy(), expected)

    fields = arr.to_numpy_fields("depth", "payload.a", ("payload", "b"))
    np.testing.assert_array_equal(fields["depth"], expected["depth"])
    np.testing.assert_array_equal(fields["payload.a"], expected["payload"]["a"])
    np.testing.assert_array_equal(fields["payload.b"], expected["payload"]["b"])

    depth = np.arange(6, dtype=np.float32) * -0.5
    payload_b = (np.arange(12, dtype=np.float32).reshape(6, 2) + 33.0).astype(np.float32)
    arr.from_numpy_fields({"depth": depth, ("payload", "b"): payload_b})
    expected["depth"] = depth
    expected["payload"]["b"] = payload_b
    np.testing.assert_array_equal(arr.to_numpy(), expected)

    item = arr.debug_getitem(2)
    assert item["payload"]["a"] == expected["payload"]["a"][2]
    item["payload"]["a"] = np.int32(123)
    arr.debug_setitem(2, item)
    expected["payload"]["a"][2] = 123
    np.testing.assert_array_equal(arr.to_numpy(), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_struct_ndarray_scalar_member_view_kernel_argument_read_write():
    payload = ti.types.struct(a=ti.i32, b=ti.types.vector(2, ti.f32))
    pixel = ti.types.struct(depth=ti.f32, payload=payload)
    arr = ti.ndarray(pixel, shape=32)
    src = np.zeros((32,), dtype=arr.numpy_dtype)
    src["depth"] = np.arange(32, dtype=np.float32) + 1.0
    src["payload"]["a"] = np.arange(32, dtype=np.int32) * 2
    src["payload"]["b"] = np.arange(64, dtype=np.float32).reshape(32, 2)
    arr.from_numpy(src)

    @ti.kernel
    def update(
        depth: ti.types.ndarray(dtype=ti.f32, ndim=1),
        nested_a: ti.types.ndarray(dtype=ti.i32, ndim=1),
        b1: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(depth.shape[0]):
            depth[i] = depth[i] * 2.0
            nested_a[i] = nested_a[i] + 7
            b1[i] = b1[i] - 3.5

    update(
        arr.field("depth"),
        arr.field("payload.a"),
        arr.field(("payload", "b"), component=1),
    )

    expected = src.copy()
    expected["depth"] *= 2.0
    expected["payload"]["a"] += 7
    expected["payload"]["b"][:, 1] -= 3.5
    np.testing.assert_array_equal(arr.to_numpy(), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_struct_ndarray_kernel_field_access_read_only():
    payload = ti.types.struct(a=ti.i32, b=ti.types.vector(2, ti.f32))
    pixel = ti.types.struct(depth=ti.f32, payload=payload, basis=ti.types.matrix(2, 2, ti.f32), tag=ti.i32)
    arr = ti.ndarray(pixel, shape=32)
    out = ti.ndarray(ti.f32, shape=1)
    out_i = ti.ndarray(ti.i32, shape=1)
    components = ti.ndarray(ti.f32, shape=4)
    src = np.zeros((32,), dtype=arr.numpy_dtype)
    src["depth"] = np.arange(32, dtype=np.float32) + 1.0
    src["payload"]["a"] = np.arange(32, dtype=np.int32) * 3 - 5
    src["payload"]["b"] = np.arange(64, dtype=np.float32).reshape(32, 2)
    src["basis"] = np.arange(128, dtype=np.float32).reshape(32, 2, 2)
    src["tag"] = np.arange(32, dtype=np.int32) + 100
    arr.from_numpy(src)
    out.fill(0)
    out_i.fill(0)
    components.fill(0)

    @ti.kernel
    def read_typed(p: ti.types.ndarray(dtype=pixel, ndim=1), result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        result[0] = p[5].depth

    @ti.kernel
    def read_untyped(p: ti.types.ndarray(), result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        result[0] += p[7].depth

    @ti.kernel
    def write_scalar_leaves(p: ti.types.ndarray(dtype=pixel, ndim=1)):
        p[0].depth = p[0].depth + 1.0
        p[1].payload.a = p[1].payload.a + 8

    @ti.kernel
    def write_tensor_members(p: ti.types.ndarray(dtype=pixel, ndim=1)):
        p[10].payload.b = ti.Vector([-1.5, 2.5])
        p[11].basis = ti.Matrix([[3.0, 4.0], [5.0, 6.0]])
        p[12].payload = ti.Struct(a=77, b=ti.Vector([7.0, 8.0]))
        p[13].payload.b += ti.Vector([1.25, -2.25])
        p[14].basis[0, 1] = p[14].basis[0, 1] + 9.0

    @ti.kernel
    def update_tensor_member_views(
        vec: ti.types.ndarray(dtype=ti.types.vector(2, ti.f32), ndim=1),
        mat: ti.types.ndarray(dtype=ti.types.matrix(2, 2, ti.f32), ndim=1),
        result: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        v = vec[15]
        m = mat[16]
        result[0] += v.dot(v) + m.trace()
        vec[17] = ti.Vector([11.0, 12.0])
        mat[18] = ti.Matrix([[13.0, 14.0], [15.0, 16.0]])
        vec[19][1] = vec[19][1] + 5.0
        mat[20][1, 0] = mat[20][1, 0] - 7.0

    @ti.kernel
    def read_nested_tensor_members(
        p: ti.types.ndarray(dtype=pixel, ndim=1),
        result_f: ti.types.ndarray(dtype=ti.f32, ndim=1),
        result_i: ti.types.ndarray(dtype=ti.i32, ndim=1),
        result_components: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        v = p[4].payload.b
        result_f[0] += v[0] * 2.0 + v[1] + p[6].basis[1, 0] - p[6].basis[0, 1]
        result_i[0] = p[3].payload.a + p[3].tag
        result_components[0] = v[0]
        result_components[1] = v[1]
        result_components[2] = p[6].basis[1, 0]
        result_components[3] = p[6].basis[0, 1]

    @ti.kernel
    def read_nested_tensor_numeric_ops(p: ti.types.ndarray(dtype=pixel, ndim=1), result: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        v = p[2].payload.b
        m = p[8].basis
        mv = m @ v
        result[0] += v.dot(v) + mv[0] - mv[1] + m.trace()

    read_typed(arr, out)
    read_untyped(arr, out)
    read_nested_tensor_members(arr, out, out_i, components)
    read_nested_tensor_numeric_ops(arr, out)
    write_scalar_leaves(arr)
    write_tensor_members(arr)
    update_tensor_member_views(arr.field("payload.b"), arr.field("basis"), out)

    expected = src.copy()
    expected["depth"][0] += 1.0
    expected["payload"]["a"][1] += 8
    expected["payload"]["b"][10] = np.array([-1.5, 2.5], dtype=np.float32)
    expected["basis"][11] = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    expected["payload"]["a"][12] = 77
    expected["payload"]["b"][12] = np.array([7.0, 8.0], dtype=np.float32)
    expected["payload"]["b"][13] += np.array([1.25, -2.25], dtype=np.float32)
    expected["basis"][14, 0, 1] += 9.0
    expected["payload"]["b"][17] = np.array([11.0, 12.0], dtype=np.float32)
    expected["basis"][18] = np.array([[13.0, 14.0], [15.0, 16.0]], dtype=np.float32)
    expected["payload"]["b"][19, 1] += 5.0
    expected["basis"][20, 1, 0] -= 7.0
    np.testing.assert_array_equal(arr.to_numpy(), expected)
    expected_out = (
        src["depth"][5]
        + src["depth"][7]
        + src["payload"]["b"][4, 0] * 2.0
        + src["payload"]["b"][4, 1]
        + src["basis"][6, 1, 0]
        - src["basis"][6, 0, 1]
        + np.dot(src["payload"]["b"][2], src["payload"]["b"][2])
        + (src["basis"][8] @ src["payload"]["b"][2])[0]
        - (src["basis"][8] @ src["payload"]["b"][2])[1]
        + np.trace(src["basis"][8])
        + np.dot(src["payload"]["b"][15], src["payload"]["b"][15])
        + np.trace(src["basis"][16])
    )
    np.testing.assert_allclose(
        components.to_numpy(),
        np.array(
            [
                src["payload"]["b"][4, 0],
                src["payload"]["b"][4, 1],
                src["basis"][6, 1, 0],
                src["basis"][6, 0, 1],
            ]
        ),
    )
    np.testing.assert_allclose(out.to_numpy()[0], expected_out)
    np.testing.assert_array_equal(out_i.to_numpy()[0], src["payload"]["a"][3] + src["tag"][3])


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_copy_from_ndarray():
    n = 16
    a = ti.ndarray(ti.i32, shape=n)
    b = ti.ndarray(ti.i32, shape=n)
    a[0] = 1
    a[4] = 2
    b[0] = 4
    b[4] = 5

    a.copy_from(b)

    assert a[0] == 4
    assert a[4] == 5

    x = ti.Vector.ndarray(10, ti.i32, 5)
    y = ti.Vector.ndarray(10, ti.i32, 5)
    x[1][0] = 1
    x[2][4] = 2
    y[1][0] = 4
    y[2][4] = 5

    x.copy_from(y)

    assert x[1][0] == 4
    assert x[2][4] == 5

    x = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    y = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    x[0][0, 0] = 1
    x[4][1, 0] = 3
    y[0][0, 0] = 4
    y[4][1, 0] = 6

    x.copy_from(y)

    assert x[0][0, 0] == 4
    assert x[4][1, 0] == 6


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_ndarray_native_copy_from():
    n = 17

    src = ti.ndarray(ti.f64, shape=n)
    dst = ti.ndarray(ti.f64, shape=n)
    src_np = np.arange(n, dtype=np.float64) * 1.5
    src.from_numpy(src_np)
    dst.fill(0)
    dst.copy_from(src)
    assert (dst.to_numpy() == src_np).all()

    src_vec = ti.Vector.ndarray(3, ti.f32, shape=n)
    dst_vec = ti.Vector.ndarray(3, ti.f32, shape=n)
    src_vec_np = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    src_vec.from_numpy(src_vec_np)
    dst_vec.fill(0)
    dst_vec.copy_from(src_vec)
    assert (dst_vec.to_numpy() == src_vec_np).all()

    src_mat = ti.Matrix.ndarray(2, 2, ti.i32, shape=n)
    dst_mat = ti.Matrix.ndarray(2, 2, ti.i32, shape=n)
    src_mat_np = np.arange(n * 4, dtype=np.int32).reshape(n, 2, 2)
    src_mat.from_numpy(src_mat_np)
    dst_mat.fill(0)
    dst_mat.copy_from(src_mat)
    assert (dst_mat.to_numpy() == src_mat_np).all()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_scalar_ndarray_native_copy_from_tape_grad_no_compile():
    n = 16
    src = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    dst = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.arange(n, dtype=np.float32))

    @ti.kernel
    def reduce_dst(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            loss[None] += 3.0 * arr[i]

    with ti.ad.Tape(loss):
        compiled_functions = impl.get_runtime().get_num_compiled_functions()
        dst.copy_from(src)
        assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
        reduce_dst(dst)

    np.testing.assert_array_equal(
        src.grad.to_numpy(), np.full((n,), 3.0, dtype=np.float32)
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_matrix_ndarray_native_copy_from_tape_grad_no_compile():
    n = 8
    src = ti.ndarray(ti.math.mat2, shape=n, needs_grad=True)
    dst = ti.ndarray(ti.math.mat2, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2))

    @ti.kernel
    def reduce_dst(arr: ti.types.ndarray(dtype=ti.math.mat2, ndim=1)):
        for i in range(n):
            for j, k in ti.static(ti.ndrange(2, 2)):
                loss[None] += (j + 2 * k + 1) * arr[i][j, k]

    expected = np.zeros((n, 2, 2), dtype=np.float32)
    for j in range(2):
        for k in range(2):
            expected[:, j, k] = j + 2 * k + 1

    with ti.ad.Tape(loss):
        compiled_functions = impl.get_runtime().get_num_compiled_functions()
        dst.copy_from(src)
        assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
        reduce_dst(dst)

    np.testing.assert_array_equal(src.grad.to_numpy(), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_ndarray_copy_from_dtype_cast_fallback():
    n = 8
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.f32, shape=n)
    src_np = np.arange(n, dtype=np.int32) - 3
    src.from_numpy(src_np)
    dst.fill(0)
    dst.copy_from(src)
    assert (dst.to_numpy() == src_np.astype(np.float32)).all()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_ndarray_native_host_staging():
    n = 19

    scalar = ti.ndarray(ti.i64, shape=n)
    scalar_np = np.arange(n, dtype=np.int64) * 3 - 11
    scalar.from_numpy(scalar_np)
    np.testing.assert_array_equal(scalar.to_numpy(), scalar_np)

    vector = ti.Vector.ndarray(3, ti.f32, shape=n)
    vector_np = (np.arange(n * 3, dtype=np.float32).reshape(n, 3) - 7.0) * 0.25
    vector.from_numpy(vector_np)
    np.testing.assert_array_equal(vector.to_numpy(), vector_np)

    matrix = ti.Matrix.ndarray(2, 2, ti.i32, shape=n)
    matrix_np = np.arange(n * 4, dtype=np.int32).reshape(n, 2, 2) - 5
    matrix.from_numpy(matrix_np)
    np.testing.assert_array_equal(matrix.to_numpy(), matrix_np)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_deepcopy():
    n = 16
    x = ti.ndarray(ti.i32, shape=n)
    x[0] = 1
    x[4] = 2

    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y[0] == 1
    assert y[4] == 2
    x[0] = 4
    x[4] = 5
    assert y[0] == 1
    assert y[4] == 2

    x = ti.Vector.ndarray(10, ti.i32, 5)
    x[1][0] = 4
    x[2][4] = 5

    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.n == x.n
    assert y[1][0] == 4
    assert y[2][4] == 5
    x[1][0] = 1
    x[2][4] = 2
    assert y[1][0] == 4
    assert y[2][4] == 5

    x = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    x[0][0, 0] = 7
    x[4][1, 0] = 9

    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.m == x.m
    assert y.n == x.n
    assert y[0][0, 0] == 7
    assert y[4][1, 0] == 9
    x[0][0, 0] = 3
    x[4][1, 0] = 5
    assert y[0][0, 0] == 7
    assert y[4][1, 0] == 9


@test_utils.test(arch=[ti.cuda])
def test_ndarray_caching_allocator():
    n = 8
    a = ti.ndarray(ti.i32, shape=(n))
    a.fill(2)
    a = 1
    b = ti.ndarray(ti.i32, shape=(n))
    b.fill(2)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_fill():
    n = 8
    a = ti.ndarray(ti.i32, shape=(n))
    anp = np.ones((n,), dtype=np.int32)
    a.fill(2)
    anp.fill(2)
    assert (a.to_numpy() == anp).all()

    b = ti.Vector.ndarray(4, ti.f32, shape=(n))
    bnp = np.ones(shape=b.arr.total_shape(), dtype=np.float32)
    b.fill(2.5)
    bnp.fill(2.5)
    assert (b.to_numpy() == bnp).all()

    c = ti.Matrix.ndarray(4, 4, ti.f32, shape=(n))
    cnp = np.ones(shape=c.arr.total_shape(), dtype=np.float32)
    c.fill(1.5)
    cnp.fill(1.5)
    assert (c.to_numpy() == cnp).all()


@test_utils.test(arch=[ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_ndarray_scalar_fill_vulkan_reset_with_live_ndarray():
    n = 4096
    arr = ti.ndarray(ti.i32, shape=n)
    arr.fill(7)
    assert (arr.to_numpy() == np.full((n,), 7, dtype=np.int32)).all()
    arr.fill(-3)
    assert (arr.to_numpy() == np.full((n,), -3, dtype=np.int32)).all()
    ti.reset()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_ndarray_native_zero_fill():
    n = 17

    scalar = ti.ndarray(ti.f64, shape=n)
    scalar.fill(3.5)
    scalar.fill(0.0)
    assert (scalar.to_numpy() == np.zeros((n,), dtype=np.float64)).all()

    vec = ti.Vector.ndarray(3, ti.f32, shape=n)
    vec.fill(2.0)
    vec.fill(0)
    assert (vec.to_numpy() == np.zeros((n, 3), dtype=np.float32)).all()

    mat = ti.Matrix.ndarray(2, 2, ti.i32, shape=n)
    mat.fill(5)
    mat.fill(0)
    assert (mat.to_numpy() == np.zeros((n, 2, 2), dtype=np.int32)).all()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_matrix_ndarray_scalar_fill_native_no_compile():
    n = 257

    vec = ti.Vector.ndarray(4, ti.f32, shape=n)
    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    vec.fill(2.5)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
    np.testing.assert_array_equal(
        vec.to_numpy(), np.full((n, 4), 2.5, dtype=np.float32)
    )

    mat = ti.Matrix.ndarray(2, 3, ti.i32, shape=n)
    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    mat.fill(7)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
    np.testing.assert_array_equal(
        mat.to_numpy(), np.full((n, 2, 3), 7, dtype=np.int32)
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_ndarray_64bit_scalar_fill_no_compile():
    n = 257

    scalar_f64 = ti.ndarray(ti.f64, shape=n)
    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    scalar_f64.fill(-3.25)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
    np.testing.assert_array_equal(
        scalar_f64.to_numpy(), np.full((n,), -3.25, dtype=np.float64)
    )

    scalar_i64 = ti.ndarray(ti.i64, shape=n)
    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    scalar_i64.fill(-1234567890123)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
    np.testing.assert_array_equal(
        scalar_i64.to_numpy(), np.full((n,), -1234567890123, dtype=np.int64)
    )

    vec_f64 = ti.Vector.ndarray(3, ti.f64, shape=n)
    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    vec_f64.fill(1.25)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
    np.testing.assert_array_equal(
        vec_f64.to_numpy(), np.full((n, 3), 1.25, dtype=np.float64)
    )

    mat_u64 = ti.Matrix.ndarray(2, 2, ti.u64, shape=n)
    fill_value = np.uint64(0xFEDCBA9876543210)
    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    mat_u64.fill(fill_value)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
    np.testing.assert_array_equal(
        mat_u64.to_numpy(), np.full((n, 2, 2), fill_value, dtype=np.uint64)
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_ndarray_large_native_host_copy_no_compile():
    n = 8192
    arr = ti.Vector.ndarray(4, ti.f32, shape=n)
    src = np.arange(n * 4, dtype=np.float32).reshape(n, 4) * 0.25

    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    arr.from_numpy(src)
    np.testing.assert_array_equal(arr.to_numpy(), src)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_rw_cache():
    a = ti.Vector.ndarray(3, ti.f32, ())
    b = ti.Vector.ndarray(3, ti.f32, 12)

    n = 100
    for i in range(n):
        c_a = copy.deepcopy(a)
        c_b = copy.deepcopy(b)
        c_a[None] = c_b[10]


def _test_ndarray_numpy_io():
    n = 7
    m = 4
    a = ti.ndarray(ti.i32, shape=(n, m))
    a.fill(2)
    b = ti.ndarray(ti.i32, shape=(n, m))
    b.from_numpy(np.ones((n, m), dtype=np.int32) * 2)
    assert (a.to_numpy() == b.to_numpy()).all()

    d = 2
    p = 4
    x = ti.Vector.ndarray(d, ti.f32, p)
    x.fill(2)
    y = ti.Vector.ndarray(d, ti.f32, p)
    y.from_numpy(np.ones((p, d), dtype=np.int32) * 2)
    assert (x.to_numpy() == y.to_numpy()).all()

    c = 2
    d = 2
    p = 4
    x = ti.Matrix.ndarray(c, d, ti.f32, p)
    x.fill(2)
    y = ti.Matrix.ndarray(c, d, ti.f32, p)
    y.from_numpy(np.ones((p, c, d), dtype=np.int32) * 2)
    assert (x.to_numpy() == y.to_numpy()).all()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_numpy_io():
    _test_ndarray_numpy_io()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_matrix_numpy_io():
    n = 5
    m = 2

    x = ti.Vector.ndarray(n, ti.i32, (m,))
    x_np = 1 + np.arange(n * m).reshape(m, n).astype(np.int32)
    x.from_numpy(x_np)
    assert (x_np.flatten() == x.to_numpy().flatten()).all()

    k = 2
    x = ti.Matrix.ndarray(m, k, ti.i32, n)
    x_np = 1 + np.arange(m * k * n).reshape(n, m, k).astype(np.int32)
    x.from_numpy(x_np)
    assert (x_np.flatten() == x.to_numpy().flatten()).all()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_matrix_ndarray_python_scope():
    a = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    for i in range(5):
        for j, k in ti.ndrange(2, 2):
            a[i][j, k] = j * j + k * k
    assert a[0][0, 0] == 0
    assert a[1][0, 1] == 1
    assert a[2][1, 0] == 1
    assert a[3][1, 1] == 2
    assert a[4][0, 1] == 1


def _test_matrix_ndarray_taichi_scope():
    @ti.kernel
    def func(a: ti.types.ndarray()):
        for i in range(5):
            for j, k in ti.ndrange(2, 2):
                a[i][j, k] = j * j + k * k

    m = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    func(m)
    assert m[0][0, 0] == 0
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_matrix_ndarray_taichi_scope():
    _test_matrix_ndarray_taichi_scope()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_matrix_ndarray_taichi_scope_real_matrix():
    _test_matrix_ndarray_taichi_scope()


def _test_matrix_ndarray_taichi_scope_struct_for():
    @ti.kernel
    def func(a: ti.types.ndarray()):
        for i in a:
            for j, k in ti.ndrange(2, 2):
                a[i][j, k] = j * j + k * k

    m = ti.Matrix.ndarray(2, 2, ti.i32, 5)
    func(m)
    assert m[0][0, 0] == 0
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_matrix_ndarray_taichi_scope_struct_for():
    _test_matrix_ndarray_taichi_scope_struct_for()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_matrix_ndarray_taichi_scope_struct_for_real_matrix():
    _test_matrix_ndarray_taichi_scope_struct_for()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_vector_ndarray_python_scope():
    a = ti.Vector.ndarray(10, ti.i32, 5)
    for i in range(5):
        for j in range(4):
            a[i][j * j] = j * j
    assert a[0][9] == 9
    assert a[1][0] == 0
    assert a[2][1] == 1
    assert a[3][4] == 4
    assert a[4][9] == 9


def _test_vector_ndarray_taichi_scope():
    @ti.kernel
    def func(a: ti.types.ndarray()):
        for i in range(5):
            for j in range(4):
                a[i][j * j] = j * j

    v = ti.Vector.ndarray(10, ti.i32, 5)
    func(v)
    assert v[0][9] == 9
    assert v[1][0] == 0
    assert v[2][1] == 1
    assert v[3][4] == 4
    assert v[4][9] == 9


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_vector_ndarray_taichi_scope():
    _test_vector_ndarray_taichi_scope()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_vector_ndarray_taichi_scope_real_matrix():
    _test_vector_ndarray_taichi_scope()


# number of compiled functions
def _test_compiled_functions():
    @ti.kernel
    def func(a: ti.types.ndarray(ti.types.vector(n=10, dtype=ti.i32))):
        for i in range(5):
            for j in range(4):
                a[i][j * j] = j * j

    v = ti.Vector.ndarray(10, ti.i32, 5)
    func(v)
    assert impl.get_runtime().get_num_compiled_functions() == 1
    v = np.zeros((6, 10), dtype=np.int32)
    func(v)
    assert impl.get_runtime().get_num_compiled_functions() == 1


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_compiled_functions():
    _test_compiled_functions()


# annotation compatibility


def _test_arg_not_match():
    @ti.kernel
    def func1(a: ti.types.ndarray(dtype=ti.types.vector(2, ti.i32))):
        pass

    x = ti.Matrix.ndarray(2, 3, ti.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required element type: VectorType\[2, i32\], but .* is provided",
    ):
        func1(x)

    x = ti.Matrix.ndarray(2, 1, ti.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required element type: VectorType\[2, i32\], but .* is provided",
    ):
        func1(x)

    @ti.kernel
    def func2(a: ti.types.ndarray(dtype=ti.types.matrix(2, 2, ti.i32))):
        pass

    x = ti.Vector.ndarray(2, ti.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required element type: MatrixType\[2,2, i32\], but .* is provided",
    ):
        func2(x)

    @ti.kernel
    def func3(a: ti.types.ndarray(dtype=ti.types.matrix(2, 1, ti.i32))):
        pass

    x = ti.Vector.ndarray(2, ti.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required element type: MatrixType\[2,1, i32\], but .* is provided",
    ):
        func3(x)

    @ti.kernel
    def func5(a: ti.types.ndarray(dtype=ti.types.matrix(2, 3, dtype=ti.i32))):
        pass

    x = ti.Vector.ndarray(2, ti.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required element type",
    ):
        func5(x)

    @ti.kernel
    def func7(a: ti.types.ndarray(ndim=2)):
        pass

    x = ti.ndarray(ti.i32, shape=(3,))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required ndim",
    ):
        func7(x)

    @ti.kernel
    def func8(x: ti.types.ndarray(dtype=ti.f32)):
        pass

    x = ti.ndarray(dtype=ti.i32, shape=(16, 16))
    with pytest.raises(TypeError, match=r"Expect element type .* for argument x, but get .*"):
        func8(x)


@test_utils.test(arch=get_host_arch_list())
def test_arg_not_match():
    _test_arg_not_match()


def _test_size_in_bytes():
    a = ti.ndarray(ti.i32, 8)
    assert a._get_element_size() == 4
    assert a._get_nelement() == 8

    b = ti.Vector.ndarray(10, ti.f64, 5)
    assert b._get_element_size() == 80
    assert b._get_nelement() == 5


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_size_in_bytes():
    _test_size_in_bytes()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_different_shape():
    n1 = 4
    x = ti.ndarray(dtype=ti.f32, shape=(n1, n1))

    @ti.kernel
    def init(d: ti.i32, arr: ti.types.ndarray()):
        for i, j in arr:
            arr[i, j] = d

    init(2, x)
    assert (x.to_numpy() == (np.ones(shape=(n1, n1)) * 2)).all()
    n2 = 8
    y = ti.ndarray(dtype=ti.f32, shape=(n2, n2))
    init(3, y)
    assert (y.to_numpy() == (np.ones(shape=(n2, n2)) * 3)).all()


def _test_ndarray_grouped():
    @ti.kernel
    def func(a: ti.types.ndarray()):
        for i in ti.grouped(a):
            for j, k in ti.ndrange(2, 2):
                a[i][j, k] = j * j

    a1 = ti.Matrix.ndarray(2, 2, ti.i32, shape=5)
    func(a1)
    for i in range(5):
        for j in range(2):
            for k in range(2):
                assert a1[i][j, k] == j * j

    a2 = ti.Matrix.ndarray(2, 2, ti.i32, shape=(3, 3))
    func(a2)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                for p in range(2):
                    assert a2[i, j][k, p] == k * k


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_grouped():
    _test_ndarray_grouped()


@test_utils.test(arch=[ti.cpu, ti.cuda], real_matrix_scalarize=False)
def test_ndarray_grouped_real_matrix():
    _test_ndarray_grouped()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_as_template():
    @ti.kernel
    def func(arr_src: ti.template(), arr_dst: ti.template()):
        for i, j in ti.ndrange(*arr_src.shape):
            arr_dst[i, j] = arr_src[i, j]

    arr_0 = ti.ndarray(ti.f32, shape=(5, 10))
    arr_1 = ti.ndarray(ti.f32, shape=(5, 10))
    with pytest.raises(ti.TaichiRuntimeTypeError, match=r"Ndarray shouldn't be passed in via"):
        func(arr_0, arr_1)


@pytest.mark.parametrize("shape", [2**31, 1.5, 0, (1, 0), (1, 0.5), (1, 2**31)])
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_shape_invalid(shape):
    with pytest.raises(TaichiRuntimeError, match=r"is not a valid shape for ndarray"):
        x = ti.ndarray(dtype=int, shape=shape)


@pytest.mark.parametrize("shape", [1, np.int32(1), (1, np.int32(1), 4096)])
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_shape_valid(shape):
    x = ti.ndarray(dtype=int, shape=shape)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_gaussian_kernel():
    M_PI = 3.14159265358979323846

    @ti.func
    def gaussian(x, sigma):
        return ti.exp(-0.5 * ti.pow(x / sigma, 2)) / (sigma * ti.sqrt(2.0 * M_PI))

    @ti.kernel
    def fill_gaussian_kernel(ker: ti.types.ndarray(ti.f32, ndim=1), N: ti.i32):
        sum = 0.0
        for i in range(2 * N + 1):
            ker[i] = gaussian(i - N, ti.sqrt(N))
            sum += ker[i]
        for i in range(2 * N + 1):
            ker[i] = ker[i] / sum

    N = 4
    arr = ti.ndarray(dtype=ti.f32, shape=(20))
    fill_gaussian_kernel(arr, N)
    res = arr.to_numpy()

    np_arr = np.zeros(20, dtype=np.float32)
    fill_gaussian_kernel(np_arr, N)

    assert test_utils.allclose(res, np_arr)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_numpy_matrix():
    boundary_box_np = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    boundary_box = ti.Vector.ndarray(3, ti.f32, shape=2)
    boundary_box.from_numpy(boundary_box_np)
    ref_numpy = boundary_box.to_numpy()

    assert (boundary_box_np == ref_numpy).all()


@pytest.mark.parametrize("dtype", [ti.i64, ti.u64, ti.f64])
@test_utils.test(arch=supported_archs_taichi_ndarray, require=ti.extension.data64)
def test_ndarray_python_scope_read_64bit(dtype):
    @ti.kernel
    def run(x: ti.types.ndarray()):
        for i in x:
            x[i] = i + ti.i64(2**40)

    n = 4
    a = ti.ndarray(dtype, shape=(n,))
    run(a)
    for i in range(n):
        assert a[i] == i + 2**40


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_init_as_zero():
    a = ti.ndarray(dtype=ti.f32, shape=(6, 10))
    v = np.zeros((6, 10), dtype=np.float32)
    assert test_utils.allclose(a.to_numpy(), v)

    b = ti.ndarray(dtype=ti.math.vec2, shape=(6, 4))
    k = np.zeros((6, 4, 2), dtype=np.float32)
    assert test_utils.allclose(b.to_numpy(), k)

    c = ti.ndarray(dtype=ti.math.mat2, shape=(6, 4))
    m = np.zeros((6, 4, 2, 2), dtype=np.float32)
    assert test_utils.allclose(c.to_numpy(), m)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_zero_fill():
    dt = ti.types.vector(n=2, dtype=ti.f32)
    arr = ti.ndarray(dtype=dt, shape=(3, 4))

    arr.fill(1.0)

    arr.to_numpy()
    no = ti.ndarray(dtype=dt, shape=(3, 5))
    assert no[0, 0][0] == 0.0


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_reset():
    n = 8
    c = ti.Matrix.ndarray(4, 4, ti.f32, shape=(n))
    del c
    d = ti.Matrix.ndarray(4, 4, ti.f32, shape=(n))
    ti.reset()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_ndarray_registry_retire_waits_for_gpu_sync():
    sink = ti.field(ti.i32, shape=())

    @ti.kernel
    def consume(arr: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        sink[None] += arr[0]

    warmup = ti.ndarray(ti.i32, shape=1)
    warmup.fill(0)
    consume(warmup)
    ti.sync()
    del warmup
    gc.collect()
    ti.sync()

    prog = impl.get_runtime().prog
    baseline = prog._debug_ndarray_resource_stats()
    arr = ti.ndarray(ti.i32, shape=1)
    arr.fill(7)
    created = prog._debug_ndarray_resource_stats()
    assert created["created_total"] == baseline["created_total"] + 1
    assert created["live"] == baseline["live"] + 1
    assert created["views"] == baseline["views"] + 1
    assert created["leases"] == baseline["leases"] + 1

    consume(arr)
    launched = prog._debug_ndarray_resource_stats()
    if impl.current_cfg().arch == ti.cpu:
        assert launched["inflight"] == baseline["inflight"]
        assert launched["leases"] == baseline["leases"] + 1
    else:
        assert launched["inflight"] == baseline["inflight"] + 1
        assert launched["leases"] == baseline["leases"] + 2

    del arr
    gc.collect()
    retired = prog._debug_ndarray_resource_stats()
    assert retired["views"] == baseline["views"]
    assert retired["retired_total"] == baseline["retired_total"] + 1
    if impl.current_cfg().arch == ti.cpu:
        assert retired["released_total"] == baseline["released_total"] + 1
    else:
        assert retired["retiring"] == baseline["retiring"] + 1
        assert retired["released_total"] == baseline["released_total"]

    ti.sync()
    completed = prog._debug_ndarray_resource_stats()
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key]
    assert completed["created_total"] == baseline["created_total"] + 1
    assert completed["retired_total"] == baseline["retired_total"] + 1
    assert completed["released_total"] == baseline["released_total"] + 1
    assert sink[None] == 7


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_ndarray_registry_concurrent_submit_and_gc():
    sink = ti.field(ti.i32, shape=())

    @ti.kernel
    def consume(arr: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        ti.atomic_add(sink[None], arr[0])

    warmup = ti.ndarray(ti.i32, shape=1)
    warmup.fill(0)
    consume(warmup)
    ti.sync()
    del warmup
    gc.collect()
    ti.sync()

    prog = impl.get_runtime().prog
    baseline = prog._debug_ndarray_resource_stats()
    errors = []
    thread_count = 4
    iterations = 8

    def worker(worker_id):
        try:
            for i in range(iterations):
                value = worker_id * iterations + i + 1
                arr = ti.ndarray(ti.i32, shape=1)
                arr.fill(value)
                consume(arr)
                del arr
                if i % 2 == 0:
                    gc.collect()
        except BaseException as exc:
            errors.append(exc)

    workers = [
        threading.Thread(target=worker, args=(i,)) for i in range(thread_count)
    ]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    assert not errors

    gc.collect()
    ti.sync()
    completed = prog._debug_ndarray_resource_stats()
    expected = thread_count * iterations
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key]
    assert completed["created_total"] - baseline["created_total"] == expected
    assert completed["retired_total"] - baseline["retired_total"] == expected
    assert completed["released_total"] - baseline["released_total"] == expected
    assert sink[None] == expected * (expected + 1) // 2


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_ndarray_registry_more_than_inline_launch_leases():
    sink = ti.field(ti.i32, shape=())

    @ti.kernel
    def consume(
        a0: ti.types.ndarray(dtype=ti.i32, ndim=1),
        a1: ti.types.ndarray(dtype=ti.i32, ndim=1),
        a2: ti.types.ndarray(dtype=ti.i32, ndim=1),
        a3: ti.types.ndarray(dtype=ti.i32, ndim=1),
        a4: ti.types.ndarray(dtype=ti.i32, ndim=1),
        a5: ti.types.ndarray(dtype=ti.i32, ndim=1),
        a6: ti.types.ndarray(dtype=ti.i32, ndim=1),
        a7: ti.types.ndarray(dtype=ti.i32, ndim=1),
        a8: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        sink[None] = (
            a0[0]
            + a1[0]
            + a2[0]
            + a3[0]
            + a4[0]
            + a5[0]
            + a6[0]
            + a7[0]
            + a8[0]
        )

    arrays = [ti.ndarray(ti.i32, shape=1) for _ in range(9)]
    for i, arr in enumerate(arrays, 1):
        arr.fill(i)
    del arr
    consume(*arrays)
    prog = impl.get_runtime().prog
    launched = prog._debug_ndarray_resource_stats()
    if impl.current_cfg().arch == ti.cpu:
        assert launched["inflight"] == 0
        assert launched["leases"] == launched["views"]
    else:
        assert launched["inflight"] == len(arrays)
        assert launched["leases"] == launched["views"] + len(arrays)

    del arrays
    gc.collect()
    ti.sync()
    completed = prog._debug_ndarray_resource_stats()
    assert completed["live"] == 0
    assert completed["retiring"] == 0
    assert completed["leases"] == 0
    assert completed["views"] == 0
    assert completed["inflight"] == 0
    assert sink[None] == sum(range(1, 10))


@test_utils.test(arch=ti.cpu)
def test_ndarray_launch_context_rejects_stale_resource_generation():
    @ti.kernel
    def bump(arr: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        arr[0] += 1

    old = ti.ndarray(ti.i32, shape=1)
    old.fill(3)
    primal = bump._primal
    key = primal.ensure_compiled(old)
    kernel_cpp = primal.compiled_kernels[key]
    prog = impl.get_runtime().prog
    compiled = prog.compile_kernel(
        prog.config(), prog.get_device_caps(), kernel_cpp
    )
    old_identity = prog._debug_ndarray_resource_identity(old.arr)

    old_native = old.arr
    old._invalidate_runtime()
    prog.delete_ndarray(old_native)

    replacement = ti.ndarray(ti.i32, shape=1)
    replacement_identity = prog._debug_ndarray_resource_identity(
        replacement.arr
    )
    assert replacement_identity["domain"] == old_identity["domain"]
    assert replacement_identity["index"] == old_identity["index"]
    assert replacement_identity["generation"] != old_identity["generation"]

    # Model the hardest aliasing case deterministically: the context points to
    # the current view address but carries the generation captured for the
    # retired occupant of the same registry slot.
    launch_ctx = kernel_cpp.make_launch_context()
    launch_ctx.set_arg_ndarray([0], replacement.arr)
    launch_ctx._debug_set_ndarray_resource_handle(
        [0],
        old_identity["domain"],
        old_identity["kind"],
        old_identity["index"],
        old_identity["generation"],
    )
    replacement.fill(17)
    with pytest.raises(RuntimeError, match="stale or retired Ndarray"):
        prog.launch_kernel(compiled, launch_ctx)
    assert replacement.to_numpy()[0] == 17


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_owned_ndarray_launch_uses_stable_slot_and_elides_snode_guard(monkeypatch):
    arch = impl.current_cfg().arch
    ti.reset()
    monkeypatch.setenv("TI_DEBUG_ORDINARY_LAUNCH_ATTRIBUTION", "1")
    ti.init(arch=arch, offline_cache=False)

    @ti.kernel
    def bump(arr: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        arr[0] += 1

    value = ti.ndarray(ti.i32, shape=1)
    value.fill(4)
    bump(value)
    ti.sync()

    prog = impl.get_runtime().prog
    prog._debug_reset_ordinary_launch_attribution()
    bump(value)
    ti.sync()
    stats = dict(prog._debug_ordinary_launch_attribution())

    assert stats["owned_ndarray_fast_path"] == 1
    assert stats["registered_execution_plan_launches"] == 1
    assert stats["owned_ndarray_only_launches"] == 1
    assert stats["ndarray_slot_validations"] == 1
    assert stats["ndarray_map_lookups"] == 0
    assert stats["snode_guard_acquisitions"] == 0
    assert stats["snode_guard_elisions"] >= 1
    if arch == ti.cpu:
        assert stats["backend_enabled"] == 1
        assert stats["backend_launches"] == 1
        assert stats["backend_task_invocations"] >= 1
        assert stats["backend_task_execution_ns"] > 0
    assert value.to_numpy()[0] == 6


def test_ordinary_launch_plan_contexts_have_call_level_ownership():
    context_lock = threading.Lock()
    next_context_id = 0
    scalar_patch_calls = 0
    full_bind_calls = 0

    class FakeLaunchContext:
        def __init__(self, context_id):
            self.context_id = context_id
            self.value = None

    def make_context():
        nonlocal next_context_id
        with context_lock:
            context_id = next_context_id
            next_context_id += 1
        return FakeLaunchContext(context_id)

    class FakeKernelCpp:
        @staticmethod
        def make_launch_context():
            return make_context()

    class FakeScalarPatchPlan:
        @staticmethod
        def apply(launch_ctx, args):
            nonlocal scalar_patch_calls
            with context_lock:
                scalar_patch_calls += 1
            launch_ctx.value = args[0]
            return -1

    caller_count = 4
    all_submitted = threading.Barrier(caller_count)
    synchronize_submissions = [True]
    submitted = []
    submitted_lock = threading.Lock()

    class FakeNativePlan:
        @staticmethod
        def launch(prog, launch_ctx):
            del prog
            if synchronize_submissions[0]:
                all_submitted.wait(timeout=5)
            with submitted_lock:
                submitted.append((launch_ctx.context_id, launch_ctx.value))

    runtime = SimpleNamespace(prog=object(), print_full_traceback=True)

    class FakeKernel:
        def __init__(self):
            self.runtime = runtime

        @staticmethod
        def _bind_ordinary_launch_context(launch_ctx, bindings, args):
            nonlocal full_bind_calls
            assert bindings in ((), ((0, "signed", None),))
            if bindings:
                with context_lock:
                    full_bind_calls += 1
                launch_ctx.value = args[0]

        @staticmethod
        def _finish_launch(launch_ctx):
            return launch_ctx.value

    plan = kernel_impl._OrdinaryLaunchPlan(
        runtime=runtime,
        key=("fake-specialization",),
        kernel_cpp=FakeKernelCpp(),
        native_plan=FakeNativePlan(),
        bindings=((0, "signed", None),),
        scalar_bindings=((0, "signed", None),),
        scalar_patch_plan=FakeScalarPatchPlan(),
        resource_guards=(),
        launch_ctx=make_context(),
        guard_context_lifecycle=False,
    )
    kernel = FakeKernel()
    results = []
    failures = []
    results_lock = threading.Lock()

    def launch(value):
        try:
            result = kernel_impl.Kernel._launch_with_ordinary_plan(
                kernel, plan, (value,)
            )
            with results_lock:
                results.append((value, result))
        except BaseException as exc:  # keep thread failures observable
            with results_lock:
                failures.append(exc)

    threads = [threading.Thread(target=launch, args=(value,)) for value in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    assert sorted(results) == [(value, value) for value in range(4)]
    assert len({context_id for context_id, _ in submitted}) == caller_count
    assert sorted(value for _, value in submitted) == list(range(caller_count))
    assert len(plan._launch_ctx_pool) <= (
        kernel_impl._ORDINARY_LAUNCH_CONTEXT_POOL_CAPACITY
    )

    # Once the overlap burst has populated the bounded pool, sequential warm
    # calls allocate no additional contexts and only patch scalar bytes.
    synchronize_submissions[0] = False
    assert next_context_id == caller_count
    assert scalar_patch_calls == 1
    assert full_bind_calls == caller_count - 1
    assert kernel_impl.Kernel._launch_with_ordinary_plan(kernel, plan, (10,)) == 10
    assert kernel_impl.Kernel._launch_with_ordinary_plan(kernel, plan, (11,)) == 11
    assert next_context_id == caller_count
    assert scalar_patch_calls == 3
    assert full_bind_calls == caller_count - 1


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_ordinary_launch_plan_lru_handles_ping_pong_resources_safely():
    @ti.kernel
    def assign(
        destination: ti.types.ndarray(dtype=ti.i32, ndim=1), value: ti.i32
    ) -> ti.i32:
        destination[0] = value
        return value + 1

    first = ti.ndarray(ti.i32, shape=1)
    second = ti.ndarray(ti.i32, shape=1)
    assert assign(first, 3) == 4
    first_plan = assign._primal._ordinary_launch_plan
    assert first_plan is not None
    first_launch_ctx = first_plan.launch_ctx
    gpu_context_reuse = impl.current_cfg().arch in (ti.cuda, ti.vulkan)
    assert (first_launch_ctx is not None) == gpu_context_reuse

    assert assign(first, 7) == 8
    assert assign._primal._ordinary_launch_plan is first_plan
    if gpu_context_reuse:
        assert assign._primal._ordinary_launch_plan.launch_ctx is first_launch_ctx
    assert first.to_numpy()[0] == 7

    # A different owned resource cannot alias the previous cached binding.
    assert assign(second, 11) == 12
    second_plan = assign._primal._ordinary_launch_plan
    assert second_plan is not None
    assert second_plan is not first_plan
    if gpu_context_reuse:
        assert second_plan.launch_ctx is not first_launch_ctx
    assert first.to_numpy()[0] == 7
    assert second.to_numpy()[0] == 11

    # Returning to the first resource must reuse its original plan rather than
    # rebuilding and replacing the second resource's entry.
    assert assign(first, 13) == 14
    assert assign._primal._ordinary_launch_plan is first_plan
    assert len(assign._primal._ordinary_launch_plans) == 2
    if gpu_context_reuse:
        assert first_plan.launch_ctx is first_launch_ctx
    assert first.to_numpy()[0] == 13


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_ordinary_launch_plan_lru_is_bounded_and_weak_guarded():
    @ti.kernel
    def assign(
        destination: ti.types.ndarray(dtype=ti.i32, ndim=1), value: ti.i32
    ):
        destination[0] = value

    arrays = [ti.ndarray(ti.i32, shape=1) for _ in range(5)]
    for index, array in enumerate(arrays):
        assign(array, index)

    plans = assign._primal._ordinary_launch_plans
    assert len(plans) == 4
    assert all(plan.resource_guards[0][1]() is not arrays[0] for plan in plans)

    retired = arrays[-1]
    del arrays[-1]
    del retired
    gc.collect()
    assign(arrays[0], 17)
    assert len(assign._primal._ordinary_launch_plans) <= 4
    assert arrays[0].to_numpy()[0] == 17


@test_utils.test(arch=[ti.cuda, ti.vulkan])
def test_ordinary_gpu_launch_plan_copies_reused_scalar_context_per_submission():
    @ti.kernel
    def write_slot(
        destination: ti.types.ndarray(dtype=ti.i32, ndim=1), slot: ti.i32
    ):
        destination[slot] = slot

    count = 128
    destination = ti.ndarray(ti.i32, shape=count)
    destination.fill(-1)
    write_slot(destination, 0)
    plan = write_slot._primal._ordinary_launch_plan
    assert plan is not None
    assert plan.launch_ctx is not None
    for slot in range(1, count):
        write_slot(destination, slot)
    ti.sync()

    np.testing.assert_array_equal(destination.to_numpy(), np.arange(count))
    assert write_slot._primal._ordinary_launch_plan is plan


@test_utils.test(arch=[ti.cuda, ti.vulkan])
def test_ordinary_gpu_launch_plan_batches_scalar_refresh_and_preserves_types():
    @ti.kernel
    def write_scalars(
        destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
        signed_value: ti.i32,
        unsigned_value: ti.u32,
        real_value: ti.f32,
    ):
        destination[0] = signed_value
        destination[1] = unsigned_value
        destination[2] = real_value

    destination = ti.ndarray(ti.f32, shape=3)
    write_scalars(destination, 1, 2, 3.0)
    plan = write_scalars._primal._ordinary_launch_plan
    assert plan is not None
    assert plan.scalar_patch_plan is not None

    write_scalars(
        destination,
        np.int32(-7),
        np.uint32(11),
        np.float32(2.5),
    )
    ti.sync()
    np.testing.assert_allclose(destination.to_numpy(), [-7.0, 11.0, 2.5])
    assert write_scalars._primal._ordinary_launch_plan is plan

    with pytest.raises(ti.TaichiRuntimeTypeError):
        write_scalars(destination, "invalid", 1, 1.0)

    write_scalars(destination, 4, 5, 6.5)
    ti.sync()
    np.testing.assert_allclose(destination.to_numpy(), [4.0, 5.0, 6.5])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_ordinary_launch_plan_excludes_snode_dependent_kernel():
    field = ti.field(ti.i32, shape=())

    @ti.kernel
    def update(value: ti.i32):
        field[None] = value

    update(5)
    update(9)
    assert update._primal._ordinary_launch_plan is None
    assert field[None] == 9


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_ordinary_launch_plan_negative_caches_stable_snode_admission(monkeypatch):
    field = ti.field(ti.i32, shape=())
    values = ti.ndarray(ti.i32, shape=1)

    @ti.kernel
    def update(source: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        field[None] = source[0]

    values[0] = 5
    build = update._primal._build_ordinary_launch_plan
    attempts = 0

    def counted_build(key, args):
        nonlocal attempts
        attempts += 1
        return build(key, args)

    monkeypatch.setattr(
        update._primal, "_build_ordinary_launch_plan", counted_build
    )
    update(values)
    values[0] = 9
    update(values)

    assert attempts == 1
    assert len(update._primal._ordinary_launch_plan_negative_keys) == 1
    assert update._primal._ordinary_launch_plan is None
    assert field[None] == 9


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.cpu)
def test_ordinary_launch_plan_is_invalidated_by_runtime_reset():
    @ti.kernel
    def assign(destination: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        destination[0] = 17

    first = ti.ndarray(ti.i32, shape=1)
    assign(first)
    assert assign._primal._ordinary_launch_plan is not None

    ti.reset()
    assert assign._primal._ordinary_launch_plan is None
    assert assign._primal._ordinary_launch_plans == []
    assert assign._primal._ordinary_launch_plan_negative_keys == set()
    ti.init(arch=ti.cpu, offline_cache=False)
    second = ti.ndarray(ti.i32, shape=1)
    assign(second)
    assert assign._primal._ordinary_launch_plan is not None
    assert second.to_numpy()[0] == 17


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_field_launch_keeps_required_snode_lifecycle_guard(monkeypatch):
    arch = impl.current_cfg().arch
    ti.reset()
    monkeypatch.setenv("TI_DEBUG_ORDINARY_LAUNCH_ATTRIBUTION", "1")
    ti.init(arch=arch, offline_cache=False)

    value = ti.field(ti.i32, shape=())

    @ti.kernel
    def bump():
        value[None] += 1

    bump()
    ti.sync()
    prog = impl.get_runtime().prog
    prog._debug_reset_ordinary_launch_attribution()
    bump()
    ti.sync()
    stats = dict(prog._debug_ordinary_launch_attribution())

    assert stats["snode_guard_acquisitions"] >= 1
    assert stats["snode_guard_elisions"] == 0
    assert value[None] == 2


@pytest.mark.run_in_serial
@test_utils.test(arch=ti.cpu)
def test_ndarray_registry_10k_create_use_release_stress():
    @ti.kernel
    def touch(arr: ti.types.ndarray(dtype=ti.i32, ndim=1), value: ti.i32):
        arr[0] = value

    warmup = ti.ndarray(ti.i32, shape=1)
    touch(warmup, 0)
    del warmup
    gc.collect()
    prog = impl.get_runtime().prog
    runtime = impl.get_runtime()
    baseline = prog._debug_ndarray_resource_stats()
    baseline_runtime_objects = len(runtime._runtime_object_refs)

    iterations = 10_000
    for i in range(iterations):
        arr = ti.ndarray(ti.i32, shape=1)
        touch(arr, i)
        del arr
        if i % 512 == 0:
            gc.collect()
    gc.collect()

    completed = prog._debug_ndarray_resource_stats()
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key]
    assert completed["created_total"] - baseline["created_total"] == iterations
    assert completed["retired_total"] - baseline["retired_total"] == iterations
    assert completed["released_total"] - baseline["released_total"] == iterations
    assert len(runtime._runtime_object_refs) == baseline_runtime_objects


@pytest.mark.run_in_serial
@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_in_python_func():
    def test():
        z = ti.ndarray(float, (8192, 8192))

    for i in range(300):
        test()


@test_utils.test(arch=[ti.cpu, ti.cuda], exclude=[ti.amdgpu])
def test_ndarray_with_fp16():
    half2 = ti.types.vector(n=2, dtype=ti.f16)

    @ti.kernel
    def init(x: ti.types.ndarray(dtype=half2, ndim=1)):
        for i in x:
            x[i] = half2(2.0)

    @ti.kernel
    def test(table: ti.types.ndarray(dtype=half2, ndim=1)):
        tmp = ti.Vector([ti.f16(0.0), ti.f16(0.0)])
        for i in ti.static(range(2)):
            tmp = tmp + 4.0 * table[i]

        table[0] = tmp

    acc = ti.ndarray(dtype=half2, shape=(40))
    table = ti.ndarray(dtype=half2, shape=(40))

    init(table)
    test(table)

    assert (table.to_numpy()[0] == 16.0).all()


@test_utils.test(
    arch=supported_archs_taichi_ndarray,
    require=ti.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_scalar_ndarray_oob():
    @ti.kernel
    def access_arr(input: ti.types.ndarray(), x: ti.i32) -> ti.f32:
        return input[x]

    input = np.random.randn(4)

    # Works
    access_arr(input, 1)

    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 4)

    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, -1)


# SOA layout for ndarray is deprecated so no need to test
@test_utils.test(
    arch=supported_archs_taichi_ndarray,
    require=ti.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_matrix_ndarray_oob():
    @ti.kernel
    def access_arr(input: ti.types.ndarray(), p: ti.i32, q: ti.i32, x: ti.i32, y: ti.i32) -> ti.f32:
        return input[p, q][x, y]

    @ti.kernel
    def valid_access(indices: ti.types.ndarray(dtype=ivec3, ndim=1), dummy: ti.types.ndarray(dtype=ivec3, ndim=1)):
        for i in indices:
            index_vec = ti.Vector([0, 0, 0])
            for j in ti.static(range(3)):
                index = indices[i][j]
                index_vec[j] = index
            dummy[i] = index_vec

    input = ti.ndarray(dtype=ti.math.mat2, shape=(4, 5))

    indices = ti.ndarray(dtype=ivec3, shape=(10))
    dummy = ti.ndarray(dtype=ivec3, shape=(10))

    # Works
    access_arr(input, 2, 3, 0, 1)
    valid_access(indices, dummy)

    # element_shape
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 2, 3, 2, 1)
    # field_shape[0]
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 4, 4, 0, 1)
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, -3, 4, 1, 1)
    # field_shape[1]
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 3, 5, 0, 1)
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 2, -10, 1, 1)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_mismatched_index_python_scope():
    x = ti.ndarray(dtype=ti.f32, shape=(4, 4))
    with pytest.raises(TaichiIndexError, match=r"2d ndarray indexed with 1d indices"):
        x[0]

    with pytest.raises(TaichiIndexError, match=r"2d ndarray indexed with 3d indices"):
        x[0, 0, 0]


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_0dim_ndarray_read_write_python_scope():
    x = ti.ndarray(dtype=ti.f32, shape=())

    x[()] = 1.0
    assert x[None] == 1.0

    y = ti.ndarray(dtype=ti.math.vec2, shape=())
    y[()] = [1.0, 2.0]
    assert y[None] == [1.0, 2.0]


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_0dim_ndarray_read_write_taichi_scope():
    x = ti.ndarray(dtype=ti.f32, shape=())

    @ti.kernel
    def write(x: ti.types.ndarray()):
        a = x[()] + 1
        x[None] = 2 * a

    write(x)
    assert x[None] == 2.0

    y = ti.ndarray(dtype=ti.math.vec2, shape=())
    write(y)
    assert y[None] == [2.0, 2.0]


@test_utils.test(arch=supported_archs_taichi_ndarray, require=ti.extension.data64)
def test_read_write_f64_python_scope():
    x = ti.ndarray(dtype=ti.f64, shape=2)

    x[0] = 1.0
    assert x[0] == 1.0

    y = ti.ndarray(dtype=ti.math.vec2, shape=2)
    y[0] = [1.0, 2.0]
    assert y[0] == [1.0, 2.0]


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_fill():
    vec2 = ti.types.vector(2, ti.f32)
    x_vec = ti.ndarray(vec2, (512, 512))
    x_vec.fill(1.0)
    assert (x_vec[2, 2] == [1.0, 1.0]).all()

    x_vec.fill(vec2(2.0, 4.0))
    assert (x_vec[3, 3] == [2.0, 4.0]).all()

    mat2x2 = ti.types.matrix(2, 2, ti.f32)
    x_mat = ti.ndarray(mat2x2, (512, 512))
    x_mat.fill(2.0)
    assert (x_mat[2, 2] == [[2.0, 2.0], [2.0, 2.0]]).all()

    x_mat.fill(mat2x2([[2.0, 4.0], [1.0, 3.0]]))
    assert (x_mat[3, 3] == [[2.0, 4.0], [1.0, 3.0]]).all()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_wrong_dtype():
    @ti.kernel
    def test2(arr: ti.types.ndarray(dtype=ti.f32)):
        for I in ti.grouped(arr):
            arr[I] = 2.0

    tp_ivec3 = ti.types.vector(3, ti.i32)

    y = ti.ndarray(tp_ivec3, shape=(12, 4))
    with pytest.raises(TypeError, match=r"get \[Tensor \(3\) i32\]"):
        test2(y)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_bad_assign():
    tp_ivec3 = ti.types.vector(3, ti.i32)

    @ti.kernel
    def test4(arr: ti.types.ndarray(dtype=tp_ivec3)):
        for I in ti.grouped(arr):
            arr[I] = [1, 2]

    y = ti.ndarray(tp_ivec3, shape=(12, 4))
    with pytest.raises(TaichiTypeError, match=r"cannot assign '\[Tensor \(2\) i32\]' to '\[Tensor \(3\) i32\]'"):
        test4(y)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_bad_ndim():
    x = ti.ndarray(ti.f32, shape=(12, 13))

    @ti.kernel
    def test5(arr: ti.types.ndarray(ndim=1)):
        for i, j in arr:
            arr[i, j] = 0

    with pytest.raises(ValueError, match=r"required ndim=1, but 2d ndarray with shape \(12, 13\) is provided"):
        test5(x)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_type_hint_matrix():
    @ti.kernel
    def test(x: ti.types.ndarray(dtype=ti.types.matrix())):
        for I in ti.grouped(x):
            x[I] = 1.0

    x = ti.ndarray(ti.math.mat2, (3))
    test(x)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    y = ti.ndarray(ti.math.mat3, (3))
    test(y)
    assert impl.get_runtime().get_num_compiled_functions() == 2

    z = ti.ndarray(ti.math.vec2, (3))
    with pytest.raises(ValueError, match=r"Invalid value for argument x"):
        test(z)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_type_hint_vector():
    @ti.kernel
    def test(x: ti.types.ndarray(dtype=ti.types.vector())):
        for I in ti.grouped(x):
            x[I] = 1.0

    x = ti.ndarray(ti.math.vec3, (3))
    test(x)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    y = ti.ndarray(ti.math.vec2, (3))
    test(y)
    assert impl.get_runtime().get_num_compiled_functions() == 2

    z = ti.ndarray(ti.math.mat2, (3))
    with pytest.raises(ValueError, match=r"Invalid value for argument x"):
        test(z)


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_pass_ndarray_to_func():
    @ti.func
    def bar(weight: ti.types.ndarray(ti.f32, ndim=3)) -> ti.f32:
        return weight[1, 1, 1]

    @ti.kernel
    def foo(weight: ti.types.ndarray(ti.f32, ndim=3)) -> ti.f32:
        return bar(weight)

    weight = ti.ndarray(dtype=ti.f32, shape=(2, 2, 2))
    weight.fill(42.0)
    assert foo(weight) == 42.0


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_pass_ndarray_to_real_func():
    @ti.real_func
    def bar(weight: ti.types.ndarray(ti.f32, ndim=3)) -> ti.f32:
        return weight[1, 1, 1]

    @ti.kernel
    def foo(weight: ti.types.ndarray(ti.f32, ndim=3)) -> ti.f32:
        return bar(weight)

    weight = ti.ndarray(dtype=ti.f32, shape=(2, 2, 2))
    weight.fill(42.0)
    assert foo(weight) == 42.0


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_pass_ndarray_outside_kernel_to_real_func():
    weight = ti.ndarray(dtype=ti.f32, shape=(2, 2, 2))

    @ti.real_func
    def bar(weight: ti.types.ndarray(ti.f32, ndim=3)) -> ti.f32:
        return weight[1, 1, 1]

    @ti.kernel
    def foo() -> ti.f32:
        return bar(weight)

    weight.fill(42.0)
    with pytest.raises(ti.TaichiTypeError, match=r"Expected ndarray in the kernel argument for argument weight"):
        foo()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_oob_clamp():
    @ti.kernel
    def test(x: ti.types.ndarray(boundary="clamp"), y: ti.i32) -> ti.f32:
        return x[y]

    x = ti.ndarray(ti.f32, shape=(3))
    for i in range(3):
        x[i] = i

    assert test(x, -1) == 0
    assert test(x, -2) == 0
    assert test(x, 3) == 2
    assert test(x, 4) == 2

    @ti.kernel
    def test_vec_arr(x: ti.types.ndarray(boundary="clamp"), y: ti.i32) -> ti.f32:
        return x[1, 2][y]

    x2 = ti.ndarray(ti.math.vec2, shape=(3, 3))
    for i in range(3):
        for j in range(3):
            x2[i, j] = [i, j]
    assert test_vec_arr(x2, -1) == 1
    assert test_vec_arr(x2, 2) == 2

    @ti.kernel
    def test_mat_arr(x: ti.types.ndarray(boundary="clamp"), i: ti.i32, j: ti.i32) -> ti.f32:
        return x[1, 2][i, j]

    x3 = ti.ndarray(ti.math.mat2, shape=(3, 3))
    for i in range(3):
        for j in range(3):
            x3[i, j] = [[i, j], [i + 1, j + 1]]
    assert test_mat_arr(x3, -1, 0) == 1
    assert test_mat_arr(x3, 1, -1) == 2
    assert test_mat_arr(x3, 2, 0) == 2
    assert test_mat_arr(x3, 1, 2) == 3
    assert test_mat_arr(x3, 0, 8) == 2


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_clamp_verify():
    height = 3
    width = 3

    @ti.kernel
    def test(ao: ti.types.ndarray(dtype=ti.f32, ndim=2, boundary="clamp")):
        for y, x in ti.ndrange(height, width):
            vis = 0.0
            ao[y, x] = vis

    ao = ti.ndarray(ti.f32, shape=(height, width))
    test(ao)
    assert (ao.to_numpy() == np.zeros((height, width))).all()


@test_utils.test(arch=supported_archs_taichi_ndarray)
def test_ndarray_arg_builtin_float_type():
    @ti.kernel
    def foo(x: ti.types.ndarray(float, ndim=0)) -> ti.f32:
        return x[None]

    x = ti.ndarray(ti.f32, shape=())
    x[None] = 42
    assert foo(x) == 42


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_real_func_vector_ndarray_arg():
    @ti.real_func
    def foo(x: ti.types.ndarray(ndim=1)) -> vec3:
        return x[0]

    @ti.kernel
    def test(x: ti.types.ndarray(ndim=1)) -> vec3:
        return foo(x)

    x = ti.Vector.ndarray(3, ti.f32, shape=(1))
    x[0] = vec3(1, 2, 3)
    assert (test(x) == vec3(1, 2, 3)).all()


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_real_func_write_ndarray_cfg():
    @ti.real_func
    def bar(a: ti.types.ndarray(ndim=1)):
        a[0] = vec3(1)

    @ti.kernel
    def foo(
        a: ti.types.ndarray(ndim=1),
    ):
        a[0] = vec3(3)
        bar(a)
        a[0] = vec3(3)

    a = ti.Vector.ndarray(3, float, shape=(2,))
    foo(a)
    assert (a[0] == vec3(3)).all()
