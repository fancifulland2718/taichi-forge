"""
To test our new `ti.field` API is functional (#1500)
"""

import numpy as np
import pytest
from taichi_forge.lang import impl
from taichi_forge.lang.misc import get_host_arch_list

import taichi_forge as ti
from tests import test_utils

data_types = [ti.i32, ti.f32, ti.i64, ti.f64]
field_shapes = [(), 8, (6, 12)]
vector_dims = [3]
matrix_dims = [(1, 2), (2, 3)]


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", field_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_scalar_field(dtype, shape):
    x = ti.field(dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)

    assert x.dtype == dtype


@pytest.mark.parametrize("n", vector_dims)
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", field_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_vector_field(n, dtype, shape):
    vec_type = ti.types.vector(n, dtype)
    x = ti.field(vec_type, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == 1


@pytest.mark.parametrize("n,m", matrix_dims)
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", field_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_matrix_field(n, m, dtype, shape):
    mat_type = ti.types.matrix(n, m, dtype)
    x = ti.field(dtype=mat_type, shape=shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == m


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", field_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_scalr_field_from_numpy(dtype, shape):
    import numpy as np

    x = ti.field(dtype, shape)
    # use the corresponding dtype for the numpy array.
    numpy_dtypes = {
        ti.i32: np.int32,
        ti.f32: np.float32,
        ti.f64: np.float64,
        ti.i64: np.int64,
    }
    arr = np.empty(shape, dtype=numpy_dtypes[dtype])
    x.from_numpy(arr)


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize(
    "shape, offset",
    [
        ((), None),
        ((), ()),
        (8, None),
        (8, 0),
        (8, 8),
        (8, -4),
        ((6, 12), None),
        ((6, 12), (0, 0)),
        ((6, 12), (-4, -4)),
        ((6, 12), (-4, 4)),
        ((6, 12), (4, -4)),
        ((6, 12), (8, 8)),
    ],
)
@test_utils.test(arch=get_host_arch_list())
def test_scalr_field_from_numpy_with_offset(dtype, shape, offset):
    import numpy as np

    x = ti.field(dtype=dtype, shape=shape, offset=offset)
    # use the corresponding dtype for the numpy array.
    numpy_dtypes = {
        ti.i32: np.int32,
        ti.f32: np.float32,
        ti.f64: np.float64,
        ti.i64: np.int64,
    }
    arr = np.ones(shape, dtype=numpy_dtypes[dtype])
    x.from_numpy(arr)

    def mat_equal(A, B, tol=1e-6):
        return np.max(np.abs(A - B)) < tol

    tol = 1e-5 if dtype == ti.f32 else 1e-12
    assert mat_equal(x.to_numpy(), arr, tol=tol)


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", field_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_scalr_field_from_numpy_with_mismatch_shape(dtype, shape):
    import numpy as np

    x = ti.field(dtype, shape)
    numpy_dtypes = {
        ti.i32: np.int32,
        ti.f32: np.float32,
        ti.f64: np.float64,
        ti.i64: np.int64,
    }
    # compose the mismatch shape for every ti.field.
    # set the shape to (2, 3) by default, if the ti.field shape is a tuple, set it to 1.
    mismatch_shape = (2, 3)
    if isinstance(shape, tuple):
        mismatch_shape = 1
    arr = np.empty(mismatch_shape, dtype=numpy_dtypes[dtype])
    with pytest.raises(ValueError):
        x.from_numpy(arr)


@pytest.mark.parametrize(
    "dtype, np_dtype, value",
    [
        (ti.i32, np.int32, -7),
        (ti.u32, np.uint32, 11),
        (ti.i64, np.int64, -13),
        (ti.u64, np.uint64, 17),
        (ti.f32, np.float32, 2.5),
        (ti.f64, np.float64, -3.25),
    ],
)
@test_utils.test(arch=[ti.cpu])
def test_scalar_field_cpu_dense_native_fill(dtype, np_dtype, value):
    x = ti.field(dtype, shape=(4, 8))
    x.fill(value)
    expected = np.full((4, 8), value, dtype=np_dtype)
    np.testing.assert_array_equal(x.to_numpy(), expected)


@pytest.mark.parametrize(
    "dtype, np_dtype, fill_value",
    [
        (ti.i32, np.int32, -11),
        (ti.u32, np.uint32, 13),
        (ti.f32, np.float32, 2.25),
    ],
)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_scalar_field_dense_native_bulk_api(dtype, np_dtype, fill_value):
    n = 257
    x = ti.field(dtype, shape=n)
    src = (np.arange(n, dtype=np_dtype) % np_dtype(23)).astype(np_dtype)
    x.from_numpy(src)
    np.testing.assert_array_equal(x.to_numpy(), src)

    x.fill(0)
    np.testing.assert_array_equal(x.to_numpy(), np.zeros(n, dtype=np_dtype))

    x.fill(fill_value)
    expected = np.full(n, fill_value, dtype=np_dtype)
    np.testing.assert_array_equal(x.to_numpy(), expected)


@pytest.mark.parametrize("dtype, np_dtype", [(ti.i32, np.int32), (ti.f32, np.float32)])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_matrix_field_dense_bulk_api_correctness(dtype, np_dtype):
    n = 17
    x = ti.Matrix.field(2, 2, dtype=dtype, shape=n)
    src = np.arange(n * 4, dtype=np_dtype).reshape(n, 2, 2)
    x.from_numpy(src)
    np.testing.assert_array_equal(x.to_numpy(), src)

    x.fill(0)
    np.testing.assert_array_equal(x.to_numpy(), np.zeros((n, 2, 2), dtype=np_dtype))


@pytest.mark.parametrize("dtype, np_dtype", [(ti.i32, np.int32), (ti.f32, np.float32)])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_matrix_field_dense_packed_bulk_no_compile(dtype, np_dtype):
    n = 23
    x = ti.Matrix.field(2, 2, dtype=dtype, shape=n)
    values = np.arange(n * 4, dtype=np_dtype).reshape(n, 2, 2)
    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    x.from_numpy(values)
    np.testing.assert_array_equal(x.to_numpy(), values)
    x.fill(0)
    np.testing.assert_array_equal(
        x.to_numpy(), np.zeros((n, 2, 2), dtype=np_dtype)
    )
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions


@test_utils.test(arch=get_host_arch_list())
def test_field_needs_grad():
    # Just make sure the usage doesn't crash, see #1545
    n = 8
    m1 = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
    m2 = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
    gr = ti.field(dtype=ti.f32, shape=n)

    @ti.kernel
    def func():
        for i in range(n):
            gr[i] = m1.grad[i] + m2.grad[i]

    func()


@test_utils.test()
def test_field_needs_grad_dtype():
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        a = ti.field(int, shape=1, needs_grad=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        b = ti.field(ti.math.ivec3, shape=1, needs_grad=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        mat_type = ti.types.matrix(2, 3, int)
        c = ti.field(dtype=mat_type, shape=1, needs_grad=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        d = ti.Struct.field(
            {
                "pos": ti.types.vector(3, int),
                "vel": ti.types.vector(3, float),
                "acc": ti.types.vector(3, float),
                "mass": ti.f32,
            },
            shape=1,
            needs_grad=True,
        )


@test_utils.test()
def test_field_needs_dual_dtype():
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        a = ti.field(int, shape=1, needs_dual=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        b = ti.field(ti.math.ivec3, shape=1, needs_dual=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        mat_type = ti.types.matrix(2, 3, int)
        c = ti.field(mat_type, shape=1, needs_dual=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        d = ti.Struct.field(
            {
                "pos": ti.types.vector(3, int),
                "vel": ti.types.vector(3, float),
                "acc": ti.types.vector(3, float),
                "mass": ti.f32,
            },
            shape=1,
            needs_dual=True,
        )


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
def test_default_fp(dtype):
    ti.init(default_fp=dtype)
    vec_type = ti.types.vector(3, dtype)

    x = ti.field(vec_type, ())

    assert x.dtype == impl.get_runtime().default_fp


@pytest.mark.parametrize("dtype", [ti.i32, ti.i64])
def test_default_ip(dtype):
    ti.init(default_ip=dtype)

    x = ti.field(ti.math.ivec2, ())

    assert x.dtype == impl.get_runtime().default_ip


@test_utils.test()
def test_field_name():
    a = ti.field(dtype=ti.f32, shape=(2, 3), name="a")
    b = ti.field(ti.math.vec3, shape=(2, 3), name="b")
    c = ti.field(ti.math.mat3, shape=(5, 4), name="c")
    assert a._name == "a"
    assert b._name == "b"
    assert c._name == "c"
    assert b.snode._name == "b"
    d = []
    for i in range(10):
        d.append(ti.field(dtype=ti.f32, shape=(2, 3), name=f"d{i}"))
        assert d[i]._name == f"d{i}"


@test_utils.test()
@pytest.mark.parametrize("shape", field_shapes)
@pytest.mark.parametrize("dtype", [ti.i32, ti.f32])
def test_field_copy_from(shape, dtype):
    x = ti.field(dtype=ti.f32, shape=shape)
    other = ti.field(dtype=dtype, shape=shape)
    other.fill(1)
    x.copy_from(other)
    convert = lambda arr: arr[0] if len(arr) == 1 else arr
    assert convert(x.shape) == shape
    assert x.dtype == ti.f32
    assert (x.to_numpy() == 1).all()


@pytest.mark.parametrize(
    "dtype, np_dtype",
    [(ti.i32, np.int32), (ti.f32, np.float32), (ti.u32, np.uint32)],
)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scalar_field_dense_native_copy_from_no_compile(dtype, np_dtype):
    n = 257
    src = ti.field(dtype=dtype, shape=n)
    dst = ti.field(dtype=dtype, shape=n)
    values = np.arange(n, dtype=np_dtype)
    src.from_numpy(values)
    dst.fill(0)

    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    dst.copy_from(src)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
    np.testing.assert_array_equal(dst.to_numpy(), values)


@pytest.mark.parametrize(
    "dtype, np_dtype",
    [(ti.i32, np.int32), (ti.f32, np.float32)],
)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_matrix_field_dense_native_copy_from_no_compile(dtype, np_dtype):
    n = 37
    src = ti.Matrix.field(2, 3, dtype=dtype, shape=n)
    dst = ti.Matrix.field(2, 3, dtype=dtype, shape=n)
    values = np.arange(n * 6, dtype=np_dtype).reshape(n, 2, 3)
    src.from_numpy(values)
    dst.fill(0)

    compiled_functions = impl.get_runtime().get_num_compiled_functions()
    dst.copy_from(src)
    assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
    np.testing.assert_array_equal(dst.to_numpy(), values)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_scalar_field_dense_native_copy_from_tape_grad_no_compile():
    n = 16
    src = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
    dst = ti.field(dtype=ti.f32, shape=n, needs_grad=True)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    src.from_numpy(np.arange(n, dtype=np.float32))

    @ti.kernel
    def reduce_dst():
        for i in range(n):
            loss[None] += 2.0 * dst[i]

    with ti.ad.Tape(loss):
        compiled_functions = impl.get_runtime().get_num_compiled_functions()
        dst.copy_from(src)
        assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
        reduce_dst()

    np.testing.assert_array_equal(
        src.grad.to_numpy(), np.full((n,), 2.0, dtype=np.float32)
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_matrix_field_dense_native_copy_from_tape_grad_no_compile():
    n = 8
    src = ti.Matrix.field(2, 3, dtype=ti.f32, shape=n, needs_grad=True)
    dst = ti.Matrix.field(2, 3, dtype=ti.f32, shape=n, needs_grad=True)
    loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
    values = np.arange(n * 6, dtype=np.float32).reshape(n, 2, 3)
    src.from_numpy(values)

    @ti.kernel
    def reduce_dst():
        for i in range(n):
            for j, k in ti.static(ti.ndrange(2, 3)):
                loss[None] += (j + k + 1) * dst[i][j, k]

    expected = np.zeros((n, 2, 3), dtype=np.float32)
    for j in range(2):
        for k in range(3):
            expected[:, j, k] = j + k + 1

    with ti.ad.Tape(loss):
        compiled_functions = impl.get_runtime().get_num_compiled_functions()
        dst.copy_from(src)
        assert impl.get_runtime().get_num_compiled_functions() == compiled_functions
        reduce_dst()

    np.testing.assert_array_equal(src.grad.to_numpy(), expected)


@test_utils.test()
def test_field_copy_from_with_mismatch_shape():
    x = ti.field(dtype=ti.f32, shape=(2, 3))
    for other_shape in [(2,), (2, 2), (2, 3, 4)]:
        other = ti.field(dtype=ti.f16, shape=other_shape)
        with pytest.raises(ValueError):
            x.copy_from(other)


@test_utils.test()
@pytest.mark.parametrize(
    "shape, x_offset, other_offset",
    [
        ((), (), ()),
        (8, 4, 0),
        (8, 0, -4),
        (8, -4, -4),
        (8, 8, -4),
        ((6, 12), (0, 0), (-6, -6)),
        ((6, 12), (-6, -6), (0, 0)),
        ((6, 12), (-6, -6), (-6, -6)),
    ],
)
@pytest.mark.parametrize("dtype", [ti.i32, ti.f32])
def test_field_copy_from_with_offset(shape, dtype, x_offset, other_offset):
    x = ti.field(dtype=ti.f32, shape=shape, offset=x_offset)
    other = ti.field(dtype=dtype, shape=shape, offset=other_offset)
    other.fill(1)
    x.copy_from(other)
    convert = lambda arr: arr[0] if len(arr) == 1 else arr
    assert convert(x.shape) == shape
    assert x.dtype == ti.f32
    assert (x.to_numpy() == 1).all()


@test_utils.test()
def test_field_copy_from_with_non_filed_object():
    import numpy as np

    x = ti.field(dtype=ti.f32, shape=(2, 3))
    other = np.zeros((2, 3))
    with pytest.raises(TypeError):
        x.copy_from(other)


@test_utils.test()
def test_field_shape_0():
    with pytest.raises(
        ti._lib.core.TaichiRuntimeError,
        match="Every dimension of a Taichi field should be positive",
    ):
        x = ti.field(dtype=ti.f32, shape=0)


@test_utils.test()
def test_index_mismatch():
    with pytest.raises(AssertionError, match="Slicing is not supported on ti.field"):
        val = ti.field(ti.i32, shape=(1, 2, 3))
        val[0, 0] = 1


@test_utils.test()
def test_invalid_slicing():
    with pytest.raises(
        TypeError,
        match="Detected illegal element of type: .*?\. Please be aware that slicing a ti.field is not supported so far.",
    ):
        val = ti.field(ti.i32, shape=(2, 2))
        val[0, :]


@test_utils.test()
def test_indexing_with_np_int():
    val = ti.field(ti.i32, shape=(2))
    idx = np.int32(0)
    val[idx]


@test_utils.test()
def test_indexing_vec_field_with_np_int():
    val = ti.field(ti.math.ivec2, shape=(2))
    idx = np.int32(0)
    val[idx][idx]


@test_utils.test()
def test_indexing_mat_field_with_np_int():
    mat_type = ti.types.matrix(2, 2, int)
    val = ti.field(mat_type, shape=(2))
    idx = np.int32(0)
    val[idx][idx, idx]


@test_utils.test()
def test_python_for_in():
    x = ti.field(int, shape=3)
    with pytest.raises(NotImplementedError, match="Struct for is only available in Taichi scope"):
        for i in x:
            pass


@test_utils.test()
def test_matrix_mult_field():
    x = ti.field(int, shape=())
    with pytest.raises(ti.TaichiTypeError, match="unsupported operand type"):

        @ti.kernel
        def foo():
            a = ti.Vector([1, 1, 1])
            b = a * x

        foo()


@test_utils.test(exclude=[ti.x64, ti.arm64, ti.cuda])
def test_sparse_not_supported():
    with pytest.raises(ti.TaichiRuntimeError, match="Pointer SNode is not supported on this backend."):
        ti.root.pointer(ti.i, 10)

    with pytest.raises(ti.TaichiRuntimeError, match="Pointer SNode is not supported on this backend."):
        a = ti.root.dense(ti.i, 10)
        a.pointer(ti.j, 10)

    with pytest.raises(ti.TaichiRuntimeError, match="Dynamic SNode is not supported on this backend."):
        ti.root.dynamic(ti.i, 10)

    with pytest.raises(ti.TaichiRuntimeError, match="Dynamic SNode is not supported on this backend."):
        a = ti.root.dense(ti.i, 10)
        a.dynamic(ti.j, 10)

    with pytest.raises(ti.TaichiRuntimeError, match="Bitmasked SNode is not supported on this backend."):
        ti.root.bitmasked(ti.i, 10)

    with pytest.raises(ti.TaichiRuntimeError, match="Bitmasked SNode is not supported on this backend."):
        a = ti.root.dense(ti.i, 10)
        a.bitmasked(ti.j, 10)


@test_utils.test(require=ti.extension.data64)
def test_write_u64():
    x = ti.field(ti.u64, shape=())
    x[None] = 2**64 - 1
    assert x[None] == 2**64 - 1


@test_utils.test(require=ti.extension.data64)
def test_field_with_dynamic_index():
    vel = ti.Vector.field(2, dtype=ti.f64, shape=(100, 100))

    @ti.func
    def foo(i, j, l):
        tmp = 1.0 / vel[i, j][l]
        return tmp

    @ti.kernel
    def collide():
        tmp0 = foo(0, 0, 0)
        print(tmp0)

    collide()
