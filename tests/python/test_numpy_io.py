import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test()
def test_to_numpy_2d():
    val = ti.field(ti.i32)

    n = 4
    m = 7

    ti.root.dense(ti.ij, (n, m)).place(val)

    for i in range(n):
        for j in range(m):
            val[i, j] = i + j * 3

    arr = val.to_numpy()

    assert arr.shape == (4, 7)
    for i in range(n):
        for j in range(m):
            assert arr[i, j] == i + j * 3


@test_utils.test()
def test_from_numpy_2d():
    val = ti.field(ti.i32)

    n = 4
    m = 7

    ti.root.dense(ti.ij, (n, m)).place(val)

    arr = np.empty(shape=(n, m), dtype=np.int32)

    for i in range(n):
        for j in range(m):
            arr[i, j] = i + j * 3

    val.from_numpy(arr)

    for i in range(n):
        for j in range(m):
            assert val[i, j] == i + j * 3


@test_utils.test()
def test_to_numpy_struct():
    n = 16
    f = ti.Struct.field({"a": ti.i32, "b": ti.f32}, shape=(n,))

    for i in range(n):
        f[i].a = i
        f[i].b = f[i].a * 2

    arr_dict = f.to_numpy()

    for i in range(n):
        assert arr_dict["a"][i] == i
        assert arr_dict["b"][i] == i * 2


@test_utils.test()
def test_from_numpy_struct():
    n = 16
    f = ti.Struct.field({"a": ti.i32, "b": ti.f32}, shape=(n,))

    arr_dict = {
        "a": np.arange(n, dtype=np.int32),
        "b": np.arange(n, dtype=np.int32) * 2,
    }

    f.from_numpy(arr_dict)

    for i in range(n):
        assert f[i].a == i
        assert f[i].b == i * 2


@test_utils.test(require=ti.extension.data64)
def test_f64():
    val = ti.field(ti.f64)

    n = 4
    m = 7

    ti.root.dense(ti.ij, (n, m)).place(val)

    for i in range(n):
        for j in range(m):
            val[i, j] = (i + j * 3) * 1e100

    val.from_numpy(val.to_numpy() * 2)

    for i in range(n):
        for j in range(m):
            assert val[i, j] == (i + j * 3) * 2e100


@test_utils.test()
def test_matrix():
    n = 4
    m = 7
    val = ti.Matrix.field(2, 3, ti.f32, shape=(n, m))

    nparr = np.empty(shape=(n, m, 2, 3), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            for k in range(2):
                for l in range(3):
                    nparr[i, j, k, l] = i + j * 2 - k - l * 3

    val.from_numpy(nparr)
    new_nparr = val.to_numpy()
    assert (nparr == new_nparr).all()


@test_utils.test()
def test_numpy_io_example():
    n = 4
    m = 7

    # Taichi tensors
    val = ti.field(ti.i32, shape=(n, m))
    vec = ti.Vector.field(3, dtype=ti.i32, shape=(n, m))
    mat = ti.Matrix.field(3, 4, dtype=ti.i32, shape=(n, m))

    # Scalar
    arr = np.ones(shape=(n, m), dtype=np.int32)
    val.from_numpy(arr)
    arr = val.to_numpy()

    # Vector
    arr = np.ones(shape=(n, m, 3), dtype=np.int32)
    vec.from_numpy(arr)

    arr = np.ones(shape=(n, m, 3, 1), dtype=np.int32)
    vec.from_numpy(arr)

    arr = np.ones(shape=(n, m, 1, 3), dtype=np.int32)
    vec.from_numpy(arr)

    arr = vec.to_numpy()
    assert arr.shape == (n, m, 3)

    arr = vec.to_numpy(keep_dims=True)
    assert arr.shape == (n, m, 3, 1)

    # Matrix
    arr = np.ones(shape=(n, m, 3, 4), dtype=np.int32)
    mat.from_numpy(arr)

    arr = mat.to_numpy()
    assert arr.shape == (n, m, 3, 4)

    arr = mat.to_numpy(keep_dims=True)
    assert arr.shape == (n, m, 3, 4)

    # For PyTorch tensors, use to_torch/from_torch instead


@test_utils.test()
def test_from_numpy_non_contiguous():
    n = 9
    m = 7
    p = 4
    arr = np.ones(shape=(n, m, p, p), dtype=np.int32)

    val = ti.field(ti.i32, shape=(2, 2))
    val.from_numpy(arr[0:6:3, 0:6:3, 0, 0])

    vec = ti.Vector.field(3, dtype=ti.i32, shape=(2, 2))
    vec.from_numpy(arr[0:6:3, 0:6:3, 0:3, 0])

    mat = ti.Matrix.field(3, 4, dtype=ti.i32, shape=(2, 2))
    mat.from_numpy(arr[0:6:3, 0:6:3, 0:3, 0:4])


@test_utils.test(arch=[ti.vulkan])
def test_dense_field_host_staging_capacity_reuse_and_runtime_generation():
    prog = impl.get_runtime().prog
    baseline = prog._debug_dense_field_staging_stats()
    assert baseline["live"] == 1
    assert baseline["leases"] == 1
    assert baseline["upload_capacity"] == 0
    assert baseline["readback_capacity"] == 0

    small_np = np.arange(8, dtype=np.int32)
    small = ti.field(ti.i32, shape=small_np.shape)
    small.from_numpy(small_np)
    np.testing.assert_array_equal(small.to_numpy(), small_np)
    small_stats = prog._debug_dense_field_staging_stats()
    assert small_stats["upload_capacity"] == small_np.nbytes
    assert small_stats["readback_capacity"] == small_np.nbytes
    assert small_stats["has_upload"] == 1
    assert small_stats["has_readback"] == 1

    smaller_np = np.arange(4, dtype=np.int32)
    smaller = ti.field(ti.i32, shape=smaller_np.shape)
    smaller.from_numpy(smaller_np)
    np.testing.assert_array_equal(smaller.to_numpy(), smaller_np)
    reused = prog._debug_dense_field_staging_stats()
    assert reused["upload_capacity"] == small_stats["upload_capacity"]
    assert reused["readback_capacity"] == small_stats["readback_capacity"]

    larger_np = np.arange(64, dtype=np.int32)
    larger = ti.field(ti.i32, shape=larger_np.shape)
    larger.from_numpy(larger_np)
    np.testing.assert_array_equal(larger.to_numpy(), larger_np)
    grown = prog._debug_dense_field_staging_stats()
    assert grown["upload_capacity"] == larger_np.nbytes
    assert grown["readback_capacity"] == larger_np.nbytes
    for key in ("live", "leases", "created_total"):
        assert grown[key] == baseline[key]

    old_domain = grown["domain"]
    ti.reset()
    ti.init(arch=ti.vulkan)
    replacement = impl.get_runtime().prog._debug_dense_field_staging_stats()
    assert replacement["domain"] != old_domain
    assert replacement["upload_capacity"] == 0
    assert replacement["readback_capacity"] == 0
