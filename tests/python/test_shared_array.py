import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.math import vec4
from tests import test_utils


@test_utils.test(arch=[ti.cuda])
def test_large_shared_array():
    # Skip the GPUs prior to Ampere which doesn't have large dynamical shared memory.
    if ti.lang.impl.get_cuda_compute_capability() < 86:
        pytest.skip("Skip the GPUs prior to Ampere")

    block_dim = 128
    nBlocks = 64
    N = nBlocks * block_dim
    v_arr = np.random.randn(N).astype(np.float32)
    d_arr = np.random.randn(N).astype(np.float32)
    a_arr = np.zeros(N).astype(np.float32)
    reference = np.zeros(N).astype(np.float32)

    @ti.kernel
    def calc(
        v: ti.types.ndarray(ndim=1),
        d: ti.types.ndarray(ndim=1),
        a: ti.types.ndarray(ndim=1),
    ):
        for i in range(N):
            acc = 0.0
            v_val = v[i]
            for j in range(N):
                acc += v_val * d[j]
            a[i] = acc

    @ti.kernel
    def calc_shared_array(
        v: ti.types.ndarray(ndim=1),
        d: ti.types.ndarray(ndim=1),
        a: ti.types.ndarray(ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(nBlocks * block_dim):
            tid = i % block_dim
            pad = ti.simt.block.SharedArray((65536 // 4,), ti.f32)
            acc = 0.0
            v_val = v[i]
            for k in range(nBlocks):
                pad[tid] = d[k * block_dim + tid]
                ti.simt.block.sync()
                for j in range(block_dim):
                    acc += v_val * pad[j]
                ti.simt.block.sync()
            a[i] = acc

    calc(v_arr, d_arr, reference)
    calc_shared_array(v_arr, d_arr, a_arr)
    assert np.allclose(reference, a_arr)


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.amdgpu])
def test_multiple_shared_array():
    block_dim = 128
    nBlocks = 64
    N = nBlocks * block_dim * 4
    v_arr = np.random.randn(N).astype(np.float32)
    d_arr = np.random.randn(N).astype(np.float32)
    a_arr = np.zeros(N).astype(np.float32)
    reference = np.zeros(N).astype(np.float32)

    @ti.kernel
    def calc(
        v: ti.types.ndarray(ndim=1),
        d: ti.types.ndarray(ndim=1),
        a: ti.types.ndarray(ndim=1),
    ):
        for i in range(N):
            acc = 0.0
            v_val = v[i]
            for j in range(N):
                acc += v_val * d[j]
            a[i] = acc

    @ti.kernel
    def calc_shared_array(
        v: ti.types.ndarray(ndim=1),
        d: ti.types.ndarray(ndim=1),
        a: ti.types.ndarray(ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(nBlocks * block_dim * 4):
            tid = i % block_dim
            pad0 = ti.simt.block.SharedArray((block_dim,), ti.f32)
            pad1 = ti.simt.block.SharedArray((block_dim,), ti.f32)
            pad2 = ti.simt.block.SharedArray((block_dim,), ti.f32)
            pad3 = ti.simt.block.SharedArray((block_dim,), ti.f32)
            acc = 0.0
            v_val = v[i]
            for k in range(nBlocks):
                pad0[tid] = d[k * block_dim * 4 + tid]
                pad1[tid] = d[k * block_dim * 4 + block_dim + tid]
                pad2[tid] = d[k * block_dim * 4 + 2 * block_dim + tid]
                pad3[tid] = d[k * block_dim * 4 + 3 * block_dim + tid]
                ti.simt.block.sync()
                for j in range(block_dim):
                    acc += v_val * pad0[j]
                    acc += v_val * pad1[j]
                    acc += v_val * pad2[j]
                    acc += v_val * pad3[j]
                ti.simt.block.sync()
            a[i] = acc

    calc(v_arr, d_arr, reference)
    calc_shared_array(v_arr, d_arr, a_arr)
    assert np.allclose(reference, a_arr, rtol=1e-4)


@test_utils.test(arch=[ti.cuda, ti.vulkan, ti.amdgpu])
def test_shared_array_atomics():
    N = 256
    block_dim = 32

    @ti.kernel
    def atomic_test(out: ti.types.ndarray()):
        ti.loop_config(block_dim=block_dim)
        for i in range(N):
            tid = i % block_dim
            val = tid
            sharr = ti.simt.block.SharedArray((block_dim,), ti.i32)
            sharr[tid] = val
            ti.simt.block.sync()
            sharr[0] += val
            ti.simt.block.sync()
            out[i] = sharr[tid]

    arr = ti.ndarray(ti.i32, (N))
    atomic_test(arr)
    ti.sync()
    sum = block_dim * (block_dim - 1) // 2
    assert arr[0] == sum
    assert arr[32] == sum
    assert arr[128] == sum
    assert arr[224] == sum


@test_utils.test(arch=[ti.cuda])
def test_shared_array_tensor_type():
    data_type = vec4
    block_dim = 16
    N = 64

    y = ti.Vector.field(4, dtype=ti.f32, shape=(block_dim))

    @ti.kernel
    def test():
        ti.loop_config(block_dim=block_dim)
        for i in range(N):
            tid = i % block_dim
            val = ti.Vector([1.0, 2.0, 3.0, 4.0])

            shared_mem = ti.simt.block.SharedArray((block_dim), data_type)
            shared_mem[tid] = val
            ti.simt.block.sync()

            y[tid] += shared_mem[tid]

    test()
    assert (y.to_numpy()[0] == [4.0, 8.0, 12.0, 16.0]).all()


@test_utils.test(arch=[ti.cuda], debug=True)
def test_shared_array_matrix():
    @ti.kernel
    def foo():
        for x in range(10):
            shared = ti.simt.block.SharedArray((10,), dtype=ti.math.vec3)
            shared[x] = ti.Vector([x + 1, x + 2, x + 3])
            assert shared[x].z == x + 3
            assert (shared[x] == ti.Vector([x + 1, x + 2, x + 3])).all()

            print(shared[x].z)
            print(shared[x])

    foo()


@test_utils.test(arch=[ti.cuda, ti.vulkan])
def test_shared_array_rejects_serial_scope():
    @ti.kernel
    def invalid(out: ti.types.ndarray()):
        shared = ti.simt.block.SharedArray((1,), ti.i32)
        for i in range(out.shape[0]):
            shared[0] = i
            ti.simt.block.sync()
            out[i] = shared[0]

    out = ti.ndarray(ti.i32, shape=8)
    with pytest.raises(
        ti.TaichiCompilationError,
        match="SharedArray must be declared inside a parallel Taichi range-for loop",
    ):
        invalid(out)


@test_utils.test(arch=[ti.cuda, ti.vulkan])
def test_shared_array_rejects_serialized_range():
    @ti.kernel
    def invalid():
        ti.loop_config(serialize=True)
        for i in range(8):
            shared = ti.simt.block.SharedArray((1,), ti.i32)
            shared[0] = i

    with pytest.raises(
        ti.TaichiCompilationError,
        match="SharedArray must be declared inside a parallel Taichi range-for loop",
    ):
        invalid()


@test_utils.test(arch=[ti.cuda, ti.vulkan])
def test_shared_array_in_inlined_function_keeps_range_scope():
    block_dim = 32
    size = block_dim * 4

    @ti.func
    def exchange_block_value(i):
        shared = ti.simt.block.SharedArray((1,), ti.i32)
        lane = i % block_dim
        if lane == 0:
            shared[0] = i // block_dim
        ti.simt.block.sync()
        return shared[0]

    @ti.kernel
    def valid(out: ti.types.ndarray()):
        ti.loop_config(block_dim=block_dim)
        for i in range(size):
            out[i] = exchange_block_value(i)

    out = ti.ndarray(ti.i32, shape=size)
    valid(out)
    expected = np.repeat(np.arange(size // block_dim, dtype=np.int32), block_dim)
    assert np.array_equal(out.to_numpy(), expected)


@test_utils.test(arch=[ti.cuda, ti.vulkan])
def test_shared_array_scope_rejected_by_aot_and_graph():
    @ti.kernel
    def invalid_aot():
        shared = ti.simt.block.SharedArray((1,), ti.i32)
        for i in range(8):
            shared[0] = i
            ti.simt.block.sync()

    module = ti.aot.Module()
    with pytest.raises(
        SyntaxError,
        match="SharedArray must be declared inside a parallel Taichi range-for loop",
    ):
        module.add_kernel(invalid_aot)

    @ti.kernel
    def invalid_graph():
        shared = ti.simt.block.SharedArray((1,), ti.i32)
        for i in range(8):
            shared[0] = i
            ti.simt.block.sync()

    builder = ti.graph.GraphBuilder()
    builder.dispatch(invalid_graph)
    with pytest.raises(
        SyntaxError,
        match="SharedArray must be declared inside a parallel Taichi range-for loop",
    ):
        builder.compile()
