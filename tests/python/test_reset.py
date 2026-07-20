import gc

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils
from taichi_forge.lang import impl


@test_utils.test(arch=[ti.vulkan])
def test_reset_invalidates_live_runtime_wrappers():
    arr = ti.ndarray(ti.i32, shape=32)
    arr.fill(7)
    vec = ti.Vector.ndarray(2, ti.f32, shape=8)
    vec.fill([1.0, 2.0])
    tex = ti.Texture(ti.Format.rgba8, (8, 8))
    pack_type = ti.types.argpack(a=ti.i32, b=ti.f32)
    pack = pack_type(a=1, b=2.0)

    ti.reset()
    del arr, vec, tex, pack
    gc.collect()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_reset_rejects_sparse_matrix_and_keeps_native_owner_alive():
    row_offsets_host = np.asarray([0, 1], dtype=np.int32)
    column_indices_host = np.asarray([0], dtype=np.int32)
    values_host = np.eye(2, dtype=np.float32).reshape(-1)
    row_offsets = ti.ndarray(ti.i32, shape=2)
    column_indices = ti.ndarray(ti.i32, shape=1)
    values = ti.ndarray(ti.f32, shape=4)
    row_offsets.from_numpy(row_offsets_host)
    column_indices.from_numpy(column_indices_host)
    values.from_numpy(values_host)
    prog = impl.get_runtime().prog
    arch = impl.current_cfg().arch
    try:
        if arch == ti.cpu:
            core = prog._create_cpu_bsr_matrix(
                1, 1, 2, row_offsets.arr, column_indices.arr, values.arr
            )
        elif arch == ti.cuda:
            core = prog._create_cuda_bsr_matrix(
                1, 1, 2, row_offsets.arr, column_indices.arr, values.arr
            )
        else:
            core = prog._create_vulkan_bsr_matrix(
                1, 1, 2, row_offsets.arr, column_indices.arr, values.arr
            )
    except RuntimeError as exc:
        if arch == ti.cuda and "does not support generic BSR SpMV" in str(
            exc
        ):
            pytest.skip("loaded cuSPARSE provider lacks generic BSR SpMV")
        raise
    matrix = ti.linalg.SparseMatrix(sm=core)
    del core

    ti.reset()
    ti.init(arch=arch)
    with pytest.raises(
        TaichiRuntimeError,
        match="SparseMatrix cannot be used after its Taichi runtime has been reset",
    ):
        matrix._debug_runtime_stats()

    del matrix
    gc.collect()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_reset_keeps_shared_sparse_pattern_owner_alive_and_rejects_rebind():
    row_offsets = ti.ndarray(ti.i32, shape=2)
    column_indices = ti.ndarray(ti.i32, shape=1)
    values = ti.ndarray(ti.f32, shape=4)
    row_offsets.from_numpy(np.asarray([0, 1], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0], dtype=np.int32))
    values.from_numpy(np.eye(2, dtype=np.float32).reshape(-1))
    prog = impl.get_runtime().prog
    arch = impl.current_cfg().arch
    pattern = prog._create_bsr_pattern(
        1, 1, 2, row_offsets.arr, column_indices.arr
    )
    try:
        core = prog._create_bsr_matrix_from_pattern(pattern, values.arr)
    except RuntimeError as exc:
        if arch == ti.cuda and "does not support generic BSR SpMV" in str(
            exc
        ):
            pytest.skip("loaded cuSPARSE provider lacks generic BSR SpMV")
        raise
    matrix = ti.linalg.SparseMatrix(sm=core)
    del core
    pattern_id = pattern._debug_runtime_stats()["identity"]["pattern_id"]

    ti.reset()
    ti.init(arch=arch)
    with pytest.raises(
        TaichiRuntimeError,
        match="SparseMatrix cannot be used after its Taichi runtime has been reset",
    ):
        matrix._debug_runtime_stats()
    assert (
        pattern._debug_runtime_stats()["identity"]["pattern_id"]
        == pattern_id
    )
    assert pattern._debug_runtime_stats()["lifecycle"][
        "operator_references"
    ] == 1

    new_values = ti.ndarray(ti.f32, shape=4)
    new_values.from_numpy(np.eye(2, dtype=np.float32).reshape(-1))
    with pytest.raises(RuntimeError, match="same Program"):
        impl.get_runtime().prog._create_bsr_matrix_from_pattern(
            pattern, new_values.arr
        )

    del matrix
    gc.collect()
    assert pattern._debug_runtime_stats()["lifecycle"]["operator_references"] == 0
    del pattern
    gc.collect()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_reset_rejects_public_sparse_pattern_and_matrix():
    row_offsets = ti.ndarray(ti.i32, shape=2)
    column_indices = ti.ndarray(ti.i32, shape=1)
    values = ti.ndarray(ti.f32, shape=4)
    row_offsets.from_numpy(np.asarray([0, 1], dtype=np.int32))
    column_indices.from_numpy(np.asarray([0], dtype=np.int32))
    values.from_numpy(np.eye(2, dtype=np.float32).reshape(-1))
    arch = impl.current_cfg().arch
    pattern = ti.linalg.SparsePattern.bsr(1, 1, 2, row_offsets, column_indices)
    try:
        matrix = pattern.matrix(values)
    except RuntimeError as exc:
        if arch == ti.cuda and "does not support generic BSR SpMV" in str(exc):
            pytest.skip("loaded cuSPARSE provider lacks generic BSR SpMV")
        raise

    ti.reset()
    ti.init(arch=arch)
    with pytest.raises(
        TaichiRuntimeError,
        match="SparsePattern cannot be used after its Taichi runtime has been reset",
    ):
        pattern._debug_runtime_stats()
    new_values = ti.ndarray(ti.f32, shape=4)
    with pytest.raises(
        TaichiRuntimeError,
        match="SparsePattern cannot be used after its Taichi runtime has been reset",
    ):
        pattern.matrix(new_values)
    with pytest.raises(
        TaichiRuntimeError,
        match="SparseMatrix cannot be used after its Taichi runtime has been reset",
    ):
        matrix.update_values(new_values)

    del matrix, pattern
    gc.collect()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_reset_retires_inflight_argpack():
    sink = ti.field(ti.i32, shape=())
    pack_type = ti.types.argpack(value=ti.i32)
    pack = pack_type(value=9)

    @ti.kernel
    def consume(value: pack_type):
        sink[None] += value.value

    consume(pack)
    arch = impl.current_cfg().arch
    ti.reset()

    # A new Program may reuse native addresses, but an invalidated Python view
    # must fail before it can resolve against that Program's registry.
    ti.init(arch=arch)

    @ti.kernel
    def consume_after_reset(value: pack_type) -> ti.i32:
        return value.value

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="Cannot submit an ArgPack after its Taichi runtime has been reset",
    ):
        consume_after_reset(pack)
    del pack
    gc.collect()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_reset_rejects_invalidated_ndarray_submission():
    arr = ti.ndarray(ti.i32, shape=1)
    arr.fill(9)
    arch = impl.current_cfg().arch
    ti.reset()
    ti.init(arch=arch)

    @ti.kernel
    def consume_after_reset(value: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        value[0] += 1

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="Cannot submit an Ndarray after its Taichi runtime has been reset",
    ):
        consume_after_reset(arr)
    del arr
    gc.collect()


@test_utils.test(arch=[ti.vulkan])
def test_reset_rejects_invalidated_texture_submission():
    tex = ti.Texture(ti.Format.rgba8, (1, 1))
    ti.reset()
    ti.init(arch=ti.vulkan)

    @ti.kernel
    def consume_after_reset(
        value: ti.types.rw_texture(
            num_dimensions=2, fmt=ti.Format.rgba8, lod=0
        ),
    ):
        value.store(ti.Vector([0, 0]), ti.Vector([1.0, 0.0, 0.0, 1.0]))

    with pytest.raises(
        ti.TaichiRuntimeError,
        match="Cannot submit a Texture after its Taichi runtime has been reset",
    ):
        consume_after_reset(tex)
    del tex
    gc.collect()
